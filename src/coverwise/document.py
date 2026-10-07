"""Sectioning, paragraph/sentence units and RAG chunks with metadata."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import CHUNK_MAX_CHARS

CATEGORY_KEYWORDS = [
    ("exclusions", ["exclusion", "not covered", "what is not covered"]),
    ("waiting", ["waiting period", "waiting periods"]),
    ("limits", ["co-payment", "copayment", "co-pay", "limit"]),
    ("claims", ["claim", "cashless", "reimbursement"]),
    ("renewal", ["renewal", "portability", "grace period", "cancellation", "migration"]),
    ("definitions", ["definition", "glossary"]),
    ("premium", ["premium", "sum insured", "schedule"]),
    ("coverage", ["coverage", "benefit", "what is covered", "scope", "hospitali", "eligibility"]),
]
ITEM_RE = re.compile(r"^(\d{1,2}[\.\)]|\([a-z0-9]{1,3}\)|[a-z][\.\)]|[-*])\s+", re.I)
KEYVAL_RE = re.compile(r"^[A-Z][A-Za-z\- /]{2,35}:\s+\S")
SENT_SPLIT = re.compile(r"(?<!\bCo\.)(?<!\bLtd\.)(?<!\bNo\.)(?<!\bInc\.)(?<!\bRs\.)(?<=[.;])\s+(?=[A-Z(])")
NUMBERED_RE = re.compile(r"^\d{1,2}(\.\d{1,2})*[\.\)]?\s+")


def categorize(title: str) -> str:
    t = title.lower()
    for cat, kws in CATEGORY_KEYWORDS:
        if any(k in t for k in kws):
            return cat
    return "general"


def is_heading(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 70 or s.endswith((".", ";", ",")):
        return False
    body = re.sub(r"^(section|part|chapter)\s+\w+\s*[:\-]?\s*", "", s, flags=re.I)
    body = NUMBERED_RE.sub("", body)
    letters = re.sub(r"[^A-Za-z]", "", body)
    if len(letters) < 3:
        return False
    if body.isupper():
        return True
    has_kw = categorize(body) != "general"
    if has_kw and len(body.split()) <= 6 and (body.istitle() or NUMBERED_RE.match(s) or s.lower().startswith("section")):
        return True
    return False


@dataclass
class Unit:
    text: str
    page: int
    section: str
    category: str
    is_item: bool = False


@dataclass
class PolicyDoc:
    name: str
    pages: list[dict]
    sections: list[dict] = field(default_factory=list)
    units: list[Unit] = field(default_factory=list)
    chunks: list[dict] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n".join(p["text"] for p in self.pages)

    @property
    def n_pages(self) -> int:
        return len(self.pages)


def _split_sections(pages: list[dict]) -> list[dict]:
    sections = [{"title": "Preamble", "category": "general", "page_start": 1, "lines": []}]
    for p in pages:
        for line in p["text"].splitlines():
            if is_heading(line):
                title = line.strip()
                sections.append({"title": title, "category": categorize(title), "page_start": p["page"], "lines": []})
            else:
                sections[-1]["lines"].append((p["page"], line.strip()))
    return [s for s in sections if s["lines"]]


def _units_from_section(sec: dict) -> list[Unit]:
    paras: list[list] = []  # [page, text, is_item]
    for page, line in sec["lines"]:
        starts_item = bool(ITEM_RE.match(line))
        starts_kv = bool(KEYVAL_RE.match(line))
        prev_done = bool(paras) and paras[-1][1].rstrip().endswith((".", ":", ";"))
        if not paras or starts_item or starts_kv or (prev_done and line[:1].isupper()):
            paras.append([page, line, starts_item])
        else:
            paras[-1][1] += " " + line
    units: list[Unit] = []
    for page, text, is_item in paras:
        parts = [text] if is_item else SENT_SPLIT.split(text)
        for part in parts:
            if part.strip():
                units.append(Unit(part.strip(), page, sec["title"], sec["category"], is_item))
    return units


def _chunk_units(units: list[Unit], doc_name: str, max_chars: int) -> list[dict]:
    chunks, buf, size = [], [], 0

    def flush():
        if not buf:
            return
        chunks.append(
            {
                "chunk_id": f"{doc_name}::c{len(chunks) + 1}",
                "policy_name": doc_name,
                "page": buf[0].page,
                "page_end": buf[-1].page,
                "section": buf[0].section,
                "category": buf[0].category,
                "text": " ".join(u.text for u in buf),
            }
        )

    cur_section = None
    for u in units:
        if (u.section != cur_section and buf) or size + len(u.text) > max_chars:
            flush()
            carry = buf[-1:] if (u.section == cur_section and buf) else []
            buf, size = list(carry), sum(len(c.text) for c in carry)
        cur_section = u.section
        buf.append(u)
        size += len(u.text)
    flush()
    return chunks


def build_document(name: str, cleaned_pages: list[dict], max_chars: int = CHUNK_MAX_CHARS) -> PolicyDoc:
    doc = PolicyDoc(name=name, pages=cleaned_pages)
    doc.sections = _split_sections(cleaned_pages)
    for sec in doc.sections:
        doc.units.extend(_units_from_section(sec))
    doc.chunks = _chunk_units(doc.units, name, max_chars)
    return doc
