"""PDF ingestion: text extraction (with OCR fallback), cleaning and PII masking."""

from __future__ import annotations

import io
import re
from collections import Counter
from pathlib import Path
from typing import BinaryIO, Union

import pdfplumber

Source = Union[str, Path, bytes, BinaryIO]


def _to_bytes(src: Source) -> bytes:
    if isinstance(src, (str, Path)):
        return Path(src).read_bytes()
    if isinstance(src, bytes):
        return src
    return src.read()


def _ocr_page(data: bytes, page_no: int, dpi: int = 200) -> str:
    """OCR a single page (1-indexed). Returns '' if OCR dependencies are missing."""
    try:
        import pytesseract
        from pdf2image import convert_from_bytes

        imgs = convert_from_bytes(data, dpi=dpi, first_page=page_no, last_page=page_no)
        return pytesseract.image_to_string(imgs[0]) if imgs else ""
    except Exception:  # pragma: no cover - environment dependent
        return ""


def extract_policy_text(src: Source, ocr_if_needed: bool = True, min_chars: int = 40) -> list[dict]:
    """Extract text page by page.

    Returns ``[{"page": 1, "text": "...", "method": "text" | "ocr" | "empty"}]``.
    Pages with almost no text layer are OCR'd (scanned / image-based PDFs).
    """
    data = _to_bytes(src)
    pages: list[dict] = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            method = "text"
            if len(text.strip()) < min_chars and ocr_if_needed:
                ocr_text = _ocr_page(data, i)
                if len(ocr_text.strip()) > len(text.strip()):
                    text, method = ocr_text, "ocr"
            if not text.strip():
                method = "empty"
            pages.append({"page": i, "text": text, "method": method})
    return pages


_PAGE_NO_RE = re.compile(r"^(page\s*)?\d{1,4}(\s*(of|/)\s*\d{1,4})?$", re.I)


def _norm(line: str) -> str:
    return re.sub(r"\d+", "#", line.lower().strip())


def clean_pages(pages: list[dict]) -> list[dict]:
    """Remove repeated headers/footers, page numbers, odd symbols and hyphenation."""
    n = len(pages)
    counts: Counter = Counter()
    for p in pages:
        lines = [ln.strip() for ln in p["text"].splitlines() if ln.strip()]
        counts.update({_norm(ln) for ln in lines[:2] + lines[-2:]})
    repeated = {k for k, v in counts.items() if n >= 3 and v >= max(3, int(0.6 * n))}

    cleaned = []
    for p in pages:
        out = []
        for raw in p["text"].splitlines():
            ln = raw.replace("\u2022", "- ").replace("\u25cf", "- ").replace("\u00a0", " ")
            ln = re.sub(r"[^\x20-\x7E\u20B9]", " ", ln)  # keep printable ASCII (+ rupee sign)
            ln = re.sub(r"\s{2,}", " ", ln).strip()
            if not ln or _PAGE_NO_RE.match(ln) or _norm(ln) in repeated:
                continue
            out.append(ln)
        text = "\n".join(out)
        text = re.sub(r"(\w)-\n([a-z])", r"\1\2", text)  # join hyphenated line breaks
        cleaned.append({**p, "text": text})
    return cleaned


_PII_RULES = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[EMAIL]"),
    (re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{9}(?!\d)"), "[PHONE]"),
    (re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b"), "[AADHAAR]"),
    (re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"), "[PAN]"),
    (re.compile(r"(?i)\b((?:policy|member|customer|claim|proposal|certificate)\s*(?:no\.?|number|id))\s*[:\-]?\s*[A-Z0-9/\-]{4,}"), r"\1: [ID]"),
    (re.compile(r"(?i)\b((?:policyholder|proposer|insured person|insured member)(?:'s)?\s*name)\s*[:\-]\s*[^\n]+"), r"\1: [NAME]"),
]


def mask_pii(text: str) -> str:
    """Mask personal identifiers before text is sent to an external LLM or stored."""
    for pattern, repl in _PII_RULES:
        text = pattern.sub(repl, text)
    return text
