"""RAG Q&A engine: answers only from the uploaded policy, cites page/section, refuses when unsupported."""

from __future__ import annotations

import re
import warnings

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import RAG_MIN_SENTENCE_SCORE, RAG_TOP_K
from .document import PolicyDoc
from .guardrails import decision_support_reply, is_advice_request, sanitize_output
from .pdf_extract import mask_pii
from .summary import risk_summary_text

# (name, question regex, regex a supporting unit must contain, query expansion)
TOPICS = [
    ("copay", r"co-?\s?pay", r"co-?\s?pay", "co-payment copayment percentage claim"),
    ("icu", r"\bicu\b|intensive care", r"\bicu\b|intensive\s+care", "ICU intensive care charges limited"),
    ("room", r"room\s*rent|room\s+(?:charges|category|cap|limit)", r"room", "room rent room category limit capped"),
    ("ped", r"pre-?\s?existing|\bped\b", r"pre-?\s?existing", "pre-existing diseases waiting period months"),
    ("waiting", r"waiting", r"waiting", "waiting period months"),
    ("maternity", r"maternity|pregnan|delivery|childbirth", r"maternity|delivery|childbirth|newborn", "maternity delivery childbirth newborn"),
    ("child", r"child|newborn|baby|kid|dependent", r"child|newborn|baby|dependent", "dependent children newborn baby covered"),
    ("restoration", r"restor|recharge|reinstat", r"restor|recharge|reinstat", "restoration restore recharge sum insured"),
    ("ncb", r"no[\s-]claim|bonus", r"bonus", "no claim bonus cumulative"),
    ("claim", r"claim|cashless|reimburs", r"claim|cashless|reimburs", "claim cashless reimbursement documents intimation"),
    ("sublimit", r"sub-?\s?limit|capped|cap\s+on", r"sub-?\s?limit|capped|limited\s+to\s+rs", "sub-limit capped per eye per event"),
    ("renewal", r"renew|portab|grace", r"renew|portab|grace", "renewal portability grace period"),
    ("premium", r"premium|cost|price", r"premium", "premium annual payable"),
    ("sum", r"sum\s+insured|coverage\s+amount|how\s+much\s+cover", r"sum\s+insured", "sum insured amount Rs"),
]
EXCLUSION_Q = re.compile(r"exclusion|not covered|excluded|what.*(?:isn'?t|is not)\s+covered", re.I)
SWITCH_Q = re.compile(r"(verify|check|ask|confirm).*(switch|port|renew|buy|moving|change)|before\s+(?:switching|porting|buying)", re.I)
LOW_PREMIUM_Q = re.compile(r"(low|lower|cheap|cheaper).*premium|premium.*(better|cheaper|lower)", re.I)


class PolicyIndex:
    def __init__(self, doc: PolicyDoc, backend: str = "tfidf"):
        self.doc = doc
        self.units = [u for u in doc.units if len(u.text) > 20]
        self.chunks = doc.chunks
        self.vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
        unit_texts = [f"{u.section} {u.text}" for u in self.units]
        self.vec.fit(unit_texts + [c["text"] for c in self.chunks])
        self.U = self.vec.transform(unit_texts)
        self.C = self.vec.transform([c["text"] for c in self.chunks])
        self.backend = "tfidf"
        if backend == "embeddings":  # optional, not exercised in the bundled tests
            try:
                from sentence_transformers import SentenceTransformer

                self._st = SentenceTransformer("all-MiniLM-L6-v2")
                self.Ue = self._st.encode(unit_texts, normalize_embeddings=True)
                self.backend = "embeddings"
            except Exception:
                warnings.warn("sentence-transformers unavailable; using TF-IDF retrieval")

    def score_units(self, query: str) -> np.ndarray:
        q = self.vec.transform([query])
        s = (self.U @ q.T).toarray().ravel()
        if self.backend == "embeddings":
            s = 0.5 * s + 0.5 * (self.Ue @ self._st.encode([query], normalize_embeddings=True)[0])
        return s

    def top_chunks(self, query: str, k: int = RAG_TOP_K) -> list[tuple[dict, float]]:
        s = (self.C @ self.vec.transform([query]).T).toarray().ravel()
        return [(self.chunks[i], float(s[i])) for i in np.argsort(-s)[:k] if s[i] > 0]


def expand_query(q: str) -> tuple[str, list[tuple]]:
    topics = [t for t in TOPICS if re.search(t[1], q, re.I)]
    if any(t[0] == "ped" for t in topics):  # "pre-existing waiting period" is one topic, not two
        topics = [t for t in topics if t[0] != "waiting"]
    return q + " " + " ".join(t[3] for t in topics), topics


def _cite(u, score=0.0) -> dict:
    return {"page": u.page, "section": u.section, "text": u.text, "score": round(float(score), 3)}


def _result(answer, citations=None, grounded=False, refused=False, reason="", mode="extractive", score=0.0):
    return {
        "answer": answer,
        "citations": citations or [],
        "grounded": grounded,
        "refused": refused,
        "reason": reason,
        "mode": mode,
        "retrieval_score": round(float(score), 3),
    }


def _flag_cites(flags: list[dict], n: int = 4) -> list[dict]:
    return [{"page": f["page"], "section": f["field"], "text": f["snippet"] or f["title"], "score": 0.0} for f in flags if f.get("page")][:n]


_STOP = set(TfidfVectorizer(stop_words="english").get_stop_words()) | {"policy", "cover", "covered", "does", "have", "there"}


def term_coverage(question: str, texts: list[str]) -> float:
    """Share of the question's content words that appear (prefix match) in the supporting text."""
    words = [w for w in re.findall(r"[a-z]{4,}", question.lower()) if w not in _STOP]
    if not words:
        return 1.0
    blob = " ".join(texts).lower()
    found = sum(1 for w in words if w[:5] in blob)
    return found / len(words)


NOT_FOUND = "Not clearly found in the uploaded policy document, so I won't guess. Please ask the insurer or advisor to confirm it in writing."


def answer_question(
    question: str, index: PolicyIndex, extraction: dict | None = None, flags: list[dict] | None = None, checklist: list[str] | None = None, llm=None
) -> dict:
    q = question.strip()
    if not q:
        return _result("Please type a question about the policy.", refused=True, reason="empty")
    flags = flags or []

    if is_advice_request(q):
        summary = risk_summary_text(extraction, flags) if extraction else None
        return _result(decision_support_reply(summary), _flag_cites(flags), refused=True, reason="advice-request", mode="guardrail")
    if SWITCH_Q.search(q) and checklist:
        return _result(
            "Before switching, consider asking the insurer or advisor:\n" + "\n".join(f"- {c}" for c in checklist[:8]),
            _flag_cites(flags),
            grounded=True,
            reason="checklist",
            mode="guardrail",
        )
    if LOW_PREMIUM_Q.search(q) and extraction:
        return _result(
            "A lower premium is not automatically better value: it can come with waiting periods, co-payment, caps or exclusions. "
            + risk_summary_text(extraction, flags)
            + " Compare these restrictions, not just the premium, and confirm with the insurer. "
            "I can't give a final buy/switch recommendation.",
            _flag_cites(flags),
            grounded=True,
            reason="premium-value",
            mode="guardrail",
        )

    expanded, topics = expand_query(q)
    scores = index.score_units(expanded)
    order = np.argsort(-scores)

    if EXCLUSION_Q.search(q):
        items = [u for u in index.units if u.category == "exclusions"][:10]
        if not items:
            return _result(NOT_FOUND, refused=True, reason="no-exclusion-section")
        cites = [_cite(u) for u in items]
        return _result(
            "Major exclusions listed in the policy:\n"
            + "\n".join(f"- (p.{c['page']}) {c['text']}" for c in cites)
            + "\n\nConfirm the full list and any conditions with the insurer.",
            cites,
            grounded=True,
        )

    missing_topics: list[str] = []
    if topics:  # gate per topic: a supporting unit must actually mention that topic
        hits, seen = [], set()
        for name, _, unit_rx, _ in topics:
            rx = re.compile(unit_rx, re.I)
            cand = [i for i in order if rx.search(index.units[i].text) and i not in seen][:2]
            if not cand:
                missing_topics.append(name)
            for i in cand:
                seen.add(i)
                hits.append((index.units[i], float(scores[i])))
        if not hits:
            return _result(
                NOT_FOUND,
                refused=True,
                reason=f"topic-not-in-document:{','.join(missing_topics)}",
                score=float(scores[order[0]]) if len(order) else 0,
            )
        hits = hits[:4]
    else:
        best = float(scores[order[0]]) if len(order) else 0.0
        if best < RAG_MIN_SENTENCE_SCORE:
            return _result(NOT_FOUND, refused=True, reason="low-retrieval-score", score=best)
        hits = [(index.units[i], float(scores[i])) for i in order[:3] if scores[i] >= max(RAG_MIN_SENTENCE_SCORE, 0.6 * scores[order[0]])]
        if term_coverage(q, [u.text for u, _ in hits]) < 0.6:  # retrieved text shares too little with the question
            return _result(NOT_FOUND, refused=True, reason="low-term-coverage", score=best)
    cites = [_cite(u, s) for u, s in hits]

    if llm is not None:
        ctx = "\n".join(f"[p.{c['page']} | {c['section']}] {mask_pii(c['text'])}" for c, _ in index.top_chunks(expanded))
        try:
            out = llm.generate_text(
                "Answer ONLY from the policy excerpts. Cite as [p.N]. If the excerpts do not state the answer, reply exactly: NOT_FOUND. "
                "Explain in simple English. Never recommend buying, switching, renewing or cancelling.",
                f"Policy excerpts:\n{ctx}\n\nQuestion: {q}",
            )
            if "NOT_FOUND" in out:
                return _result(NOT_FOUND, refused=True, reason="llm-not-found", mode="llm")
            out = sanitize_output(out)
            if not re.search(r"\[p\.\s*\d+", out):
                out += "\n\nSources: " + ", ".join(f"p.{c['page']} ({c['section']})" for c in cites)
            return _result(out + "\n\nPlease confirm with the insurer.", cites, grounded=True, mode="llm", score=hits[0][1])
        except Exception:
            pass  # fall back to extractive answer
    body = "\n".join(f"- (p.{c['page']}, {c['section']}) {c['text']}" for c in cites)
    gap = (
        ("\n\nNot clearly found in the document: " + ", ".join(missing_topics) + ". Please ask the insurer to confirm it in writing.")
        if missing_topics
        else ""
    )
    return _result(
        f"From the policy document:\n{body}{gap}\n\nPlease confirm with the insurer before relying on this.", cites, grounded=True, score=hits[0][1]
    )
