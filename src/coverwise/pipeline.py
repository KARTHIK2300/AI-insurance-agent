"""End-to-end orchestration used by the app, scripts and tests."""

from __future__ import annotations

from pathlib import Path

from .comparison import compare_policies
from .document import PolicyDoc, build_document
from .extractor import extract_clauses
from .features import build_features
from .pdf_extract import clean_pages, extract_policy_text
from .risk import advisor_questions
from .scoring import ScoringModel, score_policy
from .summary import policy_summary_lines


def load_policy(name: str, src) -> PolicyDoc:
    pages = clean_pages(extract_policy_text(src))
    return build_document(name, pages)


def analyze(
    doc: PolicyDoc,
    llm=None,
    model: ScoringModel | None = None,
    profile: dict | None = None,
    overrides: dict | None = None,
    extraction: dict | None = None,
) -> dict:
    """Run the analysis. Pass `extraction` to re-score an already extracted (e.g. human-reviewed) policy."""
    extraction = extraction if extraction is not None else extract_clauses(doc, llm)
    feats, meta = build_features(extraction, profile, overrides)
    scored = score_policy(feats, meta["policy_type"], model)
    return {
        "doc": doc,
        "extraction": extraction,
        "features": feats,
        "meta": meta,
        "score": scored,
        "flags": meta["flags"],
        "summary_lines": policy_summary_lines(extraction, feats, scored, meta),
    }


def analyze_pair(cur_doc: PolicyDoc, new_doc: PolicyDoc, llm=None, model=None, profile=None, overrides=None) -> dict:
    ov = overrides or {}
    cur = analyze(cur_doc, llm, model, profile, {"sum_insured": ov.get("current_si"), "annual_premium": ov.get("current_premium")})
    new = analyze(new_doc, llm, model, profile, {"sum_insured": ov.get("proposed_si"), "annual_premium": ov.get("proposed_premium")})
    comparison = compare_policies(cur["extraction"], new["extraction"], profile, ov)
    questions = advisor_questions(cur["flags"] + new["flags"], comparison, profile)
    return {"current": cur, "proposed": new, "comparison": comparison, "questions": questions}


def sample_paths(base: Path) -> tuple[Path, Path]:
    return (base / "sample_current_policy_surecare_plus.pdf", base / "sample_proposed_policy_horizon_max.pdf")


def refresh_pair(result: dict, profile: dict | None = None, overrides: dict | None = None) -> dict:
    """Recompute comparison and advisor questions after one side was re-analysed (e.g. after human review)."""
    if "proposed" in result:
        result["comparison"] = compare_policies(result["current"]["extraction"], result["proposed"]["extraction"], profile, overrides)
        result["questions"] = advisor_questions(result["current"]["flags"] + result["proposed"]["flags"], result["comparison"], profile)
    else:
        result["questions"] = advisor_questions(result["current"]["flags"], None, profile)
    return result
