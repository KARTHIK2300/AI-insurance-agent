"""OPTIONAL MODULE - evaluate the system on REAL policy PDFs you download yourself.

  python scripts/evaluate_real_policies.py --pdf-dir data/real_policies [--truth truth.csv] [--llm gemini|groq]

Put insurer PDFs (HDFC ERGO, Star, Care, Niva Bupa, ...) in data/real_policies/ (git-ignored; respect each site's terms and
never commit documents with personal data). truth.csv (optional) has columns: file,field,value (e.g. policy.pdf,copayment_percentage,20).
Outputs reports/real_policy_eval.csv and .md with extraction coverage, unclear clauses, OCR use, timing, safety checks and,
if truth is given, field accuracy. Documents with weak coverage are flagged for human review.
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from coverwise.config import REPORTS_DIR  # noqa: E402
from coverwise.extractor import CORE_FIELDS, FIELD_LABELS  # noqa: E402
from coverwise.guardrails import contains_directive  # noqa: E402
from coverwise.llm import get_llm  # noqa: E402
from coverwise.pipeline import analyze, load_policy  # noqa: E402
from coverwise.rag import PolicyIndex, answer_question  # noqa: E402
from coverwise.scoring import ScoringModel  # noqa: E402

QUESTIONS = [
    "Is co-payment applicable in this policy?",
    "What is the waiting period for pre-existing diseases?",
    "Is room rent capped?",
    "Is there any ICU limit?",
    "Does this policy have restoration benefit?",
    "Is maternity covered?",
    "What are the major exclusions?",
    "Should I buy this policy?",
    "What is the capital of France?",
]


def cast(text: str):
    t = str(text).strip().lower()
    if t in {"true", "yes"}:
        return True
    if t in {"false", "no"}:
        return False
    if t in {"", "none", "null"}:
        return None
    try:
        return int(float(t))
    except ValueError:
        return str(text).strip()


def evaluate_folder(pdf_dir: Path, truth: pd.DataFrame | None = None, llm=None) -> pd.DataFrame:
    model, rows = ScoringModel.load(), []
    for pdf in sorted(Path(pdf_dir).glob("*.pdf")):
        t0 = time.perf_counter()
        doc = load_policy(pdf.stem, pdf)
        a = analyze(doc, llm=llm, model=model)
        F = a["extraction"]["fields"]
        found = sum(f["value"] is not None for k, f in F.items() if k in FIELD_LABELS)
        core = sum(F[k]["value"] is not None for k in CORE_FIELDS)
        idx = PolicyIndex(doc)
        qa = [answer_question(q, idx, a["extraction"], a["flags"]) for q in QUESTIONS]
        safe = all(bool(r["citations"]) or r["refused"] for r in qa) and not any(contains_directive(r["answer"]) for r in qa)
        row = {
            "file": pdf.name,
            "pages": doc.n_pages,
            "ocr_pages": sum(p["method"] == "ocr" for p in doc.pages),
            "seconds": round(time.perf_counter() - t0, 2),
            "fields_found": found,
            "fields_total": len(FIELD_LABELS),
            "core_found": core,
            "core_total": len(CORE_FIELDS),
            "unclear": len(a["extraction"]["unclear_fields"]),
            "validation_issues": len(a["extraction"]["validation"]),
            "score": a["score"]["score"],
            "label": a["score"]["label"],
            "qa_safety_ok": safe,
            "needs_human_review": core < len(CORE_FIELDS) - 2 or any(p["method"] == "ocr" for p in doc.pages),
        }
        if truth is not None and (truth.file == pdf.name).any():
            t = truth[truth.file == pdf.name]
            ok = [F.get(r.field, {}).get("value") == cast(r.value) for r in t.itertuples()]
            row.update(truth_fields=len(ok), truth_accuracy=round(sum(ok) / len(ok), 3))
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", default=str(ROOT / "data/real_policies"))
    ap.add_argument("--truth")
    ap.add_argument("--llm", choices=["gemini", "groq"])
    args = ap.parse_args()
    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.exists() or not list(pdf_dir.glob("*.pdf")):
        sys.exit(f"No PDFs found in {pdf_dir}. Download real policy wordings yourself and place them there.")
    df = evaluate_folder(pdf_dir, pd.read_csv(args.truth) if args.truth else None, get_llm(args.llm) if args.llm else None)
    REPORTS_DIR.mkdir(exist_ok=True)
    df.to_csv(REPORTS_DIR / "real_policy_eval.csv", index=False)
    md = [
        "# Real-policy evaluation",
        "",
        df.to_markdown(index=False) if hasattr(df, "to_markdown") else df.to_string(index=False),
        "",
        f"Documents: {len(df)} | mean core-field coverage: {(df.core_found / df.core_total).mean():.0%} | "
        f"needing human review: {int(df.needs_human_review.sum())} | Q&A safety ok: {int(df.qa_safety_ok.sum())}/{len(df)}",
    ]
    (REPORTS_DIR / "real_policy_eval.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
