"""Optional human-review module (COVERWISE_HUMAN_REVIEW=1 or the sidebar toggle).

A reviewer confirms / corrects extracted values and assigns a final risk label. Every review is appended to
data/labelled/review_log.jsonl (document HASH only - never document text or personal data) and can be exported as a
labelled dataset in the training schema, so the scorer can later be retrained on real, human-verified policies.
"""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import LABELLED_DIR, LABELS
from .extractor import recompute_derived
from .features import FEATURE_COLUMNS

REVIEW_FIELDS = {
    "sum_insured": "int",
    "annual_premium": "int",
    "initial_waiting_months": "int",
    "ped_waiting_months": "int",
    "specific_waiting_months": "int",
    "copayment_percentage": "int",
    "room_rent_limit": "bool",
    "icu_limit": "bool",
    "disease_sub_limit_count": "int",
    "exclusion_count": "int",
    "restoration_benefit": "bool",
    "no_claim_bonus": "bool",
    "child_coverage": "bool",
    "maternity_coverage": "bool",
    "pre_hosp_days": "int",
    "post_hosp_days": "int",
}
LOG_PATH = LABELLED_DIR / "review_log.jsonl"
EXPORT_PATH = LABELLED_DIR / "real_labelled.csv"
BAND_MIDPOINT = {"High Risk": 25.0, "Caution Required": 48.0, "Moderate Coverage": 68.0, "Strong Coverage": 88.0}


def parse_correction(field: str, text: str):
    """Parse reviewer text. Returns (ok, value). Blank text means 'leave unchanged' and is not passed here."""
    t = str(text).strip().lower()
    if t in {"none", "not stated", "n/a", "unknown"}:
        return True, None
    kind = REVIEW_FIELDS[field]
    if kind == "bool":
        if t in {"yes", "true", "1", "y"}:
            return True, True
        if t in {"no", "false", "0", "n"}:
            return True, False
        return False, None
    try:
        return True, int(float(t.replace(",", "")))
    except ValueError:
        return False, None


def apply_review(extraction: dict, corrections: dict) -> dict:
    """Return a copy of `extraction` with reviewer values applied (source='human', confidence 1.0)."""
    ex = copy.deepcopy(extraction)
    changed = list(ex.get("reviewed_fields", []))
    for key, value in corrections.items():
        if key not in REVIEW_FIELDS:
            raise KeyError(f"{key} is not a reviewable field")
        f = ex["fields"][key]
        if f["value"] != value:
            f.update(value=value, source="human", confidence=1.0, detail="corrected by reviewer")
            if key not in changed:
                changed.append(key)
    ex["reviewed_fields"] = changed
    return recompute_derived(ex)


def doc_hash(doc) -> str:
    return hashlib.sha256(doc.full_text.encode("utf-8")).hexdigest()


def log_review(
    analysis: dict, original_extraction: dict, reviewer: str, reviewer_label: str, reviewer_score: float | None = None, path: Path = LOG_PATH
) -> dict:
    if reviewer_label not in LABELS:
        raise ValueError(f"label must be one of {LABELS}")
    if not reviewer.strip():
        raise ValueError("reviewer name/id is required")
    corrections = {
        k: {"from": original_extraction["fields"][k]["value"], "to": analysis["extraction"]["fields"][k]["value"]}
        for k in analysis["extraction"].get("reviewed_fields", [])
    }
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reviewer": reviewer.strip(),
        "doc_sha256": doc_hash(analysis["doc"]),
        "n_pages": analysis["doc"].n_pages,
        "policy_type": analysis["meta"]["policy_type"],
        "features": analysis["features"],
        "model_label": analysis["score"]["label"],
        "model_score": analysis["score"]["score"],
        "reviewer_label": reviewer_label,
        "reviewer_score": reviewer_score,
        "corrections": corrections,
        "fields_checked": len(REVIEW_FIELDS),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    return rec


def read_log(path: Path = LOG_PATH) -> list[dict]:
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def export_labelled(log_path: Path = LOG_PATH, out_path: Path = EXPORT_PATH) -> pd.DataFrame:
    """Latest review per document -> training-schema CSV (policy_id, policy_type, features, score, label, source)."""
    latest: dict[str, dict] = {}
    for rec in read_log(log_path):
        latest[rec["doc_sha256"]] = rec
    rows = []
    for sha, r in latest.items():
        score = r["reviewer_score"] if r["reviewer_score"] is not None else BAND_MIDPOINT[r["reviewer_label"]]
        rows.append(
            {
                "policy_id": f"REAL-{sha[:8]}",
                "policy_type": r["policy_type"],
                **{c: r["features"][c] for c in FEATURE_COLUMNS},
                "coverage_clarity_score": score,
                "risk_label": r["reviewer_label"],
                "source": "real_reviewed",
            }
        )
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def review_summary(log_path: Path = LOG_PATH) -> dict:
    """Real-world extraction quality (per-field correction rate) and reviewer-vs-model agreement."""
    recs = read_log(log_path)
    if not recs:
        return {"n_reviews": 0}
    n = len(recs)
    counts = {k: 0 for k in REVIEW_FIELDS}
    for r in recs:
        for k in r["corrections"]:
            counts[k] += 1
    return {
        "n_reviews": n,
        "label_agreement_with_model": round(sum(r["model_label"] == r["reviewer_label"] for r in recs) / n, 3),
        "field_correction_rate": {k: round(v / n, 3) for k, v in counts.items() if v},
        "overall_field_accuracy": round(1 - sum(counts.values()) / (n * len(REVIEW_FIELDS)), 3),
    }
