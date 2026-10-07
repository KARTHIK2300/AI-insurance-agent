"""MLOps helpers shared by CI scripts and tests: metric extraction, quality gates, data validation."""

from __future__ import annotations

import operator
from pathlib import Path

import pandas as pd

from .config import LABELS
from .scoring import score_to_label

OPS = {">=": operator.ge, "<=": operator.le, "==": operator.eq}

REQUIRED_COLUMNS = [
    "policy_id",
    "policy_type",
    "sum_insured",
    "annual_premium",
    "premium_to_coverage_ratio",
    "waiting_period_months",
    "pre_existing_disease_waiting_period_months",
    "copayment_percentage",
    "room_rent_limit_present",
    "icu_limit_present",
    "disease_sub_limit_count",
    "exclusion_count",
    "restoration_benefit_present",
    "no_claim_bonus_present",
    "child_coverage_present",
    "maternity_coverage_present",
    "claim_process_clarity_score",
    "ambiguous_clause_count",
    "risk_flags_count",
    "coverage_clarity_score",
    "risk_label",
]
BINARY = [
    "room_rent_limit_present",
    "icu_limit_present",
    "restoration_benefit_present",
    "no_claim_bonus_present",
    "child_coverage_present",
    "maternity_coverage_present",
]


# ------------------------------------------------------------------ data validation
def validate_dataframe(df: pd.DataFrame, min_class_rows: int = 20, min_band_agreement: float = 0.85) -> dict:
    errors, warnings = [], []
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return {"ok": False, "errors": [f"missing columns: {missing}"], "warnings": [], "rows": len(df)}
    if df[REQUIRED_COLUMNS].isna().any().any():
        errors.append("null values in required columns")
    if not df.policy_id.is_unique:
        errors.append("duplicate policy_id")
    bad_labels = set(df.risk_label) - set(LABELS)
    if bad_labels:
        errors.append(f"unknown labels: {sorted(bad_labels)}")
    for c in BINARY:
        if not set(df[c].unique()) <= {0, 1}:
            errors.append(f"{c} must be 0/1")
    ranges = {
        "copayment_percentage": (0, 100),
        "coverage_clarity_score": (0, 100),
        "claim_process_clarity_score": (0, 100),
        "waiting_period_months": (0, 120),
        "pre_existing_disease_waiting_period_months": (0, 120),
    }
    for c, (lo, hi) in ranges.items():
        if not df[c].between(lo, hi).all():
            errors.append(f"{c} outside [{lo}, {hi}]")
    if not ((df.sum_insured > 0) & (df.annual_premium > 0)).all():
        errors.append("sum_insured / annual_premium must be positive")
    if (((df.annual_premium / df.sum_insured) - df.premium_to_coverage_ratio).abs() > 0.002).any():
        errors.append("premium_to_coverage_ratio inconsistent with premium / sum insured")
    counts = df.risk_label.value_counts()
    for lab in LABELS:
        if counts.get(lab, 0) < min_class_rows:
            errors.append(f"class '{lab}' has {counts.get(lab, 0)} rows (< {min_class_rows})")
    agreement = float((df.risk_label == df.coverage_clarity_score.map(score_to_label)).mean())
    if agreement < min_band_agreement:
        errors.append(f"labels disagree with score bands too often ({agreement:.2f} < {min_band_agreement})")
    n_ped = int((df.pre_existing_disease_waiting_period_months < df.waiting_period_months).sum())
    if n_ped:
        warnings.append(f"{n_ped} rows have PED waiting shorter than initial waiting (kept, flagged in prepare_data)")
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "rows": len(df),
        "label_counts": counts.to_dict(),
        "band_agreement": round(agreement, 3),
    }


# ------------------------------------------------------------------ quality gates
def production_metrics(m: dict) -> dict:
    method = m["production_method"]  # e.g. "Gradient Boosting regressor -> score bands"
    reg_name = method.split(" regressor")[0]
    cls = m["classifiers"][f"{method} | core"]
    return {
        "macro_f1": cls["macro_f1"],
        "accuracy": cls["accuracy"],
        "score_mae": m["regressors"][reg_name]["mae"],
        "score_r2": m["regressors"][reg_name]["r2"],
        "high_risk_recall": m["per_class_production"]["High Risk"]["recall"],
    }


def evaluate_gates(metrics: dict, gates: dict, baseline: dict | None, evaluation: dict | None, model_features: list[str]) -> list[dict]:
    """Return a list of {gate, actual, op, threshold, passed}. Missing optional evidence fails closed."""
    pm = production_metrics(metrics)
    g, checks = gates["model"], []

    def add(name, actual, op, threshold):
        ok = actual is not None and OPS[op](actual, threshold)
        checks.append(
            {"gate": name, "actual": actual if actual is None else round(float(actual), 4), "op": op, "threshold": threshold, "passed": bool(ok)}
        )

    add("model.macro_f1", pm["macro_f1"], ">=", g["min_macro_f1"])
    add("model.accuracy", pm["accuracy"], ">=", g["min_accuracy"])
    add("model.score_mae", pm["score_mae"], "<=", g["max_score_mae"])
    add("model.score_r2", pm["score_r2"], ">=", g["min_r2"])
    add("model.high_risk_recall", pm["high_risk_recall"], ">=", g["min_high_risk_recall"])
    for banned in g["forbid_features"]:
        add(f"model.no_{banned}_feature", int(banned in model_features), "==", 0)
    if baseline:
        add("regression.macro_f1_vs_baseline", pm["macro_f1"] - baseline["macro_f1"], ">=", -g["max_macro_f1_drop_vs_baseline"])
        add("regression.score_mae_vs_baseline", pm["score_mae"] - baseline["score_mae"], "<=", g["max_mae_increase_vs_baseline"])
    p = gates["pipeline"]
    if evaluation is None:
        add("pipeline.evaluation_report_present", 0, "==", 1)
    else:
        ex = evaluation["extraction_field_accuracy"]
        add("pipeline.extraction_accuracy_min", min(ex.values()), ">=", p["min_extraction_accuracy"])
        if "field_accuracy" in evaluation.get("ocr", {}):
            add("pipeline.ocr_field_accuracy", evaluation["ocr"]["field_accuracy"], ">=", p["min_ocr_field_accuracy"])
        q = evaluation["policy_qa"]
        add("qa.answer_correctness", q["answer_correctness"], ">=", p["min_qa_correctness"])
        add("qa.citation_page_accuracy", q["citation_page_accuracy"], ">=", p["min_citation_accuracy"])
        add("qa.hallucination_rate_on_unsupported", q["hallucination_rate_on_unsupported"], "<=", p["max_hallucination_rate"])
        add("safety.directive_language_rate", q["directive_language_rate"], "<=", p["max_directive_rate"])
        t = evaluation["provided_testset_12_unique_questions_x2_policies"]
        add("safety.advice_questions_refused", t["advice_questions_refused_correctly"], ">=", p["min_advice_refusal_rate"])
        add("safety.provided_testset_directive_rate", t["directive_language_rate"], "<=", p["max_directive_rate"])
        add("qa.provided_testset_citation_or_honest", t["citation_or_honest_not_found_rate"], ">=", p["min_provided_testset_safe_rate"])
    return checks


def gates_markdown(checks: list[dict]) -> str:
    lines = ["| Gate | Actual | Rule | Result |", "|---|---|---|---|"]
    for c in checks:
        lines.append(f"| {c['gate']} | {c['actual']} | {c['op']} {c['threshold']} | {'PASS' if c['passed'] else '**FAIL**'} |")
    return "\n".join(lines)


def load_json(path: Path):
    import json

    return json.loads(Path(path).read_text()) if Path(path).exists() else None
