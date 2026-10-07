"""Feature Builder: converts extracted clauses into numeric features for scoring/ML."""

from __future__ import annotations

from .risk import generate_risk_flags

FEATURE_COLUMNS = [
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
]
# risk_flags_count is derived from the other features (correlation -0.88 with the score), so the
# production ML model excludes it to avoid leakage. It is still shown to users.
MODEL_NUMERIC = [c for c in FEATURE_COLUMNS if c != "risk_flags_count"]
MODEL_FEATURES = MODEL_NUMERIC + ["policy_type"]
DEFAULTS = {
    "sum_insured": 500_000,
    "annual_premium": 15_000,
    "waiting_period_months": 6,
    "pre_existing_disease_waiting_period_months": 24,
    "copayment_percentage": 0,
    "claim_process_clarity_score": 50,
}
KEY_FIELDS_FOR_CITATION = [
    "sum_insured",
    "annual_premium",
    "ped_waiting_months",
    "specific_waiting_months",
    "copayment_percentage",
    "room_rent_limit",
    "icu_limit",
    "exclusion_count",
    "restoration_benefit",
    "no_claim_bonus",
    "child_coverage",
    "maternity_coverage",
]


def build_features(extraction: dict, profile: dict | None = None, overrides: dict | None = None) -> tuple[dict, dict]:
    """Return (feature_row, meta). Missing values are imputed conservatively and reported."""
    F, ov = extraction["fields"], overrides or {}
    imputed: list[str] = []

    def get(key, default=None, name=None):
        val = ov.get(key) if ov.get(key) else F.get(key, {}).get("value")
        if val is None and default is not None:
            imputed.append(name or key)
            return default
        return val

    si = get("sum_insured", DEFAULTS["sum_insured"], "sum_insured")
    prem = get("annual_premium", DEFAULTS["annual_premium"], "annual_premium")
    missing_clauses = sum(F[k]["value"] is None for k in ("copayment_percentage", "ped_waiting_months", "room_rent_limit", "icu_limit"))
    conditional = 1 if "Conditional" in (F["copayment_percentage"].get("detail") or "") else 0
    for k in ("copayment_percentage", "ped_waiting_months", "room_rent_limit", "icu_limit"):
        if F[k]["value"] is None:
            imputed.append(k)

    feats = {
        "sum_insured": si,
        "annual_premium": prem,
        "premium_to_coverage_ratio": round(prem / si, 4) if si else 0.0,
        "waiting_period_months": get("waiting_period_months", DEFAULTS["waiting_period_months"]),
        "pre_existing_disease_waiting_period_months": get("ped_waiting_months", DEFAULTS["pre_existing_disease_waiting_period_months"]),
        "copayment_percentage": get("copayment_percentage", DEFAULTS["copayment_percentage"]),
        "room_rent_limit_present": int(bool(F["room_rent_limit"]["value"])),
        "icu_limit_present": int(bool(F["icu_limit"]["value"])),
        "disease_sub_limit_count": int(F["disease_sub_limit_count"]["value"] or 0),
        "exclusion_count": int(F["exclusion_count"]["value"] or 0),
        "restoration_benefit_present": int(bool(F["restoration_benefit"]["value"])),
        "no_claim_bonus_present": int(bool(F["no_claim_bonus"]["value"])),
        "child_coverage_present": int(bool(F["child_coverage"]["value"])),
        "maternity_coverage_present": int(bool(F["maternity_coverage"]["value"])),
        "claim_process_clarity_score": F["claim_process_clarity_score"]["value"] or DEFAULTS["claim_process_clarity_score"],
        # missing or conditional clauses are themselves "unclear" and count as ambiguity
        "ambiguous_clause_count": int(F["ambiguous_clause_count"]["value"] or 0) + missing_clauses + conditional,
    }
    flags = generate_risk_flags(F, profile)
    feats["risk_flags_count"] = len(flags)
    cited = sum(F[k]["page"] is not None for k in KEY_FIELDS_FOR_CITATION)
    meta = {
        "policy_type": F["policy_type"]["value"] or "Individual Health",
        "imputed": imputed,
        "flags": flags,
        "citation_coverage_score": round(100 * cited / len(KEY_FIELDS_FOR_CITATION), 1),
    }
    return feats, meta
