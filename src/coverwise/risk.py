"""Risk flags and advisor-question generation (transparent, rule-based)."""

from __future__ import annotations

SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def _flag(sev, title, detail, F=None, field=None):
    ev = F.get(field, {}) if (F and field) else {}
    return {"severity": sev, "title": title, "detail": detail, "field": field, "page": ev.get("page"), "snippet": ev.get("snippet")}


def generate_risk_flags(F: dict, profile: dict | None = None) -> list[dict]:
    profile = profile or {}
    v = lambda k: F.get(k, {}).get("value")  # noqa: E731
    flags: list[dict] = []

    copay = v("copayment_percentage")
    cdetail = F.get("copayment_percentage", {}).get("detail") or ""
    if copay is not None and copay >= 20:
        flags.append(
            _flag("high", f"Co-payment up to {copay}%", "You pay this share of every admissible claim. " + cdetail, F, "copayment_percentage")
        )
    elif copay is not None and copay >= 10:
        flags.append(_flag("medium", f"Co-payment of {copay}%", "You pay this share of admissible claims. " + cdetail, F, "copayment_percentage"))
    elif "Conditional" in cdetail:
        flags.append(_flag("medium", "Conditional co-payment", cdetail, F, "copayment_percentage"))

    ped = v("ped_waiting_months")
    if ped is not None and ped >= 48:
        flags.append(
            _flag(
                "high",
                f"Long pre-existing disease waiting period ({ped} months)",
                "Claims linked to pre-existing conditions are not payable during this period.",
                F,
                "ped_waiting_months",
            )
        )
    elif ped is not None and ped >= 36:
        flags.append(
            _flag(
                "medium",
                f"Pre-existing disease waiting period of {ped} months",
                "Claims linked to pre-existing conditions are not payable during this period.",
                F,
                "ped_waiting_months",
            )
        )
    wait = v("waiting_period_months")
    if wait is not None and wait >= 24:
        flags.append(
            _flag(
                "medium",
                f"Specific-disease waiting period of {wait} months",
                "Listed procedures (e.g. cataract, hernia, joint replacement) wait this long.",
                F,
                "specific_waiting_months",
            )
        )

    if v("room_rent_limit") is True:
        flags.append(
            _flag(
                "medium",
                "Room rent is capped",
                (F["room_rent_limit"].get("detail") or "")
                + ". Choosing a costlier room can reduce other payable charges too - confirm 'proportionate deduction'.",
                F,
                "room_rent_limit",
            )
        )
    if v("icu_limit") is True:
        flags.append(_flag("medium", "ICU charges are capped", F["icu_limit"].get("detail") or "", F, "icu_limit"))
    subs = v("disease_sub_limit_count")
    if subs is not None and subs >= 3:
        flags.append(
            _flag(
                "high" if subs >= 5 else "medium",
                f"{subs} disease-wise sub-limits",
                F["disease_sub_limit_count"].get("detail") or "",
                F,
                "disease_sub_limit_count",
            )
        )
    ex = v("exclusion_count")
    if ex is not None and ex >= 15:
        flags.append(
            _flag("medium", f"Long exclusion list ({ex} items)", "Read each exclusion against your family's health situation.", F, "exclusion_count")
        )
    amb = v("ambiguous_clause_count")
    if amb is not None and amb >= 3:
        flags.append(
            _flag(
                "medium",
                f"{amb} discretionary or vague clauses",
                "Wording such as 'at the Company's discretion' or 'from time to time' can change outcomes.",
                F,
                "ambiguous_clause_count",
            )
        )
    if v("restoration_benefit") is False:
        flags.append(
            _flag("low", "No restoration / recharge benefit", "Sum insured is not refilled after it is exhausted.", F, "restoration_benefit")
        )
    clarity = v("claim_process_clarity_score")
    if clarity is not None and clarity < 60:
        flags.append(
            _flag(
                "medium",
                "Claim process is only partly described",
                F["claim_process_clarity_score"].get("detail") or "",
                F,
                "claim_process_clarity_score",
            )
        )

    # profile-specific
    age = profile.get("age")
    if profile.get("planning_child") or profile.get("child_planned"):
        if v("maternity_coverage") is False:
            flags.append(
                _flag(
                    "medium",
                    "Maternity not covered",
                    "You mentioned planning a child; maternity expenses are stated as not covered.",
                    F,
                    "maternity_coverage",
                )
            )
        elif (v("maternity_waiting_months") or 0) >= 24:
            flags.append(
                _flag(
                    "medium",
                    f"Maternity waiting period of {v('maternity_waiting_months')} months",
                    "Check this against your family planning timeline.",
                    F,
                    "maternity_waiting_months",
                )
            )
    if age and age >= 55 and ped is not None and ped >= 36:
        flags.append(
            _flag(
                "medium",
                "Pre-existing waiting period matters more at your age",
                f"At age {age}, a {ped}-month wait can overlap years in which claims are more likely.",
                F,
                "ped_waiting_months",
            )
        )
    if age and age >= 60 and "Conditional" in cdetail:
        flags.append(_flag("high", "Age-linked co-payment may apply to you", cdetail, F, "copayment_percentage"))

    for key in ("sum_insured", "annual_premium", "ped_waiting_months", "copayment_percentage", "room_rent_limit", "icu_limit"):
        if v(key) is None:
            from .extractor import FIELD_LABELS

            flags.append(
                _flag(
                    "medium",
                    f"Not found in document: {FIELD_LABELS[key]}",
                    "The clause could not be located. Ask the insurer to confirm it in writing.",
                    F,
                    key,
                )
            )
    flags.sort(key=lambda f: SEVERITY_ORDER[f["severity"]])
    return flags


QUESTION_BY_FIELD = {
    "copayment_percentage": "Does the co-payment apply to every claim, or only for certain ages, zones or treatments? Can it be waived?",
    "ped_waiting_months": "From which date does the pre-existing disease waiting period count, and which conditions does the insurer treat as "
    "pre-existing?",
    "specific_waiting_months": "Which exact procedures carry the specific-disease waiting period?",
    "room_rent_limit": "Is there proportionate deduction on other charges if my room costs more than the limit?",
    "icu_limit": "Is there any separate cap on ICU charges? Please confirm in writing.",
    "disease_sub_limit_count": "What is the exact rupee sub-limit for each listed treatment, and is it per eye / per event / per year?",
    "exclusion_count": "Which exclusions are most likely to affect my family's current health conditions?",
    "ambiguous_clause_count": "How does the insurer exercise the discretionary clauses (underwriting, 'as decided by the Company')? Are there "
    "examples?",
    "restoration_benefit": "Does the restoration benefit apply to the same illness, and how many times per year?",
    "maternity_coverage": "Is maternity / newborn cover available, after how many months, and with what limit?",
    "maternity_waiting_months": "Can the maternity waiting period be reduced or is it fixed?",
    "claim_process_clarity_score": "What are the cashless and reimbursement time limits, and which documents are mandatory?",
    "sum_insured": "What sum insured and premium apply to my exact age and family combination?",
    "annual_premium": "Is the quoted premium inclusive of taxes, and how is it expected to change at renewal?",
}


def advisor_questions(flags: list[dict], comparison: dict | None = None, profile: dict | None = None, limit: int = 12) -> list[str]:
    qs: list[str] = []
    for f in flags:
        q = QUESTION_BY_FIELD.get(f.get("field"))
        if q and q not in qs:
            qs.append(q)
    if comparison:
        qs.append("If I move from my current policy, which waiting periods restart, and how much previous waiting-period credit will be recognised?")
        qs.append("Is the extra premium linked to specific benefits I can verify in the policy wording?")
        if comparison.get("insurer_changes"):
            qs.append("What is the portability application timeline, and what happens to my no-claim bonus?")
    if profile and profile.get("n_children"):
        qs.append("Are all my dependent children eligible, and until what age?")
    qs.append("Please share the final policy wording and customer information sheet so I can verify these points.")
    seen, out = set(), []
    for q in qs:
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out[:limit]
