"""Current vs proposed policy comparison."""

from __future__ import annotations

import re

SPEC = [  # (field, label, kind)   kind: lower/higher = numeric better direction, good/bad = boolean
    ("sum_insured", "Sum insured (Rs.)", "higher"),
    ("annual_premium", "Annual premium (Rs.)", "info"),
    ("pre_hosp_days", "Pre-hospitalisation (days)", "higher"),
    ("post_hosp_days", "Post-hospitalisation (days)", "higher"),
    ("initial_waiting_months", "Initial waiting (months)", "lower"),
    ("ped_waiting_months", "Pre-existing waiting (months)", "lower"),
    ("specific_waiting_months", "Specific-disease waiting (months)", "lower"),
    ("copayment_percentage", "Co-payment (%)", "lower"),
    ("room_rent_limit", "Room rent capped", "bad"),
    ("icu_limit", "ICU charges capped", "bad"),
    ("disease_sub_limit_count", "Disease sub-limits (count)", "lower"),
    ("exclusion_count", "Listed exclusions (count)", "lower"),
    ("restoration_benefit", "Restoration / recharge", "good"),
    ("no_claim_bonus", "No-claim bonus", "good"),
    ("child_coverage", "Dependent child cover", "good"),
    ("newborn_coverage", "Newborn cover", "good"),
    ("maternity_coverage", "Maternity cover", "good"),
    ("maternity_waiting_months", "Maternity waiting (months)", "lower"),
    ("day_care", "Day care", "good"),
    ("ambulance", "Ambulance", "good"),
    ("domiciliary", "Domiciliary", "good"),
    ("ayush", "AYUSH", "good"),
    ("opd", "OPD", "good"),
    ("claim_process_clarity_score", "Claim process clarity (0-100)", "higher"),
    ("ambiguous_clause_count", "Ambiguous clauses (count)", "lower"),
    ("grace_period_days", "Grace period (days)", "higher"),
]


def _read(f: dict, kind: str):
    """Return (value, state) where state is 'known' | 'unknown' | 'not_mentioned'."""
    v = f["value"]
    if v is None:
        return None, "unknown"
    if kind in ("good", "bad") and v is False and f["confidence"] < 0.5:
        return False, "not_mentioned"
    return v, "known"


def _show(v, state: str, kind: str, f: dict) -> str:
    if state == "unknown":
        return "Not found"
    if state == "not_mentioned":
        return "Not mentioned"
    if isinstance(v, bool):
        return "Yes" if v else "No"
    if isinstance(v, int) and v >= 1000:
        return f"{v:,}"
    txt = str(v)
    if "Conditional" in (f.get("detail") or ""):
        txt += " (conditional)"
    return txt


def _verdict(kind, a, sa, b, sb) -> str:
    if kind == "info":
        return "Info"
    if sa == "unknown" or sb == "unknown":
        return "Unclear"
    if kind in ("good", "bad"):
        if kind == "bad":
            a, b = (not a if sa == "known" else a), (not b if sb == "known" else b)
        # now True always means "benefit present"
        if a and b:
            return "Same"
        if not a and b:
            return "Improved"
        if a and not b:
            return "Worse" if sb == "known" else "Unclear"
        return "Same"
    if a == b:
        return "Same"
    better = (b > a) if kind == "higher" else (b < a)
    return "Improved" if better else "Worse"


def compare_policies(cur: dict, new: dict, profile: dict | None = None, overrides: dict | None = None) -> dict:
    profile, overrides = profile or {}, overrides or {}
    Fc, Fn = cur["fields"], new["fields"]
    rows, improved, worse, unclear = [], [], [], []
    for key, label, kind in SPEC:
        a, sa = _read(Fc[key], kind)
        b, sb = _read(Fn[key], kind)
        verdict = _verdict(kind, a, sa, b, sb)
        if "Conditional" in (Fn[key].get("detail") or "") or "Conditional" in (Fc[key].get("detail") or ""):
            verdict = "Unclear" if verdict in ("Same", "Improved") else verdict
        if key == "newborn_coverage" and verdict == "Same":  # both cover newborns - compare *when* it starts
            ta, tb = (Fc[key].get("detail") or ""), (Fn[key].get("detail") or "")
            if ta != tb and "day 1" in tb and "next renewal" in ta:
                verdict = "Improved"
            elif ta != tb and "day 1" in ta and "next renewal" in tb:
                verdict = "Worse"
            elif ta != tb:
                verdict = "Unclear"
        note = Fn[key].get("detail") if key in ("copayment_percentage", "newborn_coverage") else None
        cur_txt, new_txt = _show(a, sa, kind, Fc[key]), _show(b, sb, kind, Fn[key])
        if key == "newborn_coverage":
            cur_txt = (Fc[key].get("detail") or cur_txt).replace("Newborn cover ", "")
            new_txt = (Fn[key].get("detail") or new_txt).replace("Newborn cover ", "")
        rows.append(
            {
                "field": key,
                "label": label,
                "current": cur_txt,
                "proposed": new_txt,
                "verdict": verdict,
                "note": note,
                "current_page": Fc[key]["page"],
                "proposed_page": Fn[key]["page"],
            }
        )
        {"Improved": improved, "Worse": worse, "Unclear": unclear}.get(verdict, []).append(label)

    pc = overrides.get("current_premium") or Fc["annual_premium"]["value"]
    pn = overrides.get("proposed_premium") or Fn["annual_premium"]["value"]
    sc = overrides.get("current_si") or Fc["sum_insured"]["value"]
    sn = overrides.get("proposed_si") or Fn["sum_insured"]["value"]
    premium = {"current_premium": pc, "proposed_premium": pn, "current_si": sc, "proposed_si": sn}
    if pc and pn and sc and sn:
        premium.update(
            premium_change_pct=round(100 * (pn - pc) / pc, 1),
            si_change_pct=round(100 * (sn - sc) / sc, 1),
            per_lakh_current=round(pc / sc * 1e5),
            per_lakh_proposed=round(pn / sn * 1e5),
        )
        n_imp, n_worse = len(improved), len(worse)
        premium["message"] = (
            f"Premium changes by {premium['premium_change_pct']:+.0f}% while the sum insured changes by {premium['si_change_pct']:+.0f}%. "
            f"Premium per Rs. 1 lakh of cover moves from Rs. {premium['per_lakh_current']:,} to Rs. {premium['per_lakh_proposed']:,}. "
            f"{n_imp} compared items look better and {n_worse} look worse in the proposed policy - weigh these against what matters for your family."
        )
    else:
        premium["message"] = "Premium or sum insured was not found in one of the documents - enter them manually to compare value."

    insurer_changes = (Fc["insurer_name"]["value"] or "") != (Fn["insurer_name"]["value"] or "")
    switching = [
        "A new policy usually starts a fresh 30-day initial waiting period unless continuity / portability credit is granted - verify in writing."
    ]
    if sn and sc and sn > sc:
        switching.append(
            "Waiting-period credit is typically recognised only up to the previous sum insured; the extra sum insured may carry fresh waiting "
            "periods - confirm with the insurer."
        )
    ped_c, ped_n = Fc["ped_waiting_months"]["value"], Fn["ped_waiting_months"]["value"]
    if ped_c is not None and ped_n is not None and ped_n > ped_c:
        switching.append(
            f"The proposed policy states a longer pre-existing waiting period ({ped_n} vs {ped_c} months). Ask how much of your served waiting "
            f"period is credited."
        )
    if insurer_changes:
        switching.append("The insurer changes. Check the portability application timeline and what happens to accrued no-claim bonus.")

    cautions = []
    age, cn = profile.get("age"), Fn["copayment_percentage"]
    if age and age >= 60 and "Conditional" in (cn.get("detail") or ""):
        cautions.append(f"At age {age}, the proposed policy's age-linked co-payment ({cn['detail']}) may apply to you.")
    if profile.get("planning_child") or profile.get("child_planned"):
        mw = Fn["maternity_waiting_months"]["value"]
        if Fn["maternity_coverage"]["value"] and mw:
            cautions.append(
                f"You mentioned planning a child: the proposed policy's maternity waiting period is {mw} months; the current policy covers "
                f"maternity: {Fc['maternity_coverage']['value']}."
            )
        elif Fn["maternity_coverage"]["value"] is False:
            cautions.append("You mentioned planning a child but maternity is stated as not covered in the proposed policy.")
    m = re.search(r"up to (\d+) dependent children", (Fn["members_covered"]["value"] or ""), re.I)
    if m and profile.get("n_children") and profile["n_children"] > int(m.group(1)):
        cautions.append(f"The proposed policy lists up to {m.group(1)} dependent children; you entered {profile['n_children']}.")
    if ped_n is not None and profile.get("has_pre_existing") and ped_n >= 36:
        cautions.append("You indicated a pre-existing condition; the proposed waiting period is long - ask how it is counted after switching.")

    return {
        "rows": rows,
        "improved": improved,
        "worse": worse,
        "unclear": unclear,
        "premium": premium,
        "switching_risks": switching,
        "user_cautions": cautions,
        "insurer_changes": insurer_changes,
    }
