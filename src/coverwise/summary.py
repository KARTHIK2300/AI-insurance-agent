"""Plain-English policy summary (templated, optional LLM polish) and short risk summaries."""

from __future__ import annotations

from .config import DISCLAIMER
from .guardrails import sanitize_output


def _inr(v):
    return f"Rs. {v:,}" if isinstance(v, (int, float)) else "not found"


def policy_summary_lines(ext: dict, feats: dict, scored: dict, meta: dict) -> list[str]:
    F = ext["fields"]
    v = lambda k: F[k]["value"]  # noqa: E731
    yn = lambda k, yes, no, nf="not found in the document": yes if F[k]["value"] else (no if F[k]["confidence"] >= 0.5 else nf)  # noqa: E731
    L = []
    L.append(
        f"{v('policy_name') or ext['policy_name']} ({v('insurer_name') or 'insurer not found'}) is a {v('policy_type') or 'health'} policy "
        f"with a sum insured of {_inr(v('sum_insured'))} and an annual premium of {_inr(v('annual_premium'))}."
    )
    L.append(
        f"Hospital stays are covered with {v('pre_hosp_days') if v('pre_hosp_days') is not None else 'unstated'} days of pre-hospital and "
        f"{v('post_hosp_days') if v('post_hosp_days') is not None else 'unstated'} days of post-hospital expenses."
    )
    extras = [
        n
        for k, n in (("day_care", "day care"), ("ambulance", "ambulance"), ("domiciliary", "domiciliary care"), ("ayush", "AYUSH"), ("opd", "OPD"))
        if v(k)
    ]
    L.append("Other listed benefits: " + (", ".join(extras) if extras else "none found") + ".")
    parts = []
    if v("initial_waiting_months") is not None:
        parts.append(f"{v('initial_waiting_months')} month initial wait")
    if v("ped_waiting_months") is not None:
        parts.append(f"{v('ped_waiting_months')} months for pre-existing diseases")
    if v("specific_waiting_months") is not None:
        parts.append(f"{v('specific_waiting_months')} months for listed procedures")
    L.append("Waiting periods: " + ("; ".join(parts) if parts else "not found - please verify") + ".")
    cp = F["copayment_percentage"]
    if cp["value"] is None:
        cp_text = "not found - please verify"
    elif cp["value"] == 0:
        cp_text = "no co-payment stated"
    else:
        cp_text = cp["detail"] or f"{cp['value']}% of admissible claims"
    L.append(f"Co-payment: {cp_text}.")
    rr, icu = F["room_rent_limit"], F["icu_limit"]
    L.append(
        f"Room rent: {'capped (' + rr['detail'] + ')' if rr['value'] else 'no cap stated' if rr['value'] is False else 'not clearly found'}; "
        f"ICU: {'capped (' + icu['detail'] + ')' if icu['value'] else 'no cap stated' if icu['value'] is False else 'not clearly found'}."
    )
    L.append(
        f"{v('disease_sub_limit_count') or 0} disease-wise sub-limits"
        + (f" ({F['disease_sub_limit_count']['detail']})" if F["disease_sub_limit_count"].get("detail") else "")
        + f" and {v('exclusion_count') or 0} listed exclusions apply."
    )
    L.append(
        f"Restoration benefit: {yn('restoration_benefit', 'available', 'not offered')}. No-claim bonus: "
        f"{yn('no_claim_bonus', 'available', 'not offered')}."
    )
    mw = v("maternity_waiting_months")
    maternity = yn("maternity_coverage", "covered" + (f" after {mw} months" if mw else ""), "not covered")
    newborn = F["newborn_coverage"].get("detail") or yn("newborn_coverage", "covered", "not covered")
    L.append(f"Family: dependent children {yn('child_coverage', 'covered', 'not covered')}; newborn: {newborn}; maternity {maternity}.")
    L.append(
        f"Claims: cashless {'described' if v('cashless') else 'not described'}, reimbursement "
        f"{'described' if v('reimbursement') else 'not described'} "
        f"(process clarity {v('claim_process_clarity_score')}/100)."
    )
    L.append(
        f"Renewal: grace period {v('grace_period_days') or 'not found'} days; portability {'mentioned' if v('portability') else 'not mentioned'}."
    )
    amb = v("ambiguous_clause_count") or 0
    L.append(f"{amb} clauses use discretionary or vague wording (for example 'as decided by the Company'); these need written clarification.")
    if ext["unclear_fields"]:
        L.append("Could not be confirmed from the document: " + ", ".join(ext["unclear_fields"]) + ".")
    L.append(
        f"Coverage Clarity Score: {scored['score']:.0f}/100 - {scored['label']}"
        + (" / needs advisor review" if scored["label"] == "High Risk" else "")
        + "."
    )
    top = [f["title"] for f in meta["flags"][:3]]
    if top:
        L.append("Main caution points: " + "; ".join(top) + ".")
    L.append("This is a reading aid, not advice. Confirm every point with the insurer or a licensed advisor.")
    return L


def polish_summary(lines: list[str], llm) -> str:
    """Optional: let an LLM smooth the templated lines into plain English (no new facts allowed)."""
    base = "\n".join(lines)
    if llm is None:
        return base
    try:
        out = llm.generate_text(
            "You rewrite insurance policy facts in simple plain English for a non-expert. Use ONLY the facts given. "
            "Do not add facts, numbers, or advice. Never say to buy, switch, renew or cancel. 10-20 short lines.",
            base,
        )
        return sanitize_output(out) if len(out) > 100 else base
    except Exception:
        return base


def risk_summary_text(ext: dict, flags: list[dict]) -> str:
    F = ext["fields"]
    bits = [f"In this document the premium is {_inr(F['annual_premium']['value'])} for a sum insured of {_inr(F['sum_insured']['value'])}."]
    if flags:
        bits.append("Points that need caution: " + "; ".join(f["title"] for f in flags[:4]) + ".")
    if ext["unclear_fields"]:
        bits.append("Not confirmed in the document: " + ", ".join(ext["unclear_fields"]) + ".")
    return " ".join(bits)


DISCLAIMER_TEXT = DISCLAIMER
