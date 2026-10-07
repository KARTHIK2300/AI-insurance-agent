"""Clause extraction: transparent rule-based baseline + optional LLM extraction.

Every extracted field carries evidence (page, section, snippet). LLM values are only accepted when
the supporting quote can be found in the document, and rule-based values win on conflict.
"""

from __future__ import annotations

import re

from .document import PolicyDoc, Unit
from .guardrails import sanitize_output  # noqa: F401  (re-exported for convenience)
from .pdf_extract import mask_pii

CORE_FIELDS = [
    "sum_insured",
    "annual_premium",
    "ped_waiting_months",
    "copayment_percentage",
    "room_rent_limit",
    "icu_limit",
    "exclusion_count",
    "restoration_benefit",
]
FIELD_LABELS = {
    "insurer_name": "Insurer",
    "policy_name": "Policy name",
    "policy_type": "Policy type",
    "sum_insured": "Sum insured (Rs.)",
    "annual_premium": "Annual premium (Rs.)",
    "policy_term": "Policy term",
    "entry_age": "Entry age",
    "members_covered": "Members covered",
    "initial_waiting_months": "Initial waiting period (months)",
    "ped_waiting_months": "Pre-existing disease waiting (months)",
    "specific_waiting_months": "Specific disease waiting (months)",
    "copayment_percentage": "Co-payment (%)",
    "room_rent_limit": "Room rent limit present",
    "icu_limit": "ICU limit present",
    "disease_sub_limit_count": "Disease-wise sub-limits (count)",
    "exclusion_count": "Listed exclusions (count)",
    "restoration_benefit": "Restoration / recharge benefit",
    "no_claim_bonus": "No-claim bonus",
    "child_coverage": "Dependent child coverage",
    "newborn_coverage": "Newborn coverage",
    "maternity_coverage": "Maternity coverage",
    "maternity_waiting_months": "Maternity waiting (months)",
    "pre_hosp_days": "Pre-hospitalisation (days)",
    "post_hosp_days": "Post-hospitalisation (days)",
    "day_care": "Day care procedures",
    "ambulance": "Ambulance cover",
    "domiciliary": "Domiciliary hospitalisation",
    "ayush": "AYUSH treatment",
    "opd": "OPD cover",
    "cashless": "Cashless claims",
    "reimbursement": "Reimbursement claims",
    "claim_process_clarity_score": "Claim process clarity (0-100)",
    "ambiguous_clause_count": "Ambiguous clauses (count)",
    "grace_period_days": "Grace period (days)",
    "portability": "Portability mentioned",
}
LLM_FIELD_TYPES = {
    "insurer_name": "str",
    "policy_name": "str",
    "policy_type": "str",
    "sum_insured": "int",
    "annual_premium": "int",
    "ped_waiting_months": "int",
    "initial_waiting_months": "int",
    "specific_waiting_months": "int",
    "copayment_percentage": "int",
    "room_rent_limit": "bool",
    "icu_limit": "bool",
    "restoration_benefit": "bool",
    "no_claim_bonus": "bool",
    "child_coverage": "bool",
    "maternity_coverage": "bool",
    "pre_hosp_days": "int",
    "post_hosp_days": "int",
    "day_care": "bool",
    "ambulance": "bool",
    "ayush": "bool",
    "opd": "bool",
}

DISEASES = [
    "cataract",
    "hernia",
    "hysterectomy",
    "joint replacement",
    "knee replacement",
    "kidney stone",
    "calculi",
    "piles",
    "fistula",
    "tonsillectomy",
    "sinus",
    "gallstone",
    "varicose",
    "prostate",
    "angioplasty",
]
AMBIGUOUS_PATTERNS = [
    r"sole\s+discretion|at\s+(?:the\s+)?discretion",
    r"(?:determined|decided|specified)\s+by\s+the\s+(?:company|insurer)",
    r"reasonable\s+and\s+customary",
    r"underwriting",
    r"from\s+time\s+to\s+time",
    r"may\s+(?:be\s+)?(?:declined|rejected|reduced|loaded)",
]
POLICY_TYPES = [
    ("Senior Citizen", r"senior\s+citizen"),
    ("Critical Illness", r"critical\s+illness"),
    ("Top-up Health", r"top[\s-]?up"),
    ("Group Health", r"group\s+(?:health|policy|mediclaim)"),
    ("Family Floater", r"family\s+floater|floater"),
    ("Individual Health", r"individual"),
]


# ----------------------------------------------------------------------------- helpers
def parse_amount(text: str) -> int | None:
    m = re.search(r"(?:rs\.?|inr|\u20b9)\s*([\d,]+(?:\.\d+)?)\s*(lakhs?|lacs?|crores?|cr\b)?", text, re.I)
    if not m:
        m = re.search(r"([\d,]+(?:\.\d+)?)\s*(lakhs?|lacs?|crores?)", text, re.I)
    if not m:
        return None
    num = float(m.group(1).replace(",", "").strip(".") or 0)
    unit = (m.group(2) or "").lower()
    if unit.startswith(("lakh", "lac")):
        num *= 1e5
    elif unit.startswith("cr"):
        num *= 1e7
    return int(num) if num else None


def to_months(num: float, unit: str) -> int:
    unit = unit.lower()
    if unit.startswith("year"):
        return int(round(num * 12))
    if unit.startswith("day"):
        return max(1, int(round(num / 30)))
    return int(round(num))


def _units(doc: PolicyDoc, pattern: str, categories: tuple | None = None, flags=re.I) -> list[tuple[Unit, re.Match]]:
    rx = re.compile(pattern, flags)
    out = []
    for u in doc.units:
        if categories and u.category not in categories:
            continue
        m = rx.search(u.text)
        if m:
            out.append((u, m))
    return out


def _f(value, unit: Unit | None = None, conf: float = 0.9, detail: str | None = None, source: str = "rules") -> dict:
    return {
        "value": value,
        "detail": detail,
        "page": unit.page if unit else None,
        "section": unit.section if unit else None,
        "snippet": unit.text[:280] if unit else None,
        "source": source,
        "confidence": conf if unit else (0.5 if value is not None else 0.0),
    }


def _none() -> dict:
    return _f(None)


# ----------------------------------------------------------------------------- rule-based
def _text_field(doc, label_regex, conf=0.95):
    rx = re.compile(rf"^\s*(?:{label_regex})\s*[:\-]\s*(.+?)\s*$", re.I)
    for u in doc.units:
        m = rx.match(u.text)
        if m:
            val = m.group(1).strip()
            return _f(val if re.search(r"(Ltd|Co|Inc)\.$", val) else val.rstrip("."), u, conf)
    return None


def _amount_field(doc, label_regex) -> dict:
    hits = _units(doc, rf"(?:{label_regex})[^.\n]{{0,60}}?(?:rs\.?|inr|\u20b9|\d[\d,]*\s*(?:lakh|lac|crore))")
    hits.sort(key=lambda h: 0 if re.match(rf"\s*(?:{label_regex})", h[0].text, re.I) else 1)
    for u, _ in hits:
        amt = parse_amount(u.text)
        if amt:
            return _f(amt, u, 0.95)
    return _none()


def _months_field(doc, patterns: list[str]) -> dict:
    for pat in patterns:
        for u, m in _units(doc, pat):
            return _f(to_months(float(m.group(1)), m.group(2)), u, 0.9, detail=m.group(0)[:120])
    return _none()


def _limit_field(doc, keyword_regex: str) -> dict:
    """True = a cap/limit is mentioned, False = explicitly uncapped, None = not found."""
    no_cap = re.compile(
        r"no\s+(?:capping|cap|limit|sub-?limit|restriction)|any\s+room\s+(?:category|type)|without\s+(?:any\s+)?(?:cap|limit)|not\s+(?:capped|limited)",
        re.I,
    )
    capped = re.compile(r"limited\s+to|capped|restricted\s+to|up\s+to\s+(?:rs|\d)|\d+(?:\.\d+)?\s*%\s*of\s+(?:the\s+)?sum\s+insured|per\s+day", re.I)
    found_cap = found_nocap = None
    for u, _ in _units(doc, keyword_regex):
        if u.category == "exclusions":
            continue
        if no_cap.search(u.text) and not found_nocap:
            found_nocap = u
        elif capped.search(u.text) and not found_cap:
            found_cap = u
    if found_cap:
        pct = re.search(r"(\d+(?:\.\d+)?)\s*%", found_cap.text)
        amt = parse_amount(found_cap.text)
        detail = f"{pct.group(1)}% of sum insured per day" if pct else (f"Rs. {amt:,}" if amt else "cap mentioned")
        return _f(True, found_cap, 0.85, detail=detail)
    if found_nocap:
        return _f(False, found_nocap, 0.8, detail="no cap stated")
    return _none()


def _presence(doc, pos: str, neg: str | None = None, skip_exclusions=True, label="") -> dict:
    neg_rx = re.compile(neg, re.I) if neg else None
    neg_hit, pos_hits = None, []
    for u, m in _units(doc, pos):
        if (neg_rx and neg_rx.search(u.text)) or (skip_exclusions and u.category == "exclusions"):
            neg_hit = neg_hit or u
            continue
        pos_hits.append((m.start(), u))
    if pos_hits:
        u = min(pos_hits, key=lambda t: t[0])[1]  # earliest mention = most topical sentence
        return _f(True, u, 0.9, detail=u.text[:140])
    if neg_hit:
        return _f(False, neg_hit, 0.85, detail="stated as not covered / not offered")
    return {**_f(False), "detail": "no mention found", "confidence": 0.4}


def _extract_copay(doc) -> dict:
    pcts, zero_unit, pct_units = [], None, []
    for u, _ in _units(doc, r"co-?\s?pay(?:ment)?"):
        found = [float(x) for x in re.findall(r"(\d{1,2}(?:\.\d+)?)\s*%", u.text)]
        if found:
            pcts += found
            pct_units.append(u)
        if re.search(
            r"\bno\s+co-?\s?pay|co-?\s?pay(?:ment)?\s+(?:is\s+)?not\s+applicable|without\s+co-?\s?pay|co-?\s?pay(?:ment)?\s*[:\-]?\s*nil",
            u.text,
            re.I,
        ):
            zero_unit = zero_unit or u
    if pcts or zero_unit:
        values = set(pcts) | ({0.0} if zero_unit else set())
        unit = pct_units[0] if pct_units else zero_unit
        detail = None
        if len(values) > 1:
            detail = "Conditional: " + " / ".join(f"{v:g}%" for v in sorted(values)) + " depending on conditions (e.g. age)"
        return _f(int(max(values)), unit, 0.7 if detail else 0.9, detail=detail)
    return _none()


def _extract_maternity(doc) -> tuple[dict, dict]:
    pos_unit = neg_unit = None
    for u, _ in _units(doc, r"maternity|childbirth|delivery"):
        if u.category == "exclusions" or re.search(r"not\s+(?:covered|payable)|excluded|no\s+maternity", u.text, re.I):
            neg_unit = neg_unit or u
        elif re.search(r"covered|up\s+to|payable|benefit", u.text, re.I):
            pos_unit = pos_unit or u
    wait = _none()
    if pos_unit:
        m = re.search(r"waiting\s+period\s+of\s+(\d+)\s*(months?|years?)", pos_unit.text, re.I)
        if m:
            wait = _f(to_months(float(m.group(1)), m.group(2)), pos_unit, 0.9)
        return _f(True, pos_unit, 0.9, detail=pos_unit.text[:140]), wait
    if neg_unit:
        return _f(False, neg_unit, 0.85, detail="stated as not covered"), wait
    return {**_f(False), "detail": "no mention found", "confidence": 0.4}, wait


def _extract_claim_clarity(doc) -> dict:
    claims = ("claims", "general")
    cashless = _units(doc, r"cashless", claims)
    reimb = _units(doc, r"reimbursement", claims)
    intim = _units(doc, r"(?:within|at\s+least)\s+\d+\s*(?:hours?|days?)", ("claims",))
    docs = _units(doc, r"claim\s+form|discharge\s+summary|original\s+bills", claims)
    score = 35 + 20 * bool(cashless) + 15 * bool(reimb) + 15 * bool(intim) + 15 * bool(docs)
    unit = (cashless or reimb or intim or docs or [(None, None)])[0][0]
    parts = [n for n, ok in [("cashless", cashless), ("reimbursement", reimb), ("time limits", intim), ("document list", docs)] if ok]
    return _f(score, unit, 0.8, detail="found: " + (", ".join(parts) if parts else "nothing specific"))


def _extract_ambiguous(doc) -> dict:
    snippets, n = [], 0
    rxs = [re.compile(p, re.I) for p in AMBIGUOUS_PATTERNS]
    first = None
    for u in doc.units:
        if any(rx.search(u.text) for rx in rxs):
            n += 1
            first = first or u
            snippets.append(f"p.{u.page}: {u.text[:140]}")
    f = _f(n, first, 0.8, detail=" | ".join(snippets[:6]) if snippets else "none detected")
    f["items"] = snippets
    return f


def rules_extract(doc: PolicyDoc) -> dict:
    F: dict[str, dict] = {}
    F["insurer_name"] = _text_field(doc, "insurer|insurance company") or _none()
    if F["insurer_name"]["value"] is None:
        m = re.search(
            r"([A-Z][\w&\.\- ]{2,60}(?:Insurance|Assurance)[\w\. ]{0,30}?(?:Limited|Ltd\.?|Company)?)", doc.pages[0]["text"] if doc.pages else ""
        )
        F["insurer_name"] = _f(m.group(1).strip(), None) if m else _none()
    F["policy_name"] = _text_field(doc, "policy name|product name|plan name") or _f(doc.name)
    pt = _text_field(doc, "policy type|type of policy|plan type")
    first_page = ((pt["value"] if pt else "") or (doc.pages[0]["text"] if doc.pages else "")).lower()
    ptype = next((label for label, rx in POLICY_TYPES if re.search(rx, first_page)), None)
    F["policy_type"] = _f(ptype, None, 0.7) if ptype else _none()
    if ptype and pt:
        F["policy_type"].update({k: pt[k] for k in ("page", "section", "snippet")}, confidence=0.9)
    F["sum_insured"] = _amount_field(doc, r"sum\s+insured")
    F["annual_premium"] = _amount_field(doc, r"annual\s+premium|total\s+premium|premium\s+payable")
    F["policy_term"] = _text_field(doc, "policy term|policy period") or _none()
    F["entry_age"] = _text_field(doc, "entry age|age band|eligibility") or _none()
    F["members_covered"] = _text_field(doc, "members covered|persons covered|insured persons") or _none()

    F["initial_waiting_months"] = _months_field(doc, [r"initial\s+waiting\s+period[^.]*?(\d+)\s*(days?|months?)"])
    F["ped_waiting_months"] = _months_field(
        doc,
        [
            r"pre-?\s?existing[^.]*?waiting\s+period\s+of\s+(\d+)\s*(months?|years?)",
            r"pre-?\s?existing[^.]*?(\d+)\s*(months?|years?)",
            r"waiting\s+period\s+of\s+(\d+)\s*(months?|years?)[^.]*pre-?\s?existing",
        ],
    )
    F["specific_waiting_months"] = _months_field(doc, [r"specific\s+(?:disease|illness|procedure)s?[^.]*?(\d+)\s*(months?|years?)"])

    F["copayment_percentage"] = _extract_copay(doc)
    F["room_rent_limit"] = _limit_field(doc, r"room\s+rent|room\s+charges|room\s+category")
    F["icu_limit"] = _limit_field(doc, r"\bicu\b|intensive\s+care")

    sub_units = _units(doc, r"sub-?\s?limit|capped\s+at|restricted\s+to\s+rs|limited\s+to\s+rs")
    found = {d for u, _ in sub_units for d in DISEASES if d in u.text.lower()}
    F["disease_sub_limit_count"] = _f(
        max(len(found), len(sub_units)), sub_units[0][0] if sub_units else None, 0.8, detail=", ".join(sorted(found)) or None
    )

    ex_units = [u for u in doc.units if u.category == "exclusions"]
    items = [u for u in ex_units if u.is_item]
    if ex_units:
        F["exclusion_count"] = _f(len(items) or len(ex_units), ex_units[0], 0.85, detail="numbered/bulleted exclusion items")
    else:
        loose = _units(doc, r"not\s+covered|excluded|not\s+payable")
        F["exclusion_count"] = _f(len(loose), loose[0][0] if loose else None, 0.4, detail="no exclusions section found; counted exclusion phrases")

    F["restoration_benefit"] = _presence(
        doc, r"restor|recharge|reinstat|reset", r"does\s+not\s+(?:offer|provide|include)|not\s+(?:available|applicable|offered)|no\s+restor"
    )
    F["no_claim_bonus"] = _presence(doc, r"no[\s-]claim\s+bonus|cumulative\s+bonus", r"not\s+(?:available|offered)")
    F["child_coverage"] = _presence(doc, r"dependent\s+child|dependent\s+children|\bchildren\b", r"not\s+covered|not\s+eligible")
    F["newborn_coverage"] = _presence(doc, r"newborn|new-born|new\s+born|\bbaby\b", r"not\s+covered")
    F["maternity_coverage"], F["maternity_waiting_months"] = _extract_maternity(doc)
    nb = F["newborn_coverage"]
    if nb["snippet"]:
        t = nb["snippet"].lower()
        timing = "from day 1" if re.search(r"from\s+day\s*1|from\s+birth", t) else "only from the next renewal" if "next renewal" in t else None
        if timing:
            nb["detail"] = f"Newborn cover {timing}"

    for key, pat in (
        ("pre_hosp_days", r"pre-?\s?hospitali[sz]ation[^.]*?(\d+)\s*days"),
        ("post_hosp_days", r"post-?\s?hospitali[sz]ation[^.]*?(\d+)\s*days"),
    ):
        hit = _units(doc, pat)
        F[key] = _f(int(hit[0][1].group(1)), hit[0][0], 0.9) if hit else _none()

    F["day_care"] = _presence(doc, r"day\s?care", r"not\s+covered")
    F["ambulance"] = _presence(doc, r"ambulance", r"not\s+covered")
    F["domiciliary"] = _presence(doc, r"domiciliary", r"not\s+covered")
    F["ayush"] = _presence(doc, r"ayush|ayurveda", r"not\s+covered")
    F["opd"] = _presence(doc, r"\bopd\b|out-?\s?patient", r"not\s+covered")
    F["cashless"] = _presence(doc, r"cashless")
    F["reimbursement"] = _presence(doc, r"reimbursement")
    F["claim_process_clarity_score"] = _extract_claim_clarity(doc)
    F["ambiguous_clause_count"] = _extract_ambiguous(doc)

    gp = _units(doc, r"grace\s+period[^.]*?(\d+)\s*days")
    F["grace_period_days"] = _f(int(gp[0][1].group(1)), gp[0][0], 0.9) if gp else _none()
    F["portability"] = _presence(doc, r"portab", r"not\s+(?:available|allowed)")
    return F


# ----------------------------------------------------------------------------- LLM extraction
def _norm_tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9%]+", s.lower()))


def locate_quote(doc: PolicyDoc, quote: str, min_overlap: float = 0.8) -> Unit | None:
    """Find the document unit that supports an LLM-provided quote (grounding check)."""
    qt = _norm_tokens(quote)
    if len(qt) < 3:
        return None
    best, best_score = None, 0.0
    for u in doc.units:
        ut = _norm_tokens(u.text)
        score = len(qt & ut) / len(qt)
        if score > best_score:
            best, best_score = u, score
    return best if best_score >= min_overlap else None


def _cast(value, kind: str):
    try:
        if value is None:
            return None
        if kind == "bool":
            if isinstance(value, str):
                return value.strip().lower() in {"true", "yes", "1", "present"}
            return bool(value)
        if kind == "int":
            return int(float(str(value).replace(",", "")))
        return str(value).strip() or None
    except (ValueError, TypeError):
        return None


def build_llm_context(doc: PolicyDoc, max_chars: int = 14000) -> str:
    parts, total = [], 0
    for c in doc.chunks:
        piece = f"[p.{c['page']} | {c['section']}]\n{mask_pii(c['text'])}\n"
        if total + len(piece) > max_chars:
            break
        parts.append(piece)
        total += len(piece)
    return "\n".join(parts)


def llm_extract(doc: PolicyDoc, llm) -> dict:
    from .config import PROMPTS_DIR

    system = (PROMPTS_DIR / "clause_extraction_system.txt").read_text()
    schema = "\n".join(f'- "{k}": {v}' for k, v in LLM_FIELD_TYPES.items())
    prompt = (PROMPTS_DIR / "clause_extraction_user.txt").read_text().format(schema=schema, context=build_llm_context(doc))
    raw = llm.generate_json(system, prompt)
    out = {}
    for key, kind in LLM_FIELD_TYPES.items():
        item = raw.get(key)
        if not isinstance(item, dict):
            continue
        value = _cast(item.get("value"), kind)
        unit = locate_quote(doc, str(item.get("quote", "")))
        if value is None or unit is None:  # ungrounded -> rejected
            continue
        out[key] = _f(value, unit, 0.75, source="llm")
    return out


def merge_llm(rule_fields: dict, llm_fields: dict) -> tuple[dict, list[str]]:
    merged, conflicts = dict(rule_fields), []
    for key, lf in llm_fields.items():
        rf = merged.get(key)
        has_rule_value = rf is not None and rf["value"] is not None and rf["confidence"] >= 0.6
        if not has_rule_value:
            merged[key] = lf
        elif rf["value"] != lf["value"]:
            conflicts.append(f"{FIELD_LABELS.get(key, key)}: rules found {rf['value']!r}, LLM found {lf['value']!r}")
    return merged, conflicts


# ----------------------------------------------------------------------------- validation + entry point
def validate_fields(F: dict) -> list[str]:
    issues = []
    v = lambda k: F.get(k, {}).get("value")  # noqa: E731
    if v("sum_insured") is not None and not (50_000 <= v("sum_insured") <= 50_000_000):
        issues.append(f"Sum insured {v('sum_insured'):,} looks unusual - please verify.")
    pct = v("copayment_percentage")
    if pct is not None and not (0 <= pct <= 100):
        issues.append("Co-payment outside 0-100% - value discarded.")
        F["copayment_percentage"] = _none()
    for k in ("ped_waiting_months", "initial_waiting_months", "specific_waiting_months"):
        if v(k) is not None and not (0 <= v(k) <= 120):
            issues.append(f"{FIELD_LABELS[k]} outside 0-120 months - value discarded.")
            F[k] = _none()
    si, pr = v("sum_insured"), v("annual_premium")
    if si and pr and not (0.002 <= pr / si <= 0.15):
        issues.append("Premium-to-sum-insured ratio is outside the usual range - check premium and sum insured.")
    if v("ped_waiting_months") is not None and v("initial_waiting_months") is not None and v("ped_waiting_months") < v("initial_waiting_months"):
        issues.append("Pre-existing waiting period is shorter than the initial waiting period - verify.")
    return issues


def recompute_derived(ex: dict) -> dict:
    """(Re)compute derived fields, validation messages and the unclear list. Used after extraction and after human review."""
    F = ex["fields"]
    spec, init = F["specific_waiting_months"]["value"], F["initial_waiting_months"]["value"]
    F["waiting_period_months"] = _f(spec if spec is not None else init, detail="specific-disease waiting if stated, else initial waiting", conf=0.8)
    ex["validation"] = validate_fields(F)
    unclear = [FIELD_LABELS[k] for k in CORE_FIELDS if F[k]["value"] is None]
    if "Conditional" in (F["copayment_percentage"].get("detail") or ""):
        unclear.append("Co-payment conditions (age/zone based)")
    ex["unclear_fields"] = unclear
    return ex


def extract_clauses(doc: PolicyDoc, llm=None) -> dict:
    fields = rules_extract(doc)
    conflicts: list[str] = []
    llm_note = None
    if llm is not None:
        try:
            fields, conflicts = merge_llm(fields, llm_extract(doc, llm))
        except Exception as exc:  # LLM failure must never break the pipeline
            llm_note = f"LLM extraction skipped: {exc}"
    ex = {
        "policy_name": doc.name,
        "n_pages": doc.n_pages,
        "fields": fields,
        "validation": [],
        "unclear_fields": [],
        "llm_conflicts": conflicts,
        "llm_note": llm_note,
        "used_llm": llm is not None and llm_note is None,
        "reviewed_fields": [],
    }
    return recompute_derived(ex)
