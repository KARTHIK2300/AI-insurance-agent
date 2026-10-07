import json
from pathlib import Path

import pytest

from coverwise.comparison import compare_policies
from coverwise.document import categorize, is_heading
from coverwise.evaluation import extraction_accuracy
from coverwise.extractor import locate_quote, merge_llm, parse_amount, to_months
from coverwise.guardrails import contains_directive, is_advice_request, sanitize_output
from coverwise.llm import LLMClient, _parse_json
from coverwise.pdf_extract import clean_pages, extract_policy_text, mask_pii
from coverwise.rag import PolicyIndex, answer_question
from coverwise.report import generate_report
from coverwise.scoring import ScoringModel, rule_based_score, score_policy, score_to_label

ROOT = Path(__file__).resolve().parents[1]
TRUTH = json.loads((ROOT / "data/eval/sample_ground_truth.json").read_text())


# ---------------------------------------------------------------- ingestion
def test_clean_pages_removes_headers_and_page_numbers(samples):
    raw = extract_policy_text(samples[0])
    assert any("FICTIONAL TEST DOCUMENT" in p["text"] for p in raw)
    cleaned = clean_pages(raw)
    assert not any("FICTIONAL TEST DOCUMENT" in p["text"] for p in cleaned)
    assert not any("Page 2 of" in p["text"] for p in cleaned)


def test_mask_pii():
    out = mask_pii("Policy Number: SSH-2024-000123\nPolicyholder Name: Test User\nCall 9876543210 or a.b@x.com PAN ABCDE1234F")
    for leaked in ("SSH-2024", "Test User", "9876543210", "a.b@x.com", "ABCDE1234F"):
        assert leaked not in out


def test_ocr_fallback_reads_scanned_pdf(samples, tmp_path):
    pytest.importorskip("pytesseract")
    from pdf2image import convert_from_path

    imgs = convert_from_path(str(samples[0]), dpi=150, first_page=1, last_page=1)
    f = tmp_path / "scan.pdf"
    imgs[0].save(f, format="PDF")
    pages = extract_policy_text(f)
    if pages[0]["method"] != "ocr":
        pytest.skip("tesseract not installed")
    assert "sum insured" in pages[0]["text"].lower()


def test_sections_and_categories(analyses):
    cats = {s["category"] for s in analyses["current"]["doc"].sections}
    assert {"exclusions", "waiting", "limits", "claims", "renewal", "coverage"} <= cats
    assert is_heading("SECTION 5: EXCLUSIONS") and not is_heading("1. Cosmetic surgery unless required after an accident.")
    assert categorize("Co-payment and limits") == "limits"


def test_chunks_have_metadata(analyses):
    for c in analyses["current"]["doc"].chunks:
        assert c["page"] >= 1 and c["section"] and c["text"] and c["policy_name"]


# ---------------------------------------------------------------- extraction
def test_amount_and_month_parsing():
    assert parse_amount("Rs. 5,00,000 per year") == 500000 and parse_amount("sum insured of 10 lakh") == 1_000_000
    assert to_months(3, "years") == 36 and to_months(30, "days") == 1


@pytest.mark.parametrize("key", ["current", "proposed"])
def test_extraction_matches_ground_truth(analyses, key):
    df = extraction_accuracy(analyses[key]["extraction"], TRUTH[key])
    assert df.correct.mean() >= 0.95, df[~df.correct]


def test_missing_and_conditional_clauses_are_flagged_not_guessed(analyses):
    ex = analyses["proposed"]["extraction"]
    assert ex["fields"]["icu_limit"]["value"] is None
    assert "ICU limit present" in ex["unclear_fields"]
    assert "Conditional" in ex["fields"]["copayment_percentage"]["detail"]


def test_evidence_present_for_found_fields(analyses):
    f = analyses["current"]["extraction"]["fields"]["copayment_percentage"]
    assert f["page"] == 3 and "co-payment" in f["snippet"].lower()


def test_llm_values_must_be_grounded(analyses):
    doc = analyses["current"]["doc"]
    assert locate_quote(doc, "Co-payment: 10% of every admissible claim is payable by the insured person") is not None
    assert locate_quote(doc, "The insurer pays 100% of every claim with no limits whatsoever") is None


def test_merge_llm_fills_gaps_and_keeps_rules_on_conflict(analyses):
    rules = analyses["proposed"]["extraction"]["fields"]
    llm_fields = {
        "icu_limit": {"value": True, "confidence": 0.75, "source": "llm", "page": 3, "section": "x", "snippet": "y", "detail": None},
        "copayment_percentage": {"value": 5, "confidence": 0.75, "source": "llm", "page": 3, "section": "x", "snippet": "y", "detail": None},
    }
    merged, conflicts = merge_llm(rules, llm_fields)
    assert merged["icu_limit"]["value"] is True  # gap filled
    assert merged["copayment_percentage"]["value"] == 20  # rules win
    assert conflicts and "Co-payment" in conflicts[0]


class _FakeResp:
    def __init__(self, payload):
        self.payload, self.status_code = payload, 200

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class _FakeSession:
    def __init__(self, payload):
        self.payload, self.calls = payload, []

    def post(self, url, **kw):
        self.calls.append((url, kw))
        return _FakeResp(self.payload)


def test_llm_clients_parse_both_providers():
    g = LLMClient("gemini", "k", "m", _FakeSession({"candidates": [{"content": {"parts": [{"text": '```json\n{"a": 1}\n```'}]}}]}))
    assert g.generate_json("s", "p") == {"a": 1}
    r = LLMClient("groq", "k", "m", _FakeSession({"choices": [{"message": {"content": '{"b": 2}'}}]}))
    assert r.generate_json("s", "p") == {"b": 2} and "groq.com" in r.session.calls[0][0]
    assert _parse_json('noise {"c": 3} trailing') == {"c": 3}


def test_full_llm_extraction_path_with_fake_model(samples):
    from coverwise.pipeline import analyze, load_policy

    class Fake:
        def generate_json(self, system, prompt):
            return {
                "icu_limit": {
                    "value": False,
                    "quote": "Intimate the Company within 24 hours of emergency admission",
                },  # irrelevant quote for ICU but present in doc
                "ayush": {"value": True, "quote": "this quote does not exist anywhere in the document at all"},
            }

    a = analyze(load_policy("P", samples[1]), llm=Fake())
    assert a["extraction"]["used_llm"] is True
    assert a["extraction"]["fields"]["ayush"]["source"] == "rules"  # hallucinated quote rejected, rules value kept


# ---------------------------------------------------------------- scoring
def test_rule_score_monotonic_and_bands(analyses):
    base = dict(analyses["current"]["features"])
    worse = {**base, "copayment_percentage": base["copayment_percentage"] + 20}
    assert rule_based_score(worse)["score"] < rule_based_score(base)["score"]
    better = {**base, "restoration_benefit_present": 1}
    assert rule_based_score(better)["score"] > rule_based_score(base)["score"]
    assert [score_to_label(x) for x in (10, 40, 60, 90)] == ["High Risk", "Caution Required", "Moderate Coverage", "Strong Coverage"]


def test_ml_model_loads_and_is_consistent(analyses):
    model = ScoringModel.load()
    assert model is not None
    for a in analyses.values():
        s = a["score"]
        assert 0 <= s["score"] <= 100 and s["label"] == score_to_label(s["score"]) and 0 <= s["confidence"] <= 1
        assert abs(sum(s["probabilities"].values()) - 1) < 1e-6
    assert "risk_flags_count" not in model.meta["features"]  # leakage guard


def test_current_policy_scores_lower_than_clean_policy(analyses):
    clean = {
        **analyses["current"]["features"],
        "copayment_percentage": 0,
        "room_rent_limit_present": 0,
        "icu_limit_present": 0,
        "restoration_benefit_present": 1,
        "ambiguous_clause_count": 0,
        "exclusion_count": 8,
        "disease_sub_limit_count": 0,
    }
    assert score_policy(clean, "Family Floater")["score"] > analyses["current"]["score"]["rule_score"] + 20


# ---------------------------------------------------------------- comparison / flags
def test_comparison_verdicts_and_user_cautions(analyses):
    prof = {"age": 62, "planning_child": True, "n_children": 3}
    c = compare_policies(analyses["current"]["extraction"], analyses["proposed"]["extraction"], prof)
    v = {r["field"]: r["verdict"] for r in c["rows"]}
    assert v["ped_waiting_months"] == "Worse" and v["restoration_benefit"] == "Improved" and v["maternity_coverage"] == "Improved"
    assert v["copayment_percentage"] == "Worse" and v["icu_limit"] == "Unclear" and v["newborn_coverage"] == "Improved"
    assert any("age-linked co-payment" in x for x in c["user_cautions"]) and any("maternity" in x.lower() for x in c["user_cautions"])
    assert c["insurer_changes"] and "Premium changes by +39%" in c["premium"]["message"]
    assert not contains_directive(json.dumps(c))


def test_flags_have_evidence(analyses):
    flags = analyses["current"]["flags"]
    assert flags and flags[0]["severity"] == "high" or any(f["severity"] == "medium" for f in flags)
    assert any(f["page"] for f in flags)


# ---------------------------------------------------------------- RAG + guardrails
def test_guardrail_detection():
    assert is_advice_request("Should I buy this policy?") and is_advice_request("should i switch to the new plan")
    assert not is_advice_request("Is co-payment applicable?")
    assert contains_directive("You should buy this policy") and contains_directive("Do not buy this plan")
    assert not contains_directive(sanitize_output("I recommend switching to the new plan"))


def test_rag_answers_with_citations_and_refuses(analyses):
    a = analyses["current"]
    idx = PolicyIndex(a["doc"])
    r = answer_question("Is co-payment applicable in this policy?", idx, a["extraction"], a["flags"])
    assert not r["refused"] and r["citations"][0]["page"] == 3 and "10%" in r["answer"]
    assert answer_question("What is the capital of France?", idx)["refused"]
    assert answer_question("Is hair transplant surgery covered?", idx)["refused"]
    adv = answer_question("Should I buy this policy?", idx, a["extraction"], a["flags"])
    assert adv["refused"] and adv["reason"] == "advice-request" and not contains_directive(adv["answer"])


def test_rag_reports_missing_topic_instead_of_guessing(analyses):
    a = analyses["proposed"]
    r = answer_question("Is there any ICU limit?", PolicyIndex(a["doc"]), a["extraction"], a["flags"])
    assert r["refused"] and "Not clearly found" in r["answer"]


def test_rag_checklist_for_switching(analyses):
    a = analyses["current"]
    r = answer_question(
        "What should I verify before switching from my current policy?", PolicyIndex(a["doc"]), a["extraction"], a["flags"], ["Q1?", "Q2?"]
    )
    assert "Q1?" in r["answer"] and r["citations"]


# ---------------------------------------------------------------- report
def test_pdf_report_contains_disclaimer_and_no_directives(analyses, tmp_path):
    import pdfplumber

    from coverwise.config import SAMPLES_DIR
    from coverwise.pipeline import analyze_pair, load_policy, sample_paths

    c, n = sample_paths(SAMPLES_DIR)
    res = analyze_pair(load_policy("C", c), load_policy("P", n), model=ScoringModel.load(), profile={"age": 40})
    out = generate_report(tmp_path / "r.pdf", res, {"age": 40})
    with pdfplumber.open(out) as pdf:
        text = "\n".join(p.extract_text() or "" for p in pdf.pages)
    assert "decision-support" in text and "Questions to ask" in text and "Current vs proposed" in text
    assert not contains_directive(text)
