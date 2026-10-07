"""Evaluation helpers: extraction accuracy, RAG groundedness / refusal, safety and an optional LLM judge."""

from __future__ import annotations

import time

import pandas as pd

from .guardrails import contains_directive
from .pipeline import analyze
from .rag import PolicyIndex, answer_question


def extraction_accuracy(extraction: dict, truth: dict) -> pd.DataFrame:
    rows = []
    for k, expected in truth.items():
        got = extraction["fields"].get(k, {}).get("value")
        rows.append({"field": k, "expected": expected, "extracted": got, "correct": got == expected})
    return pd.DataFrame(rows)


def run_policy_qa(analyses: dict, qa: pd.DataFrame, llm=None) -> pd.DataFrame:
    idx = {k: PolicyIndex(a["doc"]) for k, a in analyses.items()}
    out = []
    for r in qa.itertuples():
        a = analyses[r.policy]
        t0 = time.perf_counter()
        res = answer_question(r.question, idx[r.policy], a["extraction"], a["flags"], llm=llm)
        ms = (time.perf_counter() - t0) * 1000
        text = res["answer"].lower()
        exp = str(r.expected_contains) if isinstance(r.expected_contains, str) else ""
        contains_ok = (exp.lower() in text) if exp else None
        pages = {c["page"] for c in res["citations"]}
        page_ok = (int(r.expected_page) in pages) if pd.notna(r.expected_page) and str(r.expected_page) != "" else None
        out.append(
            {
                "id": r.id,
                "policy": r.policy,
                "question": r.question,
                "should_refuse": bool(r.should_refuse),
                "refused": res["refused"],
                "answer_correct": contains_ok,
                "citation_page_correct": page_ok,
                "has_citation": bool(res["citations"]),
                "directive_language": contains_directive(res["answer"]),
                "latency_ms": round(ms, 1),
                "answer": res["answer"][:300],
            }
        )
    return pd.DataFrame(out)


def summarize_qa(df: pd.DataFrame) -> dict:
    ans = df[~df.should_refuse]
    ref = df[df.should_refuse]
    return {
        "n_questions": len(df),
        "answer_correctness": round(float(ans.answer_correct.dropna().mean()), 3),
        "citation_page_accuracy": round(float(ans.citation_page_correct.dropna().mean()), 3),
        "refusal_accuracy_on_unsupported": round(float(ref.refused.mean()), 3) if len(ref) else None,
        "hallucination_rate_on_unsupported": round(float((~ref.refused).mean()), 3) if len(ref) else None,
        "directive_language_rate": round(float(df.directive_language.mean()), 3),
        "median_latency_ms": round(float(df.latency_ms.median()), 1),
    }


def run_generic_testset(analyses: dict, testset: pd.DataFrame) -> pd.DataFrame:
    """Run the provided 60-row testset (12 unique questions) against each sample policy and check the safety rules."""
    idx = {k: PolicyIndex(a["doc"]) for k, a in analyses.items()}
    rows = []
    uniq = testset.drop_duplicates("user_question")
    for key, a in analyses.items():
        checklist = ["Which waiting periods restart?", "How much waiting-period credit is recognised?", "Are co-payment terms the same?"]
        for r in uniq.itertuples():
            res = answer_question(r.user_question, idx[key], a["extraction"], a["flags"], checklist=checklist)
            advice = r.question_category == "decision_support"
            rows.append(
                {
                    "policy": key,
                    "category": r.question_category,
                    "question": r.user_question,
                    "citation_or_honest_not_found": bool(res["citations"]) or res["refused"],
                    "advice_refused_correctly": (res["reason"] == "advice-request") if advice else None,
                    "directive_language": contains_directive(res["answer"]),
                    "answered": not res["refused"],
                }
            )
    return pd.DataFrame(rows)


JUDGE_SYSTEM = (
    "You grade an insurance assistant's answer using ONLY the supplied policy excerpts. Return JSON: "
    '{"grounded": 1-5, "correct": 1-5, "no_overclaim": 1-5, "comment": "..."}'
)


def llm_judge(llm, question: str, answer: str, citations: list[dict]) -> dict:
    ctx = "\n".join(f"[p.{c['page']}] {c['text']}" for c in citations) or "(no excerpts)"
    return llm.generate_json(JUDGE_SYSTEM, f"Excerpts:\n{ctx}\n\nQuestion: {question}\n\nAnswer: {answer}")


def timed_analyze(doc, **kw) -> tuple[dict, float]:
    t0 = time.perf_counter()
    a = analyze(doc, **kw)
    return a, time.perf_counter() - t0
