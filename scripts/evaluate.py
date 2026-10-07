"""Run the full evaluation: extraction accuracy (text + scanned/OCR), policy Q&A grounding, safety, timing."""

import io
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from coverwise.config import REPORTS_DIR, SAMPLES_DIR  # noqa: E402
from coverwise.evaluation import extraction_accuracy, run_generic_testset, run_policy_qa, summarize_qa, timed_analyze  # noqa: E402
from coverwise.pipeline import load_policy, sample_paths  # noqa: E402
from coverwise.scoring import ScoringModel  # noqa: E402

truth = json.loads((ROOT / "data/eval/sample_ground_truth.json").read_text())
cur_p, new_p = sample_paths(SAMPLES_DIR)
model = ScoringModel.load()
results = {}

docs = {"current": load_policy("Current policy", cur_p), "proposed": load_policy("Proposed policy", new_p)}
analyses, times = {}, {}
for k, d in docs.items():
    analyses[k], times[k] = timed_analyze(d, model=model)

acc = {}
detail = []
for k in docs:
    df = extraction_accuracy(analyses[k]["extraction"], truth[k])
    df.insert(0, "policy", k)
    detail.append(df)
    acc[k] = round(float(df.correct.mean()), 3)
results["extraction_field_accuracy"] = acc
results["extraction_fields_wrong"] = (
    pd.concat(detail).query("not correct")[["policy", "field", "expected", "extracted"]].astype(str).to_dict("records")
)

# scanned / OCR robustness: rasterise the current policy into an image-only PDF
try:
    from pdf2image import convert_from_path

    imgs = convert_from_path(str(cur_p), dpi=200)
    buf = io.BytesIO()
    imgs[0].save(buf, format="PDF", save_all=True, append_images=imgs[1:])
    t0 = time.perf_counter()
    doc_scan = load_policy("Scanned current", buf.getvalue())
    ocr_pages = sum(p["method"] == "ocr" for p in doc_scan.pages)
    from coverwise.pipeline import analyze

    a_scan = analyze(doc_scan, model=model)
    sdf = extraction_accuracy(a_scan["extraction"], truth["current"])
    results["ocr"] = {
        "pages_ocr": ocr_pages,
        "field_accuracy": round(float(sdf.correct.mean()), 3),
        "seconds": round(time.perf_counter() - t0, 1),
        "wrong": sdf.query("not correct")[["field", "expected", "extracted"]].astype(str).to_dict("records"),
    }
except Exception as exc:  # OCR system deps missing
    results["ocr"] = {"skipped": str(exc)}

qa = pd.read_csv(ROOT / "data/eval/sample_policy_qa.csv")
qa_res = run_policy_qa(analyses, qa)
results["policy_qa"] = summarize_qa(qa_res)
results["policy_qa_failures"] = (
    qa_res[qa_res.answer_correct.eq(False) | qa_res.citation_page_correct.eq(False) | ((qa_res.should_refuse) & (~qa_res.refused))][
        ["id", "question", "answer_correct", "citation_page_correct", "refused"]
    ]
    .astype(str)
    .to_dict("records")
)

gen = run_generic_testset(analyses, pd.read_csv(ROOT / "data/raw/coverwise_policy_qa_risk_testset_60.csv"))
adv = gen[gen.advice_refused_correctly.notna()]
results["provided_testset_12_unique_questions_x2_policies"] = {
    "citation_or_honest_not_found_rate": round(float(gen.citation_or_honest_not_found.mean()), 3),
    "advice_questions_refused_correctly": round(float(adv.advice_refused_correctly.mean()), 3),
    "directive_language_rate": round(float(gen.directive_language.mean()), 3),
}
results["processing_seconds_per_document"] = {k: round(v, 2) for k, v in times.items()}

REPORTS_DIR.mkdir(exist_ok=True)
qa_res.to_csv(REPORTS_DIR / "qa_eval_results.csv", index=False)
gen.to_csv(REPORTS_DIR / "provided_testset_eval.csv", index=False)
(REPORTS_DIR / "evaluation_results.json").write_text(json.dumps(results, indent=2, default=str))
print(json.dumps(results, indent=2, default=str))
