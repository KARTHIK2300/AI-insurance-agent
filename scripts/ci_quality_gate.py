"""CI step: quality gates for the ML model and the pipeline. Exit 1 if any gate fails (blocks merge / release).

python scripts/ci_quality_gate.py                  # check
python scripts/ci_quality_gate.py --update-baseline # promote current metrics as the new baseline (after review)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from coverwise.config import CI_DIR, MODELS_DIR, REPORTS_DIR  # noqa: E402
from coverwise.mlops import evaluate_gates, gates_markdown, load_json, production_metrics  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--update-baseline", action="store_true")
args = ap.parse_args()

metrics = load_json(MODELS_DIR / "metrics.json")
if metrics is None:
    sys.exit("models/metrics.json missing - run scripts/train_model.py first")
bundle = joblib.load(MODELS_DIR / "coverwise_model.joblib")
checks = evaluate_gates(
    metrics,
    load_json(CI_DIR / "quality_gates.json"),
    load_json(CI_DIR / "baseline_metrics.json"),
    load_json(REPORTS_DIR / "evaluation_results.json"),
    bundle["meta"]["features"],
)
passed = all(c["passed"] for c in checks)
result = {"passed": passed, "checks": checks, "metrics": production_metrics(metrics)}
REPORTS_DIR.mkdir(exist_ok=True)
(REPORTS_DIR / "quality_gate_result.json").write_text(json.dumps(result, indent=2))
md = gates_markdown(checks)
print(md, "\n\nQUALITY GATES:", "PASSED" if passed else "FAILED")
if os.getenv("GITHUB_STEP_SUMMARY"):
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as fh:
        fh.write("## Model quality gates: " + ("PASSED" if passed else "FAILED") + "\n\n" + md + "\n")
if args.update_baseline:
    if not passed:
        sys.exit("refusing to promote a baseline that fails the gates")
    (CI_DIR / "baseline_metrics.json").write_text(json.dumps(production_metrics(metrics), indent=2))
    print("baseline updated")
sys.exit(0 if passed else 1)
