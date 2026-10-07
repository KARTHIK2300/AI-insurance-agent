"""CD step: package the trained model = MANIFEST.json (hash, versions, data hash, metrics, gate result) + MODEL_CARD.md.

The app verifies the manifest before loading the pickle. Use --require-gates in release pipelines.
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import sklearn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from coverwise.config import LABELLED_DIR, MODELS_DIR, REPORTS_DIR  # noqa: E402
from coverwise.features import MODEL_FEATURES  # noqa: E402
from coverwise.mlops import load_json, production_metrics  # noqa: E402
from coverwise.scoring import sha256_file  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--version", default=os.getenv("MODEL_VERSION"))
ap.add_argument("--require-gates", action="store_true")
args = ap.parse_args()


def git(*a):
    try:
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


gate = load_json(REPORTS_DIR / "quality_gate_result.json")
if args.require_gates and not (gate and gate["passed"]):
    sys.exit("quality gates missing or failed - refusing to package")
metrics = load_json(MODELS_DIR / "metrics.json")
model_path = MODELS_DIR / "coverwise_model.joblib"
data_files = [ROOT / "data/raw/coverwise_synthetic_coverage_risk_dataset_500.csv", LABELLED_DIR / "real_labelled.csv"]
h = hashlib.sha256()
for f in data_files:
    if f.exists() and (f.name != "real_labelled.csv" or metrics.get("n_real_rows", 0)):
        h.update(f.read_bytes())
version = args.version or git("describe", "--tags", "--always") or datetime.now(timezone.utc).strftime("dev-%Y%m%d")
pm = production_metrics(metrics)
manifest = {
    "model_version": version,
    "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "git_sha": git("rev-parse", "HEAD"),
    "artifact": model_path.name,
    "artifact_sha256": sha256_file(model_path),
    "features": MODEL_FEATURES,
    "training_data_sha256": h.hexdigest(),
    "n_rows": metrics["n_rows"],
    "n_real_rows": metrics.get("n_real_rows", 0),
    "sklearn_version": sklearn.__version__,
    "python_version": platform.python_version(),
    "production_method": metrics["production_method"],
    "metrics": pm,
    "gates_passed": gate["passed"] if gate else None,
}
(MODELS_DIR / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
card = f"""# Model card - CoverWise coverage clarity scorer {version}

**Intended use.** Estimate how clear / restrictive an Indian health-insurance policy's wording looks (score 0-100 and one of
four labels) to support *human* review. It is **not** a claim-settlement predictor and must not be used to tell anyone to
buy, switch or cancel a policy.

**Method.** {metrics["production_method"]}. Reasons shown to users come from a transparent linear rule set, not from the ML model.
`risk_flags_count` is excluded from the features because it is derived from the other features (leakage).

**Training data.** {metrics["n_rows"]} rows ({metrics.get("n_real_rows", 0)} human-reviewed real rows), mostly a *synthetic* dataset whose
score is a function of its own features; the model learns that recipe, not real claim outcomes. Data hash `{manifest["training_data_sha256"][:16]}`.

**Evaluation (5-fold out-of-fold).** macro-F1 {pm["macro_f1"]:.3f},
accuracy {pm["accuracy"]:.3f}, score MAE {pm["score_mae"]:.2f}, R2 {pm["score_r2"]:.3f},
High Risk recall {pm["high_risk_recall"]:.2f} (weakest class - treat borderline scores as "needs advisor review").

**Quality gates.** {"passed" if manifest["gates_passed"] else "NOT passed / not run"} (see `reports/quality_gate_result.json`).

**Provenance.** git `{manifest["git_sha"]}`, scikit-learn {sklearn.__version__},
Python {platform.python_version()}, artifact sha256 `{manifest["artifact_sha256"][:16]}`.

**Known limitations.** Small data; tree models overfit; rules tuned on two fictional policies; real documents may need the LLM path or human review.
"""
(MODELS_DIR / "MODEL_CARD.md").write_text(card)
print(json.dumps({k: manifest[k] for k in ("model_version", "artifact_sha256", "gates_passed")}, indent=2))
