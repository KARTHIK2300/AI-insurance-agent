"""CI step 1: validate the training data (schema, ranges, labels, class sizes). Exit 1 on errors."""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from coverwise.config import LABELLED_DIR, REPORTS_DIR  # noqa: E402
from coverwise.mlops import validate_dataframe  # noqa: E402

report = {}
raw = ROOT / "data/raw/coverwise_synthetic_coverage_risk_dataset_500.csv"
report["synthetic"] = validate_dataframe(pd.read_csv(raw))
real = LABELLED_DIR / "real_labelled.csv"
if real.exists():  # optional human-reviewed data: same schema, smaller class-size requirement
    report["real_labelled"] = validate_dataframe(pd.read_csv(real), min_class_rows=0, min_band_agreement=0.0)
REPORTS_DIR.mkdir(exist_ok=True)
(REPORTS_DIR / "data_validation.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
sys.exit(0 if all(r["ok"] for r in report.values()) else 1)
