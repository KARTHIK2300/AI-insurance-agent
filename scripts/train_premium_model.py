"""Train the optional premium benchmark on the Acko CSV (log1p target, RMSLE evaluation).

Usage: python scripts/train_premium_model.py /path/to/acko_dataset_modified.csv [--rows 300000]
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from coverwise import premium_model as pm  # noqa: E402
from coverwise.config import MODELS_DIR  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("csv")
ap.add_argument("--rows", type=int, default=300_000)
args = ap.parse_args()

df = pd.read_csv(args.csv)
n_all, n_null = len(df), int(df["Premium Amount"].isna().sum())
df = df.dropna(subset=["Premium Amount"])
df = df[df["Premium Amount"] > 0]
if len(df) > args.rows:
    df = df.sample(args.rows, random_state=42)
X, y = pm.prepare(df), np.log1p(df["Premium Amount"])
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, categorical_features="from_dtype", random_state=42)
model.fit(Xtr, ytr)
pred = np.clip(model.predict(Xte), 0, None)
rmsle = float(np.sqrt(mean_squared_log_error(np.expm1(yte), np.expm1(pred))))
baseline = float(np.sqrt(mean_squared_log_error(np.expm1(yte), np.expm1(np.full_like(yte, ytr.mean())))))

defaults = {c: float(X[c].median()) for c in pm.NUMERIC}
defaults.update({c: X[c].mode().iloc[0] for c in pm.CATEGORICAL})
bundle = {"model": model, "defaults": defaults, "categories": {c: list(X[c].cat.categories) for c in pm.CATEGORICAL}}
MODELS_DIR.mkdir(exist_ok=True)
joblib.dump(bundle, pm.PATH, compress=3)
metrics = {
    "rows_total": n_all,
    "rows_without_target": n_null,
    "rows_used": len(df),
    "rmsle_holdout": round(rmsle, 4),
    "rmsle_mean_baseline": round(baseline, 4),
}
(MODELS_DIR / "premium_metrics.json").write_text(json.dumps(metrics, indent=2))
print(metrics)
