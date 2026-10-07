"""Optional practice model: indicative premium benchmark trained on the (synthetic) Acko / Kaggle
'Regression with an Insurance Dataset' style data. It is NOT used for policy risk scoring: that data
has no clause-level features (waiting period, co-pay, room rent...). Shown in the app as a labelled
benchmark only."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import MODELS_DIR

NUMERIC = [
    "Age",
    "Annual Income",
    "Number of Dependents",
    "Health Score",
    "Previous Claims",
    "Credit Score",
    "Insurance Duration",
    "start_year",
    "start_month",
]
CATEGORICAL = [
    "Gender",
    "Marital Status",
    "Education Level",
    "Occupation",
    "Location",
    "Policy Type",
    "Customer Feedback",
    "Smoking Status",
    "Exercise Frequency",
    "Property Type",
]
FEATURES = NUMERIC + CATEGORICAL
PATH = MODELS_DIR / "premium_model.joblib"


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "Policy Start Date" in d:
        start = pd.to_datetime(d["Policy Start Date"], errors="coerce")
        d["start_year"], d["start_month"] = start.dt.year.astype("float"), start.dt.month.astype("float")
    else:  # single-row prediction: start_year / start_month come from the stored defaults
        for c in ("start_year", "start_month"):
            d[c] = d[c] if c in d else np.nan
    for c in CATEGORICAL:
        d[c] = d[c].astype("category") if c in d else pd.Categorical([None] * len(d))
    return d[FEATURES]


def load():
    return joblib.load(PATH) if Path(PATH).exists() else None


def predict_premium(bundle: dict, profile: dict) -> float:
    row = pd.DataFrame([{**bundle["defaults"], **profile}])
    X = prepare(row)
    for c in CATEGORICAL:
        X[c] = pd.Categorical(X[c], categories=bundle["categories"][c])
    return float(np.expm1(bundle["model"].predict(X)[0]))
