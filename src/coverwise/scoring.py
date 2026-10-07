"""Coverage Clarity / Risk scoring: transparent rule baseline + optional ML model."""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import joblib
import pandas as pd

from .config import MODELS_DIR
from .features import MODEL_FEATURES

# Weights are rounded from a linear fit of coverage_clarity_score on the synthetic dataset
# (R^2 = 0.91 without the derived risk_flags_count), so every point is explainable.
RULE_WEIGHTS = {
    "copayment_percentage": (-1.0, "Co-payment (per %)"),
    "pre_existing_disease_waiting_period_months": (-0.4, "Pre-existing disease waiting (per month)"),
    "waiting_period_months": (-0.2, "Specific/initial waiting (per month)"),
    "room_rent_limit_present": (-7.0, "Room rent capped"),
    "icu_limit_present": (-7.0, "ICU charges capped"),
    "disease_sub_limit_count": (-1.7, "Disease-wise sub-limits (each)"),
    "exclusion_count": (-0.5, "Listed exclusions (each)"),
    "restoration_benefit_present": (17.0, "Restoration / recharge benefit"),
    "no_claim_bonus_present": (12.0, "No-claim bonus"),
    "claim_process_clarity_score": (0.13, "Claim process clarity (per point)"),
    "ambiguous_clause_count": (-3.7, "Ambiguous / unclear clauses (each)"),
    "maternity_coverage_present": (1.5, "Maternity covered"),
}
# Average policy in the synthetic dataset; contributions are measured against it.
REFERENCE = {
    "copayment_percentage": 7.6,
    "pre_existing_disease_waiting_period_months": 22.5,
    "waiting_period_months": 13.7,
    "room_rent_limit_present": 0.35,
    "icu_limit_present": 0.22,
    "disease_sub_limit_count": 1.8,
    "exclusion_count": 12.8,
    "restoration_benefit_present": 0.66,
    "no_claim_bonus_present": 0.66,
    "claim_process_clarity_score": 68.6,
    "ambiguous_clause_count": 2.1,
    "maternity_coverage_present": 0.17,
}
MEAN_SCORE = 69.3


def score_to_label(score: float) -> str:
    if score < 38:
        return "High Risk"
    if score < 58:
        return "Caution Required"
    if score < 78:
        return "Moderate Coverage"
    return "Strong Coverage"


def rule_based_score(feats: dict) -> dict:
    total, reasons = MEAN_SCORE, []
    for key, (w, label) in RULE_WEIGHTS.items():
        delta = w * (float(feats.get(key, REFERENCE[key])) - REFERENCE[key])
        total += delta
        if abs(delta) >= 1.0:
            reasons.append({"factor": label, "value": feats.get(key), "points": round(delta, 1)})
    score = max(0.0, min(100.0, total))
    reasons.sort(key=lambda r: r["points"])
    return {"score": round(score, 1), "label": score_to_label(score), "reasons": reasons}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_artifact(path: Path) -> tuple[bool, str]:
    """Check a model file against its manifest (written by scripts/package_model.py) BEFORE unpickling it."""
    manifest = Path(path).parent / "MANIFEST.json"
    if not manifest.exists():
        return True, "no manifest (development mode)"
    m = json.loads(manifest.read_text())
    if sha256_file(path) != m.get("artifact_sha256"):
        return False, "artifact hash does not match MANIFEST.json"
    if m.get("features") != MODEL_FEATURES:
        return False, "feature schema does not match this code version"
    if m.get("gates_passed") is False:
        return False, "model did not pass the CI quality gates"
    return True, "verified"


def model_status(path: Path | None = None) -> dict:
    path = Path(path or MODELS_DIR / "coverwise_model.joblib")
    if not path.exists():
        return {"available": False, "message": "no trained model - using rule-based scoring"}
    ok, msg = verify_artifact(path)
    manifest = path.parent / "MANIFEST.json"
    info = json.loads(manifest.read_text()) if manifest.exists() else {}
    return {
        "available": True,
        "verified": ok,
        "message": msg,
        "version": info.get("model_version", "dev"),
        "sha256": (info.get("artifact_sha256") or sha256_file(path))[:16],
        "gates_passed": info.get("gates_passed"),
        "created_utc": info.get("created_utc"),
    }


class ScoringModel:
    """Wraps the trained classifier + regressor saved by scripts/train_model.py."""

    def __init__(self, bundle: dict):
        self.clf, self.reg, self.meta = bundle["classifier"], bundle["regressor"], bundle.get("meta", {})

    @classmethod
    def load(cls, path: Path | None = None, verify: bool = True):
        """Load the model. When a MANIFEST.json sits next to it, the file hash and feature schema must match,
        otherwise the model is rejected (callers fall back to the rule-based scorer)."""
        path = Path(path or MODELS_DIR / "coverwise_model.joblib")
        if not path.exists():
            return None
        if verify:
            ok, msg = verify_artifact(path)
            if not ok:
                warnings.warn(f"Model rejected ({msg}); using rule-based scoring.")
                return None
        return cls(joblib.load(path))

    def predict(self, feats: dict, policy_type: str) -> dict:
        """Score from the regressor; label from the score bands (keeps score and label consistent).

        The classifier is a second opinion: its probabilities give a confidence for the final label.
        """
        row = pd.DataFrame([{**{k: feats[k] for k in MODEL_FEATURES if k != "policy_type"}, "policy_type": policy_type}])[MODEL_FEATURES]
        score = float(max(0, min(100, self.reg.predict(row)[0])))
        label = score_to_label(score)
        proba = self.clf.predict_proba(row)[0]
        probs = {c: float(p) for c, p in zip(self.clf.classes_, proba)}
        return {
            "label": label,
            "score": round(score, 1),
            "probabilities": probs,
            "confidence": probs.get(label, 0.0),
            "classifier_label": max(probs, key=probs.get),
        }


def score_policy(feats: dict, policy_type: str, model: ScoringModel | None = None) -> dict:
    """Combine the transparent rule score (always) with the ML prediction (when a model exists)."""
    rule = rule_based_score(feats)
    out = {"rule_score": rule["score"], "rule_label": rule["label"], "reasons": rule["reasons"], "method": "rule-based", "probabilities": None}
    if model is not None:
        ml = model.predict(feats, policy_type)
        out.update(
            score=ml["score"],
            label=ml["label"],
            probabilities=ml["probabilities"],
            confidence=ml["confidence"],
            classifier_label=ml["classifier_label"],
            method="Gradient-boosting score model + score bands; reasons from transparent rules",
        )
    else:
        out.update(score=rule["score"], label=rule["label"], confidence=None)
    out["agreement"] = out["label"] == out["rule_label"]
    return out


def load_metrics() -> dict:
    p = MODELS_DIR / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else {}
