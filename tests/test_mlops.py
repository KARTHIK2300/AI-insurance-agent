import json
import sys
import types
import warnings
from pathlib import Path

import pandas as pd
import pytest

from coverwise import secrets as secrets_mod
from coverwise.config import LABELS, env_flag, feature_enabled
from coverwise.features import FEATURE_COLUMNS
from coverwise.mlops import evaluate_gates, gates_markdown, load_json, production_metrics, validate_dataframe
from coverwise.review import (
    REVIEW_FIELDS,
    apply_review,
    export_labelled,
    log_review,
    parse_correction,
    review_summary,
)
from coverwise.scoring import ScoringModel, model_status, sha256_file, verify_artifact

ROOT = Path(__file__).resolve().parents[1]
GATES = load_json(ROOT / "ci/quality_gates.json")


# ------------------------------------------------------------ data validation
def test_clean_dataset_passes_validation():
    df = pd.read_csv(ROOT / "data/raw/coverwise_synthetic_coverage_risk_dataset_500.csv")
    assert validate_dataframe(df)["ok"]


@pytest.mark.parametrize(
    "mutate,expected",
    [
        (lambda d: d.assign(copayment_percentage=150), "copayment_percentage"),
        (lambda d: d.assign(risk_label="Maybe"), "unknown labels"),
        (lambda d: d.drop(columns=["exclusion_count"]), "missing columns"),
        (lambda d: pd.concat([d, d.head(1)]), "duplicate policy_id"),
        (lambda d: d[d.risk_label != "High Risk"], "High Risk"),
    ],
)
def test_validation_catches_bad_data(mutate, expected):
    df = pd.read_csv(ROOT / "data/raw/coverwise_synthetic_coverage_risk_dataset_500.csv")
    report = validate_dataframe(mutate(df))
    assert not report["ok"] and any(expected in e for e in report["errors"])


# ------------------------------------------------------------ quality gates
def _gate_inputs():
    return (
        load_json(ROOT / "models/metrics.json"),
        load_json(ROOT / "ci/baseline_metrics.json"),
        load_json(ROOT / "reports/evaluation_results.json"),
    )


def test_current_model_passes_all_gates():
    metrics, baseline, evaluation = _gate_inputs()
    checks = evaluate_gates(metrics, GATES, baseline, evaluation, ScoringModel.load().meta["features"])
    assert all(c["passed"] for c in checks), gates_markdown([c for c in checks if not c["passed"]])


def test_gates_fail_on_regression_leakage_and_safety():
    metrics, baseline, evaluation = _gate_inputs()
    feats = ScoringModel.load().meta["features"]
    # 1) model got much worse than the baseline
    worse = json.loads(json.dumps(metrics))
    worse["classifiers"][f"{worse['production_method']} | core"]["macro_f1"] = 0.60
    names = {c["gate"] for c in evaluate_gates(worse, GATES, baseline, evaluation, feats) if not c["passed"]}
    assert {"model.macro_f1", "regression.macro_f1_vs_baseline"} <= names
    # 2) leakage feature sneaks in
    assert not all(c["passed"] for c in evaluate_gates(metrics, GATES, baseline, evaluation, feats + ["risk_flags_count"]))
    # 3) safety regression: directive language or hallucination
    bad_eval = json.loads(json.dumps(evaluation))
    bad_eval["policy_qa"]["directive_language_rate"] = 0.04
    bad_eval["policy_qa"]["hallucination_rate_on_unsupported"] = 0.33
    failed = {c["gate"] for c in evaluate_gates(metrics, GATES, baseline, bad_eval, feats) if not c["passed"]}
    assert {"safety.directive_language_rate", "qa.hallucination_rate_on_unsupported"} <= failed
    # 4) missing evaluation evidence fails closed
    assert not all(c["passed"] for c in evaluate_gates(metrics, GATES, baseline, None, feats))
    assert production_metrics(metrics)["macro_f1"] > 0.75


# ------------------------------------------------------------ artifact integrity (CD safety)
def _make_release(tmp_path, gates_passed=True):
    from coverwise.features import MODEL_FEATURES

    model = tmp_path / "coverwise_model.joblib"
    model.write_bytes((ROOT / "models/coverwise_model.joblib").read_bytes())
    (tmp_path / "MANIFEST.json").write_text(
        json.dumps({"artifact_sha256": sha256_file(model), "features": MODEL_FEATURES, "gates_passed": gates_passed, "model_version": "t"})
    )
    return model


def test_verified_model_loads_and_tampered_or_ungated_model_is_rejected(tmp_path):
    model = _make_release(tmp_path)
    assert verify_artifact(model)[0] and ScoringModel.load(model) is not None
    assert model_status(model)["verified"]
    model.write_bytes(model.read_bytes() + b"tamper")
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        assert ScoringModel.load(model) is None
    sub = tmp_path / "x"
    sub.mkdir()
    ungated = _make_release(sub)
    (ungated.parent / "MANIFEST.json").write_text(json.dumps({**json.loads((ungated.parent / "MANIFEST.json").read_text()), "gates_passed": False}))
    assert not verify_artifact(ungated)[0]


def test_feature_schema_mismatch_is_rejected(tmp_path):
    model = _make_release(tmp_path)
    m = json.loads((tmp_path / "MANIFEST.json").read_text())
    m["features"] = m["features"][:-1]
    (tmp_path / "MANIFEST.json").write_text(json.dumps(m))
    assert verify_artifact(model) == (False, "feature schema does not match this code version")


# ------------------------------------------------------------ optional modules + secrets
def test_optional_modules_default_off_and_switch_with_env(monkeypatch):
    monkeypatch.delenv("COVERWISE_HUMAN_REVIEW", raising=False)
    assert feature_enabled("human_review") is False
    monkeypatch.setenv("COVERWISE_HUMAN_REVIEW", "1")
    assert feature_enabled("human_review") is True and env_flag("COVERWISE_HUMAN_REVIEW")


def test_secrets_env_first_then_aws(monkeypatch):
    monkeypatch.setenv("MY_KEY", "from-env")
    assert secrets_mod.get_secret("MY_KEY") == "from-env"
    monkeypatch.delenv("MY_KEY")
    assert secrets_mod.get_secret("MY_KEY") is None
    fake = types.SimpleNamespace(
        client=lambda *a, **k: types.SimpleNamespace(get_secret_value=lambda SecretId: {"SecretString": json.dumps({"MY_KEY": "from-aws"})})
    )
    monkeypatch.setitem(sys.modules, "boto3", fake)
    monkeypatch.setenv("COVERWISE_AWS_SECRET_ID", "coverwise/test")
    secrets_mod._cache.clear()
    assert secrets_mod.get_secret("MY_KEY") == "from-aws"


# ------------------------------------------------------------ human review
def test_review_applies_corrections_and_rescores(analyses):
    from coverwise.pipeline import analyze

    cur = analyses["current"]
    reviewed = apply_review(cur["extraction"], {"copayment_percentage": 25, "icu_limit": False})
    f = reviewed["fields"]["copayment_percentage"]
    assert f["value"] == 25 and f["source"] == "human" and f["confidence"] == 1.0
    assert reviewed["reviewed_fields"] == ["copayment_percentage", "icu_limit"]
    assert cur["extraction"]["fields"]["copayment_percentage"]["value"] == 10  # original untouched
    again = analyze(cur["doc"], model=None, extraction=reviewed)
    assert again["features"]["copayment_percentage"] == 25 and again["score"]["rule_score"] < cur["score"]["rule_score"]
    with pytest.raises(KeyError):
        apply_review(cur["extraction"], {"policy_name": "x"})


def test_parse_correction():
    assert parse_correction("copayment_percentage", "1,000") == (True, 1000)
    assert parse_correction("room_rent_limit", "Yes") == (True, True) and parse_correction("room_rent_limit", "maybe") == (False, None)
    assert parse_correction("icu_limit", "not stated") == (True, None)


def test_review_log_export_and_summary_hold_no_document_text(analyses, tmp_path):
    from coverwise.pipeline import analyze

    cur = analyses["current"]
    reviewed = analyze(cur["doc"], model=None, extraction=apply_review(cur["extraction"], {"copayment_percentage": 25}))
    log, out = tmp_path / "log.jsonl", tmp_path / "real.csv"
    log_review(reviewed, cur["extraction"], "reviewer-1", "Caution Required", path=log)
    log_review(analyses["proposed"], analyses["proposed"]["extraction"], "reviewer-1", "Moderate Coverage", path=log)
    raw = log.read_text()
    assert "SureCare" not in raw and "9876543210" not in raw  # hash + numbers only
    df = export_labelled(log, out)
    assert len(df) == 2 and set(df.risk_label) <= set(LABELS) and {"policy_id", "coverage_clarity_score", "source"} <= set(df.columns)
    assert set(FEATURE_COLUMNS) <= set(df.columns)  # same schema the trainer expects
    s = review_summary(log)
    assert s["n_reviews"] == 2 and "copayment_percentage" in s["field_correction_rate"] and len(REVIEW_FIELDS) == 16
    with pytest.raises(ValueError):
        log_review(reviewed, cur["extraction"], " ", "High Risk", path=log)
    with pytest.raises(ValueError):
        log_review(reviewed, cur["extraction"], "r", "Bogus", path=log)


# ------------------------------------------------------------ real-policy evaluation + hygiene scripts
def test_real_policy_evaluation_runs_on_a_folder(samples):
    sys.path.insert(0, str(ROOT / "scripts"))
    import evaluate_real_policies as erp

    df = erp.evaluate_folder(samples[0].parent)
    assert len(df) == 2 and df.qa_safety_ok.all() and (df.core_found >= 6).all() and not df.needs_human_review.any()
    truth = pd.DataFrame(
        [{"file": samples[0].name, "field": "copayment_percentage", "value": "10"}, {"file": samples[0].name, "field": "icu_limit", "value": "false"}]
    )
    t = erp.evaluate_folder(samples[0].parent, truth)
    assert t.loc[t.file == samples[0].name, "truth_accuracy"].iloc[0] == 0.5  # icu really is capped -> 1 of 2 correct


def test_hygiene_scan_detects_leaked_secret_and_unlisted_pii(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "scripts"))
    import check_repo_hygiene as h

    (tmp_path / "ci").mkdir()
    (tmp_path / "ci/pii_allowlist.txt").write_text("test.user@example.com\n")
    (tmp_path / "data/samples").mkdir(parents=True)
    leaked = "GEMINI_KEY = 'AIza" + "A" * 35 + "'"
    (tmp_path / "config.py").write_text(leaked)
    (tmp_path / "data/samples/notes.txt").write_text("contact real.person@gmail.com or test.user@example.com")
    monkeypatch.setattr(h, "ROOT", tmp_path)
    monkeypatch.setattr(h, "tracked_files", lambda: [tmp_path / "config.py", tmp_path / "data/samples/notes.txt"])
    assert h.main() == 1
    (tmp_path / "config.py").write_text("x = 1")
    (tmp_path / "data/samples/notes.txt").write_text("test.user@example.com")
    assert h.main() == 0
