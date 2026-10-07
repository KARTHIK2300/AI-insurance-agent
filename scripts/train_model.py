"""Train and evaluate the Coverage Clarity / Risk scoring models.

Compares a transparent rule baseline with Logistic Regression, Decision Tree, Random Forest and
Gradient Boosting (5-fold stratified CV, out-of-fold predictions), with and without the derived
`risk_flags_count` feature (leakage check). Saves the best leakage-free models + metrics.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_data  # noqa: E402

from coverwise.config import LABELS, MODELS_DIR, REPORTS_DIR  # noqa: E402
from coverwise.features import FEATURE_COLUMNS, MODEL_FEATURES, MODEL_NUMERIC  # noqa: E402
from coverwise.scoring import rule_based_score, score_to_label  # noqa: E402

warnings.filterwarnings("ignore")
SEED = 42


def make_pre(numeric):
    return ColumnTransformer([("num", StandardScaler(), numeric), ("cat", OneHotEncoder(handle_unknown="ignore"), ["policy_type"])])


def classifiers():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED),
        "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
    }


def main(include_labelled: bool = False):
    df = prepare_data.main()
    n_real = 0
    real_path = ROOT / "data" / "labelled" / "real_labelled.csv"
    if include_labelled:  # OPTIONAL: add human-reviewed real policies (see src/coverwise/review.py)
        if real_path.exists() and len(pd.read_csv(real_path)) >= 20:
            real = pd.read_csv(real_path)
            real["dq_ped_shorter_than_waiting"] = 0
            df = pd.concat([df, real.reindex(columns=df.columns)], ignore_index=True)
            n_real = len(real)
            print(f"including {n_real} human-reviewed real rows")
        else:
            print("--include-labelled: fewer than 20 reviewed rows found; training on synthetic data only")
    y = df["risk_label"]
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    results, oof_best = {}, {}

    # 1) rule baseline
    rb = [rule_based_score(r) for r in df.to_dict("records")]
    rb_score = np.array([r["score"] for r in rb])
    rb_label = [r["label"] for r in rb]
    results["Rule-based baseline"] = {
        "accuracy": accuracy_score(y, rb_label),
        "macro_f1": f1_score(y, rb_label, average="macro"),
        "score_mae": mean_absolute_error(df.coverage_clarity_score, rb_score),
        "score_r2": r2_score(df.coverage_clarity_score, rb_score),
    }

    # 2) ML classifiers, with and without derived risk_flags_count
    for set_name, numeric in (("core (no risk_flags_count)", MODEL_NUMERIC), ("with risk_flags_count", FEATURE_COLUMNS)):
        cols = numeric + ["policy_type"]
        for name, clf in classifiers().items():
            pipe = Pipeline([("pre", make_pre(numeric)), ("clf", clf)])
            pred = cross_val_predict(pipe, df[cols], y, cv=skf)
            key = f"{name} | {set_name}"
            results[key] = {"accuracy": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro")}
            if set_name.startswith("core"):
                oof_best[name] = pred

    core = {k: v for k, v in results.items() if k.endswith("core (no risk_flags_count)")}  # classifiers only
    best_key = max(core, key=lambda k: core[k]["macro_f1"])
    best_name = best_key.split(" | ")[0]
    best_pred = oof_best[best_name]
    per_class = classification_report(y, best_pred, labels=LABELS, output_dict=True, zero_division=0)
    cm = confusion_matrix(y, best_pred, labels=LABELS)

    # 3) score regressors
    reg_results, oof_reg = {}, {}
    for name, reg in {"Ridge": Ridge(alpha=1.0), "Gradient Boosting": GradientBoostingRegressor(random_state=SEED)}.items():
        pipe = Pipeline([("pre", make_pre(MODEL_NUMERIC)), ("reg", reg)])
        pred = cross_val_predict(pipe, df[MODEL_FEATURES], df.coverage_clarity_score, cv=skf.split(df, y))
        oof_reg[name] = pred
        band = [score_to_label(x) for x in pred]
        reg_results[name] = {
            "mae": mean_absolute_error(df.coverage_clarity_score, pred),
            "r2": r2_score(df.coverage_clarity_score, pred),
            "label_accuracy_via_bands": accuracy_score(y, band),
            "label_macro_f1_via_bands": f1_score(y, band, average="macro"),
        }
        results[f"{name} regressor -> score bands | core"] = {
            "accuracy": reg_results[name]["label_accuracy_via_bands"],
            "macro_f1": reg_results[name]["label_macro_f1_via_bands"],
        }
    best_reg_name = min(reg_results, key=lambda k: reg_results[k]["mae"])

    # production label = score bands applied to the regressor's out-of-fold score
    prod_pred = [score_to_label(x) for x in oof_reg[best_reg_name]]
    per_class = classification_report(y, prod_pred, labels=LABELS, output_dict=True, zero_division=0)
    cm = confusion_matrix(y, prod_pred, labels=LABELS)
    best_pred = np.array(prod_pred)

    # 4) fit final models on all rows
    final_clf = Pipeline([("pre", make_pre(MODEL_NUMERIC)), ("clf", classifiers()[best_name])]).fit(df[MODEL_FEATURES], y)
    reg = Ridge(alpha=1.0) if best_reg_name == "Ridge" else GradientBoostingRegressor(random_state=SEED)
    final_reg = Pipeline([("pre", make_pre(MODEL_NUMERIC)), ("reg", reg)]).fit(df[MODEL_FEATURES], df.coverage_clarity_score)

    names = final_clf.named_steps["pre"].get_feature_names_out()
    est = final_clf.named_steps["clf"]
    imp = getattr(est, "feature_importances_", None)
    if imp is None:
        imp = np.abs(est.coef_).mean(axis=0)
    fi = pd.Series(imp, index=[n.split("__")[-1] for n in names]).sort_values(ascending=False)

    # 5) consistency subset check (rows where PED >= waiting)
    mask = df.dq_ped_shorter_than_waiting.eq(0).values
    clean_f1 = f1_score(y[mask], np.array(best_pred)[mask], average="macro")

    MODELS_DIR.mkdir(exist_ok=True)
    (MODELS_DIR / "MANIFEST.json").unlink(missing_ok=True)  # model changed: re-run scripts/package_model.py after the gates
    meta = {"classifier": best_name, "regressor": best_reg_name, "features": MODEL_FEATURES, "n_rows": len(df)}
    joblib.dump({"classifier": final_clf, "regressor": final_reg, "meta": meta}, MODELS_DIR / "coverwise_model.joblib")
    metrics = {
        "n_rows": len(df),
        "n_real_rows": n_real,
        "label_counts": y.value_counts().to_dict(),
        "cv": "5-fold stratified, out-of-fold",
        "classifiers": results,
        "second_opinion_classifier": best_key,
        "production_method": f"{best_reg_name} regressor -> score bands",
        "per_class_production": {k: {m: round(v, 3) for m, v in per_class[k].items()} for k in LABELS},
        "confusion_matrix": {"labels": LABELS, "matrix": cm.tolist()},
        "regressors": reg_results,
        "selected_regressor": best_reg_name,
        "macro_f1_on_consistent_waiting_rows": round(clean_f1, 3),
        "top_features": {k: round(float(v), 3) for k, v in fi.head(10).items()},
    }
    (MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    fi.to_csv(MODELS_DIR / "feature_importance.csv", header=["importance"])

    # figures for README / deck
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = REPORTS_DIR / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(4), [lab.replace(" ", "\n") for lab in LABELS], fontsize=7)
    ax.set_yticks(range(4), LABELS, fontsize=7)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title("Production scorer: out-of-fold confusion matrix", fontsize=9)
    plt.tight_layout()
    fig.savefig(figdir / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    fi.head(10)[::-1].plot.barh(ax=ax, color="#1f6feb")
    ax.set_title("Top features (leakage-free classifier)", fontsize=9)
    ax.tick_params(labelsize=7)
    plt.tight_layout()
    fig.savefig(figdir / "feature_importance.png", dpi=160)
    plt.close(fig)

    print(json.dumps({k: {m: round(v, 3) for m, v in r.items()} for k, r in results.items()}, indent=1))
    print("selected:", best_key, "| regressor:", best_reg_name, reg_results)
    print("per-class F1:", {k: per_class[k]["f1-score"].__round__(2) for k in LABELS})
    print("top features:", list(fi.head(6).index))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-labelled", action="store_true", help="also train on data/labelled/real_labelled.csv (needs >= 20 rows)")
    main(ap.parse_args().include_labelled)
