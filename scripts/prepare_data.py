"""Validate and clean the synthetic coverage risk dataset -> data/processed/coverwise_clean.csv."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "coverwise_synthetic_coverage_risk_dataset_500.csv"
OUT = ROOT / "data" / "processed" / "coverwise_clean.csv"


def main() -> pd.DataFrame:
    df = pd.read_csv(RAW)
    n0 = len(df)
    assert df.policy_id.is_unique and df.isna().sum().sum() == 0, "unexpected nulls / duplicate ids"
    ratio_ok = ((df.annual_premium / df.sum_insured) - df.premium_to_coverage_ratio).abs() < 0.002
    assert ratio_ok.all(), "premium_to_coverage_ratio inconsistent"
    # Data-quality flags (kept, not silently altered)
    df["dq_ped_shorter_than_waiting"] = (df.pre_existing_disease_waiting_period_months < df.waiting_period_months).astype(int)
    # Implausible: maternity cover on top-up / critical illness / senior products -> dropped
    implausible = df.maternity_coverage_present.eq(1) & df.policy_type.isin(["Top-up Health", "Critical Illness", "Senior Citizen"])
    df = df[~implausible].reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(
        f"rows {n0} -> {len(df)} (dropped {int(implausible.sum())} implausible maternity rows); "
        f"{int(df.dq_ped_shorter_than_waiting.sum())} rows flagged PED < waiting period (kept)"
    )
    print(df.risk_label.value_counts().to_string())
    return df


if __name__ == "__main__":
    main()
