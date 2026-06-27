#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def add_bins(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def qbin(series: pd.Series, labels: list[str]) -> pd.Series:
        valid = series.dropna()
        if valid.nunique() < len(labels):
            return pd.Series(["all"] * len(series), index=series.index)
        return pd.qcut(series, q=len(labels), labels=labels, duplicates="drop")

    out["elongation_bin"] = qbin(out["input_elongation_ratio"], ["low", "medium", "high"])
    out["compactness_bin"] = qbin(out["input_compactness_proxy"], ["low", "medium", "high"])
    return out


def summarize_categorical(df: pd.DataFrame, category_col: str) -> pd.DataFrame:
    counts = (
        df.groupby(["case_type", category_col])
        .size()
        .reset_index(name="count")
        .sort_values(["case_type", category_col])
    )
    counts["fraction"] = counts["count"] / counts.groupby("case_type")["count"].transform("sum")
    return counts


def summarize_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    available = [c for c in cols if c in df.columns]
    out = df.groupby("case_type")[available].agg(["mean", "median", "std"])
    out.columns = [f"{col}_{stat}" for col, stat in out.columns]
    return out.reset_index()


def write_report(
    path: Path,
    numeric_summary: pd.DataFrame,
    elongation_summary: pd.DataFrame,
    compactness_summary: pd.DataFrame,
    treatment_summary: pd.DataFrame,
    tier_treatment_summary: pd.DataFrame,
) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Morphology And Treatment Analysis\n\n")
        f.write("## Numeric Summary\n\n")
        f.write(numeric_summary.to_markdown(index=False))
        f.write("\n\n## By Elongation Bin\n\n")
        f.write(elongation_summary.to_markdown(index=False))
        f.write("\n\n## By Compactness Bin\n\n")
        f.write(compactness_summary.to_markdown(index=False))
        f.write("\n\n## By Treatment At Input\n\n")
        f.write(treatment_summary.to_markdown(index=False))
        f.write("\n\n## By Tier And Treatment At Input\n\n")
        f.write(tier_treatment_summary.to_markdown(index=False))
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze morphology and treatment patterns across forecasting case types.")
    parser.add_argument("--case_type_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.case_type_csv)
    df = add_bins(df)

    numeric_summary = summarize_numeric(
        df,
        [
            "input_elongation_ratio",
            "input_compactness_proxy",
            "input_bbox_x",
            "input_bbox_y",
            "input_bbox_z",
            "treated_at_input",
            "treated_at_target",
            "treatment_on_any",
            "treatment_started_before_input",
        ],
    )
    elongation_summary = summarize_categorical(df, "elongation_bin")
    compactness_summary = summarize_categorical(df, "compactness_bin")
    treatment_summary = summarize_categorical(df, "treated_at_input")

    tier_treatment_summary = (
        df.groupby(["case_type", "tier", "treated_at_input"])
        .size()
        .reset_index(name="count")
        .sort_values(["case_type", "tier", "treated_at_input"])
    )
    tier_treatment_summary["fraction_within_case_type"] = (
        tier_treatment_summary["count"] / tier_treatment_summary.groupby("case_type")["count"].transform("sum")
    )

    df.to_csv(out_dir / "case_type_samples_with_morphology_bins.csv", index=False)
    numeric_summary.to_csv(out_dir / "morphology_treatment_numeric_summary.csv", index=False)
    elongation_summary.to_csv(out_dir / "case_type_by_elongation_bin.csv", index=False)
    compactness_summary.to_csv(out_dir / "case_type_by_compactness_bin.csv", index=False)
    treatment_summary.to_csv(out_dir / "case_type_by_treated_at_input.csv", index=False)
    tier_treatment_summary.to_csv(out_dir / "case_type_by_tier_and_treated_at_input.csv", index=False)
    write_report(
        out_dir / "morphology_treatment_report.md",
        numeric_summary,
        elongation_summary,
        compactness_summary,
        treatment_summary,
        tier_treatment_summary,
    )
    print(f"Saved morphology/treatment analysis to: {out_dir}")


if __name__ == "__main__":
    main()
