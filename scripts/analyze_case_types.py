#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def assign_case_type(
    df: pd.DataFrame,
    gap_margin: float,
    high_dice: float,
    low_dice: float,
) -> pd.DataFrame:
    out = df.copy()
    out["case_type"] = "close_mixed"

    both_easy = (out["baseline_dice"] >= high_dice) & (out["target_dice"] >= high_dice)
    both_hard = (out["baseline_dice"] <= low_dice) & (out["target_dice"] <= low_dice)
    target_wins = out["dice_gap"] >= gap_margin
    baseline_wins = out["dice_gap"] <= -gap_margin

    out.loc[both_easy, "case_type"] = "both_easy"
    out.loc[both_hard, "case_type"] = "both_hard"
    out.loc[~both_easy & ~both_hard & target_wins, "case_type"] = "target_wins"
    out.loc[~both_easy & ~both_hard & baseline_wins, "case_type"] = "baseline_wins"
    return out


def summarize_numeric(df: pd.DataFrame, group_col: str, value_cols: list[str]) -> pd.DataFrame:
    available = [c for c in value_cols if c in df.columns]
    if not available:
        return pd.DataFrame()
    agg = {c: ["mean", "median", "std"] for c in available}
    out = df.groupby(group_col).agg(agg)
    out.columns = [f"{col}_{stat}" for col, stat in out.columns]
    return out.reset_index()


def summarize_categorical(df: pd.DataFrame, group_col: str, category_col: str) -> pd.DataFrame:
    if category_col not in df.columns:
        return pd.DataFrame()
    counts = (
        df.groupby([group_col, category_col])
        .size()
        .reset_index(name="count")
        .sort_values([group_col, category_col])
    )
    totals = counts.groupby(group_col)["count"].transform("sum")
    counts["fraction"] = counts["count"] / totals
    return counts


def write_markdown_report(
    out_path: Path,
    counts: pd.DataFrame,
    numeric_summary: pd.DataFrame,
    tier_summary: pd.DataFrame,
    horizon_summary: pd.DataFrame,
    volume_summary: pd.DataFrame,
    growth_summary: pd.DataFrame,
    treatment_summary: pd.DataFrame,
) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# Case Type Report\n\n")
        f.write("## Case Type Counts\n\n")
        f.write(counts.to_markdown(index=False))
        f.write("\n\n## Numeric Feature Summary\n\n")
        if not numeric_summary.empty:
            f.write(numeric_summary.to_markdown(index=False))
        else:
            f.write("No numeric summary available.")
        f.write("\n\n## By Tier\n\n")
        if not tier_summary.empty:
            f.write(tier_summary.to_markdown(index=False))
        else:
            f.write("No tier summary available.")
        f.write("\n\n## By Horizon\n\n")
        if not horizon_summary.empty:
            f.write(horizon_summary.to_markdown(index=False))
        else:
            f.write("No horizon summary available.")
        f.write("\n\n## By Input Volume Bin\n\n")
        if not volume_summary.empty:
            f.write(volume_summary.to_markdown(index=False))
        else:
            f.write("No volume-bin summary available.")
        f.write("\n\n## By Recent Growth Bin\n\n")
        if not growth_summary.empty:
            f.write(growth_summary.to_markdown(index=False))
        else:
            f.write("No recent-growth summary available.")
        f.write("\n\n## By Treatment At Input\n\n")
        if not treatment_summary.empty:
            f.write(treatment_summary.to_markdown(index=False))
        else:
            f.write("No treatment summary available.")
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze model-suitability case types from a pairwise forecasting comparison.")
    parser.add_argument("--pairwise_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--gap_margin", type=float, default=0.05)
    parser.add_argument("--high_dice", type=float, default=0.85)
    parser.add_argument("--low_dice", type=float, default=0.70)
    args = parser.parse_args()

    pairwise_csv = Path(args.pairwise_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pair = pd.read_csv(pairwise_csv)
    case_df = assign_case_type(
        pair,
        gap_margin=args.gap_margin,
        high_dice=args.high_dice,
        low_dice=args.low_dice,
    )

    count_summary = (
        case_df.groupby("case_type")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    count_summary["fraction"] = count_summary["count"] / max(1, len(case_df))

    numeric_cols = [
        "baseline_dice",
        "target_dice",
        "dice_gap",
        "input_volume_vox",
        "future_delta_volume_vox",
        "future_relative_growth",
        "recent_delta_volume_vox",
        "recent_relative_growth",
        "delta_days",
        "input_elongation_ratio",
        "input_compactness_proxy",
        "input_connected_component_count",
        "n_sessions",
        "followup_days",
        "mean_interval_days",
    ]
    numeric_summary = summarize_numeric(case_df, "case_type", numeric_cols)
    tier_summary = summarize_categorical(case_df, "case_type", "tier")
    horizon_summary = summarize_categorical(case_df, "case_type", "horizon")
    volume_summary = summarize_categorical(case_df, "case_type", "input_volume_bin")
    growth_summary = summarize_categorical(case_df, "case_type", "recent_growth_bin")
    treatment_summary = summarize_categorical(case_df, "case_type", "treated_at_input")

    case_df.to_csv(output_dir / "case_type_samples.csv", index=False)
    count_summary.to_csv(output_dir / "case_type_counts.csv", index=False)
    numeric_summary.to_csv(output_dir / "case_type_numeric_summary.csv", index=False)
    tier_summary.to_csv(output_dir / "case_type_by_tier.csv", index=False)
    horizon_summary.to_csv(output_dir / "case_type_by_horizon.csv", index=False)
    volume_summary.to_csv(output_dir / "case_type_by_input_volume_bin.csv", index=False)
    growth_summary.to_csv(output_dir / "case_type_by_recent_growth_bin.csv", index=False)
    treatment_summary.to_csv(output_dir / "case_type_by_treated_at_input.csv", index=False)

    write_markdown_report(
        output_dir / "case_type_report.md",
        count_summary,
        numeric_summary,
        tier_summary,
        horizon_summary,
        volume_summary,
        growth_summary,
        treatment_summary,
    )

    print(f"Saved case-type analysis to: {output_dir}")


if __name__ == "__main__":
    main()
