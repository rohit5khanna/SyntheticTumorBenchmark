#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from analyze_case_types import assign_case_type
from analyze_soft_regime_membership import (
    assign_soft_membership,
    build_profiles,
    compute_scores,
    stable_profile_names,
)


DEFAULT_MIN_CASES = 20
DEFAULT_DOMINANT_FRACTION = 0.50
DEFAULT_TEMPERATURE = 0.25
DEFAULT_RATIO_THRESHOLD = 0.80
DEFAULT_PROB_THRESHOLD = 0.55


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def summarize_soft_story(df: pd.DataFrame, stable_types: list[str]) -> dict[str, object]:
    out: dict[str, object] = {}
    out["stable_types"] = ",".join(stable_types)
    out["n_stable_types"] = len(stable_types)
    out["has_both_easy_stable"] = int("both_easy" in stable_types)
    out["has_target_wins_stable"] = int("target_wins" in stable_types)
    out["stable_anchor_pair"] = int(("both_easy" in stable_types) and ("target_wins" in stable_types))

    total = max(1, len(df))
    label_counts = df["soft_regime_label"].value_counts(normalize=True)
    out["core_aligned_frac"] = float(label_counts.get("core_aligned", 0.0))
    out["cross_regime_pull_frac"] = float(label_counts.get("cross_regime_pull", 0.0))
    out["transition_frac"] = float(label_counts.get("transition", 0.0))

    for case_type in ["both_easy", "target_wins", "both_hard", "close_mixed"]:
        sub = df[df["case_type"] == case_type]
        denom = max(1, len(sub))
        out[f"{case_type}_count"] = int(len(sub))
        out[f"{case_type}_core_frac"] = float((sub["soft_regime_label"] == "core_aligned").mean()) if len(sub) else 0.0
        out[f"{case_type}_cross_pull_frac"] = float((sub["soft_regime_label"] == "cross_regime_pull").mean()) if len(sub) else 0.0
        out[f"{case_type}_transition_frac"] = float((sub["soft_regime_label"] == "transition").mean()) if len(sub) else 0.0

    for horizon in [1, 2, 3]:
        sub = df[df["horizon"] == horizon]
        out[f"h{horizon}_core_frac"] = float((sub["soft_regime_label"] == "core_aligned").mean()) if len(sub) else 0.0
        out[f"h{horizon}_cross_pull_frac"] = float((sub["soft_regime_label"] == "cross_regime_pull").mean()) if len(sub) else 0.0
        out[f"h{horizon}_transition_frac"] = float((sub["soft_regime_label"] == "transition").mean()) if len(sub) else 0.0

    for tier in ["A", "B", "C"]:
        sub = df[df["tier"] == tier]
        out[f"tier_{tier}_core_frac"] = float((sub["soft_regime_label"] == "core_aligned").mean()) if len(sub) else 0.0
        out[f"tier_{tier}_cross_pull_frac"] = float((sub["soft_regime_label"] == "cross_regime_pull").mean()) if len(sub) else 0.0
        out[f"tier_{tier}_transition_frac"] = float((sub["soft_regime_label"] == "transition").mean()) if len(sub) else 0.0

    out["h3_minus_h1_cross_pull_frac"] = float(out["h3_cross_pull_frac"] - out["h1_cross_pull_frac"])
    out["tierB_minus_tierA_transition_frac"] = float(out["tier_B_transition_frac"] - out["tier_A_transition_frac"])
    out["n_rows"] = int(total)
    return out


def run_soft_story(
    case_df: pd.DataFrame,
    min_cases: int,
    min_dominant_fraction: float,
    temperature: float,
    ratio_threshold: float,
    prob_threshold: float,
) -> tuple[pd.DataFrame, list[str], dict[str, object]]:
    scored = compute_scores(case_df)
    profiles = build_profiles(scored)
    stable_types = stable_profile_names(
        profiles,
        min_cases=min_cases,
        min_dominant_fraction=min_dominant_fraction,
    )
    if len(stable_types) < 2:
        scored = scored.copy()
        scored["soft_regime_label"] = "unassigned"
        return scored, stable_types, summarize_soft_story(scored, stable_types)

    assigned = assign_soft_membership(
        scored,
        profiles,
        stable_types=stable_types,
        temperature=temperature,
        ratio_threshold=ratio_threshold,
        prob_threshold=prob_threshold,
    )
    return assigned, stable_types, summarize_soft_story(assigned, stable_types)


def hard_sweep(
    pair: pd.DataFrame,
    gap_margins: list[float],
    high_dice_values: list[float],
    low_dice_values: list[float],
) -> pd.DataFrame:
    rows = []
    for gap_margin in gap_margins:
        for high_dice in high_dice_values:
            for low_dice in low_dice_values:
                case_df = assign_case_type(pair, gap_margin=gap_margin, high_dice=high_dice, low_dice=low_dice)
                _, stable_types, summary = run_soft_story(
                    case_df,
                    min_cases=DEFAULT_MIN_CASES,
                    min_dominant_fraction=DEFAULT_DOMINANT_FRACTION,
                    temperature=DEFAULT_TEMPERATURE,
                    ratio_threshold=DEFAULT_RATIO_THRESHOLD,
                    prob_threshold=DEFAULT_PROB_THRESHOLD,
                )
                summary.update(
                    {
                        "gap_margin": gap_margin,
                        "high_dice": high_dice,
                        "low_dice": low_dice,
                    }
                )
                rows.append(summary)
    return pd.DataFrame(rows)


def soft_sweep(
    case_df: pd.DataFrame,
    dominant_fractions: list[float],
    ratio_thresholds: list[float],
    prob_thresholds: list[float],
) -> pd.DataFrame:
    rows = []
    for dominant_fraction in dominant_fractions:
        for ratio_threshold in ratio_thresholds:
            for prob_threshold in prob_thresholds:
                _, stable_types, summary = run_soft_story(
                    case_df,
                    min_cases=DEFAULT_MIN_CASES,
                    min_dominant_fraction=dominant_fraction,
                    temperature=DEFAULT_TEMPERATURE,
                    ratio_threshold=ratio_threshold,
                    prob_threshold=prob_threshold,
                )
                summary.update(
                    {
                        "min_dominant_fraction": dominant_fraction,
                        "distance_ratio_threshold": ratio_threshold,
                        "top_prob_threshold": prob_threshold,
                    }
                )
                rows.append(summary)
    return pd.DataFrame(rows)


def build_robustness_summary(hard_df: pd.DataFrame, soft_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add_block(name: str, df: pd.DataFrame) -> None:
        rows.append(
            {
                "sweep": name,
                "n_settings": int(len(df)),
                "stable_anchor_pair_rate": float(df["stable_anchor_pair"].mean()),
                "both_easy_stable_rate": float(df["has_both_easy_stable"].mean()),
                "target_wins_stable_rate": float(df["has_target_wins_stable"].mean()),
                "both_easy_core_frac_mean": float(df["both_easy_core_frac"].mean()),
                "both_easy_core_frac_min": float(df["both_easy_core_frac"].min()),
                "target_wins_core_frac_mean": float(df["target_wins_core_frac"].mean()),
                "target_wins_core_frac_min": float(df["target_wins_core_frac"].min()),
                "cross_pull_h3_gt_h1_rate": float((df["h3_minus_h1_cross_pull_frac"] > 0).mean()),
                "tierB_transition_gt_tierA_rate": float((df["tierB_minus_tierA_transition_frac"] > 0).mean()),
            }
        )

    add_block("hard_thresholds", hard_df)
    add_block("soft_thresholds", soft_df)
    return pd.DataFrame(rows)


def write_report(path: Path, overall: pd.DataFrame, hard_df: pd.DataFrame, soft_df: pd.DataFrame) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Threshold Sensitivity Report\n\n")
        f.write("This report checks whether the SRD regime story survives threshold changes in the hard case labeling and soft membership layers.\n\n")
        f.write("## Overall Robustness Summary\n\n")
        f.write(overall.to_markdown(index=False))
        f.write("\n\n## Hard-Threshold Sweep\n\n")
        f.write(hard_df.to_markdown(index=False))
        f.write("\n\n## Soft-Threshold Sweep\n\n")
        f.write(soft_df.to_markdown(index=False))
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stress-test the SRD regime story under hard and soft threshold changes.")
    parser.add_argument("--pairwise_csv", type=str, default=None)
    parser.add_argument("--case_type_csv", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--gap_margins", type=str, default="0.03,0.05,0.07")
    parser.add_argument("--high_dice_values", type=str, default="0.82,0.85,0.88")
    parser.add_argument("--low_dice_values", type=str, default="0.65,0.70,0.75")
    parser.add_argument("--dominant_fractions", type=str, default="0.45,0.50,0.55")
    parser.add_argument("--ratio_thresholds", type=str, default="0.75,0.80,0.85")
    parser.add_argument("--prob_thresholds", type=str, default="0.50,0.55,0.60")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if bool(args.pairwise_csv) == bool(args.case_type_csv):
        raise ValueError("Provide exactly one of --pairwise_csv or --case_type_csv.")

    source_csv = args.pairwise_csv if args.pairwise_csv else args.case_type_csv
    pair = pd.read_csv(source_csv)
    hard_df = hard_sweep(
        pair,
        gap_margins=parse_float_list(args.gap_margins),
        high_dice_values=parse_float_list(args.high_dice_values),
        low_dice_values=parse_float_list(args.low_dice_values),
    )

    default_case_df = assign_case_type(pair, gap_margin=0.05, high_dice=0.85, low_dice=0.70)
    soft_df = soft_sweep(
        default_case_df,
        dominant_fractions=parse_float_list(args.dominant_fractions),
        ratio_thresholds=parse_float_list(args.ratio_thresholds),
        prob_thresholds=parse_float_list(args.prob_thresholds),
    )

    overall = build_robustness_summary(hard_df, soft_df)

    hard_df.to_csv(out_dir / "hard_threshold_sweep.csv", index=False)
    soft_df.to_csv(out_dir / "soft_threshold_sweep.csv", index=False)
    overall.to_csv(out_dir / "threshold_robustness_summary.csv", index=False)
    write_report(out_dir / "threshold_sensitivity_report.md", overall, hard_df, soft_df)

    manifest = {
        "source_csv": str(Path(source_csv).resolve()),
        "source_kind": "pairwise_csv" if args.pairwise_csv else "case_type_csv",
        "gap_margins": parse_float_list(args.gap_margins),
        "high_dice_values": parse_float_list(args.high_dice_values),
        "low_dice_values": parse_float_list(args.low_dice_values),
        "dominant_fractions": parse_float_list(args.dominant_fractions),
        "ratio_thresholds": parse_float_list(args.ratio_thresholds),
        "prob_thresholds": parse_float_list(args.prob_thresholds),
        "output_dir": str(out_dir.resolve()),
    }
    with (out_dir / "threshold_sensitivity_summary.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
