#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ACTIVITY_FEATURES = [
    ("input_volume_vox", 1.0),
    ("recent_relative_growth", 1.0),
    ("delta_days", 1.0),
    ("treated_at_input", -1.0),
]

STRUCTURE_FEATURES = [
    ("input_connected_component_count", 1.0),
    ("input_compactness_proxy", 1.0),
    ("input_elongation_ratio", -1.0),
]


def zscore(series: pd.Series) -> pd.Series:
    arr = series.astype(float)
    std = float(arr.std(ddof=0))
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(arr)), index=arr.index)
    return (arr - float(arr.mean())) / std


def build_signed_z_columns(df: pd.DataFrame, signed_features: list[tuple[str, float]]) -> pd.DataFrame:
    out = df.copy()
    for feature, sign in signed_features:
        if feature not in out.columns:
            raise KeyError(f"Missing required feature: {feature}")
        out[f"z_{feature}"] = zscore(out[feature]) * float(sign)
    return out


def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = build_signed_z_columns(df, ACTIVITY_FEATURES)
    out = build_signed_z_columns(out, STRUCTURE_FEATURES)

    activity_cols = [f"z_{feature}" for feature, _ in ACTIVITY_FEATURES]
    structure_cols = [f"z_{feature}" for feature, _ in STRUCTURE_FEATURES]
    out["activity_score"] = out[activity_cols].mean(axis=1)
    out["structure_score"] = out[structure_cols].mean(axis=1)

    def quadrant(row: pd.Series) -> str:
        a = "highA" if row["activity_score"] >= 0 else "lowA"
        s = "highS" if row["structure_score"] >= 0 else "lowS"
        return f"{a}_{s}"

    out["regime_quadrant"] = out.apply(quadrant, axis=1)
    return out


def summarize_scores(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("case_type")
        .agg(
            count=("case_type", "size"),
            activity_score_mean=("activity_score", "mean"),
            activity_score_median=("activity_score", "median"),
            activity_score_std=("activity_score", "std"),
            structure_score_mean=("structure_score", "mean"),
            structure_score_median=("structure_score", "median"),
            structure_score_std=("structure_score", "std"),
        )
        .reset_index()
        .sort_values("activity_score_mean", ascending=False)
    )


def summarize_quadrants(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby(["case_type", "regime_quadrant"])
        .size()
        .reset_index(name="count")
        .sort_values(["case_type", "regime_quadrant"])
    )
    out["fraction"] = out["count"] / out.groupby("case_type")["count"].transform("sum")
    return out


def summarize_tier_quadrants(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby(["tier", "regime_quadrant"])
        .size()
        .reset_index(name="count")
        .sort_values(["tier", "regime_quadrant"])
    )
    out["fraction"] = out["count"] / out.groupby("tier")["count"].transform("sum")
    return out


def save_scatter(df: pd.DataFrame, out_path: Path) -> None:
    colors = {
        "both_easy": "#4C78A8",
        "target_wins": "#E45756",
        "both_hard": "#54A24B",
        "close_mixed": "#B279A2",
        "baseline_wins": "#F58518",
    }
    markers = {"A": "o", "B": "s", "C": "^"}

    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    for case_type, sub in df.groupby("case_type"):
        for tier, tier_sub in sub.groupby("tier"):
            ax.scatter(
                tier_sub["activity_score"],
                tier_sub["structure_score"],
                label=f"{case_type} | tier {tier}",
                c=colors.get(case_type, "#777777"),
                marker=markers.get(str(tier), "o"),
                alpha=0.7,
                s=42,
                edgecolors="none",
            )
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.35)
    ax.axvline(0.0, color="black", linewidth=1.0, alpha=0.35)
    ax.set_xlabel("Activity score")
    ax.set_ylabel("Structure score")
    ax.set_title("SRD regime map: activity vs structure")
    ax.grid(alpha=0.20, linestyle="--")

    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    filtered_h = []
    filtered_l = []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            filtered_h.append(h)
            filtered_l.append(l)
    ax.legend(filtered_h, filtered_l, frameon=False, fontsize=8, loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_boxplot(df: pd.DataFrame, col: str, out_path: Path, title: str) -> None:
    order = ["both_easy", "target_wins", "both_hard", "close_mixed", "baseline_wins"]
    groups = [df.loc[df["case_type"] == c, col].dropna().to_numpy() for c in order if c in set(df["case_type"])]
    labels = [c for c in order if c in set(df["case_type"])]
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.boxplot(groups, labels=labels, patch_artist=True)
    ax.set_title(title)
    ax.set_ylabel(col.replace("_", " "))
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(path: Path, score_summary: pd.DataFrame, quadrant_summary: pd.DataFrame, tier_quadrant_summary: pd.DataFrame) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Regime Map Report\n\n")
        f.write("## Score Summary By Case Type\n\n")
        f.write(score_summary.to_markdown(index=False))
        f.write("\n\n## Regime Quadrants By Case Type\n\n")
        f.write(quadrant_summary.to_markdown(index=False))
        f.write("\n\n## Regime Quadrants By Tier\n\n")
        f.write(tier_quadrant_summary.to_markdown(index=False))
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a two-axis regime map from forecast-origin descriptors.")
    parser.add_argument("--case_type_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    case_df = pd.read_csv(args.case_type_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scored = compute_scores(case_df)
    score_summary = summarize_scores(scored)
    quadrant_summary = summarize_quadrants(scored)
    tier_quadrant_summary = summarize_tier_quadrants(scored)

    scored.to_csv(out_dir / "regime_map_scored_cases.csv", index=False)
    score_summary.to_csv(out_dir / "regime_map_score_summary.csv", index=False)
    quadrant_summary.to_csv(out_dir / "regime_map_quadrants_by_case_type.csv", index=False)
    tier_quadrant_summary.to_csv(out_dir / "regime_map_quadrants_by_tier.csv", index=False)

    save_scatter(scored, out_dir / "regime_map_scatter.png")
    save_boxplot(scored, "activity_score", out_dir / "activity_score_by_case_type.png", "Activity score by case type")
    save_boxplot(scored, "structure_score", out_dir / "structure_score_by_case_type.png", "Structure score by case type")
    write_report(out_dir / "regime_map_report.md", score_summary, quadrant_summary, tier_quadrant_summary)

    manifest = {
        "case_type_csv": str(Path(args.case_type_csv).resolve()),
        "activity_features": ACTIVITY_FEATURES,
        "structure_features": STRUCTURE_FEATURES,
        "output_dir": str(out_dir.resolve()),
    }
    with (out_dir / "regime_map_summary.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
