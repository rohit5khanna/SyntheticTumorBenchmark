#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


GROUP_ORDER = [
    "both_easy_core",
    "target_wins_core",
    "cross_regime_pull",
    "transition",
]

GROUP_LABELS = {
    "both_easy_core": "Both-Easy Core",
    "target_wins_core": "Target-Wins Core",
    "cross_regime_pull": "Cross-Regime Pull",
    "transition": "Transition",
}

GROUP_COLORS = {
    "both_easy_core": "#4C78A8",
    "target_wins_core": "#E45756",
    "cross_regime_pull": "#72B7B2",
    "transition": "#E9C46A",
}

PROFILE_FEATURES = [
    "activity_score",
    "structure_score",
    "input_volume_vox",
    "recent_relative_growth",
    "delta_days",
    "treated_at_input",
    "input_connected_component_count",
    "input_compactness_proxy",
    "input_elongation_ratio",
]


def assign_profile_group(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["profile_group"] = "other"

    both_easy_core = (out["case_type"] == "both_easy") & (out["soft_regime_label"] == "core_aligned")
    target_wins_core = (out["case_type"] == "target_wins") & (out["soft_regime_label"] == "core_aligned")
    cross_pull = out["soft_regime_label"] == "cross_regime_pull"
    transition = out["soft_regime_label"] == "transition"

    out.loc[both_easy_core, "profile_group"] = "both_easy_core"
    out.loc[target_wins_core, "profile_group"] = "target_wins_core"
    out.loc[cross_pull, "profile_group"] = "cross_regime_pull"
    out.loc[transition, "profile_group"] = "transition"
    return out


def summarize_counts(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df[df["profile_group"].isin(GROUP_ORDER)]
        .groupby("profile_group")
        .size()
        .reindex(GROUP_ORDER, fill_value=0)
        .reset_index(name="count")
    )
    out["fraction"] = out["count"] / max(1, int(out["count"].sum()))
    out["group_label"] = out["profile_group"].map(GROUP_LABELS)
    return out


def summarize_numeric(df: pd.DataFrame) -> pd.DataFrame:
    available = [c for c in PROFILE_FEATURES if c in df.columns]
    out = (
        df[df["profile_group"].isin(GROUP_ORDER)]
        .groupby("profile_group")[available]
        .agg(["mean", "median", "std"])
    )
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out = out.reset_index()
    out["group_label"] = out["profile_group"].map(GROUP_LABELS)
    return out


def summarize_categorical(df: pd.DataFrame, cat_col: str) -> pd.DataFrame:
    out = (
        df[df["profile_group"].isin(GROUP_ORDER)]
        .groupby(["profile_group", cat_col])
        .size()
        .reset_index(name="count")
        .sort_values(["profile_group", cat_col])
    )
    if out.empty:
        return out
    out["fraction_within_group"] = out["count"] / out.groupby("profile_group")["count"].transform("sum")
    out["group_label"] = out["profile_group"].map(GROUP_LABELS)
    return out


def save_boxplot(df: pd.DataFrame, feature: str, out_path: Path) -> None:
    groups = []
    labels = []
    colors = []
    filtered = df[df["profile_group"].isin(GROUP_ORDER)]
    for group in GROUP_ORDER:
        vals = filtered.loc[filtered["profile_group"] == group, feature].dropna().to_numpy()
        if len(vals) == 0:
            continue
        groups.append(vals)
        labels.append(GROUP_LABELS[group])
        colors.append(GROUP_COLORS[group])

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    bp = ax.boxplot(groups, tick_labels=labels, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.70)
    ax.set_title(f"{feature.replace('_', ' ').title()} by soft regime profile")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_stacked_bar(summary_df: pd.DataFrame, group_col: str, frac_col: str, out_path: Path, title: str) -> None:
    pivot = (
        summary_df.pivot(index=group_col, columns="profile_group", values=frac_col)
        .fillna(0.0)
        .reindex(columns=[g for g in GROUP_ORDER if g in summary_df["profile_group"].unique()])
    )
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    bottom = None
    x = range(len(pivot.index))
    for group in pivot.columns:
        vals = pivot[group].to_numpy()
        ax.bar(
            x,
            vals,
            bottom=bottom,
            label=GROUP_LABELS[group],
            color=GROUP_COLORS[group],
            alpha=0.85,
        )
        bottom = vals if bottom is None else bottom + vals

    ax.set_xticks(list(x))
    ax.set_xticklabels([str(v) for v in pivot.index])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Fraction within group")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(
    path: Path,
    count_summary: pd.DataFrame,
    numeric_summary: pd.DataFrame,
    tier_summary: pd.DataFrame,
    horizon_summary: pd.DataFrame,
) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Soft Regime Profile Report\n\n")
        f.write("This report characterizes the main soft-regime populations: persistence core, learned-advantage core, cross-regime pull, and transition.\n\n")
        f.write("## Group Counts\n\n")
        f.write(count_summary.to_markdown(index=False))
        f.write("\n\n## Numeric Feature Summary\n\n")
        f.write(numeric_summary.to_markdown(index=False))
        f.write("\n\n## By Tier\n\n")
        f.write(tier_summary.to_markdown(index=False))
        f.write("\n\n## By Horizon\n\n")
        f.write(horizon_summary.to_markdown(index=False))
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Characterize the core and ambiguous soft-regime populations.")
    parser.add_argument("--soft_membership_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.soft_membership_csv)
    df = assign_profile_group(df)
    df = df[df["profile_group"].isin(GROUP_ORDER)].copy()

    count_summary = summarize_counts(df)
    numeric_summary = summarize_numeric(df)
    tier_summary = summarize_categorical(df, "tier")
    horizon_summary = summarize_categorical(df, "horizon")

    count_summary.to_csv(out_dir / "soft_regime_profile_counts.csv", index=False)
    numeric_summary.to_csv(out_dir / "soft_regime_profile_numeric_summary.csv", index=False)
    tier_summary.to_csv(out_dir / "soft_regime_profile_by_tier.csv", index=False)
    horizon_summary.to_csv(out_dir / "soft_regime_profile_by_horizon.csv", index=False)
    df.to_csv(out_dir / "soft_regime_profile_samples.csv", index=False)

    for feature in PROFILE_FEATURES:
        if feature in df.columns:
            save_boxplot(df, feature, out_dir / f"{feature}_by_soft_regime_profile.png")

    save_stacked_bar(tier_summary, "tier", "fraction_within_group", out_dir / "soft_regime_profiles_by_tier.png", "Soft regime profile composition by tier")
    save_stacked_bar(horizon_summary, "horizon", "fraction_within_group", out_dir / "soft_regime_profiles_by_horizon.png", "Soft regime profile composition by horizon")

    write_report(
        out_dir / "soft_regime_profile_report.md",
        count_summary,
        numeric_summary,
        tier_summary,
        horizon_summary,
    )

    manifest = {
        "soft_membership_csv": str(Path(args.soft_membership_csv).resolve()),
        "groups": GROUP_ORDER,
        "features": PROFILE_FEATURES,
        "output_dir": str(out_dir.resolve()),
    }
    with (out_dir / "soft_regime_profile_summary.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
