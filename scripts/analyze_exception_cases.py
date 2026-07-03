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


def build_case_type_profiles(df: pd.DataFrame) -> pd.DataFrame:
    counts = (
        df.groupby(["case_type", "regime_quadrant"])
        .size()
        .reset_index(name="count")
    )
    totals = counts.groupby("case_type")["count"].transform("sum")
    counts["fraction"] = counts["count"] / totals

    dominant = (
        counts.sort_values(["case_type", "fraction", "count"], ascending=[True, False, False])
        .drop_duplicates("case_type")
        .rename(
            columns={
                "regime_quadrant": "dominant_quadrant",
                "count": "dominant_quadrant_count",
                "fraction": "dominant_quadrant_fraction",
            }
        )
    )

    centroids = (
        df.groupby("case_type")
        .agg(
            case_count=("case_type", "size"),
            centroid_activity=("activity_score", "mean"),
            centroid_structure=("structure_score", "mean"),
            activity_std=("activity_score", "std"),
            structure_std=("structure_score", "std"),
        )
        .reset_index()
    )
    return centroids.merge(
        dominant[["case_type", "dominant_quadrant", "dominant_quadrant_count", "dominant_quadrant_fraction"]],
        on="case_type",
        how="left",
    )


def attach_centroid_distances(df: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    centroid_lookup = {
        row["case_type"]: (float(row["centroid_activity"]), float(row["centroid_structure"]))
        for _, row in profiles.iterrows()
    }

    own_distances: list[float] = []
    nearest_other_type: list[str] = []
    nearest_other_distance: list[float] = []
    closer_to_other: list[bool] = []

    for _, row in out.iterrows():
        own = centroid_lookup[row["case_type"]]
        ax = float(row["activity_score"])
        sx = float(row["structure_score"])
        own_dist = float(np.hypot(ax - own[0], sx - own[1]))

        other_dists = []
        for case_type, centroid in centroid_lookup.items():
            if case_type == row["case_type"]:
                continue
            dist = float(np.hypot(ax - centroid[0], sx - centroid[1]))
            other_dists.append((case_type, dist))

        other_dists.sort(key=lambda x: x[1])
        if other_dists:
            nearest_type, nearest_dist = other_dists[0]
        else:
            nearest_type, nearest_dist = row["case_type"], own_dist

        own_distances.append(own_dist)
        nearest_other_type.append(str(nearest_type))
        nearest_other_distance.append(float(nearest_dist))
        closer_to_other.append(bool(nearest_dist < own_dist))

    out["distance_to_own_centroid"] = own_distances
    out["nearest_other_case_type"] = nearest_other_type
    out["distance_to_nearest_other_centroid"] = nearest_other_distance
    out["closer_to_other_centroid"] = closer_to_other
    return out


def classify_exceptions(
    df: pd.DataFrame,
    profiles: pd.DataFrame,
    quadrant_share_threshold: float,
    rare_count_threshold: int,
) -> pd.DataFrame:
    out = df.merge(
        profiles[
            [
                "case_type",
                "case_count",
                "dominant_quadrant",
                "dominant_quadrant_fraction",
            ]
        ],
        on="case_type",
        how="left",
    ).copy()

    out["has_stable_quadrant"] = out["dominant_quadrant_fraction"] >= quadrant_share_threshold
    out["quadrant_exception"] = out["has_stable_quadrant"] & (out["regime_quadrant"] != out["dominant_quadrant"])
    out["rare_case_type_exception"] = out["case_count"] <= rare_count_threshold
    out["centroid_exception"] = out["closer_to_other_centroid"]

    reason_labels: list[str] = []
    for _, row in out.iterrows():
        reasons = []
        if bool(row["rare_case_type_exception"]):
            reasons.append("rare_case_type")
        if bool(row["quadrant_exception"]):
            reasons.append("quadrant_mismatch")
        if bool(row["centroid_exception"]):
            reasons.append("closer_to_other_centroid")
        reason_labels.append(";".join(reasons) if reasons else "none")

    out["exception_reasons"] = reason_labels
    out["is_exception_case"] = out["exception_reasons"] != "none"
    return out


def summarize_reason_counts(df: pd.DataFrame) -> pd.DataFrame:
    exploded = (
        df.loc[df["is_exception_case"], ["exception_reasons"]]
        .assign(exception_reason=lambda x: x["exception_reasons"].str.split(";"))
        .explode("exception_reason")
    )
    if exploded.empty:
        return pd.DataFrame(columns=["exception_reason", "count", "fraction"])
    out = exploded.groupby("exception_reason").size().reset_index(name="count")
    out["fraction"] = out["count"] / max(1, len(df.loc[df["is_exception_case"]]))
    return out.sort_values("count", ascending=False)


def summarize_categorical(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = (
        df.loc[df["is_exception_case"]]
        .groupby(["case_type", col])
        .size()
        .reset_index(name="count")
        .sort_values(["case_type", col])
    )
    if out.empty:
        return out
    out["fraction_within_case_type"] = out["count"] / out.groupby("case_type")["count"].transform("sum")
    return out


def summarize_numeric(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "dice_gap",
        "baseline_dice",
        "target_dice",
        "activity_score",
        "structure_score",
        "distance_to_own_centroid",
        "distance_to_nearest_other_centroid",
        "input_volume_vox",
        "recent_relative_growth",
        "delta_days",
        "input_connected_component_count",
    ]
    available = [c for c in cols if c in df.columns]
    if not available:
        return pd.DataFrame()
    out = (
        df.loc[df["is_exception_case"]]
        .groupby("case_type")[available]
        .agg(["mean", "median", "std"])
    )
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    return out.reset_index()


def build_top_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    flagged = df.loc[df["is_exception_case"]].copy()
    if flagged.empty:
        empty = pd.DataFrame()
        return {
            "largest_positive_gap_off_profile": empty,
            "largest_negative_gap_off_profile": empty,
            "closest_cross_type_cases": empty,
        }

    cols = [
        "patient_id",
        "tier",
        "horizon",
        "case_type",
        "regime_quadrant",
        "dominant_quadrant",
        "dice_gap",
        "baseline_dice",
        "target_dice",
        "activity_score",
        "structure_score",
        "nearest_other_case_type",
        "distance_to_own_centroid",
        "distance_to_nearest_other_centroid",
        "exception_reasons",
    ]
    cols = [c for c in cols if c in flagged.columns]

    return {
        "largest_positive_gap_off_profile": flagged.sort_values("dice_gap", ascending=False).head(15)[cols],
        "largest_negative_gap_off_profile": flagged.sort_values("dice_gap", ascending=True).head(15)[cols],
        "closest_cross_type_cases": flagged.sort_values("distance_to_nearest_other_centroid", ascending=True).head(15)[cols],
    }


def save_scatter(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 6.0))
    normal = df.loc[~df["is_exception_case"]]
    flagged = df.loc[df["is_exception_case"]]

    if not normal.empty:
        ax.scatter(
            normal["activity_score"],
            normal["structure_score"],
            c="#AAB7C4",
            alpha=0.45,
            s=26,
            label="non-exception",
            edgecolors="none",
        )
    if not flagged.empty:
        ax.scatter(
            flagged["activity_score"],
            flagged["structure_score"],
            c="#D1495B",
            alpha=0.90,
            s=46,
            label="exception",
            edgecolors="black",
            linewidths=0.25,
        )

    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.30)
    ax.axvline(0.0, color="black", linewidth=1.0, alpha=0.30)
    ax.set_xlabel("Activity score")
    ax.set_ylabel("Structure score")
    ax.set_title("Exception-case map in descriptor space")
    ax.grid(alpha=0.18, linestyle="--")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(
    path: Path,
    profiles: pd.DataFrame,
    reason_counts: pd.DataFrame,
    case_type_counts: pd.DataFrame,
    tier_summary: pd.DataFrame,
    horizon_summary: pd.DataFrame,
    numeric_summary: pd.DataFrame,
    top_tables: dict[str, pd.DataFrame],
) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Exception-Case Audit\n\n")
        f.write("This audit isolates cases that break the dominant descriptor-level profile of their case type.\n\n")

        f.write("## Case-Type Profiles\n\n")
        f.write(profiles.to_markdown(index=False))
        f.write("\n\n## Exception Reasons\n\n")
        if not reason_counts.empty:
            f.write(reason_counts.to_markdown(index=False))
        else:
            f.write("No exception cases were flagged.")
        f.write("\n\n## Exception Counts By Case Type\n\n")
        if not case_type_counts.empty:
            f.write(case_type_counts.to_markdown(index=False))
        else:
            f.write("No exception cases were flagged.")
        f.write("\n\n## Exception Counts By Tier\n\n")
        if not tier_summary.empty:
            f.write(tier_summary.to_markdown(index=False))
        else:
            f.write("No tier-level exceptions were flagged.")
        f.write("\n\n## Exception Counts By Horizon\n\n")
        if not horizon_summary.empty:
            f.write(horizon_summary.to_markdown(index=False))
        else:
            f.write("No horizon-level exceptions were flagged.")
        f.write("\n\n## Numeric Summary For Exception Cases\n\n")
        if not numeric_summary.empty:
            f.write(numeric_summary.to_markdown(index=False))
        else:
            f.write("No numeric exception summary available.")

        for title, table in top_tables.items():
            f.write(f"\n\n## {title.replace('_', ' ').title()}\n\n")
            if not table.empty:
                f.write(table.to_markdown(index=False))
            else:
                f.write("No rows available.")
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit exception cases that break the dominant regime-profile story.")
    parser.add_argument("--case_type_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--quadrant_share_threshold", type=float, default=0.50)
    parser.add_argument("--rare_count_threshold", type=int, default=3)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    case_df = pd.read_csv(args.case_type_csv)
    scored = compute_scores(case_df)
    profiles = build_case_type_profiles(scored)
    scored = attach_centroid_distances(scored, profiles)
    flagged = classify_exceptions(
        scored,
        profiles,
        quadrant_share_threshold=args.quadrant_share_threshold,
        rare_count_threshold=args.rare_count_threshold,
    )

    reason_counts = summarize_reason_counts(flagged)
    exception_counts = (
        flagged.loc[flagged["is_exception_case"]]
        .groupby("case_type")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    if not exception_counts.empty:
        exception_counts["fraction_within_case_type"] = (
            exception_counts["count"]
            / exception_counts["case_type"].map(flagged["case_type"].value_counts())
        )

    tier_summary = summarize_categorical(flagged, "tier")
    horizon_summary = summarize_categorical(flagged, "horizon")
    numeric_summary = summarize_numeric(flagged)
    top_tables = build_top_tables(flagged)

    flagged.to_csv(out_dir / "exception_case_samples.csv", index=False)
    profiles.to_csv(out_dir / "case_type_profiles.csv", index=False)
    reason_counts.to_csv(out_dir / "exception_reason_counts.csv", index=False)
    exception_counts.to_csv(out_dir / "exception_counts_by_case_type.csv", index=False)
    tier_summary.to_csv(out_dir / "exception_counts_by_tier.csv", index=False)
    horizon_summary.to_csv(out_dir / "exception_counts_by_horizon.csv", index=False)
    numeric_summary.to_csv(out_dir / "exception_numeric_summary.csv", index=False)
    for name, table in top_tables.items():
        table.to_csv(out_dir / f"{name}.csv", index=False)

    save_scatter(flagged, out_dir / "exception_case_map.png")
    write_report(
        out_dir / "exception_case_report.md",
        profiles,
        reason_counts,
        exception_counts,
        tier_summary,
        horizon_summary,
        numeric_summary,
        top_tables,
    )

    manifest = {
        "case_type_csv": str(Path(args.case_type_csv).resolve()),
        "activity_features": ACTIVITY_FEATURES,
        "structure_features": STRUCTURE_FEATURES,
        "quadrant_share_threshold": args.quadrant_share_threshold,
        "rare_count_threshold": args.rare_count_threshold,
        "output_dir": str(out_dir.resolve()),
    }
    with (out_dir / "exception_case_summary.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
