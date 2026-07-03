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


def build_profiles(df: pd.DataFrame) -> pd.DataFrame:
    quadrant_counts = (
        df.groupby(["case_type", "regime_quadrant"])
        .size()
        .reset_index(name="count")
    )
    quadrant_counts["fraction"] = quadrant_counts["count"] / quadrant_counts.groupby("case_type")["count"].transform("sum")
    dominant = (
        quadrant_counts.sort_values(["case_type", "fraction", "count"], ascending=[True, False, False])
        .drop_duplicates("case_type")
        .rename(
            columns={
                "regime_quadrant": "dominant_quadrant",
                "count": "dominant_quadrant_count",
                "fraction": "dominant_quadrant_fraction",
            }
        )
    )

    profiles = (
        df.groupby("case_type")
        .agg(
            case_count=("case_type", "size"),
            centroid_activity=("activity_score", "mean"),
            centroid_structure=("structure_score", "mean"),
            activity_std=("activity_score", "std"),
            structure_std=("structure_score", "std"),
        )
        .reset_index()
        .merge(
            dominant[["case_type", "dominant_quadrant", "dominant_quadrant_count", "dominant_quadrant_fraction"]],
            on="case_type",
            how="left",
        )
    )
    return profiles


def stable_profile_names(profiles: pd.DataFrame, min_cases: int, min_dominant_fraction: float) -> list[str]:
    stable = profiles[
        (profiles["case_count"] >= min_cases)
        & (profiles["dominant_quadrant_fraction"] >= min_dominant_fraction)
    ].copy()
    return stable["case_type"].tolist()


def softmax_negative_distance(distances: np.ndarray, temperature: float) -> np.ndarray:
    scaled = -distances / max(temperature, 1e-6)
    scaled = scaled - scaled.max()
    weights = np.exp(scaled)
    denom = weights.sum()
    if denom <= 0:
        return np.ones_like(weights) / len(weights)
    return weights / denom


def assign_soft_membership(
    df: pd.DataFrame,
    profiles: pd.DataFrame,
    stable_types: list[str],
    temperature: float,
    ratio_threshold: float,
    prob_threshold: float,
) -> pd.DataFrame:
    stable_profiles = profiles[profiles["case_type"].isin(stable_types)].copy()
    stable_profiles = stable_profiles.sort_values("case_type").reset_index(drop=True)

    centroids = {
        row["case_type"]: (float(row["centroid_activity"]), float(row["centroid_structure"]))
        for _, row in stable_profiles.iterrows()
    }
    names = list(centroids.keys())
    if len(names) < 2:
        raise ValueError("Need at least two stable profiles for soft membership analysis.")

    out = df.copy()
    top_types: list[str] = []
    second_types: list[str] = []
    top_distances: list[float] = []
    second_distances: list[float] = []
    distance_ratios: list[float] = []
    top_probs: list[float] = []
    second_probs: list[float] = []
    labels: list[str] = []

    membership_cols: dict[str, list[float]] = {f"membership_{name}": [] for name in names}

    for _, row in out.iterrows():
        point = np.array([float(row["activity_score"]), float(row["structure_score"])])
        dists = np.array([np.linalg.norm(point - np.array(centroids[name])) for name in names], dtype=float)
        order = np.argsort(dists)
        top_idx = int(order[0])
        second_idx = int(order[1])

        probs = softmax_negative_distance(dists, temperature=temperature)

        top_name = names[top_idx]
        second_name = names[second_idx]
        top_dist = float(dists[top_idx])
        second_dist = float(dists[second_idx])
        ratio = top_dist / max(second_dist, 1e-6)
        top_prob = float(probs[top_idx])
        second_prob = float(probs[second_idx])

        if (ratio <= ratio_threshold) and (top_prob >= prob_threshold):
            if row["case_type"] == top_name:
                label = "core_aligned"
            else:
                label = "cross_regime_pull"
        else:
            label = "transition"

        top_types.append(top_name)
        second_types.append(second_name)
        top_distances.append(top_dist)
        second_distances.append(second_dist)
        distance_ratios.append(ratio)
        top_probs.append(top_prob)
        second_probs.append(second_prob)
        labels.append(label)

        for name, prob in zip(names, probs):
            membership_cols[f"membership_{name}"].append(float(prob))

    out["top_stable_regime"] = top_types
    out["second_stable_regime"] = second_types
    out["distance_to_top_stable_regime"] = top_distances
    out["distance_to_second_stable_regime"] = second_distances
    out["distance_ratio_top_to_second"] = distance_ratios
    out["top_stable_membership_prob"] = top_probs
    out["second_stable_membership_prob"] = second_probs
    out["soft_regime_label"] = labels

    for col, values in membership_cols.items():
        out[col] = values
    return out


def summarize_soft_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.groupby("soft_regime_label").size().reset_index(name="count").sort_values("count", ascending=False)
    out["fraction"] = out["count"] / max(1, len(df))
    return out


def summarize_by_case_type(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby(["case_type", "soft_regime_label"])
        .size()
        .reset_index(name="count")
        .sort_values(["case_type", "soft_regime_label"])
    )
    out["fraction_within_case_type"] = out["count"] / out.groupby("case_type")["count"].transform("sum")
    return out


def summarize_by_tier(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby(["tier", "soft_regime_label"])
        .size()
        .reset_index(name="count")
        .sort_values(["tier", "soft_regime_label"])
    )
    out["fraction_within_tier"] = out["count"] / out.groupby("tier")["count"].transform("sum")
    return out


def summarize_by_horizon(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby(["horizon", "soft_regime_label"])
        .size()
        .reset_index(name="count")
        .sort_values(["horizon", "soft_regime_label"])
    )
    out["fraction_within_horizon"] = out["count"] / out.groupby("horizon")["count"].transform("sum")
    return out


def summarize_transition_flows(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["case_type", "top_stable_regime", "soft_regime_label"]
    out = (
        df.groupby(cols)
        .size()
        .reset_index(name="count")
        .sort_values(cols)
    )
    out["fraction_within_case_type"] = out["count"] / out.groupby("case_type")["count"].transform("sum")
    return out


def summarize_membership_confidence(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "top_stable_membership_prob",
        "distance_ratio_top_to_second",
        "distance_to_top_stable_regime",
        "distance_to_second_stable_regime",
    ]
    out = (
        df.groupby("soft_regime_label")[cols]
        .agg(["mean", "median", "std"])
    )
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    return out.reset_index()


def save_scatter(df: pd.DataFrame, stable_types: list[str], out_path: Path) -> None:
    colors = {
        "core_aligned": "#2A9D8F",
        "transition": "#E9C46A",
        "cross_regime_pull": "#D1495B",
    }
    fig, ax = plt.subplots(figsize=(7.6, 6.0))
    for label, sub in df.groupby("soft_regime_label"):
        ax.scatter(
            sub["activity_score"],
            sub["structure_score"],
            c=colors.get(label, "#777777"),
            alpha=0.75,
            s=34,
            label=label,
            edgecolors="none",
        )

    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.30)
    ax.axvline(0.0, color="black", linewidth=1.0, alpha=0.30)
    ax.set_xlabel("Activity score")
    ax.set_ylabel("Structure score")
    ax.set_title(f"Soft regime membership map ({', '.join(stable_types)})")
    ax.grid(alpha=0.18, linestyle="--")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(
    path: Path,
    profiles: pd.DataFrame,
    stable_types: list[str],
    soft_summary: pd.DataFrame,
    by_case_type: pd.DataFrame,
    by_tier: pd.DataFrame,
    by_horizon: pd.DataFrame,
    transition_flows: pd.DataFrame,
    confidence_summary: pd.DataFrame,
) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Soft Regime Membership Report\n\n")
        f.write("This analysis replaces hard exception logic with a softer view of regime support, using only the stable descriptor cores.\n\n")
        f.write("## Stable Profiles Used\n\n")
        f.write(f"- Stable regime anchors: `{', '.join(stable_types)}`\n\n")
        f.write(profiles.to_markdown(index=False))
        f.write("\n\n## Soft Label Summary\n\n")
        f.write(soft_summary.to_markdown(index=False))
        f.write("\n\n## Soft Labels By Original Case Type\n\n")
        f.write(by_case_type.to_markdown(index=False))
        f.write("\n\n## Soft Labels By Tier\n\n")
        f.write(by_tier.to_markdown(index=False))
        f.write("\n\n## Soft Labels By Horizon\n\n")
        f.write(by_horizon.to_markdown(index=False))
        f.write("\n\n## Transition Flows\n\n")
        f.write(transition_flows.to_markdown(index=False))
        f.write("\n\n## Membership Confidence Summary\n\n")
        f.write(confidence_summary.to_markdown(index=False))
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a softer regime-membership view from stable descriptor cores.")
    parser.add_argument("--case_type_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--min_cases_for_stable_profile", type=int, default=20)
    parser.add_argument("--min_dominant_fraction", type=float, default=0.50)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--distance_ratio_threshold", type=float, default=0.80)
    parser.add_argument("--top_prob_threshold", type=float, default=0.55)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    case_df = pd.read_csv(args.case_type_csv)
    scored = compute_scores(case_df)
    profiles = build_profiles(scored)
    stable_types = stable_profile_names(
        profiles,
        min_cases=args.min_cases_for_stable_profile,
        min_dominant_fraction=args.min_dominant_fraction,
    )
    if len(stable_types) < 2:
        raise ValueError("Fewer than two stable profiles identified. Relax thresholds or inspect the descriptor map.")

    assigned = assign_soft_membership(
        scored,
        profiles,
        stable_types=stable_types,
        temperature=args.temperature,
        ratio_threshold=args.distance_ratio_threshold,
        prob_threshold=args.top_prob_threshold,
    )

    soft_summary = summarize_soft_labels(assigned)
    by_case_type = summarize_by_case_type(assigned)
    by_tier = summarize_by_tier(assigned)
    by_horizon = summarize_by_horizon(assigned)
    transition_flows = summarize_transition_flows(assigned)
    confidence_summary = summarize_membership_confidence(assigned)

    profiles.to_csv(out_dir / "stable_case_type_profiles.csv", index=False)
    assigned.to_csv(out_dir / "soft_regime_membership_samples.csv", index=False)
    soft_summary.to_csv(out_dir / "soft_regime_label_summary.csv", index=False)
    by_case_type.to_csv(out_dir / "soft_regime_by_case_type.csv", index=False)
    by_tier.to_csv(out_dir / "soft_regime_by_tier.csv", index=False)
    by_horizon.to_csv(out_dir / "soft_regime_by_horizon.csv", index=False)
    transition_flows.to_csv(out_dir / "soft_regime_transition_flows.csv", index=False)
    confidence_summary.to_csv(out_dir / "soft_regime_confidence_summary.csv", index=False)

    save_scatter(assigned, stable_types, out_dir / "soft_regime_membership_map.png")
    write_report(
        out_dir / "soft_regime_membership_report.md",
        profiles,
        stable_types,
        soft_summary,
        by_case_type,
        by_tier,
        by_horizon,
        transition_flows,
        confidence_summary,
    )

    manifest = {
        "case_type_csv": str(Path(args.case_type_csv).resolve()),
        "activity_features": ACTIVITY_FEATURES,
        "structure_features": STRUCTURE_FEATURES,
        "stable_types": stable_types,
        "min_cases_for_stable_profile": args.min_cases_for_stable_profile,
        "min_dominant_fraction": args.min_dominant_fraction,
        "temperature": args.temperature,
        "distance_ratio_threshold": args.distance_ratio_threshold,
        "top_prob_threshold": args.top_prob_threshold,
        "output_dir": str(out_dir.resolve()),
    }
    with (out_dir / "soft_regime_membership_summary.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
