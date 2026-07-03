#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from analyze_anchor_separation import FULL_NUMERIC_FEATURES, RAW_NUMERIC_FEATURES


GROUP_ORDER = ["both_easy_core", "target_wins_core", "cross_regime_pull", "transition"]
GROUP_COLORS = {
    "both_easy_core": "#4C78A8",
    "target_wins_core": "#E45756",
    "cross_regime_pull": "#72B7B2",
    "transition": "#E9C46A",
}


def assign_profile_group(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["profile_group"] = "other"
    out.loc[(out["case_type"] == "both_easy") & (out["soft_regime_label"] == "core_aligned"), "profile_group"] = "both_easy_core"
    out.loc[(out["case_type"] == "target_wins") & (out["soft_regime_label"] == "core_aligned"), "profile_group"] = "target_wins_core"
    out.loc[out["soft_regime_label"] == "cross_regime_pull", "profile_group"] = "cross_regime_pull"
    out.loc[out["soft_regime_label"] == "transition", "profile_group"] = "transition"
    return out


def fit_pca(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    available = [c for c in features if c in df.columns]
    X = df[available].copy()

    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=2)),
        ]
    )
    scores = pipe.fit_transform(X)
    pca = pipe.named_steps["pca"]

    score_df = df.copy()
    score_df["pc1"] = scores[:, 0]
    score_df["pc2"] = scores[:, 1]

    loading_df = pd.DataFrame(
        {
            "feature": available,
            "pc1_loading": pca.components_[0],
            "pc2_loading": pca.components_[1],
            "pc1_abs_loading": np.abs(pca.components_[0]),
            "pc2_abs_loading": np.abs(pca.components_[1]),
        }
    ).sort_values("pc1_abs_loading", ascending=False)

    variance_df = pd.DataFrame(
        {
            "component": ["PC1", "PC2"],
            "explained_variance_ratio": pca.explained_variance_ratio_,
        }
    )
    return score_df, loading_df, variance_df


def summarize_centroids(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    out = (
        df.groupby(group_col)[["pc1", "pc2"]]
        .agg(["mean", "std", "count"])
    )
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    return out.reset_index()


def save_scatter(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    filtered = df[df["profile_group"].isin(GROUP_ORDER)].copy()
    for group in GROUP_ORDER:
        sub = filtered[filtered["profile_group"] == group]
        if sub.empty:
            continue
        ax.scatter(
            sub["pc1"],
            sub["pc2"],
            s=34,
            alpha=0.72,
            c=GROUP_COLORS[group],
            label=group,
            edgecolors="none",
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Descriptor PCA by soft-regime profile")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(
    path: Path,
    variance_df: pd.DataFrame,
    loading_df: pd.DataFrame,
    profile_centroids: pd.DataFrame,
    tier_centroids: pd.DataFrame,
    horizon_centroids: pd.DataFrame,
) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Descriptor PCA Report\n\n")
        f.write("PCA is used here as a structure probe rather than a predictive model.\n\n")
        f.write("## Explained Variance\n\n")
        f.write(variance_df.to_markdown(index=False))
        f.write("\n\n## Loadings\n\n")
        f.write(loading_df.to_markdown(index=False))
        f.write("\n\n## Profile Centroids\n\n")
        f.write(profile_centroids.to_markdown(index=False))
        f.write("\n\n## Tier Centroids\n\n")
        f.write(tier_centroids.to_markdown(index=False))
        f.write("\n\n## Horizon Centroids\n\n")
        f.write(horizon_centroids.to_markdown(index=False))
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore descriptor structure with PCA.")
    parser.add_argument("--soft_profile_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--feature_mode", choices=["full", "raw_only"], default="raw_only")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.soft_profile_csv)
    if "profile_group" not in df.columns:
        df = assign_profile_group(df)
    df = df[df["profile_group"].isin(GROUP_ORDER)].copy()

    features = FULL_NUMERIC_FEATURES if args.feature_mode == "full" else RAW_NUMERIC_FEATURES
    score_df, loading_df, variance_df = fit_pca(df, features)
    profile_centroids = summarize_centroids(score_df, "profile_group")
    tier_centroids = summarize_centroids(score_df, "tier")
    horizon_centroids = summarize_centroids(score_df, "horizon")

    score_df.to_csv(out_dir / "descriptor_pca_scores.csv", index=False)
    loading_df.to_csv(out_dir / "descriptor_pca_loadings.csv", index=False)
    variance_df.to_csv(out_dir / "descriptor_pca_variance.csv", index=False)
    profile_centroids.to_csv(out_dir / "descriptor_pca_profile_centroids.csv", index=False)
    tier_centroids.to_csv(out_dir / "descriptor_pca_tier_centroids.csv", index=False)
    horizon_centroids.to_csv(out_dir / "descriptor_pca_horizon_centroids.csv", index=False)
    save_scatter(score_df, out_dir / "descriptor_pca_scatter.png")
    write_report(
        out_dir / "descriptor_pca_report.md",
        variance_df,
        loading_df,
        profile_centroids,
        tier_centroids,
        horizon_centroids,
    )

    manifest = {
        "soft_profile_csv": str(Path(args.soft_profile_csv).resolve()),
        "feature_mode": args.feature_mode,
        "output_dir": str(out_dir.resolve()),
    }
    with (out_dir / "descriptor_pca_summary.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
