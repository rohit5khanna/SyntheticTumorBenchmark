#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FORECAST_ORIGIN_FEATURES = [
    "input_volume_vox",
    "recent_relative_growth",
    "treated_at_input",
    "delta_days",
    "input_elongation_ratio",
    "input_compactness_proxy",
    "input_connected_component_count",
    "n_sessions",
    "followup_days",
    "mean_interval_days",
]


def summarize_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for feature in features:
        s = pd.to_numeric(df[feature], errors="coerce")
        rows.append(
            {
                "feature": feature,
                "count_nonnull": int(s.notna().sum()),
                "count_missing": int(s.isna().sum()),
                "n_unique": int(s.nunique(dropna=True)),
                "mean": float(s.mean()),
                "std": float(s.std()),
                "min": float(s.min()),
                "max": float(s.max()),
            }
        )
    return pd.DataFrame(rows).sort_values("feature")


def correlation_pairs(corr: pd.DataFrame, method: str, threshold: float) -> pd.DataFrame:
    feats = list(corr.columns)
    rows: list[dict] = []
    for i, fi in enumerate(feats):
        for j in range(i + 1, len(feats)):
            fj = feats[j]
            val = float(corr.loc[fi, fj])
            rows.append(
                {
                    "feature_a": fi,
                    "feature_b": fj,
                    "method": method,
                    "corr": val,
                    "abs_corr": abs(val),
                    "flag_high": int(abs(val) >= threshold),
                }
            )
    return pd.DataFrame(rows).sort_values("abs_corr", ascending=False)


def build_group_correlation_summary(
    df: pd.DataFrame,
    features: list[str],
    group_col: str,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict] = []
    high_rows: list[dict] = []

    if group_col not in df.columns:
        return pd.DataFrame(), pd.DataFrame()

    for group_value, sub in df.groupby(group_col):
        available = [f for f in features if f in sub.columns and pd.to_numeric(sub[f], errors="coerce").notna().sum() >= 3]
        if len(available) < 2:
            continue
        num = sub[available].apply(pd.to_numeric, errors="coerce")
        pearson = num.corr(method="pearson")
        pairs = correlation_pairs(pearson, "pearson", threshold=threshold)
        pairs[group_col] = group_value
        high_rows.append(pairs[pairs["flag_high"] == 1].copy())

        if len(pairs) == 0:
            continue
        summary_rows.append(
            {
                group_col: group_value,
                "n_rows": int(len(sub)),
                "n_features": int(len(available)),
                "max_abs_corr": float(pairs["abs_corr"].max()),
                "mean_abs_corr": float(pairs["abs_corr"].mean()),
                "n_high_corr_pairs": int((pairs["flag_high"] == 1).sum()),
                "top_pair_a": str(pairs.iloc[0]["feature_a"]),
                "top_pair_b": str(pairs.iloc[0]["feature_b"]),
                "top_pair_corr": float(pairs.iloc[0]["corr"]),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    high_df = pd.concat(high_rows, ignore_index=True) if high_rows else pd.DataFrame()
    return summary_df, high_df


def write_report(
    out_path: Path,
    feature_summary: pd.DataFrame,
    top_pearson: pd.DataFrame,
    top_spearman: pd.DataFrame,
    tier_summary: pd.DataFrame,
    horizon_summary: pd.DataFrame,
) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# Descriptor Redundancy Report\n\n")
        f.write("## Feature Summary\n\n")
        f.write(feature_summary.to_markdown(index=False))
        f.write("\n\n## Top Pearson Correlation Pairs\n\n")
        f.write(top_pearson.head(15).to_markdown(index=False))
        f.write("\n\n## Top Spearman Correlation Pairs\n\n")
        f.write(top_spearman.head(15).to_markdown(index=False))
        if not tier_summary.empty:
            f.write("\n\n## Tier-Level Correlation Summary\n\n")
            f.write(tier_summary.to_markdown(index=False))
        if not horizon_summary.empty:
            f.write("\n\n## Horizon-Level Correlation Summary\n\n")
            f.write(horizon_summary.to_markdown(index=False))
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit redundancy and dependency structure in forecast-origin descriptors.")
    parser.add_argument("--case_type_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.70)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.case_type_csv)
    features = [f for f in FORECAST_ORIGIN_FEATURES if f in df.columns]
    num = df[features].apply(pd.to_numeric, errors="coerce")

    feature_summary = summarize_features(df, features)
    pearson = num.corr(method="pearson")
    spearman = num.corr(method="spearman")
    pearson_pairs = correlation_pairs(pearson, "pearson", threshold=args.threshold)
    spearman_pairs = correlation_pairs(spearman, "spearman", threshold=args.threshold)
    tier_summary, tier_high = build_group_correlation_summary(df, features, "tier", threshold=args.threshold)
    horizon_summary, horizon_high = build_group_correlation_summary(df, features, "horizon", threshold=args.threshold)

    feature_summary.to_csv(out_dir / "feature_summary.csv", index=False)
    pearson.to_csv(out_dir / "pearson_correlation_matrix.csv")
    spearman.to_csv(out_dir / "spearman_correlation_matrix.csv")
    pearson_pairs.to_csv(out_dir / "pearson_correlation_pairs.csv", index=False)
    spearman_pairs.to_csv(out_dir / "spearman_correlation_pairs.csv", index=False)
    tier_summary.to_csv(out_dir / "tier_correlation_summary.csv", index=False)
    horizon_summary.to_csv(out_dir / "horizon_correlation_summary.csv", index=False)
    tier_high.to_csv(out_dir / "tier_high_correlation_pairs.csv", index=False)
    horizon_high.to_csv(out_dir / "horizon_high_correlation_pairs.csv", index=False)

    write_report(
        out_dir / "descriptor_redundancy_report.md",
        feature_summary,
        pearson_pairs,
        spearman_pairs,
        tier_summary,
        horizon_summary,
    )

    manifest = {
        "case_type_csv": str(Path(args.case_type_csv).resolve()),
        "features": features,
        "threshold": args.threshold,
        "output_dir": str(out_dir.resolve()),
    }
    with (out_dir / "descriptor_redundancy_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
