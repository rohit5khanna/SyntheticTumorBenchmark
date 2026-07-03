#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_anchor_separation import FULL_NUMERIC_FEATURES, RAW_NUMERIC_FEATURES, cohen_d


def assign_profile_group(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["profile_group"] = "other"
    out.loc[(out["case_type"] == "both_easy") & (out["soft_regime_label"] == "core_aligned"), "profile_group"] = "both_easy_core"
    out.loc[(out["case_type"] == "target_wins") & (out["soft_regime_label"] == "core_aligned"), "profile_group"] = "target_wins_core"
    out.loc[out["soft_regime_label"] == "cross_regime_pull", "profile_group"] = "cross_regime_pull"
    out.loc[out["soft_regime_label"] == "transition", "profile_group"] = "transition"
    return out


def bootstrap_gap(x1: np.ndarray, x0: np.ndarray, n_boot: int, rng: np.random.Generator) -> dict[str, float]:
    if len(x1) < 2 or len(x0) < 2:
        return {
            "mean_gap": np.nan,
            "mean_gap_ci_low": np.nan,
            "mean_gap_ci_high": np.nan,
            "cohen_d": np.nan,
            "cohen_d_ci_low": np.nan,
            "cohen_d_ci_high": np.nan,
        }

    gap_samples = []
    d_samples = []
    for _ in range(n_boot):
        b1 = rng.choice(x1, size=len(x1), replace=True)
        b0 = rng.choice(x0, size=len(x0), replace=True)
        gap_samples.append(float(np.mean(b1) - np.mean(b0)))
        d_samples.append(float(cohen_d(b1, b0)))

    gap_arr = np.asarray(gap_samples, dtype=float)
    d_arr = np.asarray(d_samples, dtype=float)
    return {
        "mean_gap": float(np.mean(x1) - np.mean(x0)),
        "mean_gap_ci_low": float(np.quantile(gap_arr, 0.025)),
        "mean_gap_ci_high": float(np.quantile(gap_arr, 0.975)),
        "cohen_d": float(cohen_d(x1, x0)),
        "cohen_d_ci_low": float(np.quantile(d_arr, 0.025)),
        "cohen_d_ci_high": float(np.quantile(d_arr, 0.975)),
    }


def build_summary(df: pd.DataFrame, features: list[str], n_boot: int, seed: int) -> pd.DataFrame:
    anchor = df[df["profile_group"].isin(["both_easy_core", "target_wins_core"])].copy()
    pos = anchor[anchor["profile_group"] == "target_wins_core"]
    neg = anchor[anchor["profile_group"] == "both_easy_core"]
    rng = np.random.default_rng(seed)

    rows = []
    for feature in features:
        if feature not in anchor.columns:
            continue
        x1 = pos[feature].dropna().to_numpy(dtype=float)
        x0 = neg[feature].dropna().to_numpy(dtype=float)
        stats = bootstrap_gap(x1, x0, n_boot=n_boot, rng=rng)
        rows.append(
            {
                "feature": feature,
                "target_wins_core_mean": float(np.mean(x1)) if len(x1) else np.nan,
                "both_easy_core_mean": float(np.mean(x0)) if len(x0) else np.nan,
                "n_target_wins_core": int(len(x1)),
                "n_both_easy_core": int(len(x0)),
                **stats,
            }
        )
    out = pd.DataFrame(rows)
    out["abs_cohen_d"] = out["cohen_d"].abs()
    return out.sort_values("abs_cohen_d", ascending=False)


def write_report(path: Path, summary: pd.DataFrame) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Anchor Bootstrap Report\n\n")
        f.write("Bootstrap confidence intervals for anchor feature gaps between both-easy core and target-wins core.\n\n")
        f.write(summary.to_markdown(index=False))
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap confidence intervals for anchor feature gaps.")
    parser.add_argument("--soft_profile_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--feature_mode", type=str, choices=["full", "raw_only"], default="raw_only")
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.soft_profile_csv)
    if "profile_group" not in df.columns:
        df = assign_profile_group(df)

    features = FULL_NUMERIC_FEATURES if args.feature_mode == "full" else RAW_NUMERIC_FEATURES
    summary = build_summary(df, features=features, n_boot=args.n_boot, seed=args.seed)
    summary.to_csv(out_dir / "anchor_bootstrap_summary.csv", index=False)
    write_report(out_dir / "anchor_bootstrap_report.md", summary)

    manifest = {
        "soft_profile_csv": str(Path(args.soft_profile_csv).resolve()),
        "feature_mode": args.feature_mode,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "output_dir": str(out_dir.resolve()),
    }
    with (out_dir / "anchor_bootstrap_summary.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
