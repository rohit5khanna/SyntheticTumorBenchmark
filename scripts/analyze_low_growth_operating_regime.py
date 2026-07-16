#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def read_samples(bottleneck_dir: Path) -> pd.DataFrame:
    path = bottleneck_dir / "hybrid_vs_direct_samples.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"No rows found in {path}")
    return df


def add_operating_regimes(df: pd.DataFrame, low_growth_bins: set[str]) -> pd.DataFrame:
    out = df.copy()
    out["is_low_growth_bin"] = out["absolute_growth_bin"].isin(low_growth_bins)
    out["is_short_horizon"] = out["horizon"].astype(int) == 1
    out["is_short_low_growth"] = out["is_short_horizon"] & out["is_low_growth_bin"]
    out["best_method"] = np.select(
        [
            (out["hybrid_policy_dice"] >= out["direct_model_dice"]) & (out["hybrid_policy_dice"] >= out["locf_dice"]),
            (out["direct_model_dice"] >= out["hybrid_policy_dice"]) & (out["direct_model_dice"] >= out["locf_dice"]),
        ],
        ["hybrid", "direct_resunet"],
        default="locf",
    )
    out["hybrid_minus_direct"] = out["hybrid_policy_dice"] - out["direct_model_dice"]
    out["hybrid_minus_locf"] = out["hybrid_policy_dice"] - out["locf_dice"]
    out["direct_minus_locf"] = out["direct_model_dice"] - out["locf_dice"]
    return out


def summarize(df: pd.DataFrame, group_cols: Iterable[str]) -> pd.DataFrame:
    cols = [c for c in group_cols if c in df.columns]
    group_df = df if cols else df.assign(_overall="overall")
    by = cols if cols else ["_overall"]
    out = (
        group_df.groupby(by, observed=True, dropna=False)
        .agg(
            count=("hybrid_policy_dice", "size"),
            low_growth_rate=("is_low_growth_bin", "mean"),
            mean_locf_dice=("locf_dice", "mean"),
            mean_hybrid_dice=("hybrid_policy_dice", "mean"),
            mean_direct_dice=("direct_model_dice", "mean"),
            mean_hybrid_minus_direct=("hybrid_minus_direct", "mean"),
            mean_hybrid_minus_locf=("hybrid_minus_locf", "mean"),
            mean_direct_minus_locf=("direct_minus_locf", "mean"),
            hybrid_beats_direct_rate=("hybrid_minus_direct", lambda x: float((x > 0).mean())),
            hybrid_beats_locf_rate=("hybrid_minus_locf", lambda x: float((x > 0).mean())),
            direct_beats_locf_rate=("direct_minus_locf", lambda x: float((x > 0).mean())),
            gate_active_rate=("gate_active", "mean"),
            mean_growth_volume_vox=("growth_volume_vox", "mean"),
            median_growth_volume_vox=("growth_volume_vox", "median"),
            mean_relative_new_growth=("relative_new_growth", "mean"),
        )
        .reset_index()
    )
    if "_overall" in out.columns:
        out = out.drop(columns=["_overall"])
    return out.sort_values(cols).reset_index(drop=True) if cols else out


def method_counts(df: pd.DataFrame, group_cols: Iterable[str]) -> pd.DataFrame:
    cols = [c for c in group_cols if c in df.columns]
    if not cols:
        group_df = df.assign(_overall="overall")
        cols = ["_overall"]
    else:
        group_df = df
    out = (
        group_df.groupby(cols + ["best_method"], observed=True, dropna=False)
        .size()
        .reset_index(name="count")
    )
    totals = out.groupby(cols, observed=True, dropna=False)["count"].transform("sum")
    out["fraction"] = out["count"] / totals
    if "_overall" in out.columns:
        out = out.drop(columns=["_overall"])
    return out.sort_values(cols + ["best_method"]).reset_index(drop=True)


def bootstrap_mean(values: np.ndarray, n_bootstrap: int, seed: int) -> dict:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return {"n": 0, "mean": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(vals), len(vals))
        boot.append(float(vals[idx].mean()))
    boot = np.asarray(boot)
    return {
        "n": int(len(vals)),
        "mean": float(vals.mean()),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
    }


def bootstrap_table(df: pd.DataFrame, n_bootstrap: int, seed: int) -> pd.DataFrame:
    regimes = {
        "all": df,
        "low_growth": df[df["is_low_growth_bin"]],
        "short_horizon": df[df["is_short_horizon"]],
        "short_low_growth": df[df["is_short_low_growth"]],
    }
    rows = []
    for regime, sub in regimes.items():
        for metric in ["hybrid_minus_direct", "hybrid_minus_locf", "direct_minus_locf"]:
            rows.append({"regime": regime, "metric": metric, **bootstrap_mean(sub[metric].to_numpy(), n_bootstrap, seed)})
    return pd.DataFrame(rows)


def write_report(
    path: Path,
    low_growth_bins: list[str],
    overall: pd.DataFrame,
    by_horizon: pd.DataFrame,
    low_growth_summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Low-Growth Operating Regime Analysis\n\n")
        f.write(
            "This analysis asks whether the hybrid persistence-plus-growth policy is useful in the short-horizon, "
            "low-growth regime where persistence is expected to be strong.\n\n"
        )
        f.write(f"Low-growth bins: `{', '.join(low_growth_bins)}`\n\n")
        f.write("## Overall\n\n")
        f.write(overall.to_markdown(index=False))
        f.write("\n\n## By Horizon\n\n")
        f.write(by_horizon.to_markdown(index=False))
        f.write("\n\n## Low-Growth Subset\n\n")
        f.write(low_growth_summary.to_markdown(index=False))
        f.write("\n\n## Bootstrap\n\n")
        f.write(bootstrap.to_markdown(index=False))
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze low-growth short-horizon operating regimes.")
    parser.add_argument("--bottleneck_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--low_growth_bins",
        type=str,
        default="zero,small_nonzero",
        help="Comma-separated absolute_growth_bin values treated as low-growth.",
    )
    parser.add_argument("--n_bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    bottleneck_dir = Path(args.bottleneck_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    low_growth_bins = [x.strip() for x in args.low_growth_bins.split(",") if x.strip()]

    samples = add_operating_regimes(read_samples(bottleneck_dir), set(low_growth_bins))
    overall = summarize(samples, [])
    by_horizon = summarize(samples, ["horizon"])
    by_growth = summarize(samples, ["absolute_growth_bin"])
    by_horizon_growth = summarize(samples, ["horizon", "absolute_growth_bin"])
    low_growth_summary = summarize(samples[samples["is_low_growth_bin"]], [])
    short_low_growth_summary = summarize(samples[samples["is_short_low_growth"]], [])
    best_by_growth = method_counts(samples, ["absolute_growth_bin"])
    best_by_horizon_growth = method_counts(samples, ["horizon", "absolute_growth_bin"])
    bootstrap = bootstrap_table(samples, args.n_bootstrap, args.seed)

    samples.to_csv(output_dir / "low_growth_operating_samples.csv", index=False)
    overall.to_csv(output_dir / "low_growth_operating_overall.csv", index=False)
    by_horizon.to_csv(output_dir / "low_growth_operating_by_horizon.csv", index=False)
    by_growth.to_csv(output_dir / "low_growth_operating_by_growth_bin.csv", index=False)
    by_horizon_growth.to_csv(output_dir / "low_growth_operating_by_horizon_growth_bin.csv", index=False)
    low_growth_summary.to_csv(output_dir / "low_growth_subset_summary.csv", index=False)
    short_low_growth_summary.to_csv(output_dir / "short_horizon_low_growth_summary.csv", index=False)
    best_by_growth.to_csv(output_dir / "best_method_counts_by_growth_bin.csv", index=False)
    best_by_horizon_growth.to_csv(output_dir / "best_method_counts_by_horizon_growth_bin.csv", index=False)
    bootstrap.to_csv(output_dir / "low_growth_operating_bootstrap.csv", index=False)
    write_report(output_dir / "low_growth_operating_regime_report.md", low_growth_bins, overall, by_horizon, low_growth_summary, bootstrap)

    print(
        json.dumps(
            {
                "bottleneck_dir": str(bottleneck_dir),
                "low_growth_bins": low_growth_bins,
                "n_samples": int(len(samples)),
                "n_low_growth_samples": int(samples["is_low_growth_bin"].sum()),
                "n_short_low_growth_samples": int(samples["is_short_low_growth"].sum()),
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
