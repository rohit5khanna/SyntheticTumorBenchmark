#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


KEY_COLS = ["patient_id", "input_idx", "target_idx", "horizon", "delta_days"]


def load_per_sample_tables(output_dir: Path, methods: Iterable[str]) -> pd.DataFrame:
    rows = []
    for method in methods:
        path = output_dir / f"{method}_per_sample.json"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        for row in payload:
            out = dict(row)
            out["method"] = method
            rows.append(out)
    if not rows:
        raise FileNotFoundError(f"No per-sample JSONs found in {output_dir}")
    return pd.DataFrame(rows)


def build_dice_pairwise(per_sample: pd.DataFrame, methods: List[str]) -> pd.DataFrame:
    locf = per_sample[per_sample["method"] == "locf"][KEY_COLS + ["dice"]].rename(columns={"dice": "locf_dice"})
    rows = []
    for method in methods:
        cur = per_sample[per_sample["method"] == method][KEY_COLS + ["dice"]].rename(columns={"dice": "model_dice"})
        pair = cur.merge(locf, on=KEY_COLS, how="inner")
        pair["model"] = method
        pair["dice_gap_vs_locf"] = pair["model_dice"] - pair["locf_dice"]
        rows.append(pair)
    if not rows:
        return pd.DataFrame(columns=KEY_COLS + ["model", "model_dice", "locf_dice", "dice_gap_vs_locf"])
    return pd.concat(rows, ignore_index=True)


def build_ranking_pairwise(ranking: pd.DataFrame, methods: List[str]) -> pd.DataFrame:
    dist = ranking[ranking["method"] == "distance_to_input_mask"][
        KEY_COLS + ["growth_average_precision", "growth_recall_at_growth_volume"]
    ].rename(
        columns={
            "growth_average_precision": "distance_ap",
            "growth_recall_at_growth_volume": "distance_recall_at_growth_volume",
        }
    )
    rows = []
    for method in methods:
        cur = ranking[ranking["method"] == method][
            KEY_COLS + ["growth_average_precision", "growth_recall_at_growth_volume"]
        ].rename(
            columns={
                "growth_average_precision": "model_ap",
                "growth_recall_at_growth_volume": "model_recall_at_growth_volume",
            }
        )
        pair = cur.merge(dist, on=KEY_COLS, how="left")
        pair["model"] = method
        pair["ap_gap_vs_distance"] = pair["model_ap"] - pair["distance_ap"]
        pair["recall_gap_vs_distance"] = (
            pair["model_recall_at_growth_volume"] - pair["distance_recall_at_growth_volume"]
        )
        rows.append(pair)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def add_absolute_growth_bins(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    nonzero = out.loc[out["growth_volume_vox"] > 0, "growth_volume_vox"].dropna()
    if nonzero.empty:
        small_max = 0.0
        large_min = 0.0
    else:
        small_max = float(nonzero.quantile(0.33))
        large_min = float(nonzero.quantile(0.67))

    def label_growth(v: float) -> str:
        if pd.isna(v):
            return "unknown"
        if v <= 0:
            return "zero"
        if v <= small_max:
            return "small_nonzero"
        if v <= large_min:
            return "medium_nonzero"
        return "large_nonzero"

    out["absolute_growth_bin"] = out["growth_volume_vox"].apply(label_growth)
    thresholds = {
        "small_nonzero_max_vox": small_max,
        "large_nonzero_min_vox": large_min,
    }
    return out, thresholds


def label_quadrants(df: pd.DataFrame, min_ap_gap: float, min_dice_gap: float) -> pd.DataFrame:
    out = df.copy()
    labels = []
    for _, row in out.iterrows():
        ap_gap = row.get("ap_gap_vs_distance")
        dice_gap = row.get("dice_gap_vs_locf")
        if pd.isna(ap_gap):
            labels.append("no_growth_ranking_metric")
            continue
        good_rank = ap_gap > min_ap_gap
        bad_rank = ap_gap < -min_ap_gap
        good_dice = dice_gap > min_dice_gap
        bad_dice = dice_gap < -min_dice_gap

        if good_rank and good_dice:
            labels.append("good_ranking_good_dice")
        elif good_rank and bad_dice:
            labels.append("good_ranking_bad_dice")
        elif bad_rank and good_dice:
            labels.append("bad_ranking_good_dice")
        elif bad_rank and bad_dice:
            labels.append("bad_ranking_bad_dice")
        elif good_rank:
            labels.append("good_ranking_neutral_dice")
        elif bad_rank:
            labels.append("bad_ranking_neutral_dice")
        elif good_dice:
            labels.append("neutral_ranking_good_dice")
        elif bad_dice:
            labels.append("neutral_ranking_bad_dice")
        else:
            labels.append("neutral")
    out["ranking_dice_quadrant"] = labels
    return out


def summarize(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    available = [c for c in group_cols if c in df.columns]
    if not available:
        return pd.DataFrame()
    grouped = (
        df.groupby(available, dropna=False, observed=True)
        .agg(
            count=("patient_id", "size"),
            mean_ap_gap_vs_distance=("ap_gap_vs_distance", "mean"),
            mean_dice_gap_vs_locf=("dice_gap_vs_locf", "mean"),
            median_growth_volume_vox=("growth_volume_vox", "median"),
            mean_relative_new_growth=("relative_new_growth", "mean"),
            mean_model_ap=("model_ap", "mean"),
            mean_distance_ap=("distance_ap", "mean"),
            mean_model_dice=("model_dice", "mean"),
            mean_locf_dice=("locf_dice", "mean"),
        )
        .reset_index()
    )
    return grouped.sort_values(available)


def summarize_quadrants(df: pd.DataFrame) -> pd.DataFrame:
    base = (
        df.groupby(["model", "ranking_dice_quadrant"], dropna=False, observed=True)
        .agg(
            count=("patient_id", "size"),
            mean_ap_gap_vs_distance=("ap_gap_vs_distance", "mean"),
            mean_dice_gap_vs_locf=("dice_gap_vs_locf", "mean"),
            median_growth_volume_vox=("growth_volume_vox", "median"),
            mean_relative_new_growth=("relative_new_growth", "mean"),
        )
        .reset_index()
    )
    totals = base.groupby("model")["count"].transform("sum")
    base["fraction_within_model"] = base["count"] / totals
    return base.sort_values(["model", "count"], ascending=[True, False])


def correlation_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in df.dropna(subset=["ap_gap_vs_distance", "dice_gap_vs_locf"]).groupby(
        ["model", "absolute_growth_bin"], observed=True
    ):
        model, growth_bin = keys
        if len(group) < 3:
            pearson = np.nan
            spearman = np.nan
        else:
            pearson = group["ap_gap_vs_distance"].corr(group["dice_gap_vs_locf"], method="pearson")
            spearman = group["ap_gap_vs_distance"].corr(group["dice_gap_vs_locf"], method="spearman")
        rows.append(
            {
                "model": model,
                "absolute_growth_bin": growth_bin,
                "count": int(len(group)),
                "pearson_apgap_dicegap": pearson,
                "spearman_apgap_dicegap": spearman,
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "absolute_growth_bin"])


def make_plots(df: pd.DataFrame, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    plot_df = df.dropna(subset=["ap_gap_vs_distance", "dice_gap_vs_locf"]).copy()
    if plot_df.empty:
        return

    models = sorted(plot_df["model"].unique())
    bins = ["zero", "small_nonzero", "medium_nonzero", "large_nonzero"]
    colors = {
        "zero": "#737373",
        "small_nonzero": "#4C78A8",
        "medium_nonzero": "#F58518",
        "large_nonzero": "#54A24B",
    }

    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5), squeeze=False)
    for ax, model in zip(axes[0], models):
        cur = plot_df[plot_df["model"] == model]
        for growth_bin in bins:
            sub = cur[cur["absolute_growth_bin"] == growth_bin]
            if sub.empty:
                continue
            ax.scatter(
                sub["ap_gap_vs_distance"],
                sub["dice_gap_vs_locf"],
                label=growth_bin,
                alpha=0.75,
                s=35,
                color=colors.get(growth_bin),
            )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(model)
        ax.set_xlabel("AP gap vs distance baseline")
        ax.set_ylabel("Dice gap vs LOCF")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "ranking_dice_tradeoff_scatter.png", dpi=200)
    plt.close(fig)


def write_report(path: Path, thresholds: dict, quadrant_summary: pd.DataFrame, growth_bin_summary: pd.DataFrame) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Ranking-Dice Tradeoff Analysis\n\n")
        f.write("This analysis tests whether growth-ranking gains align with full-mask Dice gains.\n\n")
        f.write("## Absolute Growth Bins\n\n")
        f.write(f"- zero: growth volume <= 0 voxels\n")
        f.write(f"- small_nonzero: 0 < growth volume <= {thresholds['small_nonzero_max_vox']:.3f} voxels\n")
        f.write(
            f"- medium_nonzero: {thresholds['small_nonzero_max_vox']:.3f} < growth volume <= "
            f"{thresholds['large_nonzero_min_vox']:.3f} voxels\n"
        )
        f.write(f"- large_nonzero: growth volume > {thresholds['large_nonzero_min_vox']:.3f} voxels\n\n")
        f.write("## Quadrant Summary\n\n")
        f.write(quadrant_summary.to_markdown(index=False) if not quadrant_summary.empty else "No quadrant summary.")
        f.write("\n\n## Growth-Bin Summary\n\n")
        f.write(growth_bin_summary.to_markdown(index=False) if not growth_bin_summary.empty else "No growth-bin summary.")
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze tradeoffs between new-growth ranking and full-mask Dice.")
    parser.add_argument("--growth_eval_dir", type=str, required=True)
    parser.add_argument("--baseline_output_dir", type=str, required=True)
    parser.add_argument("--methods", type=str, default="unet_image_mask,resunet_image_mask")
    parser.add_argument("--min_ap_gap", type=float, default=0.05)
    parser.add_argument("--min_dice_gap", type=float, default=0.02)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    growth_eval_dir = Path(args.growth_eval_dir)
    baseline_output_dir = Path(args.baseline_output_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    per_sample = load_per_sample_tables(baseline_output_dir, ["locf"] + methods)
    dice_pair = build_dice_pairwise(per_sample, methods)
    ranking = pd.read_csv(growth_eval_dir / "growth_ranking_metrics.csv")
    rank_pair = build_ranking_pairwise(ranking, methods)
    features = pd.read_csv(growth_eval_dir / "growth_sample_features.csv")

    feature_cols = [
        "tier",
        "new_growth_bin",
        "abs_change_bin",
        "net_growth_bin",
        "relative_new_growth",
        "relative_abs_change",
        "growth_volume_vox",
        "input_volume_vox",
        "target_volume_vox",
        "locf_dice_from_masks",
    ]
    merged = rank_pair.merge(dice_pair, on=KEY_COLS + ["model"], how="left")
    merged = merged.merge(features[KEY_COLS + feature_cols], on=KEY_COLS, how="left")
    merged, thresholds = add_absolute_growth_bins(merged)
    merged = label_quadrants(merged, min_ap_gap=args.min_ap_gap, min_dice_gap=args.min_dice_gap)

    quadrant_summary = summarize_quadrants(merged)
    growth_bin_summary = summarize(merged, ["model", "absolute_growth_bin"])
    tier_summary = summarize(merged, ["model", "tier", "absolute_growth_bin"])
    horizon_summary = summarize(merged, ["model", "horizon", "absolute_growth_bin"])
    corr_summary = correlation_summary(merged)

    merged.to_csv(output_dir / "ranking_dice_tradeoff_samples.csv", index=False)
    quadrant_summary.to_csv(output_dir / "ranking_dice_quadrant_summary.csv", index=False)
    growth_bin_summary.to_csv(output_dir / "ranking_dice_by_absolute_growth_bin.csv", index=False)
    tier_summary.to_csv(output_dir / "ranking_dice_by_tier_growth_bin.csv", index=False)
    horizon_summary.to_csv(output_dir / "ranking_dice_by_horizon_growth_bin.csv", index=False)
    corr_summary.to_csv(output_dir / "ranking_dice_correlation_summary.csv", index=False)
    make_plots(merged, output_dir)
    write_report(output_dir / "ranking_dice_tradeoff_report.md", thresholds, quadrant_summary, growth_bin_summary)

    print(
        json.dumps(
            {
                "growth_eval_dir": str(growth_eval_dir),
                "baseline_output_dir": str(baseline_output_dir),
                "methods": methods,
                "n_rows": int(len(merged)),
                "absolute_growth_thresholds": thresholds,
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
