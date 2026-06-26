#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _save_barplot(
    df: pd.DataFrame,
    category_col: str,
    value_col: str,
    title: str,
    ylabel: str,
    out_path: Path,
    rotate_xticks: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cats = [str(v) for v in df[category_col].tolist()]
    vals = df[value_col].astype(float).tolist()
    bars = ax.bar(cats, vals, color="#3A6EA5")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 1.05 if "rate" in value_col else max(vals) * 1.15 if vals else 1.0)
    if rotate_xticks:
        ax.tick_params(axis="x", rotation=20)
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + (0.02 if "rate" in value_col else 0.01 * max(vals)),
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_grouped_horizon_plot(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    methods = ["locf", "unet_mask", "unet_image_mask", "resunet_image_mask", "plain_cnn_image_mask"]
    method_labels = {
        "locf": "LOCF",
        "unet_mask": "UNet-mask",
        "unet_image_mask": "UNet-image+mask",
        "resunet_image_mask": "ResUNet-image+mask",
        "plain_cnn_image_mask": "PlainCNN-image+mask",
    }
    colors = {
        "locf": "#777777",
        "unet_mask": "#4C78A8",
        "unet_image_mask": "#72B7B2",
        "resunet_image_mask": "#E45756",
        "plain_cnn_image_mask": "#54A24B",
    }
    for method in methods:
        sub = df[df["method"] == method].sort_values("horizon")
        if sub.empty:
            continue
        ax.plot(
            sub["horizon"],
            sub["mean"],
            marker="o",
            linewidth=2.0,
            markersize=6,
            color=colors[method],
            label=method_labels[method],
        )
    ax.set_title("Forecast Dice by Horizon")
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Mean Dice")
    ax.set_xticks(sorted(df["horizon"].unique()))
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False)
    ax.grid(alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export key regime-analysis figures from saved CSV outputs.")
    parser.add_argument("--analysis_root", type=str, required=True)
    parser.add_argument("--baseline_output_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    analysis_root = Path(args.analysis_root)
    baseline_output_dir = Path(args.baseline_output_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairwise_by_tier = pd.read_csv(analysis_root / "pairwise_by_tier.csv")
    pairwise_by_input_volume = pd.read_csv(analysis_root / "pairwise_by_input_volume_bin.csv")
    pairwise_by_recent_growth = pd.read_csv(analysis_root / "pairwise_by_recent_growth_bin.csv")
    pairwise_by_future_growth = pd.read_csv(analysis_root / "pairwise_by_future_growth_bin.csv")
    horizon_table = pd.read_json(baseline_output_dir / "locf_per_sample.json")

    # Rebuild horizon summary from all methods if available through enriched CSV.
    enriched = pd.read_csv(analysis_root / "all_methods_enriched_samples.csv")
    horizon_summary = (
        enriched.groupby(["method", "horizon"])
        .agg(mean=("dice", "mean"))
        .reset_index()
    )

    _save_barplot(
        pairwise_by_tier,
        category_col="tier",
        value_col="win_rate",
        title="ResUNet Win Rate Over LOCF by Regime",
        ylabel="Win rate",
        out_path=output_dir / "win_rate_by_tier.png",
    )

    _save_barplot(
        pairwise_by_input_volume,
        category_col="input_volume_bin",
        value_col="win_rate",
        title="ResUNet Win Rate Over LOCF by Input Volume",
        ylabel="Win rate",
        out_path=output_dir / "win_rate_by_input_volume_bin.png",
    )

    _save_barplot(
        pairwise_by_recent_growth,
        category_col="recent_growth_bin",
        value_col="win_rate",
        title="ResUNet Win Rate Over LOCF by Recent Growth",
        ylabel="Win rate",
        out_path=output_dir / "win_rate_by_recent_growth_bin.png",
    )

    _save_barplot(
        pairwise_by_future_growth,
        category_col="future_growth_bin",
        value_col="win_rate",
        title="ResUNet Win Rate Over LOCF by Future Growth",
        ylabel="Win rate",
        out_path=output_dir / "win_rate_by_future_growth_bin.png",
    )

    _save_grouped_horizon_plot(horizon_summary, output_dir / "dice_by_horizon_all_methods.png")

    key_tables = {
        "pairwise_by_tier.csv": pairwise_by_tier,
        "pairwise_by_input_volume_bin.csv": pairwise_by_input_volume,
        "pairwise_by_recent_growth_bin.csv": pairwise_by_recent_growth,
        "pairwise_by_future_growth_bin.csv": pairwise_by_future_growth,
    }
    for name, df in key_tables.items():
        df.to_csv(output_dir / name, index=False)

    summary_md = output_dir / "figure_manifest.md"
    with summary_md.open("w", encoding="utf-8") as f:
        f.write("# Figure Manifest\n\n")
        f.write("- `win_rate_by_tier.png`\n")
        f.write("- `win_rate_by_input_volume_bin.png`\n")
        f.write("- `win_rate_by_recent_growth_bin.png`\n")
        f.write("- `win_rate_by_future_growth_bin.png`\n")
        f.write("- `dice_by_horizon_all_methods.png`\n")

    print(f"Saved figures and tables to: {output_dir}")


if __name__ == "__main__":
    main()
