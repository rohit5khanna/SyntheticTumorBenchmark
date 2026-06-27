#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _make_matrix(df: pd.DataFrame, value_col: str = "mean_eval_dice") -> pd.DataFrame:
    mat = df.pivot_table(
        index="train_tier",
        columns="eval_tier",
        values=value_col,
        aggfunc="mean",
    )
    return mat.reindex(index=["A", "B", "C"], columns=["A", "B", "C"])


def _save_heatmap(
    mat: pd.DataFrame,
    title: str,
    out_path: Path,
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap: str = "YlOrRd",
) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(mat.values, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Eval tier")
    ax.set_ylabel("Train tier")
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels(mat.columns)
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(mat.index)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=10, color="black")
            else:
                ax.text(j, i, "NA", ha="center", va="center", fontsize=10, color="black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean Dice")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_locf_barplot(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    sub = df.sort_values("eval_tier")
    tiers = sub["eval_tier"].astype(str).tolist()
    vals = sub["mean_eval_dice"].astype(float).tolist()
    bars = ax.bar(tiers, vals, color="#777777")
    ax.set_title("LOCF by Evaluation Regime")
    ax.set_xlabel("Eval tier")
    ax.set_ylabel("Mean Dice")
    ax.set_ylim(0.0, 1.0)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2.0, val + 0.015, f"{val:.3f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_delta_heatmap(
    model_mat: pd.DataFrame,
    locf_df: pd.DataFrame,
    title: str,
    out_path: Path,
) -> None:
    locf_map = dict(zip(locf_df["eval_tier"].astype(str), locf_df["mean_eval_dice"].astype(float)))
    delta = model_mat.copy()
    for col in delta.columns:
        delta[col] = delta[col] - locf_map.get(str(col), float("nan"))

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    vmax = float(max(abs(delta.min().min()), abs(delta.max().max())))
    vmax = max(vmax, 0.05)
    im = ax.imshow(delta.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Eval tier")
    ax.set_ylabel("Train tier")
    ax.set_xticks(range(len(delta.columns)))
    ax.set_xticklabels(delta.columns)
    ax.set_yticks(range(len(delta.index)))
    ax.set_yticklabels(delta.index)
    for i in range(delta.shape[0]):
        for j in range(delta.shape[1]):
            val = delta.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:+.3f}", ha="center", va="center", fontsize=10, color="black")
            else:
                ax.text(j, i, "NA", ha="center", va="center", fontsize=10, color="black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Dice minus LOCF")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export cross-regime transfer figures and tables.")
    parser.add_argument("--resunet_dir", type=str, required=True)
    parser.add_argument("--locf_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--unetr_dir", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    res_df = pd.read_csv(Path(args.resunet_dir) / "cross_regime_transfer_overall.csv")
    locf_df = pd.read_csv(Path(args.locf_dir) / "cross_regime_transfer_locf.csv")
    unetr_df = None
    if args.unetr_dir:
        unetr_path = Path(args.unetr_dir) / "cross_regime_transfer_overall.csv"
        if unetr_path.exists():
            unetr_df = pd.read_csv(unetr_path)

    res_mat = _make_matrix(res_df)
    _save_heatmap(res_mat, "ResUNet Cross-Regime Transfer", output_dir / "resunet_transfer_heatmap.png")
    _save_delta_heatmap(res_mat, locf_df, "ResUNet Advantage Over LOCF", output_dir / "resunet_vs_locf_delta_heatmap.png")

    if unetr_df is not None:
        unetr_mat = _make_matrix(unetr_df)
        _save_heatmap(unetr_mat, "UNETR Cross-Regime Transfer", output_dir / "unetr_transfer_heatmap.png")
        _save_delta_heatmap(unetr_mat, locf_df, "UNETR Advantage Over LOCF", output_dir / "unetr_vs_locf_delta_heatmap.png")
        unetr_mat.to_csv(output_dir / "unetr_transfer_matrix.csv")

    _save_locf_barplot(locf_df, output_dir / "locf_by_eval_tier.png")

    res_mat.to_csv(output_dir / "resunet_transfer_matrix.csv")
    res_df.to_csv(output_dir / "resunet_transfer_runs.csv", index=False)
    locf_df.to_csv(output_dir / "locf_by_eval_tier.csv", index=False)
    if unetr_df is not None:
        unetr_df.to_csv(output_dir / "unetr_transfer_runs.csv", index=False)

    with (output_dir / "transfer_figure_manifest.md").open("w", encoding="utf-8") as f:
        f.write("# Transfer Figure Manifest\n\n")
        f.write("- `resunet_transfer_heatmap.png`\n")
        f.write("- `resunet_vs_locf_delta_heatmap.png`\n")
        f.write("- `locf_by_eval_tier.png`\n")
        if unetr_df is not None:
            f.write("- `unetr_transfer_heatmap.png`\n")
            f.write("- `unetr_vs_locf_delta_heatmap.png`\n")

    print(f"Saved transfer figures and tables to: {output_dir}")


if __name__ == "__main__":
    main()
