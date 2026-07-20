#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


KEY_CANDIDATES = ["split", "patient_id", "input_idx", "target_idx", "horizon"]


def parse_model_csvs(items: Iterable[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Model CSV must be formatted as label=/path/to/file.csv, got: {item}")
        label, path = item.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"Empty model label in: {item}")
        out[label] = Path(path.strip())
    return out


def qbin(series: pd.Series, labels: List[str]) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce")
    if clean.dropna().nunique() < 2:
        return pd.Series(["all"] * len(series), index=series.index, dtype="object")
    q = min(len(labels), int(clean.dropna().nunique()))
    try:
        cats = pd.qcut(clean, q=q, duplicates="drop")
    except ValueError:
        return pd.Series(["all"] * len(series), index=series.index, dtype="object")
    codes = cats.cat.codes
    n_cats = len(cats.cat.categories)
    out = pd.Series(pd.NA, index=series.index, dtype="object")
    for code, label in enumerate(labels[:n_cats]):
        out[codes == code] = label
    return out


def normalize_locf_samples(path: Path, change_col: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "new_growth_volume_vox" not in df.columns and "growth_volume_vox" in df.columns:
        df["new_growth_volume_vox"] = df["growth_volume_vox"]
    if "relative_absolute_change" not in df.columns:
        if {"relative_new_growth", "relative_loss"}.issubset(df.columns):
            df["relative_absolute_change"] = df["relative_new_growth"] + df["relative_loss"]
        elif {"new_growth_volume_vox", "loss_volume_vox", "input_volume_vox"}.issubset(df.columns):
            df["relative_absolute_change"] = (
                pd.to_numeric(df["new_growth_volume_vox"], errors="coerce")
                + pd.to_numeric(df["loss_volume_vox"], errors="coerce")
            ) / pd.to_numeric(df["input_volume_vox"], errors="coerce").clip(lower=1)
    if "locf_dice" not in df.columns:
        raise ValueError(f"LOCF samples file must contain locf_dice: {path}")
    if change_col not in df.columns:
        raise ValueError(f"Change column '{change_col}' not found. Available columns: {list(df.columns)}")
    if "split" not in df.columns:
        df["split"] = "all"
    if "net_direction" not in df.columns and "net_delta_volume_vox" in df.columns:
        df["net_direction"] = np.select(
            [df["net_delta_volume_vox"] > 0, df["net_delta_volume_vox"] < 0],
            ["net_growth", "net_shrinkage"],
            default="net_stable",
        )
    df["operating_change_bin"] = qbin(df[change_col], ["low_change", "medium_change", "high_change"])
    return df


def find_dice_col(df: pd.DataFrame, explicit: str | None = None) -> str:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"Explicit dice column '{explicit}' not found in model CSV.")
        return explicit
    for col in ["dice", "mean_dice", "model_dice", "direct_model_dice", "hybrid_policy_dice"]:
        if col in df.columns:
            return col
    raise ValueError(f"Could not infer model Dice column. Available columns: {list(df.columns)}")


def normalize_model_samples(path: Path, label: str, dice_col: str | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "input_idx" not in df.columns and "input_end_idx" in df.columns:
        df["input_idx"] = df["input_end_idx"]
    if "split" not in df.columns:
        df["split"] = "all"
    col = find_dice_col(df, dice_col)
    keys = [c for c in KEY_CANDIDATES if c in df.columns]
    if not {"patient_id", "input_idx", "target_idx", "horizon"}.issubset(set(keys)):
        raise ValueError(f"Model CSV missing required keys. File={path}, columns={list(df.columns)}")
    out = df[keys + [col]].copy()
    out = out.rename(columns={col: f"{label}_dice"})
    return out


def merge_models(locf: pd.DataFrame, models: Dict[str, Path]) -> pd.DataFrame:
    merged = locf.copy()
    for label, path in models.items():
        model = normalize_model_samples(path, label)
        keys = [c for c in KEY_CANDIDATES if c in merged.columns and c in model.columns]
        merged = merged.merge(model, on=keys, how="left")
        dice_col = f"{label}_dice"
        gap_col = f"{label}_gap_vs_locf"
        win_col = f"{label}_beats_locf"
        merged[gap_col] = merged[dice_col] - merged["locf_dice"]
        merged[win_col] = merged[gap_col] > 0
    return merged


def summarize(df: pd.DataFrame, group_cols: List[str], model_labels: List[str]) -> pd.DataFrame:
    work = df.copy()
    by = [c for c in group_cols if c in work.columns]
    if not by:
        by = ["_overall"]
        work["_overall"] = "overall"

    base_aggs = {
        "n_samples": ("patient_id", "size"),
        "n_patients": ("patient_id", "nunique"),
        "mean_delta_days": ("delta_days", "mean"),
        "median_delta_days": ("delta_days", "median"),
        "mean_locf_dice": ("locf_dice", "mean"),
        "median_locf_dice": ("locf_dice", "median"),
    }
    optional_cols = {
        "mean_relative_absolute_change": "relative_absolute_change",
        "mean_relative_new_growth": "relative_new_growth",
        "mean_relative_loss": "relative_loss",
        "mean_new_growth_volume_vox": "new_growth_volume_vox",
        "mean_loss_volume_vox": "loss_volume_vox",
        "mean_new_growth_rate_vox_per_day": "new_growth_rate_vox_per_day",
    }
    for out_col, src_col in optional_cols.items():
        if src_col in work.columns:
            base_aggs[out_col] = (src_col, "mean")

    for label in model_labels:
        dice_col = f"{label}_dice"
        gap_col = f"{label}_gap_vs_locf"
        win_col = f"{label}_beats_locf"
        if dice_col in work.columns:
            base_aggs[f"{label}_mean_dice"] = (dice_col, "mean")
            base_aggs[f"{label}_mean_gap_vs_locf"] = (gap_col, "mean")
            base_aggs[f"{label}_median_gap_vs_locf"] = (gap_col, "median")
            base_aggs[f"{label}_win_rate_vs_locf"] = (win_col, "mean")
            base_aggs[f"{label}_n_available"] = (dice_col, lambda s: int(s.notna().sum()))

    out = work.groupby(by, dropna=False, observed=True).agg(**base_aggs).reset_index()
    if "_overall" in out.columns:
        out = out.drop(columns=["_overall"])
    return out


def write_plots(df: pd.DataFrame, output_dir: Path, model_labels: List[str]) -> List[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: List[str] = []

    by_change = summarize(df, ["operating_change_bin"], model_labels)
    if by_change.empty:
        return paths

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    order = [x for x in ["low_change", "medium_change", "high_change", "all"] if x in set(by_change["operating_change_bin"])]
    plot_df = by_change.set_index("operating_change_bin").loc[order].reset_index() if order else by_change
    x = np.arange(len(plot_df))
    ax.plot(x, plot_df["mean_locf_dice"], marker="o", label="LOCF", linewidth=2)
    for label in model_labels:
        col = f"{label}_mean_dice"
        if col in plot_df.columns:
            ax.plot(x, plot_df[col], marker="o", label=label, linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["operating_change_bin"], rotation=20, ha="right")
    ax.set_ylabel("Mean Dice")
    ax.set_title("Forecast Dice by observed change burden")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    path = output_dir / "operating_regime_dice_by_change.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    if model_labels:
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        width = 0.8 / max(1, len(model_labels))
        for i, label in enumerate(model_labels):
            col = f"{label}_mean_gap_vs_locf"
            if col not in plot_df.columns:
                continue
            ax.bar(x + (i - (len(model_labels) - 1) / 2) * width, plot_df[col], width=width, label=label)
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df["operating_change_bin"], rotation=20, ha="right")
        ax.set_ylabel("Mean Dice gap vs LOCF")
        ax.set_title("Model gap over LOCF by observed change burden")
        ax.legend()
        fig.tight_layout()
        path = output_dir / "operating_regime_gap_by_change.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(str(path))

    return paths


def write_report(path: Path, tables: Dict[str, pd.DataFrame], model_labels: List[str], plot_paths: List[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# LOCF Operating-Regime Bridge\n\n")
        f.write(
            "This descriptive bridge links observed tumor-change burden to LOCF performance and optional model gaps. "
            "The strata are descriptive quantiles, not learned clinical categories or test-tuned decision rules.\n\n"
        )
        f.write("## Interpretation Guardrail\n\n")
        f.write(
            "LOCF Dice is mathematically related to relative new growth and relative loss, so this analysis should be used "
            "to define the persistence operating range, not to claim that a regression discovered a surprising causal law.\n\n"
        )
        if model_labels:
            f.write("Merged model labels: " + ", ".join(f"`{m}`" for m in model_labels) + "\n\n")
        for name, table in tables.items():
            f.write(f"## {name}\n\n")
            f.write(table.to_markdown(index=False) if not table.empty else "No rows.")
            f.write("\n\n")
        if plot_paths:
            f.write("## Figures\n\n")
            for path_s in plot_paths:
                f.write(f"- `{path_s}`\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bridge LOCF operating regimes with optional model behavior.")
    parser.add_argument("--locf_operating_csv", type=str, required=True)
    parser.add_argument("--model_csv", action="append", default=[], help="Optional model CSV as label=/path/to/file.csv")
    parser.add_argument("--change_col", type=str, default="relative_absolute_change")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--no_plots", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    locf = normalize_locf_samples(Path(args.locf_operating_csv), args.change_col)
    model_paths = parse_model_csvs(args.model_csv)
    samples = merge_models(locf, model_paths) if model_paths else locf
    model_labels = list(model_paths.keys())

    tables = {
        "Overall": summarize(samples, [], model_labels),
        "By Operating Change Bin": summarize(samples, ["operating_change_bin"], model_labels),
        "By Split And Operating Change Bin": summarize(samples, ["split", "operating_change_bin"], model_labels),
        "By Net Direction And Operating Change Bin": summarize(samples, ["net_direction", "operating_change_bin"], model_labels),
        "By Patient": summarize(samples, ["patient_id"], model_labels),
    }

    samples.to_csv(output_dir / "operating_regime_bridge_samples.csv", index=False)
    for name, table in tables.items():
        filename = name.lower().replace(" ", "_").replace("and", "by")
        table.to_csv(output_dir / f"operating_regime_bridge_{filename}.csv", index=False)
    plot_paths = [] if args.no_plots else write_plots(samples, output_dir, model_labels)
    write_report(output_dir / "operating_regime_bridge_report.md", tables, model_labels, plot_paths)

    payload = {
        "locf_operating_csv": args.locf_operating_csv,
        "change_col": args.change_col,
        "model_csvs": {k: str(v) for k, v in model_paths.items()},
        "n_samples": int(len(samples)),
        "n_patients": int(samples["patient_id"].nunique()),
        "output_dir": str(output_dir),
        "outputs": {
            "samples_csv": str(output_dir / "operating_regime_bridge_samples.csv"),
            "report_md": str(output_dir / "operating_regime_bridge_report.md"),
            "plots": plot_paths,
        },
    }
    with (output_dir / "operating_regime_bridge_summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
