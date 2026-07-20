#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.metrics import dice_np
from baselines.tasks import ForecastSample, build_samples_for_split, infer_tier_from_patient_id, patient_paths


EPS_DAYS = 1e-6


def _standardize_label(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 5 and arr.shape[1] == 1:
        return (arr[:, 0] > 0)
    if arr.ndim == 4:
        return (arr > 0)
    raise ValueError(f"Unsupported label shape: {arr.shape}")


def _parse_csv(payload: str | None) -> List[str]:
    if payload is None:
        return []
    return [x.strip() for x in payload.split(",") if x.strip()]


def _parse_float_bins(payload: str) -> List[float]:
    vals: List[float] = []
    for item in payload.split(","):
        item = item.strip().lower()
        if not item:
            continue
        vals.append(float("inf") if item in {"inf", "infinity", "np.inf"} else float(item))
    if len(vals) < 2:
        raise ValueError("Need at least two interval bin edges.")
    if any(vals[i] >= vals[i + 1] for i in range(len(vals) - 1)):
        raise ValueError(f"Bin edges must be strictly increasing: {vals}")
    return vals


def _interval_labels(edges: List[float], unit: str = "d") -> List[str]:
    labels = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        lo_s = f"{lo:g}"
        hi_s = "inf" if math.isinf(hi) else f"{hi:g}"
        labels.append(f"{lo_s}-{hi_s}{unit}")
    return labels


def _qbin(series: pd.Series, labels: List[str]) -> pd.Series:
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
    use_labels = labels[:n_cats]
    out = pd.Series(pd.NA, index=series.index, dtype="object")
    for code, label in enumerate(use_labels):
        out[codes == code] = label
    out[codes < 0] = pd.NA
    return out


def _safe_rate(numerator: pd.Series, delta_days: pd.Series) -> pd.Series:
    denom = pd.to_numeric(delta_days, errors="coerce").clip(lower=EPS_DAYS)
    return pd.to_numeric(numerator, errors="coerce") / denom


def normalize_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    out = manifest.copy()
    if "input_idx" not in out.columns and "input_end_idx" in out.columns:
        out["input_idx"] = out["input_end_idx"]
    if "split" not in out.columns:
        out["split"] = "all"
    return out


def build_samples_from_manifest(manifest: pd.DataFrame, splits: Iterable[str]) -> List[ForecastSample]:
    splits_l = list(splits)
    rows = manifest[manifest["split"].isin(splits_l)].copy() if splits_l else manifest.copy()
    if rows.empty:
        raise ValueError(f"No rows found for splits={splits_l} in manifest.")

    samples: List[ForecastSample] = []
    for _, row in rows.iterrows():
        input_idx = int(row["input_idx"])
        current_treatment = float(
            row["input_end_treatment"]
            if "input_end_treatment" in row
            else row.get("current_treatment", row.get("input_treatment", 0.0))
        )
        target_treatment = float(row.get("target_treatment", current_treatment))
        samples.append(
            ForecastSample(
                patient_id=str(row["patient_id"]),
                input_idx=input_idx,
                target_idx=int(row["target_idx"]),
                horizon=int(row.get("horizon", int(row["target_idx"]) - input_idx)),
                delta_days=float(row["delta_days"]),
                current_treatment=current_treatment,
                target_treatment=target_treatment,
            )
        )
    return samples


def build_samples(args: argparse.Namespace) -> tuple[List[ForecastSample], pd.DataFrame | None]:
    if args.manifest_csv:
        manifest = normalize_manifest(pd.read_csv(args.manifest_csv))
        splits = _parse_csv(args.splits) or sorted(str(x) for x in manifest["split"].dropna().unique())
        return build_samples_from_manifest(manifest, splits), manifest

    samples = build_samples_for_split(
        dataset_root=Path(args.dataset_root),
        split=args.split,
        fit_sessions=args.fit_sessions,
        horizons=args.horizons,
        allowed_tiers=args.allowed_tiers,
    )
    return samples, None


def compute_samples(dataset_root: Path, samples: List[ForecastSample], manifest: pd.DataFrame | None) -> pd.DataFrame:
    split_lookup = {}
    if manifest is not None and "split" in manifest.columns:
        for _, row in manifest.iterrows():
            input_idx = int(row["input_idx"] if "input_idx" in row else row.get("input_end_idx"))
            horizon = int(row.get("horizon", int(row["target_idx"]) - input_idx))
            split_lookup[(str(row["patient_id"]), input_idx, int(row["target_idx"]), horizon)] = str(row["split"])

    label_cache: dict[str, np.ndarray] = {}
    rows = []
    for s in samples:
        if s.patient_id not in label_cache:
            label_cache[s.patient_id] = _standardize_label(np.load(patient_paths(dataset_root, s.patient_id)["label"]))
        labels = label_cache[s.patient_id]
        input_mask = labels[s.input_idx] > 0
        target_mask = labels[s.target_idx] > 0

        growth = target_mask & ~input_mask
        loss = input_mask & ~target_mask
        persistent = input_mask & target_mask
        union = input_mask | target_mask

        input_volume = int(input_mask.sum())
        target_volume = int(target_mask.sum())
        growth_volume = int(growth.sum())
        loss_volume = int(loss.sum())
        persistent_volume = int(persistent.sum())
        union_volume = int(union.sum())
        delta_days = float(s.delta_days)
        key = (s.patient_id, int(s.input_idx), int(s.target_idx), int(s.horizon))

        rows.append(
            {
                "split": split_lookup.get(key, "all"),
                "patient_id": s.patient_id,
                "tier": infer_tier_from_patient_id(s.patient_id, default_tier="REAL"),
                "input_idx": int(s.input_idx),
                "target_idx": int(s.target_idx),
                "horizon": int(s.horizon),
                "delta_days": delta_days,
                "current_treatment": float(s.current_treatment),
                "target_treatment": float(s.target_treatment),
                "input_volume_vox": input_volume,
                "target_volume_vox": target_volume,
                "persistent_volume_vox": persistent_volume,
                "union_volume_vox": union_volume,
                "new_growth_volume_vox": growth_volume,
                "loss_volume_vox": loss_volume,
                "net_delta_volume_vox": target_volume - input_volume,
                "absolute_change_volume_vox": growth_volume + loss_volume,
                "relative_new_growth": growth_volume / max(1, input_volume),
                "relative_loss": loss_volume / max(1, input_volume),
                "relative_net_growth": (target_volume - input_volume) / max(1, input_volume),
                "relative_absolute_change": (growth_volume + loss_volume) / max(1, input_volume),
                "locf_dice": float(dice_np(input_mask.astype(np.float32), target_mask.astype(np.float32))),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No samples were computed.")
    return add_derived_features(df)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["new_growth_rate_vox_per_day"] = _safe_rate(out["new_growth_volume_vox"], out["delta_days"])
    out["loss_rate_vox_per_day"] = _safe_rate(out["loss_volume_vox"], out["delta_days"])
    out["absolute_change_rate_vox_per_day"] = _safe_rate(out["absolute_change_volume_vox"], out["delta_days"])
    out["net_volume_rate_vox_per_day"] = _safe_rate(out["net_delta_volume_vox"], out["delta_days"])
    out["relative_new_growth_rate_per_day"] = _safe_rate(out["relative_new_growth"], out["delta_days"])
    out["relative_loss_rate_per_day"] = _safe_rate(out["relative_loss"], out["delta_days"])
    out["relative_absolute_change_rate_per_day"] = _safe_rate(out["relative_absolute_change"], out["delta_days"])
    out["relative_net_growth_rate_per_day"] = _safe_rate(out["relative_net_growth"], out["delta_days"])
    out["net_direction"] = np.select(
        [out["net_delta_volume_vox"] > 0, out["net_delta_volume_vox"] < 0],
        ["net_growth", "net_shrinkage"],
        default="net_stable",
    )
    return out


def add_bins(df: pd.DataFrame, interval_edges: List[float]) -> pd.DataFrame:
    out = df.copy()
    out["delta_days_bin"] = pd.cut(
        out["delta_days"],
        bins=interval_edges,
        labels=_interval_labels(interval_edges),
        include_lowest=True,
        right=True,
    ).astype("object")
    out["delta_days_qbin"] = _qbin(out["delta_days"], ["short", "medium", "long"])
    out["new_growth_rate_qbin"] = _qbin(out["new_growth_rate_vox_per_day"], ["low_rate", "medium_rate", "high_rate"])
    out["absolute_change_rate_qbin"] = _qbin(
        out["absolute_change_rate_vox_per_day"], ["low_change_rate", "medium_change_rate", "high_change_rate"]
    )
    out["relative_new_growth_qbin"] = _qbin(out["relative_new_growth"], ["low_growth", "medium_growth", "high_growth"])
    out["relative_abs_change_qbin"] = _qbin(
        out["relative_absolute_change"], ["low_abs_change", "medium_abs_change", "high_abs_change"]
    )
    return out


def summarize(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    cols = [c for c in group_cols if c in df.columns]
    work = df.copy()
    if not cols:
        work["_overall"] = "overall"
        cols = ["_overall"]

    out = (
        work.groupby(cols, dropna=False, observed=True)
        .agg(
            n_samples=("locf_dice", "size"),
            n_patients=("patient_id", "nunique"),
            mean_locf_dice=("locf_dice", "mean"),
            median_locf_dice=("locf_dice", "median"),
            std_locf_dice=("locf_dice", "std"),
            mean_delta_days=("delta_days", "mean"),
            median_delta_days=("delta_days", "median"),
            mean_new_growth_volume_vox=("new_growth_volume_vox", "mean"),
            median_new_growth_volume_vox=("new_growth_volume_vox", "median"),
            mean_loss_volume_vox=("loss_volume_vox", "mean"),
            mean_relative_new_growth=("relative_new_growth", "mean"),
            mean_new_growth_rate_vox_per_day=("new_growth_rate_vox_per_day", "mean"),
            mean_absolute_change_rate_vox_per_day=("absolute_change_rate_vox_per_day", "mean"),
        )
        .reset_index()
    )
    if "_overall" in out.columns:
        out = out.drop(columns=["_overall"])
    return out


def correlations(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "delta_days",
        "new_growth_volume_vox",
        "loss_volume_vox",
        "absolute_change_volume_vox",
        "relative_new_growth",
        "relative_loss",
        "relative_absolute_change",
        "new_growth_rate_vox_per_day",
        "absolute_change_rate_vox_per_day",
        "relative_new_growth_rate_per_day",
        "relative_absolute_change_rate_per_day",
    ]
    rows = []
    for col in metrics:
        if col not in df.columns or df[col].dropna().nunique() < 2:
            continue
        rows.append(
            {
                "feature": col,
                "pearson_corr_with_locf_dice": float(df[[col, "locf_dice"]].corr(method="pearson").iloc[0, 1]),
                "spearman_corr_with_locf_dice": float(df[[col, "locf_dice"]].corr(method="spearman").iloc[0, 1]),
            }
        )
    return pd.DataFrame(rows).sort_values("spearman_corr_with_locf_dice")


def standardized_regression(df: pd.DataFrame) -> pd.DataFrame:
    predictors = [
        "delta_days",
        "new_growth_rate_vox_per_day",
        "loss_rate_vox_per_day",
        "relative_new_growth",
        "relative_loss",
    ]
    work = df[predictors + ["locf_dice"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(work) < len(predictors) + 3:
        return pd.DataFrame()

    x = work[predictors].astype(float)
    x = np.log1p(x.clip(lower=0.0))
    x = (x - x.mean(axis=0)) / x.std(axis=0).replace(0, np.nan)
    x = x.fillna(0.0)
    y = work["locf_dice"].astype(float)
    y_std = (y - y.mean()) / (y.std() if y.std() > 0 else 1.0)

    design = np.column_stack([np.ones(len(x)), x.to_numpy()])
    coef, *_ = np.linalg.lstsq(design, y_std.to_numpy(), rcond=None)
    pred = design @ coef
    ss_res = float(np.sum((y_std.to_numpy() - pred) ** 2))
    ss_tot = float(np.sum((y_std.to_numpy() - y_std.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    rows = [{"feature": "intercept", "standardized_coefficient": float(coef[0]), "model_r2": r2, "n": int(len(work))}]
    for feature, value in zip(predictors, coef[1:]):
        rows.append({"feature": feature, "standardized_coefficient": float(value), "model_r2": r2, "n": int(len(work))})
    return pd.DataFrame(rows)


def write_plots(samples: pd.DataFrame, output_dir: Path) -> List[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: List[str] = []

    def save(fig, name: str) -> None:
        path = output_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(str(path))

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.scatter(samples["delta_days"], samples["locf_dice"], s=22, alpha=0.7)
    ax.set_xlabel("Delta days")
    ax.set_ylabel("LOCF Dice")
    ax.set_title("LOCF Dice vs calendar interval")
    save(fig, "locf_dice_vs_delta_days.png")

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.scatter(samples["new_growth_rate_vox_per_day"], samples["locf_dice"], s=22, alpha=0.7)
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_xlabel("New-growth rate (vox/day)")
    ax.set_ylabel("LOCF Dice")
    ax.set_title("LOCF Dice vs new-growth rate")
    save(fig, "locf_dice_vs_new_growth_rate.png")

    heat = (
        samples.groupby(["delta_days_qbin", "new_growth_rate_qbin"], observed=True, dropna=False)
        .agg(mean_locf_dice=("locf_dice", "mean"), n_samples=("locf_dice", "size"))
        .reset_index()
    )
    if not heat.empty:
        pivot = heat.pivot(index="delta_days_qbin", columns="new_growth_rate_qbin", values="mean_locf_dice")
        fig, ax = plt.subplots(figsize=(6.5, 4.6))
        im = ax.imshow(pivot.to_numpy(dtype=float), vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel("New-growth rate quantile")
        ax.set_ylabel("Delta-days quantile")
        ax.set_title("LOCF operating range")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.iloc[i, j]
                if pd.notna(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if val < 0.55 else "black")
        fig.colorbar(im, ax=ax, label="Mean LOCF Dice")
        save(fig, "locf_operating_range_heatmap.png")

    return paths


def write_report(
    path: Path,
    samples: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    plot_paths: List[str],
    args: argparse.Namespace,
) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# LOCF Operating Range Analysis\n\n")
        f.write(
            "This analysis quantifies when the persistence prior behind LOCF remains competitive. "
            "It separates session horizon from calendar interval and observed biological/change rate.\n\n"
        )
        f.write("## Inputs\n\n")
        f.write(f"- dataset_root: `{args.dataset_root}`\n")
        if args.manifest_csv:
            f.write(f"- manifest_csv: `{args.manifest_csv}`\n")
            f.write(f"- splits: `{args.splits}`\n")
        else:
            f.write(f"- split: `{args.split}`\n")
            f.write(f"- fit_sessions: `{args.fit_sessions}`\n")
            f.write(f"- horizons: `{args.horizons}`\n")
        f.write(f"- n_samples: `{len(samples)}`\n")
        f.write(f"- n_patients: `{samples['patient_id'].nunique()}`\n\n")

        f.write("## Core Definition\n\n")
        f.write("Short-term forecasting is treated as a combination of:\n\n")
        f.write("- session horizon;\n- calendar horizon (`delta_days`);\n- biological/change horizon (tumor change per unit time).\n\n")
        f.write("The main question is whether LOCF degrades as calendar interval and/or observed change rate increases.\n\n")

        for name, table in tables.items():
            f.write(f"## {name}\n\n")
            f.write(table.to_markdown(index=False) if not table.empty else "No rows.")
            f.write("\n\n")

        if plot_paths:
            f.write("## Figures\n\n")
            for p in plot_paths:
                f.write(f"- `{p}`\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantify the operating range of LOCF as a persistence prior.")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--manifest_csv", type=str, default=None, help="Optional longitudinal manifest CSV.")
    parser.add_argument("--splits", type=str, default="val,test", help="Manifest splits to include.")
    parser.add_argument("--split", type=str, default="test", help="Non-manifest split name.")
    parser.add_argument("--fit_sessions", type=int, default=3)
    parser.add_argument("--horizons", type=str, default="1,2,3")
    parser.add_argument("--allowed_tiers", type=str, default=None)
    parser.add_argument("--interval_bins", type=str, default="0,30,60,90,180,365,inf")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--no_plots", action="store_true")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples_l, manifest = build_samples(args)
    samples = add_bins(compute_samples(dataset_root, samples_l, manifest), _parse_float_bins(args.interval_bins))

    tables = {
        "Overall": summarize(samples, []),
        "By Split": summarize(samples, ["split"]),
        "By Horizon": summarize(samples, ["horizon"]),
        "By Delta Days Bin": summarize(samples, ["delta_days_bin"]),
        "By Delta Days Quantile": summarize(samples, ["delta_days_qbin"]),
        "By New-Growth Rate Quantile": summarize(samples, ["new_growth_rate_qbin"]),
        "By Absolute-Change Rate Quantile": summarize(samples, ["absolute_change_rate_qbin"]),
        "By Delta Quantile x Growth-Rate Quantile": summarize(samples, ["delta_days_qbin", "new_growth_rate_qbin"]),
        "By Net Direction": summarize(samples, ["net_direction"]),
        "By Patient": summarize(samples, ["patient_id"]),
        "Correlations": correlations(samples),
        "Standardized Regression": standardized_regression(samples),
    }

    samples.to_csv(output_dir / "locf_operating_samples.csv", index=False)
    for name, table in tables.items():
        filename = name.lower().replace(" ", "_").replace("-", "_").replace("x", "by")
        table.to_csv(output_dir / f"locf_operating_{filename}.csv", index=False)

    plot_paths = [] if args.no_plots else write_plots(samples, output_dir)
    write_report(output_dir / "locf_operating_range_report.md", samples, tables, plot_paths, args)

    payload = {
        "dataset_root": str(dataset_root),
        "manifest_csv": args.manifest_csv,
        "n_samples": int(len(samples)),
        "n_patients": int(samples["patient_id"].nunique()),
        "output_dir": str(output_dir),
        "outputs": {
            "samples_csv": str(output_dir / "locf_operating_samples.csv"),
            "report_md": str(output_dir / "locf_operating_range_report.md"),
            "plots": plot_paths,
        },
    }
    with (output_dir / "locf_operating_range_summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
