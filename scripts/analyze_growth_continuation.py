#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.tasks import build_samples_for_split, infer_tier_from_patient_id, patient_paths
from benchmark.audit import session_shape_metrics


PREDICTION_TIME_FEATURES = [
    "input_volume_vox",
    "prev_volume_vox",
    "prev_new_growth_vox",
    "prev_lost_volume_vox",
    "prev_net_delta_volume_vox",
    "prev_abs_change_vox",
    "prev_relative_new_growth",
    "prev_relative_net_delta",
    "prev_interval_days",
    "delta_days",
    "input_treatment",
    "prev_treatment",
    "treatment_started_at_input",
    "input_elongation_ratio",
    "input_compactness_proxy",
    "input_connected_component_count",
    "input_bbox_volume_vox",
]


def _standardize_label(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 5 and arr.shape[1] == 1:
        arr = arr[:, 0]
    elif arr.ndim != 4:
        raise ValueError(f"Unsupported label shape: {arr.shape}")
    return (arr > 0).astype(bool)


def _load_patient_arrays(dataset_root: Path, patient_id: str) -> Dict[str, np.ndarray]:
    paths = patient_paths(dataset_root, patient_id)
    return {
        "label": _standardize_label(np.load(paths["label"])),
        "days": np.asarray(np.load(paths["days"]), dtype=np.float32),
        "treatment": np.asarray(np.load(paths["treatment"]), dtype=np.float32),
    }


def _growth_features(prefix: str, from_mask: np.ndarray, to_mask: np.ndarray) -> Dict[str, float]:
    from_vol = int(from_mask.sum())
    to_vol = int(to_mask.sum())
    new_growth = int((to_mask & ~from_mask).sum())
    lost = int((from_mask & ~to_mask).sum())
    net_delta = to_vol - from_vol
    abs_change = int(np.logical_xor(to_mask, from_mask).sum())
    denom = max(1, from_vol)
    return {
        f"{prefix}_from_volume_vox": float(from_vol),
        f"{prefix}_to_volume_vox": float(to_vol),
        f"{prefix}_new_growth_vox": float(new_growth),
        f"{prefix}_lost_volume_vox": float(lost),
        f"{prefix}_net_delta_volume_vox": float(net_delta),
        f"{prefix}_abs_change_vox": float(abs_change),
        f"{prefix}_relative_new_growth": float(new_growth / denom),
        f"{prefix}_relative_lost_volume": float(lost / denom),
        f"{prefix}_relative_net_delta": float(net_delta / denom),
        f"{prefix}_relative_abs_change": float(abs_change / denom),
    }


def _state_label(prev_active: bool, future_active: bool) -> str:
    if prev_active and future_active:
        return "continued_growth"
    if prev_active and not future_active:
        return "stopped_growth"
    if not prev_active and future_active:
        return "newly_active"
    return "stable"


def _qbin(series: pd.Series, labels: List[str]) -> pd.Series:
    valid = series.replace([np.inf, -np.inf], np.nan).dropna()
    if valid.nunique() < len(labels):
        return pd.Series(["all"] * len(series), index=series.index)
    return pd.qcut(series, q=len(labels), labels=labels, duplicates="drop")


def build_continuation_table(
    dataset_root: Path,
    split: str,
    fit_sessions: int,
    horizons: str,
    allowed_tiers: str | None,
    min_growth_vox: int,
) -> pd.DataFrame:
    samples = build_samples_for_split(
        dataset_root=dataset_root,
        split=split,
        fit_sessions=fit_sessions,
        horizons=horizons,
        allowed_tiers=allowed_tiers,
    )
    cache: Dict[str, Dict[str, np.ndarray]] = {}
    rows = []

    for sample in samples:
        if sample.patient_id not in cache:
            cache[sample.patient_id] = _load_patient_arrays(dataset_root, sample.patient_id)
        arr = cache[sample.patient_id]
        labels = arr["label"]
        days = arr["days"]
        treatment = arr["treatment"]

        if sample.input_idx <= 0:
            continue

        prev_idx = sample.input_idx - 1
        input_idx = sample.input_idx
        target_idx = sample.target_idx
        prev_mask = labels[prev_idx]
        input_mask = labels[input_idx]
        target_mask = labels[target_idx]

        prev_feats = _growth_features("prev", prev_mask, input_mask)
        future_feats = _growth_features("future", input_mask, target_mask)
        prev_active = prev_feats["prev_new_growth_vox"] > min_growth_vox
        future_active = future_feats["future_new_growth_vox"] > min_growth_vox
        state = _state_label(prev_active, future_active)
        input_shape = session_shape_metrics(input_mask.astype(np.uint8))

        rows.append(
            {
                "patient_id": sample.patient_id,
                "tier": infer_tier_from_patient_id(sample.patient_id),
                "split": split,
                "prev_idx": prev_idx,
                "input_idx": input_idx,
                "target_idx": target_idx,
                "horizon": sample.horizon,
                "prev_day": float(days[prev_idx]),
                "input_day": float(days[input_idx]),
                "target_day": float(days[target_idx]),
                "prev_interval_days": float(days[input_idx] - days[prev_idx]),
                "delta_days": float(days[target_idx] - days[input_idx]),
                "prev_treatment": float(treatment[prev_idx]),
                "input_treatment": float(treatment[input_idx]),
                "target_treatment": float(treatment[target_idx]),
                "treatment_started_at_input": int(treatment[prev_idx] <= 0 and treatment[input_idx] > 0),
                "prev_growth_active": int(prev_active),
                "future_growth_active": int(future_active),
                "continuation_state": state,
                **prev_feats,
                **future_feats,
                "input_volume_vox": float(input_shape["volume_vox"]),
                "input_bbox_volume_vox": float(input_shape["bbox_volume_vox"]),
                "input_elongation_ratio": float(input_shape["elongation_ratio"]),
                "input_compactness_proxy": float(input_shape["compactness_proxy"]),
                "input_connected_component_count": float(input_shape["connected_component_count"]),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df["prev_new_growth_bin"] = _qbin(df["prev_new_growth_vox"], ["low", "medium", "high"])
        df["future_new_growth_bin"] = _qbin(df["future_new_growth_vox"], ["low", "medium", "high"])
        df["input_volume_bin"] = _qbin(df["input_volume_vox"], ["small", "medium", "large"])
        df["delta_days_bin"] = _qbin(df["delta_days"], ["short", "medium", "long"])
    return df


def summarize_state_counts(df: pd.DataFrame, group_cols: Iterable[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    cols = [c for c in group_cols if c in df.columns]
    by = cols + ["continuation_state"]
    counts = df.groupby(by, observed=True, dropna=False).size().reset_index(name="count")
    denom_cols = cols if cols else []
    if denom_cols:
        counts["fraction"] = counts["count"] / counts.groupby(denom_cols)["count"].transform("sum")
    else:
        counts["fraction"] = counts["count"] / counts["count"].sum()
    return counts.sort_values(by).reset_index(drop=True)


def summarize_numeric_by_state(df: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
    rows = []
    for state, group in df.groupby("continuation_state", observed=True, dropna=False):
        for feature in features:
            if feature not in group.columns:
                continue
            values = group[feature].replace([np.inf, -np.inf], np.nan).dropna().astype(float)
            if values.empty:
                continue
            rows.append(
                {
                    "continuation_state": state,
                    "feature": feature,
                    "count": int(len(values)),
                    "mean": float(values.mean()),
                    "median": float(values.median()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "q25": float(values.quantile(0.25)),
                    "q75": float(values.quantile(0.75)),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )
    return pd.DataFrame(rows)


def stopped_vs_continued_contrast(df: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
    sub = df[df["continuation_state"].isin(["stopped_growth", "continued_growth"])].copy()
    rows = []
    stopped = sub[sub["continuation_state"] == "stopped_growth"]
    continued = sub[sub["continuation_state"] == "continued_growth"]
    for feature in features:
        if feature not in sub.columns:
            continue
        a = stopped[feature].replace([np.inf, -np.inf], np.nan).dropna().astype(float).to_numpy()
        b = continued[feature].replace([np.inf, -np.inf], np.nan).dropna().astype(float).to_numpy()
        if len(a) == 0 or len(b) == 0:
            continue
        pooled = np.sqrt(((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)) / max(1, len(a) + len(b) - 2))
        cohen_d = float((np.mean(a) - np.mean(b)) / pooled) if pooled > 0 else 0.0
        rows.append(
            {
                "feature": feature,
                "stopped_count": int(len(a)),
                "continued_count": int(len(b)),
                "stopped_mean": float(np.mean(a)),
                "continued_mean": float(np.mean(b)),
                "mean_gap_stopped_minus_continued": float(np.mean(a) - np.mean(b)),
                "stopped_median": float(np.median(a)),
                "continued_median": float(np.median(b)),
                "median_gap_stopped_minus_continued": float(np.median(a) - np.median(b)),
                "cohen_d_stopped_minus_continued": cohen_d,
                "abs_cohen_d": abs(cohen_d),
            }
        )
    return pd.DataFrame(rows).sort_values("abs_cohen_d", ascending=False).reset_index(drop=True)


def write_report(
    path: Path,
    args: argparse.Namespace,
    counts: pd.DataFrame,
    by_tier: pd.DataFrame,
    contrast: pd.DataFrame,
) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Growth Continuation Analysis\n\n")
        f.write("This report analyzes whether previous growth continues, stops, newly appears, or remains stable.\n\n")
        f.write("## Setup\n\n")
        f.write(f"- dataset root: `{args.dataset_root}`\n")
        f.write(f"- split: `{args.split}`\n")
        f.write(f"- fit sessions: `{args.fit_sessions}`\n")
        f.write(f"- horizons: `{args.horizons}`\n")
        f.write(f"- minimum growth threshold: `{args.min_growth_vox}` voxels\n\n")
        f.write("## State Counts\n\n")
        f.write(counts.to_markdown(index=False) if not counts.empty else "No state counts.")
        f.write("\n\n## State Counts By Tier\n\n")
        f.write(by_tier.to_markdown(index=False) if not by_tier.empty else "No tier counts.")
        f.write("\n\n## Stopped Versus Continued Growth Feature Contrast\n\n")
        f.write(contrast.to_markdown(index=False) if not contrast.empty else "No stopped/continued contrast available.")
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze growth continuation and cessation states from longitudinal masks.")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--fit_sessions", type=int, default=3)
    parser.add_argument("--horizons", type=str, default="1,2,3")
    parser.add_argument("--allowed_tiers", type=str, default=None)
    parser.add_argument("--min_growth_vox", type=int, default=0)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = build_continuation_table(
        dataset_root=Path(args.dataset_root),
        split=args.split,
        fit_sessions=args.fit_sessions,
        horizons=args.horizons,
        allowed_tiers=args.allowed_tiers,
        min_growth_vox=args.min_growth_vox,
    )

    counts = summarize_state_counts(df, [])
    by_tier = summarize_state_counts(df, ["tier"])
    by_horizon = summarize_state_counts(df, ["horizon"])
    by_tier_horizon = summarize_state_counts(df, ["tier", "horizon"])
    feature_profiles = summarize_numeric_by_state(df, PREDICTION_TIME_FEATURES)
    contrast = stopped_vs_continued_contrast(df, PREDICTION_TIME_FEATURES)

    df.to_csv(output_dir / "growth_continuation_samples.csv", index=False)
    counts.to_csv(output_dir / "growth_continuation_state_counts.csv", index=False)
    by_tier.to_csv(output_dir / "growth_continuation_state_by_tier.csv", index=False)
    by_horizon.to_csv(output_dir / "growth_continuation_state_by_horizon.csv", index=False)
    by_tier_horizon.to_csv(output_dir / "growth_continuation_state_by_tier_horizon.csv", index=False)
    feature_profiles.to_csv(output_dir / "growth_continuation_feature_profiles.csv", index=False)
    contrast.to_csv(output_dir / "stopped_vs_continued_feature_contrast.csv", index=False)
    write_report(output_dir / "growth_continuation_report.md", args, counts, by_tier, contrast)

    print(
        json.dumps(
            {
                "dataset_root": args.dataset_root,
                "split": args.split,
                "fit_sessions": args.fit_sessions,
                "horizons": args.horizons,
                "min_growth_vox": args.min_growth_vox,
                "n_samples": int(len(df)),
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
