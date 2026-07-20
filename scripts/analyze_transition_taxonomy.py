#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.metrics import dice_np
from baselines.tasks import ForecastSample, build_samples_for_split, infer_tier_from_patient_id, patient_paths

EPS = 1e-8
EPS_DAYS = 1e-6


def _parse_csv(payload: str | None) -> List[str]:
    if payload is None:
        return []
    return [x.strip() for x in payload.split(",") if x.strip()]


def _parse_float_bins(payload: str) -> List[float]:
    out: List[float] = []
    for item in payload.split(","):
        item = item.strip().lower()
        if not item:
            continue
        out.append(float("inf") if item in {"inf", "infinity", "np.inf"} else float(item))
    if len(out) < 2:
        raise ValueError("Need at least two bin edges.")
    if any(out[i] >= out[i + 1] for i in range(len(out) - 1)):
        raise ValueError(f"Bin edges must be strictly increasing: {out}")
    return out


def _interval_labels(edges: List[float], suffix: str = "d") -> List[str]:
    labels = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        lo_s = f"{lo:g}"
        hi_s = "inf" if math.isinf(hi) else f"{hi:g}"
        labels.append(f"{lo_s}-{hi_s}{suffix}")
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


def _standardize_label(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 5 and arr.shape[1] == 1:
        return arr[:, 0] > 0
    if arr.ndim == 4:
        return arr > 0
    raise ValueError(f"Unsupported label shape: {arr.shape}")


def normalize_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    out = manifest.copy()
    if "input_idx" not in out.columns and "input_end_idx" in out.columns:
        out["input_idx"] = out["input_end_idx"]
    if "split" not in out.columns:
        out["split"] = "all"
    if "tier" not in out.columns:
        out["tier"] = out["patient_id"].astype(str).map(lambda x: infer_tier_from_patient_id(x, default_tier="REAL"))
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


def manifest_has_core_features(manifest: pd.DataFrame) -> bool:
    required = {
        "patient_id",
        "input_idx",
        "target_idx",
        "horizon",
        "delta_days",
        "input_volume_vox",
        "target_volume_vox",
        "growth_volume_vox",
        "loss_volume_vox",
        "relative_new_growth",
        "relative_loss",
        "relative_net_growth",
        "locf_dice",
    }
    return required.issubset(set(manifest.columns))


def build_core_from_manifest(manifest: pd.DataFrame, splits: Iterable[str]) -> pd.DataFrame:
    splits_l = list(splits)
    rows = manifest[manifest["split"].isin(splits_l)].copy() if splits_l else manifest.copy()
    if rows.empty:
        raise ValueError(f"No rows found for splits={splits_l} in manifest.")
    input_volume = pd.to_numeric(rows["input_volume_vox"], errors="coerce")
    target_volume = pd.to_numeric(rows["target_volume_vox"], errors="coerce")
    growth = pd.to_numeric(rows["growth_volume_vox"], errors="coerce")
    loss = pd.to_numeric(rows["loss_volume_vox"], errors="coerce")
    persistent = input_volume - loss
    union = input_volume + growth
    out = pd.DataFrame(
        {
            "split": rows["split"].astype(str),
            "patient_id": rows["patient_id"].astype(str),
            "tier": rows["tier"].astype(str)
            if "tier" in rows
            else rows["patient_id"].astype(str).map(lambda x: infer_tier_from_patient_id(x, default_tier="REAL")),
            "input_idx": rows["input_idx"].astype(int),
            "target_idx": rows["target_idx"].astype(int),
            "horizon": rows["horizon"].astype(int),
            "delta_days": rows["delta_days"].astype(float),
            "current_treatment": rows["input_end_treatment"].astype(float)
            if "input_end_treatment" in rows
            else rows.get("current_treatment", pd.Series(0.0, index=rows.index)).astype(float),
            "target_treatment": rows["target_treatment"].astype(float)
            if "target_treatment" in rows
            else rows.get("input_end_treatment", pd.Series(0.0, index=rows.index)).astype(float),
            "input_volume_vox": input_volume,
            "target_volume_vox": target_volume,
            "persistent_volume_vox": persistent.clip(lower=0),
            "union_volume_vox": union,
            "new_growth_volume_vox": growth,
            "loss_volume_vox": loss,
            "net_delta_volume_vox": pd.to_numeric(rows.get("net_delta_volume_vox", target_volume - input_volume), errors="coerce"),
            "absolute_change_volume_vox": growth + loss,
            "relative_new_growth": rows["relative_new_growth"].astype(float),
            "relative_loss": rows["relative_loss"].astype(float),
            "relative_net_growth": rows["relative_net_growth"].astype(float),
            "relative_absolute_change": rows["relative_new_growth"].astype(float) + rows["relative_loss"].astype(float),
            "locf_dice": rows["locf_dice"].astype(float),
        }
    )
    if "input_span_days" in rows:
        out["input_span_days"] = rows["input_span_days"].astype(float)
    if "previous_growth_volume_vox" in rows:
        out["previous_growth_volume_vox"] = rows["previous_growth_volume_vox"].astype(float)
    if "previous_loss_volume_vox" in rows:
        out["previous_loss_volume_vox"] = rows["previous_loss_volume_vox"].astype(float)
    if "previous_growth_ratio" in rows:
        out["previous_growth_ratio"] = rows["previous_growth_ratio"].astype(float)
    return out


def spatial_features(input_mask: np.ndarray, target_mask: np.ndarray, boundary_radius: float) -> dict:
    growth = target_mask & ~input_mask
    loss = input_mask & ~target_mask
    persistent = input_mask & target_mask
    union = input_mask | target_mask

    row = {
        "input_volume_vox": int(input_mask.sum()),
        "target_volume_vox": int(target_mask.sum()),
        "persistent_volume_vox": int(persistent.sum()),
        "union_volume_vox": int(union.sum()),
        "new_growth_volume_vox": int(growth.sum()),
        "loss_volume_vox": int(loss.sum()),
        "net_delta_volume_vox": int(target_mask.sum()) - int(input_mask.sum()),
        "absolute_change_volume_vox": int(growth.sum()) + int(loss.sum()),
        "locf_dice": float(dice_np(input_mask.astype(np.float32), target_mask.astype(np.float32))),
    }

    try:
        from scipy import ndimage as ndi
    except Exception:
        return row

    if input_mask.any():
        dist_to_input = ndi.distance_transform_edt(~input_mask)
        growth_dist = dist_to_input[growth]
        row["boundary_growth_volume_vox"] = int((growth_dist <= boundary_radius).sum()) if growth_dist.size else 0
        row["distant_growth_volume_vox"] = int((growth_dist > boundary_radius).sum()) if growth_dist.size else 0
        row["mean_growth_distance_to_input_vox"] = float(growth_dist.mean()) if growth_dist.size else 0.0
        row["max_growth_distance_to_input_vox"] = float(growth_dist.max()) if growth_dist.size else 0.0
    else:
        row["boundary_growth_volume_vox"] = 0
        row["distant_growth_volume_vox"] = int(growth.sum())
        row["mean_growth_distance_to_input_vox"] = float("nan")
        row["max_growth_distance_to_input_vox"] = float("nan")

    if target_mask.any():
        dist_to_target = ndi.distance_transform_edt(~target_mask)
        loss_dist = dist_to_target[loss]
        row["boundary_loss_volume_vox"] = int((loss_dist <= boundary_radius).sum()) if loss_dist.size else 0
        row["core_loss_volume_vox"] = int((loss_dist > boundary_radius).sum()) if loss_dist.size else 0
        row["mean_loss_distance_to_target_vox"] = float(loss_dist.mean()) if loss_dist.size else 0.0
        row["max_loss_distance_to_target_vox"] = float(loss_dist.max()) if loss_dist.size else 0.0
    else:
        row["boundary_loss_volume_vox"] = 0
        row["core_loss_volume_vox"] = int(loss.sum())
        row["mean_loss_distance_to_target_vox"] = float("nan")
        row["max_loss_distance_to_target_vox"] = float("nan")

    growth_structure, n_growth_components = ndi.label(growth)
    loss_structure, n_loss_components = ndi.label(loss)
    row["growth_component_count"] = int(n_growth_components)
    row["loss_component_count"] = int(n_loss_components)
    if n_growth_components > 0:
        sizes = np.bincount(growth_structure[growth].ravel())
        sizes = sizes[1:] if len(sizes) > 1 else np.asarray([], dtype=int)
        row["largest_growth_component_vox"] = int(sizes.max()) if len(sizes) else 0
    else:
        row["largest_growth_component_vox"] = 0
    if n_loss_components > 0:
        sizes = np.bincount(loss_structure[loss].ravel())
        sizes = sizes[1:] if len(sizes) > 1 else np.asarray([], dtype=int)
        row["largest_loss_component_vox"] = int(sizes.max()) if len(sizes) else 0
    else:
        row["largest_loss_component_vox"] = 0
    return row


def build_core_from_masks(dataset_root: Path, samples: List[ForecastSample], manifest: pd.DataFrame | None, boundary_radius: float) -> pd.DataFrame:
    split_lookup = {}
    extra_lookup = {}
    if manifest is not None:
        for _, row in manifest.iterrows():
            input_idx = int(row["input_idx"] if "input_idx" in row else row.get("input_end_idx"))
            horizon = int(row.get("horizon", int(row["target_idx"]) - input_idx))
            key = (str(row["patient_id"]), input_idx, int(row["target_idx"]), horizon)
            split_lookup[key] = str(row.get("split", "all"))
            extra_lookup[key] = row.to_dict()

    label_cache: dict[str, np.ndarray] = {}
    rows = []
    for s in samples:
        if s.patient_id not in label_cache:
            label_cache[s.patient_id] = _standardize_label(np.load(patient_paths(dataset_root, s.patient_id)["label"]))
        labels = label_cache[s.patient_id]
        input_mask = labels[s.input_idx] > 0
        target_mask = labels[s.target_idx] > 0
        spatial = spatial_features(input_mask, target_mask, boundary_radius=boundary_radius)
        input_volume = max(1, spatial["input_volume_vox"])
        key = (s.patient_id, int(s.input_idx), int(s.target_idx), int(s.horizon))
        extra = extra_lookup.get(key, {})
        row = {
            "split": split_lookup.get(key, "all"),
            "patient_id": s.patient_id,
            "tier": str(extra.get("tier", infer_tier_from_patient_id(s.patient_id, default_tier="REAL"))),
            "input_idx": int(s.input_idx),
            "target_idx": int(s.target_idx),
            "horizon": int(s.horizon),
            "delta_days": float(s.delta_days),
            "current_treatment": float(s.current_treatment),
            "target_treatment": float(s.target_treatment),
            **spatial,
        }
        row["relative_new_growth"] = row["new_growth_volume_vox"] / input_volume
        row["relative_loss"] = row["loss_volume_vox"] / input_volume
        row["relative_net_growth"] = row["net_delta_volume_vox"] / input_volume
        row["relative_absolute_change"] = row["absolute_change_volume_vox"] / input_volume
        for col in ["input_span_days", "previous_growth_volume_vox", "previous_loss_volume_vox", "previous_growth_ratio"]:
            if col in extra:
                row[col] = extra[col]
        rows.append(row)
    if not rows:
        raise ValueError("No transition rows computed.")
    return pd.DataFrame(rows)


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


def add_derived_features(df: pd.DataFrame, interval_edges: List[float]) -> pd.DataFrame:
    out = df.copy()
    input_volume = pd.to_numeric(out["input_volume_vox"], errors="coerce").clip(lower=1)
    target_volume = pd.to_numeric(out["target_volume_vox"], errors="coerce").clip(lower=1)
    union_volume = pd.to_numeric(out["union_volume_vox"], errors="coerce").clip(lower=1)
    delta_days = pd.to_numeric(out["delta_days"], errors="coerce").clip(lower=EPS_DAYS)

    out["persistent_input_fraction"] = out["persistent_volume_vox"] / input_volume
    out["target_covered_by_input_fraction"] = out["persistent_volume_vox"] / target_volume
    out["jaccard_index"] = out["persistent_volume_vox"] / union_volume
    out["new_growth_rate_vox_per_day"] = out["new_growth_volume_vox"] / delta_days
    out["loss_rate_vox_per_day"] = out["loss_volume_vox"] / delta_days
    out["absolute_change_rate_vox_per_day"] = out["absolute_change_volume_vox"] / delta_days
    out["relative_new_growth_rate_per_day"] = out["relative_new_growth"] / delta_days
    out["relative_loss_rate_per_day"] = out["relative_loss"] / delta_days
    out["relative_absolute_change_rate_per_day"] = out["relative_absolute_change"] / delta_days
    out["growth_loss_balance"] = (out["new_growth_volume_vox"] - out["loss_volume_vox"]) / (out["new_growth_volume_vox"] + out["loss_volume_vox"] + EPS)
    out["net_direction"] = np.select(
        [out["net_delta_volume_vox"] > 0, out["net_delta_volume_vox"] < 0],
        ["net_growth", "net_shrinkage"],
        default="net_stable",
    )

    if "boundary_growth_volume_vox" in out.columns:
        growth_volume = pd.to_numeric(out["new_growth_volume_vox"], errors="coerce").clip(lower=1)
        loss_volume = pd.to_numeric(out["loss_volume_vox"], errors="coerce").clip(lower=1)
        out["boundary_growth_fraction"] = out["boundary_growth_volume_vox"] / growth_volume
        out["distant_growth_fraction"] = out["distant_growth_volume_vox"] / growth_volume
        out["boundary_loss_fraction"] = out["boundary_loss_volume_vox"] / loss_volume
        out["core_loss_fraction"] = out["core_loss_volume_vox"] / loss_volume
    else:
        out["boundary_growth_fraction"] = np.nan
        out["distant_growth_fraction"] = np.nan
        out["boundary_loss_fraction"] = np.nan
        out["core_loss_fraction"] = np.nan

    out["delta_days_bin"] = pd.cut(
        out["delta_days"],
        bins=interval_edges,
        labels=_interval_labels(interval_edges),
        include_lowest=True,
        right=True,
    ).astype("object")
    out["relative_abs_change_qbin"] = _qbin(out["relative_absolute_change"], ["low_change", "medium_change", "high_change"])
    out["relative_growth_qbin"] = _qbin(out["relative_new_growth"], ["low_growth", "medium_growth", "high_growth"])
    out["relative_loss_qbin"] = _qbin(out["relative_loss"], ["low_loss", "medium_loss", "high_loss"])
    out["change_rate_qbin"] = _qbin(
        out["relative_absolute_change_rate_per_day"], ["low_change_rate", "medium_change_rate", "high_change_rate"]
    )
    return assign_transition_types(out)


def assign_transition_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rel_abs = out["relative_absolute_change"].fillna(0)
    rel_growth = out["relative_new_growth"].fillna(0)
    rel_loss = out["relative_loss"].fillna(0)
    boundary_growth = out["boundary_growth_fraction"].fillna(np.nan)
    distant_growth = out["distant_growth_fraction"].fillna(np.nan)

    conditions = [
        rel_abs <= 0.20,
        (rel_growth >= 0.20) & (rel_loss >= 0.20),
        (rel_growth >= 0.20) & (rel_loss < 0.20),
        (rel_loss >= 0.20) & (rel_growth < 0.20),
        (rel_growth > 0) & (boundary_growth >= 0.80),
        (rel_growth > 0) & (distant_growth >= 0.20),
    ]
    choices = [
        "persistence_dominant",
        "mixed_growth_loss",
        "growth_dominant",
        "loss_dominant",
        "boundary_growth_dominant",
        "distant_growth_present",
    ]
    out["transition_type"] = np.select(conditions, choices, default="moderate_mixed_change")
    out["has_distant_growth"] = (out["distant_growth_fraction"].fillna(0) >= 0.20) & (out["new_growth_volume_vox"] > 0)
    out["has_core_loss"] = (out["core_loss_fraction"].fillna(0) >= 0.20) & (out["loss_volume_vox"] > 0)
    return out


def summarize(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    cols = [c for c in group_cols if c in df.columns]
    work = df.copy()
    if not cols:
        work["_overall"] = "overall"
        cols = ["_overall"]
    out = (
        work.groupby(cols, observed=True, dropna=False)
        .agg(
            n_transitions=("patient_id", "size"),
            n_patients=("patient_id", "nunique"),
            mean_delta_days=("delta_days", "mean"),
            median_delta_days=("delta_days", "median"),
            mean_locf_dice=("locf_dice", "mean"),
            median_locf_dice=("locf_dice", "median"),
            mean_input_volume_vox=("input_volume_vox", "mean"),
            mean_target_volume_vox=("target_volume_vox", "mean"),
            mean_persistent_input_fraction=("persistent_input_fraction", "mean"),
            mean_target_covered_by_input_fraction=("target_covered_by_input_fraction", "mean"),
            mean_relative_new_growth=("relative_new_growth", "mean"),
            mean_relative_loss=("relative_loss", "mean"),
            mean_relative_absolute_change=("relative_absolute_change", "mean"),
            mean_relative_abs_change_rate_per_day=("relative_absolute_change_rate_per_day", "mean"),
            mean_boundary_growth_fraction=("boundary_growth_fraction", "mean"),
            mean_distant_growth_fraction=("distant_growth_fraction", "mean"),
            distant_growth_rate=("has_distant_growth", "mean"),
            core_loss_rate=("has_core_loss", "mean"),
        )
        .reset_index()
    )
    if "_overall" in out.columns:
        out = out.drop(columns=["_overall"])
    return out


def patient_trajectory_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.sort_values(["patient_id", "input_idx", "target_idx"])
        .groupby("patient_id", observed=True)
        .agg(
            n_transitions=("patient_id", "size"),
            split_modes=("split", lambda x: ",".join(sorted(set(map(str, x))))),
            first_input_idx=("input_idx", "min"),
            last_target_idx=("target_idx", "max"),
            mean_delta_days=("delta_days", "mean"),
            mean_locf_dice=("locf_dice", "mean"),
            mean_relative_new_growth=("relative_new_growth", "mean"),
            mean_relative_loss=("relative_loss", "mean"),
            mean_relative_absolute_change=("relative_absolute_change", "mean"),
            net_growth_fraction=("net_direction", lambda x: float((x == "net_growth").mean())),
            net_shrinkage_fraction=("net_direction", lambda x: float((x == "net_shrinkage").mean())),
            distant_growth_fraction=("has_distant_growth", "mean"),
            core_loss_fraction=("has_core_loss", "mean"),
            n_transition_types=("transition_type", lambda x: int(pd.Series(x).nunique())),
            transition_type_sequence=("transition_type", lambda x: " -> ".join(map(str, x))),
        )
        .reset_index()
    )
    return out


def write_plots(df: pd.DataFrame, out_dir: Path) -> List[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: List[str] = []

    def save(fig, name: str) -> None:
        path = out_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(str(path))

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.scatter(df["relative_new_growth"], df["relative_loss"], c=df["locf_dice"], cmap="viridis", s=28, alpha=0.75)
    ax.set_xlabel("Relative new growth")
    ax.set_ylabel("Relative apparent loss")
    ax.set_title("Transition decomposition colored by LOCF Dice")
    cb = fig.colorbar(ax.collections[0], ax=ax)
    cb.set_label("LOCF Dice")
    save(fig, "transition_growth_loss_scatter.png")

    counts = df["transition_type"].value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.barh(counts.index, counts.values, color="#4477aa")
    ax.set_xlabel("Number of transitions")
    ax.set_title("Transition type counts")
    save(fig, "transition_type_counts.png")

    heat = (
        df.groupby(["delta_days_bin", "relative_abs_change_qbin"], observed=True, dropna=False)
        .agg(mean_locf_dice=("locf_dice", "mean"), n=("locf_dice", "size"))
        .reset_index()
    )
    if not heat.empty:
        pivot = heat.pivot(index="delta_days_bin", columns="relative_abs_change_qbin", values="mean_locf_dice")
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        im = ax.imshow(pivot.to_numpy(dtype=float), vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=25, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel("Relative absolute change bin")
        ax.set_ylabel("Delta-days bin")
        ax.set_title("Persistence operating surface")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.iloc[i, j]
                if pd.notna(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if val < 0.55 else "black")
        fig.colorbar(im, ax=ax, label="Mean LOCF Dice")
        save(fig, "transition_operating_surface.png")

    return paths


def write_report(path: Path, tables: dict[str, pd.DataFrame], plot_paths: List[str], args: argparse.Namespace, feature_source: str) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Transition Taxonomy Analysis\n\n")
        f.write(
            "This analysis decomposes each forecasting transition into persistent tumor, new growth, and apparent loss. "
            "When masks are recomputed, it also separates boundary-adjacent from distant growth and boundary from core loss.\n\n"
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
        f.write(f"- feature_source: `{feature_source}`\n")
        f.write(f"- boundary_radius_vox: `{args.boundary_radius_vox}`\n\n")
        f.write("## Notes\n\n")
        f.write("- `new growth` means target-mask voxels not present in the input mask.\n")
        f.write("- `apparent loss` means input-mask voxels absent from the target mask.\n")
        f.write("- Boundary/distant labels use voxel distance to the opposite mask and should be treated as geometric descriptors, not biology by themselves.\n\n")
        for name, table in tables.items():
            f.write(f"## {name}\n\n")
            f.write(table.to_markdown(index=False) if not table.empty else "No rows.")
            f.write("\n\n")
        if plot_paths:
            f.write("## Figures\n\n")
            for p in plot_paths:
                f.write(f"- `{p}`\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Decompose longitudinal tumor forecasting transitions into interpretable components.")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--manifest_csv", type=str, default=None)
    parser.add_argument("--splits", type=str, default="train,val,test")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--fit_sessions", type=int, default=3)
    parser.add_argument("--horizons", type=str, default="1,2,3")
    parser.add_argument("--allowed_tiers", type=str, default=None)
    parser.add_argument("--interval_bins", type=str, default="0,30,60,90,180,365,inf")
    parser.add_argument("--spatial_mode", choices=["auto", "manifest", "recompute"], default="auto")
    parser.add_argument("--boundary_radius_vox", type=float, default=3.0)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--no_plots", action="store_true")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples, manifest = build_samples(args)
    splits = _parse_csv(args.splits)
    use_manifest = (
        manifest is not None
        and args.spatial_mode in {"auto", "manifest"}
        and manifest_has_core_features(manifest)
    )
    if args.spatial_mode == "manifest" and not use_manifest:
        raise ValueError("Manifest mode requested, but the manifest does not contain the required transition columns.")

    if use_manifest:
        transitions = build_core_from_manifest(manifest, splits)
        feature_source = "manifest_core_features"
    else:
        transitions = build_core_from_masks(dataset_root, samples, manifest, boundary_radius=args.boundary_radius_vox)
        feature_source = "recomputed_from_masks"

    transitions = add_derived_features(transitions, _parse_float_bins(args.interval_bins))
    patient_traj = patient_trajectory_summary(transitions)

    tables = {
        "Overall": summarize(transitions, []),
        "By Split": summarize(transitions, ["split"]),
        "By Tier": summarize(transitions, ["tier"]),
        "By Horizon": summarize(transitions, ["horizon"]),
        "By Net Direction": summarize(transitions, ["net_direction"]),
        "By Transition Type": summarize(transitions, ["transition_type"]),
        "By Delta Days Bin": summarize(transitions, ["delta_days_bin"]),
        "By Relative Absolute Change Bin": summarize(transitions, ["relative_abs_change_qbin"]),
        "By Change Rate Bin": summarize(transitions, ["change_rate_qbin"]),
        "By Split x Transition Type": summarize(transitions, ["split", "transition_type"]),
        "Patient Trajectory Summary": patient_traj,
    }

    transitions.to_csv(out_dir / "transition_taxonomy_samples.csv", index=False)
    patient_traj.to_csv(out_dir / "transition_taxonomy_patient_trajectories.csv", index=False)
    for name, table in tables.items():
        fname = name.lower().replace(" ", "_").replace("/", "_").replace("-", "_").replace("x", "by")
        table.to_csv(out_dir / f"transition_taxonomy_{fname}.csv", index=False)

    plot_paths = [] if args.no_plots else write_plots(transitions, out_dir)
    write_report(out_dir / "transition_taxonomy_report.md", tables, plot_paths, args, feature_source)

    payload = {
        "dataset_root": str(dataset_root),
        "manifest_csv": args.manifest_csv,
        "n_transitions": int(len(transitions)),
        "n_patients": int(transitions["patient_id"].nunique()),
        "feature_source": feature_source,
        "output_dir": str(out_dir),
        "outputs": {
            "samples_csv": str(out_dir / "transition_taxonomy_samples.csv"),
            "patient_trajectories_csv": str(out_dir / "transition_taxonomy_patient_trajectories.csv"),
            "report_md": str(out_dir / "transition_taxonomy_report.md"),
            "plots": plot_paths,
        },
    }
    with (out_dir / "transition_taxonomy_summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
