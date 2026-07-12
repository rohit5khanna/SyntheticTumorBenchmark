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

from baselines.metrics import dice_np
from baselines.tasks import build_samples_for_split, infer_tier_from_patient_id, patient_paths


KEY_COLS = ["patient_id", "input_idx", "target_idx", "horizon", "delta_days"]


def _standardize_label(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 5 and arr.shape[1] == 1:
        return (arr > 0).astype(np.float32)
    if arr.ndim == 4:
        return (arr[:, None, ...] > 0).astype(np.float32)
    raise ValueError(f"Unsupported label shape: {arr.shape}")


def _distance_to_input_score(input_mask: np.ndarray) -> np.ndarray | None:
    if int(input_mask.sum()) == 0:
        return None
    try:
        from scipy.ndimage import distance_transform_edt
    except Exception as e:
        print(f"[WARN] Distance baseline unavailable because scipy import failed: {e}")
        return None
    return -distance_transform_edt(~input_mask).astype(np.float32)


def _rank_normalize_score(score: np.ndarray) -> np.ndarray:
    s = np.asarray(score, dtype=np.float32).reshape(-1)
    if len(s) <= 1:
        return np.zeros_like(s, dtype=np.float32)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.arange(len(s), dtype=np.float32)
    return ranks / float(len(s) - 1)


def _topk_growth_mask(input_mask: np.ndarray, score_full: np.ndarray, k: int) -> np.ndarray:
    pred = input_mask.copy()
    outside = ~input_mask
    outside_idx = np.flatnonzero(outside.reshape(-1))
    if k <= 0 or len(outside_idx) == 0:
        return pred
    k = int(max(0, min(k, len(outside_idx))))
    scores = score_full.reshape(-1)[outside_idx]
    order = np.argsort(-scores, kind="mergesort")[:k]
    flat = pred.reshape(-1)
    flat[outside_idx[order]] = True
    return flat.reshape(pred.shape)


def _load_summary(output_dir: Path, method: str) -> Dict | None:
    path = output_dir / f"{method}_summary.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _infer_checkpoint(output_dir: Path, method: str) -> Path | None:
    summary = _load_summary(output_dir, method)
    if summary and summary.get("checkpoint"):
        ckpt = Path(summary["checkpoint"])
        if ckpt.exists():
            return ckpt
        local = output_dir / ckpt.name
        if local.exists():
            return local
    matches = sorted(output_dir.glob(f"model_best_{method}.pt"))
    if matches:
        return matches[0]
    return None


def _checkpoint_specs(output_dir: Path, methods: Iterable[str]) -> Dict[str, Dict]:
    specs = {}
    for method in methods:
        summary = _load_summary(output_dir, method)
        ckpt = _infer_checkpoint(output_dir, method)
        if summary is None or ckpt is None:
            continue
        specs[method] = {
            "checkpoint": ckpt,
            "model_variant": summary.get("model_variant", method.split("_")[0]),
            "input_mode": summary.get("input_mode", "image_mask" if "image_mask" in method else "mask"),
        }
    return specs


def _qbin(series: pd.Series, labels: List[str]) -> pd.Series:
    if series.dropna().nunique() < len(labels):
        return pd.Series(["all"] * len(series), index=series.index)
    return pd.qcut(series, q=len(labels), labels=labels, duplicates="drop")


def _absolute_growth_bins(df: pd.DataFrame) -> tuple[pd.Series, Dict]:
    nonzero = df.loc[df["growth_volume_vox"] > 0, "growth_volume_vox"].dropna()
    if nonzero.empty:
        small_max = 0.0
        large_min = 0.0
    else:
        small_max = float(nonzero.quantile(0.33))
        large_min = float(nonzero.quantile(0.67))

    def label(v: float) -> str:
        if pd.isna(v):
            return "unknown"
        if v <= 0:
            return "zero"
        if v <= small_max:
            return "small_nonzero"
        if v <= large_min:
            return "medium_nonzero"
        return "large_nonzero"

    return df["growth_volume_vox"].apply(label), {
        "small_nonzero_max_vox": small_max,
        "large_nonzero_min_vox": large_min,
    }


def _budget_values(labels: np.ndarray, input_idx: int, input_mask: np.ndarray, true_growth_count: int) -> Dict[str, int]:
    prev_growth = 0
    if input_idx > 0:
        prev_mask = (labels[input_idx - 1] > 0)[0]
        prev_growth = int((input_mask & ~prev_mask).sum())
    candidate_count = int((~input_mask).sum())
    return {
        "oracle_true_growth_volume": int(true_growth_count),
        "previous_growth_volume": int(prev_growth),
        "one_pct_candidates": int(max(1, round(0.01 * candidate_count))) if true_growth_count > 0 else 0,
        "five_pct_candidates": int(max(1, round(0.05 * candidate_count))) if true_growth_count > 0 else 0,
    }


def _add_rows_for_score(
    rows: List[Dict],
    score_name: str,
    score_full: np.ndarray,
    sample_meta: Dict,
    input_mask: np.ndarray,
    target_mask: np.ndarray,
    budgets: Dict[str, int],
) -> None:
    locf_dice = float(sample_meta["locf_dice"])
    for budget_policy, k in budgets.items():
        pred = _topk_growth_mask(input_mask=input_mask, score_full=score_full, k=int(k))
        d = dice_np(pred.astype(np.float32), target_mask.astype(np.float32))
        rows.append(
            {
                **sample_meta,
                "score_source": score_name,
                "budget_policy": budget_policy,
                "growth_budget_vox": int(k),
                "budget_to_true_growth_ratio": int(k) / max(1, int(sample_meta["growth_volume_vox"])),
                "persistence_growth_dice": d,
                "dice_gap_vs_locf": d - locf_dice,
            }
        )


def compute_budget_predictions(
    dataset_root: Path,
    split: str,
    fit_sessions: int,
    horizons: str,
    allowed_tiers: str | None,
    baseline_output_dir: Path,
    methods: Iterable[str],
    device: str,
) -> tuple[pd.DataFrame, Dict]:
    samples = build_samples_for_split(dataset_root, split, fit_sessions, horizons, allowed_tiers=allowed_tiers)
    label_cache: Dict[str, np.ndarray] = {}
    base_rows = []
    rows = []

    def labels_for(pid: str) -> np.ndarray:
        if pid not in label_cache:
            p = patient_paths(dataset_root, pid)
            label_cache[pid] = _standardize_label(np.load(p["label"]))
        return label_cache[pid]

    for s in samples:
        labels = labels_for(s.patient_id)
        input_mask = (labels[s.input_idx] > 0)[0]
        target_mask = (labels[s.target_idx] > 0)[0]
        growth = target_mask & ~input_mask
        locf_dice = dice_np(input_mask.astype(np.float32), target_mask.astype(np.float32))
        sample_meta = {
            "patient_id": s.patient_id,
            "input_idx": s.input_idx,
            "target_idx": s.target_idx,
            "horizon": s.horizon,
            "delta_days": s.delta_days,
            "tier": infer_tier_from_patient_id(s.patient_id),
            "input_volume_vox": int(input_mask.sum()),
            "target_volume_vox": int(target_mask.sum()),
            "growth_volume_vox": int(growth.sum()),
            "relative_new_growth": int(growth.sum()) / max(1, int(input_mask.sum())),
            "locf_dice": locf_dice,
        }
        base_rows.append(sample_meta)
        budgets = _budget_values(labels, s.input_idx, input_mask, int(growth.sum()))
        distance_score = _distance_to_input_score(input_mask)
        if distance_score is not None:
            _add_rows_for_score(rows, "distance_to_input_mask", distance_score, sample_meta, input_mask, target_mask, budgets)

    specs = _checkpoint_specs(baseline_output_dir, methods)
    if specs:
        try:
            import torch

            from baselines.unet import _TorchForecastDataset, _build_torch_model
        except Exception as e:
            print(f"[WARN] Skipping model scores because PyTorch/model code is unavailable: {e}")
            specs = {}

    if specs:
        import torch

        dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))
        for method, spec in specs.items():
            ds = _TorchForecastDataset(dataset_root, samples, input_mode=spec["input_mode"], cache_arrays=True)
            sample_x, _, _ = ds[0]
            ckpt = torch.load(spec["checkpoint"], map_location=dev, weights_only=False)
            base_channels = int(ckpt.get("base_channels", 12))
            in_channels = int(ckpt.get("in_channels", sample_x.shape[0]))
            model = _build_torch_model(in_channels, base_channels, model_variant=spec["model_variant"])
            model.load_state_dict(ckpt["model_state_dict"])
            model.to(dev)
            model.eval()

            for i, s in enumerate(samples):
                labels = labels_for(s.patient_id)
                input_mask = (labels[s.input_idx] > 0)[0]
                target_mask = (labels[s.target_idx] > 0)[0]
                growth = target_mask & ~input_mask
                locf_dice = dice_np(input_mask.astype(np.float32), target_mask.astype(np.float32))
                sample_meta = {
                    "patient_id": s.patient_id,
                    "input_idx": s.input_idx,
                    "target_idx": s.target_idx,
                    "horizon": s.horizon,
                    "delta_days": s.delta_days,
                    "tier": infer_tier_from_patient_id(s.patient_id),
                    "input_volume_vox": int(input_mask.sum()),
                    "target_volume_vox": int(target_mask.sum()),
                    "growth_volume_vox": int(growth.sum()),
                    "relative_new_growth": int(growth.sum()) / max(1, int(input_mask.sum())),
                    "locf_dice": locf_dice,
                }
                budgets = _budget_values(labels, s.input_idx, input_mask, int(growth.sum()))
                x, _, _ = ds[i]
                with torch.no_grad():
                    logits = model(x[None].to(dev))
                    prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()

                _add_rows_for_score(rows, method, prob, sample_meta, input_mask, target_mask, budgets)

                distance_score = _distance_to_input_score(input_mask)
                if distance_score is not None:
                    outside = ~input_mask
                    model_rank = np.zeros_like(prob, dtype=np.float32)
                    dist_rank = np.zeros_like(prob, dtype=np.float32)
                    model_rank[outside] = _rank_normalize_score(prob[outside])
                    dist_rank[outside] = _rank_normalize_score(distance_score[outside])
                    hybrid_score = 0.25 * dist_rank + 0.75 * model_rank
                    _add_rows_for_score(
                        rows,
                        f"hybrid_distance_{method}_a0.75",
                        hybrid_score,
                        sample_meta,
                        input_mask,
                        target_mask,
                        budgets,
                    )

    base_df = pd.DataFrame(base_rows)
    if not base_df.empty:
        base_df["absolute_growth_bin"], thresholds = _absolute_growth_bins(base_df)
        base_df["relative_growth_bin"] = _qbin(base_df["relative_new_growth"], ["low", "medium", "high"])
    else:
        thresholds = {"small_nonzero_max_vox": 0.0, "large_nonzero_min_vox": 0.0}

    pred_df = pd.DataFrame(rows)
    if not pred_df.empty:
        pred_df = pred_df.merge(
            base_df[KEY_COLS + ["absolute_growth_bin", "relative_growth_bin"]],
            on=KEY_COLS,
            how="left",
        )
    return pred_df, thresholds


def summarize(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    available = [c for c in group_cols if c in df.columns]
    return (
        df.groupby(available, dropna=False, observed=True)
        .agg(
            count=("persistence_growth_dice", "size"),
            mean_dice=("persistence_growth_dice", "mean"),
            mean_locf_dice=("locf_dice", "mean"),
            mean_gap_vs_locf=("dice_gap_vs_locf", "mean"),
            win_rate_vs_locf=("dice_gap_vs_locf", lambda x: float((x > 0).mean())),
            median_growth_volume_vox=("growth_volume_vox", "median"),
            mean_growth_budget_vox=("growth_budget_vox", "mean"),
            mean_budget_to_true_growth_ratio=("budget_to_true_growth_ratio", "mean"),
        )
        .reset_index()
        .sort_values(available)
    )


def write_report(path: Path, thresholds: Dict, overall: pd.DataFrame, by_growth: pd.DataFrame) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Persistence Plus Ranked-Growth Budget Analysis\n\n")
        f.write("This analysis keeps the input mask and adds only top-ranked candidate growth voxels.\n\n")
        f.write("## Absolute Growth Bins\n\n")
        f.write(f"- zero: growth volume <= 0 voxels\n")
        f.write(f"- small_nonzero: 0 < growth volume <= {thresholds['small_nonzero_max_vox']:.3f} voxels\n")
        f.write(
            f"- medium_nonzero: {thresholds['small_nonzero_max_vox']:.3f} < growth volume <= "
            f"{thresholds['large_nonzero_min_vox']:.3f} voxels\n"
        )
        f.write(f"- large_nonzero: growth volume > {thresholds['large_nonzero_min_vox']:.3f} voxels\n\n")
        f.write("## Overall\n\n")
        f.write(overall.to_markdown(index=False) if not overall.empty else "No overall summary.")
        f.write("\n\n## By Absolute Growth Bin\n\n")
        f.write(by_growth.to_markdown(index=False) if not by_growth.empty else "No growth-bin summary.")
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate persistence-preserving masks formed by adding top-ranked growth voxels."
    )
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--fit_sessions", type=int, default=3)
    parser.add_argument("--horizons", type=str, default="1,2,3")
    parser.add_argument("--allowed_tiers", type=str, default=None)
    parser.add_argument("--baseline_output_dir", type=str, required=True)
    parser.add_argument("--methods", type=str, default="unet_image_mask,resunet_image_mask")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    pred_df, thresholds = compute_budget_predictions(
        dataset_root=Path(args.dataset_root),
        split=args.split,
        fit_sessions=args.fit_sessions,
        horizons=args.horizons,
        allowed_tiers=args.allowed_tiers,
        baseline_output_dir=Path(args.baseline_output_dir),
        methods=methods,
        device=args.device,
    )

    overall = summarize(pred_df, ["score_source", "budget_policy"])
    by_growth = summarize(pred_df, ["score_source", "budget_policy", "absolute_growth_bin"])
    by_tier_growth = summarize(pred_df, ["score_source", "budget_policy", "tier", "absolute_growth_bin"])
    by_horizon_growth = summarize(pred_df, ["score_source", "budget_policy", "horizon", "absolute_growth_bin"])

    pred_df.to_csv(output_dir / "persistence_growth_budget_samples.csv", index=False)
    overall.to_csv(output_dir / "persistence_growth_budget_overall.csv", index=False)
    by_growth.to_csv(output_dir / "persistence_growth_budget_by_absolute_growth_bin.csv", index=False)
    by_tier_growth.to_csv(output_dir / "persistence_growth_budget_by_tier_growth_bin.csv", index=False)
    by_horizon_growth.to_csv(output_dir / "persistence_growth_budget_by_horizon_growth_bin.csv", index=False)
    write_report(output_dir / "persistence_growth_budget_report.md", thresholds, overall, by_growth)

    print(
        json.dumps(
            {
                "n_rows": int(len(pred_df)),
                "methods": methods,
                "absolute_growth_thresholds": thresholds,
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
