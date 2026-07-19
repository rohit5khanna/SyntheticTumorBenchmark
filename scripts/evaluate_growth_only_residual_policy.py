#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.metrics import dice_np
from baselines.tasks import ForecastSample, patient_paths
from baselines.unet import _TorchForecastDataset, _build_torch_model


def normalize_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    out = manifest.copy()
    if "input_idx" not in out.columns and "input_end_idx" in out.columns:
        out["input_idx"] = out["input_end_idx"]
    return out


def build_samples_from_manifest(manifest: pd.DataFrame, split: str) -> List[ForecastSample]:
    rows = manifest[manifest["split"] == split].copy()
    if rows.empty:
        raise ValueError(f"No rows found for split='{split}' in manifest.")

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


def parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def add_by_threshold(input_mask: np.ndarray, growth_prob: np.ndarray, threshold: float) -> tuple[np.ndarray, int]:
    candidate = (~input_mask) & (growth_prob >= threshold)
    pred = input_mask | candidate
    return pred, int(candidate.sum())


def add_by_budget(input_mask: np.ndarray, growth_prob: np.ndarray, budget_fraction: float) -> tuple[np.ndarray, int]:
    outside = ~input_mask
    n_outside = int(outside.sum())
    input_volume = int(input_mask.sum())
    k = int(math.ceil(max(0.0, budget_fraction) * max(1, input_volume)))
    k = min(k, n_outside)
    if k <= 0:
        return input_mask.copy(), 0

    flat_scores = growth_prob.reshape(-1)
    outside_idx = np.flatnonzero(outside.reshape(-1))
    if k >= len(outside_idx):
        chosen = outside_idx
    else:
        outside_scores = flat_scores[outside_idx]
        chosen_local = np.argpartition(outside_scores, -k)[-k:]
        chosen = outside_idx[chosen_local]
    pred_flat = input_mask.reshape(-1).copy()
    pred_flat[chosen] = True
    return pred_flat.reshape(input_mask.shape), int(len(chosen))


def evaluate_policy(
    model,
    dataset_root: Path,
    manifest: pd.DataFrame,
    split: str,
    input_mode: str,
    device,
    spatial_stride: int,
    mode: str,
    value: float,
    batch_size: int,
) -> pd.DataFrame:
    import torch
    from torch.utils.data import DataLoader

    samples = build_samples_from_manifest(manifest, split)
    ds = _TorchForecastDataset(dataset_root, samples, input_mode=input_mode)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))
    rows = []
    label_cache: Dict[str, np.ndarray] = {}
    model.eval()
    with torch.no_grad():
        for x, _, idx in loader:
            if spatial_stride > 1:
                s = spatial_stride
                x = x[:, :, ::s, ::s, ::s]
            x = x.to(device, non_blocking=True)
            probs = torch.sigmoid(model(x)).detach().cpu().numpy()
            for j in range(probs.shape[0]):
                sample = samples[int(idx[j])]
                if sample.patient_id not in label_cache:
                    labels = np.load(patient_paths(dataset_root, sample.patient_id)["label"])
                    label_cache[sample.patient_id] = _TorchForecastDataset._standardize_label_sessions(labels) > 0
                labels = label_cache[sample.patient_id]
                input_mask = labels[sample.input_idx, 0]
                target_mask = labels[sample.target_idx, 0]
                if spatial_stride > 1:
                    s = spatial_stride
                    input_mask = input_mask[::s, ::s, ::s]
                    target_mask = target_mask[::s, ::s, ::s]

                growth_prob = probs[j, 0]
                if mode == "threshold":
                    pred, added = add_by_threshold(input_mask, growth_prob, value)
                elif mode == "budget":
                    pred, added = add_by_budget(input_mask, growth_prob, value)
                else:
                    raise ValueError(f"Unknown mode: {mode}")

                true_growth = target_mask & ~input_mask
                pred_growth = pred & ~input_mask
                growth_tp = int((pred_growth & true_growth).sum())
                growth_fp = int((pred_growth & ~true_growth).sum())
                input_volume = int(input_mask.sum())
                target_volume = int(target_mask.sum())
                locf_dice = float(dice_np(input_mask.astype(np.float32), target_mask.astype(np.float32)))
                dice = float(dice_np(pred.astype(np.float32), target_mask.astype(np.float32)))
                rows.append(
                    {
                        "split": split,
                        "patient_id": sample.patient_id,
                        "input_idx": int(sample.input_idx),
                        "target_idx": int(sample.target_idx),
                        "horizon": int(sample.horizon),
                        "delta_days": float(sample.delta_days),
                        "policy_mode": mode,
                        "policy_value": float(value),
                        "dice": dice,
                        "locf_dice_recomputed": locf_dice,
                        "gap_vs_locf": dice - locf_dice,
                        "input_volume_vox": input_volume,
                        "target_volume_vox": target_volume,
                        "net_direction": "net_growth"
                        if target_volume > input_volume
                        else "net_shrinkage"
                        if target_volume < input_volume
                        else "net_stable",
                        "true_growth_volume_vox": int(true_growth.sum()),
                        "added_growth_volume_vox": int(added),
                        "growth_tp_vox": growth_tp,
                        "growth_fp_vox": growth_fp,
                        "growth_precision": float(growth_tp / added) if added else np.nan,
                        "growth_recall": float(growth_tp / int(true_growth.sum())) if int(true_growth.sum()) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    work = df.copy()
    by = group_cols if group_cols else ["_overall"]
    if not group_cols:
        work["_overall"] = "overall"
    out = (
        work.groupby(by, observed=True, dropna=False)
        .agg(
            n=("patient_id", "size"),
            n_patients=("patient_id", "nunique"),
            mean_dice=("dice", "mean"),
            locf_mean=("locf_dice_recomputed", "mean"),
            mean_gap_vs_locf=("gap_vs_locf", "mean"),
            win_rate_vs_locf=("gap_vs_locf", lambda s: float((s > 0).mean())),
            true_growth_volume_mean=("true_growth_volume_vox", "mean"),
            added_growth_volume_mean=("added_growth_volume_vox", "mean"),
            growth_precision_mean=("growth_precision", "mean"),
            growth_recall_mean=("growth_recall", "mean"),
        )
        .reset_index()
    )
    if "_overall" in out.columns:
        out = out.drop(columns=["_overall"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LOCF plus growth-only residual correction policies.")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--manifest_csv", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_variant", type=str, choices=["unet", "resunet", "plain_cnn"], default="resunet")
    parser.add_argument("--input_mode", type=str, choices=["mask", "image_mask"], default="image_mask")
    parser.add_argument("--validation_split", type=str, default="val")
    parser.add_argument("--eval_splits", type=str, default="val,test")
    parser.add_argument("--base_channels", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--spatial_stride", type=int, default=1)
    parser.add_argument("--thresholds", type=str, default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--budget_fractions", type=str, default="0,0.005,0.01,0.02,0.05,0.1,0.2")
    parser.add_argument("--selection_objective", type=str, choices=["mean_dice", "mean_gap_vs_locf", "net_growth_gap"], default="mean_dice")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    try:
        import torch
    except Exception as e:
        raise RuntimeError("PyTorch is required for growth-only residual policy evaluation.") from e

    dataset_root = Path(args.dataset_root)
    manifest = normalize_manifest(pd.read_csv(args.manifest_csv))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dev = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=dev, weights_only=False)
    in_channels = int(ckpt["in_channels"])
    base_channels = int(ckpt.get("base_channels", args.base_channels))
    model = _build_torch_model(
        in_channels=in_channels,
        base_channels=base_channels,
        model_variant=args.model_variant,
        out_channels=2,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(dev)

    candidates = [("threshold", v) for v in parse_float_list(args.thresholds)]
    candidates += [("budget", v) for v in parse_float_list(args.budget_fractions)]

    val_rows = []
    val_summaries = []
    for mode, value in candidates:
        part = evaluate_policy(
            model,
            dataset_root,
            manifest,
            args.validation_split,
            args.input_mode,
            dev,
            args.spatial_stride,
            mode,
            value,
            args.batch_size,
        )
        val_rows.append(part)
        overall = summarize(part, [])
        net_growth_gap = float(part.loc[part["net_direction"] == "net_growth", "gap_vs_locf"].mean())
        val_summaries.append(
            {
                "policy_mode": mode,
                "policy_value": float(value),
                "mean_dice": float(overall["mean_dice"].iloc[0]),
                "locf_mean": float(overall["locf_mean"].iloc[0]),
                "mean_gap_vs_locf": float(overall["mean_gap_vs_locf"].iloc[0]),
                "win_rate_vs_locf": float(overall["win_rate_vs_locf"].iloc[0]),
                "net_growth_gap": net_growth_gap,
                "added_growth_volume_mean": float(overall["added_growth_volume_mean"].iloc[0]),
                "growth_precision_mean": float(overall["growth_precision_mean"].iloc[0]),
                "growth_recall_mean": float(overall["growth_recall_mean"].iloc[0]),
            }
        )

    val_selection = pd.DataFrame(val_summaries)
    selected_row = val_selection.sort_values(args.selection_objective, ascending=False).iloc[0]
    selected_mode = str(selected_row["policy_mode"])
    selected_value = float(selected_row["policy_value"])

    eval_splits = [s.strip() for s in args.eval_splits.split(",") if s.strip()]
    eval_rows = []
    for split in eval_splits:
        part = evaluate_policy(
            model,
            dataset_root,
            manifest,
            split,
            args.input_mode,
            dev,
            args.spatial_stride,
            selected_mode,
            selected_value,
            args.batch_size,
        )
        eval_rows.append(part)
    samples = pd.concat(eval_rows, ignore_index=True)
    by_split = summarize(samples, ["split"])
    by_direction = summarize(samples, ["split", "net_direction"])

    val_all = pd.concat(val_rows, ignore_index=True)
    val_selection.to_csv(output_dir / "growth_only_policy_validation_sweep.csv", index=False)
    val_all.to_csv(output_dir / "growth_only_policy_validation_samples_all.csv", index=False)
    samples.to_csv(output_dir / "growth_only_policy_selected_samples.csv", index=False)
    by_split.to_csv(output_dir / "growth_only_policy_selected_by_split.csv", index=False)
    by_direction.to_csv(output_dir / "growth_only_policy_selected_by_net_direction.csv", index=False)

    report = {
        "dataset_root": str(dataset_root.resolve()),
        "manifest_csv": str(Path(args.manifest_csv).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "in_channels": in_channels,
        "base_channels": base_channels,
        "input_mode": args.input_mode,
        "model_variant": args.model_variant,
        "spatial_stride": int(args.spatial_stride),
        "selection_objective": args.selection_objective,
        "selected_policy_mode": selected_mode,
        "selected_policy_value": selected_value,
        "selected_validation_row": selected_row.to_dict(),
        "by_split": by_split.to_dict(orient="records"),
        "outputs": {
            "validation_sweep_csv": str(output_dir / "growth_only_policy_validation_sweep.csv"),
            "selected_samples_csv": str(output_dir / "growth_only_policy_selected_samples.csv"),
            "selected_by_split_csv": str(output_dir / "growth_only_policy_selected_by_split.csv"),
            "selected_by_net_direction_csv": str(output_dir / "growth_only_policy_selected_by_net_direction.csv"),
        },
    }
    with (output_dir / "growth_only_policy_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
