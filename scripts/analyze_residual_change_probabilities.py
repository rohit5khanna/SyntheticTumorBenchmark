#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


def limit_samples(samples: List[ForecastSample], max_samples: int) -> List[ForecastSample]:
    if max_samples <= 0 or max_samples >= len(samples):
        return samples
    return samples[:max_samples]


def safe_mean(values: np.ndarray, mask: np.ndarray) -> float:
    if int(mask.sum()) == 0:
        return float("nan")
    return float(values[mask].mean())


def safe_sum(values: np.ndarray, mask: np.ndarray) -> float:
    if int(mask.sum()) == 0:
        return 0.0
    return float(values[mask].sum())


def binary_metrics(y_true: np.ndarray, score: np.ndarray, rng: np.random.Generator, max_pos: int, max_neg: int) -> Dict[str, float]:
    y = y_true.reshape(-1).astype(np.uint8)
    s = score.reshape(-1).astype(np.float32)
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return {
            "average_precision": float("nan"),
            "roc_auc": float("nan"),
            "n_pos_eval": int(len(pos_idx)),
            "n_neg_eval": int(len(neg_idx)),
        }

    if max_pos > 0 and len(pos_idx) > max_pos:
        pos_idx = rng.choice(pos_idx, size=max_pos, replace=False)
    if max_neg > 0 and len(neg_idx) > max_neg:
        neg_idx = rng.choice(neg_idx, size=max_neg, replace=False)

    idx = np.concatenate([pos_idx, neg_idx])
    rng.shuffle(idx)
    yy = y[idx]
    ss = s[idx]

    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        ap = float(average_precision_score(yy, ss))
        auc = float(roc_auc_score(yy, ss))
    except Exception:
        ap = float("nan")
        auc = float("nan")

    return {
        "average_precision": ap,
        "roc_auc": auc,
        "n_pos_eval": int(len(pos_idx)),
        "n_neg_eval": int(len(neg_idx)),
    }


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
            growth_ap_mean=("growth_average_precision", "mean"),
            growth_auc_mean=("growth_roc_auc", "mean"),
            loss_ap_mean=("loss_average_precision", "mean"),
            loss_auc_mean=("loss_roc_auc", "mean"),
            true_growth_volume_mean=("true_growth_volume_vox", "mean"),
            true_loss_volume_mean=("true_loss_volume_vox", "mean"),
            growth_prob_true_growth_mean=("growth_prob_true_growth", "mean"),
            growth_prob_outside_non_growth_mean=("growth_prob_outside_non_growth", "mean"),
            loss_prob_true_loss_mean=("loss_prob_true_loss", "mean"),
            loss_prob_stable_core_mean=("loss_prob_stable_core", "mean"),
            growth_prob_mass_outside_input_mean=("growth_prob_mass_outside_input", "mean"),
            loss_prob_mass_inside_input_mean=("loss_prob_mass_inside_input", "mean"),
        )
        .reset_index()
    )
    if "_overall" in out.columns:
        out = out.drop(columns=["_overall"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose residual growth/loss probability maps before hard thresholding.")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--manifest_csv", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_variant", type=str, choices=["unet", "resunet", "plain_cnn"], default="resunet")
    parser.add_argument("--input_mode", type=str, choices=["mask", "image_mask"], default="image_mask")
    parser.add_argument("--eval_splits", type=str, default="val,test")
    parser.add_argument("--base_channels", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--spatial_stride", type=int, default=1)
    parser.add_argument("--max_samples_per_split", type=int, default=0)
    parser.add_argument("--max_pos_voxels", type=int, default=20000)
    parser.add_argument("--max_neg_voxels", type=int, default=80000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    try:
        import torch
        from torch.utils.data import DataLoader
    except Exception as e:
        raise RuntimeError("PyTorch is required for residual probability diagnostics.") from e

    if args.spatial_stride < 1:
        raise ValueError("spatial_stride must be >= 1.")

    rng = np.random.default_rng(args.seed)
    dataset_root = Path(args.dataset_root)
    manifest = normalize_manifest(pd.read_csv(args.manifest_csv))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(args.device)

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
    model.eval()

    eval_splits = [s.strip() for s in args.eval_splits.split(",") if s.strip()]
    rows = []
    label_cache: Dict[str, np.ndarray] = {}
    for split in eval_splits:
        samples_all = build_samples_from_manifest(manifest, split)
        samples = limit_samples(samples_all, args.max_samples_per_split)
        ds = _TorchForecastDataset(dataset_root, samples, input_mode=args.input_mode)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=(dev.type == "cuda"))
        print(f"[INFO] split={split} samples={len(samples)}/{len(samples_all)}", flush=True)
        with torch.no_grad():
            for x, _, idx in loader:
                if args.spatial_stride > 1:
                    s = args.spatial_stride
                    x = x[:, :, ::s, ::s, ::s]
                x = x.to(dev, non_blocking=True)
                probs = torch.sigmoid(model(x)).detach().cpu().numpy()
                for j in range(probs.shape[0]):
                    sample = samples[int(idx[j])]
                    if sample.patient_id not in label_cache:
                        labels = np.load(patient_paths(dataset_root, sample.patient_id)["label"])
                        label_cache[sample.patient_id] = _TorchForecastDataset._standardize_label_sessions(labels) > 0

                    labels = label_cache[sample.patient_id]
                    input_mask = labels[sample.input_idx, 0]
                    target_mask = labels[sample.target_idx, 0]
                    if args.spatial_stride > 1:
                        s = args.spatial_stride
                        input_mask = input_mask[::s, ::s, ::s]
                        target_mask = target_mask[::s, ::s, ::s]

                    growth_prob = probs[j, 0]
                    loss_prob = probs[j, 1]
                    true_growth = target_mask & ~input_mask
                    true_loss = input_mask & ~target_mask
                    stable_core = input_mask & target_mask
                    outside_input = ~input_mask
                    inside_input = input_mask
                    outside_non_growth = outside_input & ~true_growth
                    background = (~input_mask) & (~target_mask)

                    growth_metrics = binary_metrics(
                        true_growth[outside_input],
                        growth_prob[outside_input],
                        rng,
                        args.max_pos_voxels,
                        args.max_neg_voxels,
                    )
                    loss_metrics = binary_metrics(
                        true_loss[inside_input],
                        loss_prob[inside_input],
                        rng,
                        args.max_pos_voxels,
                        args.max_neg_voxels,
                    )

                    input_volume = int(input_mask.sum())
                    target_volume = int(target_mask.sum())
                    true_growth_volume = int(true_growth.sum())
                    true_loss_volume = int(true_loss.sum())

                    rows.append(
                        {
                            "split": split,
                            "patient_id": sample.patient_id,
                            "input_idx": int(sample.input_idx),
                            "target_idx": int(sample.target_idx),
                            "horizon": int(sample.horizon),
                            "delta_days": float(sample.delta_days),
                            "input_volume_vox": input_volume,
                            "target_volume_vox": target_volume,
                            "net_direction": "net_growth"
                            if target_volume > input_volume
                            else "net_shrinkage"
                            if target_volume < input_volume
                            else "net_stable",
                            "true_growth_volume_vox": true_growth_volume,
                            "true_loss_volume_vox": true_loss_volume,
                            "growth_average_precision": growth_metrics["average_precision"],
                            "growth_roc_auc": growth_metrics["roc_auc"],
                            "growth_n_pos_eval": growth_metrics["n_pos_eval"],
                            "growth_n_neg_eval": growth_metrics["n_neg_eval"],
                            "loss_average_precision": loss_metrics["average_precision"],
                            "loss_roc_auc": loss_metrics["roc_auc"],
                            "loss_n_pos_eval": loss_metrics["n_pos_eval"],
                            "loss_n_neg_eval": loss_metrics["n_neg_eval"],
                            "growth_prob_true_growth": safe_mean(growth_prob, true_growth),
                            "growth_prob_outside_non_growth": safe_mean(growth_prob, outside_non_growth),
                            "growth_prob_background": safe_mean(growth_prob, background),
                            "growth_prob_stable_core": safe_mean(growth_prob, stable_core),
                            "loss_prob_true_loss": safe_mean(loss_prob, true_loss),
                            "loss_prob_stable_core": safe_mean(loss_prob, stable_core),
                            "loss_prob_outside_input": safe_mean(loss_prob, outside_input),
                            "growth_prob_mass_outside_input": safe_sum(growth_prob, outside_input),
                            "loss_prob_mass_inside_input": safe_sum(loss_prob, inside_input),
                        }
                    )

    sample_df = pd.DataFrame(rows)
    overall = summarize(sample_df, [])
    by_split = summarize(sample_df, ["split"])
    by_direction = summarize(sample_df, ["split", "net_direction"])

    sample_df.to_csv(output_dir / "residual_probability_sample_summary.csv", index=False)
    overall.to_csv(output_dir / "residual_probability_overall.csv", index=False)
    by_split.to_csv(output_dir / "residual_probability_by_split.csv", index=False)
    by_direction.to_csv(output_dir / "residual_probability_by_net_direction.csv", index=False)
    report = {
        "dataset_root": str(dataset_root.resolve()),
        "manifest_csv": str(Path(args.manifest_csv).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "input_mode": args.input_mode,
        "model_variant": args.model_variant,
        "in_channels": in_channels,
        "base_channels": base_channels,
        "spatial_stride": int(args.spatial_stride),
        "max_samples_per_split": int(args.max_samples_per_split),
        "max_pos_voxels": int(args.max_pos_voxels),
        "max_neg_voxels": int(args.max_neg_voxels),
        "outputs": {
            "sample_summary_csv": str(output_dir / "residual_probability_sample_summary.csv"),
            "overall_csv": str(output_dir / "residual_probability_overall.csv"),
            "by_split_csv": str(output_dir / "residual_probability_by_split.csv"),
            "by_net_direction_csv": str(output_dir / "residual_probability_by_net_direction.csv"),
        },
    }
    with (output_dir / "residual_probability_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
