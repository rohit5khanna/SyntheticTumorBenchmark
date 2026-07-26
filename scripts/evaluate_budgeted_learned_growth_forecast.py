#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.metrics import dice_np
from baselines.unet import _build_torch_model
from scripts.evaluate_budgeted_distance_forecast import add_budget_predictions
from scripts.run_growth_only_manifest_baseline import GrowthOnlyDataset, build_samples_from_manifest, normalize_manifest
from scripts.analyze_forecast_origin_predictability import parse_csv


def topk_add(input_mask: np.ndarray, growth_prob: np.ndarray, k: int) -> tuple[np.ndarray, int]:
    outside = ~input_mask
    idx = np.flatnonzero(outside.reshape(-1))
    if len(idx) == 0 or k <= 0:
        return input_mask.copy(), 0
    k = min(int(k), len(idx))
    scores = growth_prob.reshape(-1)[idx]
    if k >= len(idx):
        chosen = idx
    else:
        chosen_local = np.argpartition(scores, -k)[-k:]
        chosen = idx[chosen_local]
    pred = input_mask.reshape(-1).copy()
    pred[chosen] = True
    return pred.reshape(input_mask.shape), int(len(chosen))


def budget_to_downsampled_k(row: pd.Series, input_volume_downsampled: int, budget_col: str, projection: str, spatial_stride: int) -> int:
    budget = max(0.0, float(row[budget_col]))
    if projection == "stride_volume":
        return int(round(budget / max(1, int(spatial_stride) ** 3)))
    if projection == "input_fraction":
        input_volume_original = max(1.0, float(row.get("input_volume_vox", input_volume_downsampled)))
        frac = budget / input_volume_original
        return int(round(frac * max(1, input_volume_downsampled)))
    raise ValueError(f"Unknown budget_projection={projection}")


def evaluate_one(input_mask: np.ndarray, target_mask: np.ndarray, growth_prob: np.ndarray, row: pd.Series, args) -> List[dict]:
    locf_dice = float(dice_np(input_mask.astype(np.float32), target_mask.astype(np.float32)))
    true_growth = target_mask & ~input_mask
    pred_growth_k = budget_to_downsampled_k(
        row,
        input_volume_downsampled=int(input_mask.sum()),
        budget_col="pred_growth_budget_vox",
        projection=args.budget_projection,
        spatial_stride=args.spatial_stride,
    )
    true_growth_k = int(true_growth.sum())
    pred_is_growth = bool(float(row["pred_net_growth_prob"]) >= float(args.direction_threshold))
    true_is_growth = str(row["net_direction"]) == "net_growth"

    policies = {
        "locf": input_mask,
        "pred_growth_budget_learned_field": topk_add(input_mask, growth_prob, pred_growth_k)[0],
        "pred_direction_growth_only_learned_field": topk_add(input_mask, growth_prob, pred_growth_k)[0]
        if pred_is_growth
        else input_mask,
        "true_growth_budget_learned_field": topk_add(input_mask, growth_prob, true_growth_k)[0],
        "true_direction_growth_only_learned_field": topk_add(input_mask, growth_prob, true_growth_k)[0]
        if true_is_growth
        else input_mask,
    }

    rows = []
    for policy, pred in policies.items():
        pred_growth = pred & ~input_mask
        growth_tp = int((pred_growth & true_growth).sum())
        growth_fp = int((pred_growth & ~true_growth).sum())
        added = int(pred_growth.sum())
        d = float(dice_np(pred.astype(np.float32), target_mask.astype(np.float32)))
        rows.append(
            {
                "policy": policy,
                "dice": d,
                "locf_dice": locf_dice,
                "gap_vs_locf": d - locf_dice,
                "predicted_net_growth": int(pred_is_growth),
                "true_net_growth": int(true_is_growth),
                "pred_growth_budget_vox_original": float(row["pred_growth_budget_vox"]),
                "pred_growth_budget_vox_downsampled": int(pred_growth_k),
                "true_growth_volume_vox_downsampled": int(true_growth_k),
                "added_growth_vox": added,
                "growth_tp_vox": growth_tp,
                "growth_fp_vox": growth_fp,
                "growth_precision": growth_tp / added if added else np.nan,
                "growth_recall": growth_tp / true_growth_k if true_growth_k else np.nan,
                "growth_prob_true_growth_mean": float(growth_prob[true_growth].mean()) if true_growth.any() else np.nan,
                "growth_prob_outside_non_growth_mean": float(growth_prob[(~input_mask) & (~true_growth)].mean())
                if ((~input_mask) & (~true_growth)).any()
                else np.nan,
            }
        )
    return rows


def summarize(samples: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    group_cols_l = [c for c in group_cols if c in samples.columns]
    return (
        samples.groupby(group_cols_l, observed=True, dropna=False)
        .agg(
            n=("dice", "size"),
            n_patients=("patient_id", "nunique"),
            mean_dice=("dice", "mean"),
            std_dice=("dice", "std"),
            mean_locf_dice=("locf_dice", "mean"),
            mean_gap_vs_locf=("gap_vs_locf", "mean"),
            median_gap_vs_locf=("gap_vs_locf", "median"),
            win_rate_vs_locf=("gap_vs_locf", lambda x: float((x > 0).mean())),
            added_growth_mean=("added_growth_vox", "mean"),
            pred_budget_downsampled_mean=("pred_growth_budget_vox_downsampled", "mean"),
            true_growth_downsampled_mean=("true_growth_volume_vox_downsampled", "mean"),
            growth_precision_mean=("growth_precision", "mean"),
            growth_recall_mean=("growth_recall", "mean"),
            growth_prob_true_growth_mean=("growth_prob_true_growth_mean", "mean"),
            growth_prob_outside_non_growth_mean=("growth_prob_outside_non_growth_mean", "mean"),
        )
        .reset_index()
        .sort_values(group_cols_l)
    )


def write_report(path: Path, overall: pd.DataFrame, by_direction: pd.DataFrame) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Budgeted Learned Growth-Field Forecast Evaluation\n\n")
        f.write(
            "This evaluation replaces the naive distance-to-mask spatial score with a trained growth-only model's "
            "voxelwise growth probability. Forecast-origin features predict the growth budget, and the learned "
            "probability field decides where that budget is added outside the current mask.\n\n"
        )
        f.write("## Overall\n\n")
        f.write(overall.to_markdown(index=False))
        f.write("\n\n## By Net Direction\n\n")
        f.write(by_direction.to_markdown(index=False))
        f.write(
            "\n\nInterpretation rule: compare predicted-budget learned-field policies to LOCF and to "
            "true-budget learned-field policies. If true-budget learned-field helps but predicted-budget does not, "
            "budget calibration remains the bottleneck. If both fail, the learned spatial field is not yet useful. "
            "If predicted-budget policies improve mainly in net-growth cases but hurt shrinkage cases, direction gating "
            "or loss-specific modeling is required.\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate forecast-origin budgets with a learned growth-probability spatial field.")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--manifest_csv", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--eval_splits", type=str, default="val,test")
    parser.add_argument("--feature_set", type=str, default="history_only")
    parser.add_argument("--budget_model", type=str, default="ridge_log")
    parser.add_argument("--model_variant", type=str, default="resunet", choices=["unet", "resunet", "plain_cnn"])
    parser.add_argument("--input_mode", type=str, default="image_mask", choices=["mask", "image_mask"])
    parser.add_argument("--base_channels", type=int, default=6)
    parser.add_argument("--spatial_stride", type=int, default=2)
    parser.add_argument("--budget_projection", type=str, default="input_fraction", choices=["input_fraction", "stride_volume"])
    parser.add_argument("--direction_threshold", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    try:
        import torch
        from torch.utils.data import DataLoader
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("PyTorch is required for learned growth-field evaluation.") from exc

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = normalize_manifest(pd.read_csv(args.manifest_csv))
    eval_splits = parse_csv(args.eval_splits)
    predictions = add_budget_predictions(
        manifest,
        train_split=args.train_split,
        eval_splits=eval_splits,
        feature_set=args.feature_set,
        budget_model=args.budget_model,
        seed=args.seed,
    )

    if args.device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=dev, weights_only=False)
    in_channels = int(ckpt.get("in_channels"))
    base_channels = int(ckpt.get("base_channels", args.base_channels))
    model_variant = str(ckpt.get("model_variant", args.model_variant))
    input_mode = str(ckpt.get("input_mode", args.input_mode))
    spatial_stride = int(ckpt.get("spatial_stride", args.spatial_stride))
    args.spatial_stride = spatial_stride

    model = _build_torch_model(
        in_channels=in_channels,
        base_channels=base_channels,
        model_variant=model_variant,
        out_channels=1,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(dev)
    model.eval()

    rows = []
    with torch.no_grad():
        for split in eval_splits:
            part_pred = predictions[predictions["split"].astype(str) == split].reset_index(drop=True)
            samples = build_samples_from_manifest(part_pred, split)
            ds = GrowthOnlyDataset(args.dataset_root, samples, input_mode, spatial_stride=spatial_stride, cache_arrays=True)
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=(dev.type == "cuda"))
            for x, _, target, input_mask, _, idx in loader:
                x = x.to(dev, non_blocking=True)
                probs = torch.sigmoid(model(x)).detach().cpu().numpy()
                target_np = target.numpy()
                input_np = input_mask.numpy()
                for j in range(probs.shape[0]):
                    row = part_pred.iloc[int(idx[j])]
                    input_j = input_np[j, 0] > 0
                    target_j = target_np[j, 0] > 0
                    growth_prob = probs[j, 0]
                    base = row.to_dict()
                    for result in evaluate_one(input_j, target_j, growth_prob, row, args):
                        rows.append({**base, **result})

    samples_df = pd.DataFrame(rows)
    overall = summarize(samples_df, ["split", "policy"])
    by_direction = summarize(samples_df, ["split", "net_direction", "policy"])

    samples_df.to_csv(output_dir / "budgeted_learned_growth_forecast_samples.csv", index=False)
    overall.to_csv(output_dir / "budgeted_learned_growth_forecast_summary_by_split.csv", index=False)
    by_direction.to_csv(output_dir / "budgeted_learned_growth_forecast_summary_by_direction.csv", index=False)
    write_report(output_dir / "budgeted_learned_growth_forecast_report.md", overall, by_direction)

    run_summary = {
        "dataset_root": args.dataset_root,
        "manifest_csv": args.manifest_csv,
        "checkpoint": args.checkpoint,
        "train_split": args.train_split,
        "eval_splits": eval_splits,
        "feature_set": args.feature_set,
        "budget_model": args.budget_model,
        "budget_projection": args.budget_projection,
        "direction_threshold": float(args.direction_threshold),
        "seed": int(args.seed),
        "model_variant": model_variant,
        "input_mode": input_mode,
        "in_channels": in_channels,
        "base_channels": base_channels,
        "spatial_stride": spatial_stride,
        "n_eval_windows": int(predictions.shape[0]),
        "n_output_rows": int(samples_df.shape[0]),
        "output_dir": str(output_dir),
        "outputs": {
            "samples_csv": str(output_dir / "budgeted_learned_growth_forecast_samples.csv"),
            "summary_by_split_csv": str(output_dir / "budgeted_learned_growth_forecast_summary_by_split.csv"),
            "summary_by_direction_csv": str(output_dir / "budgeted_learned_growth_forecast_summary_by_direction.csv"),
            "report_md": str(output_dir / "budgeted_learned_growth_forecast_report.md"),
        },
    }
    with (output_dir / "budgeted_learned_growth_forecast_run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)
    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    main()
