#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
from scripts.evaluate_budgeted_learned_growth_forecast import budget_to_downsampled_k, topk_add
from scripts.run_growth_only_manifest_baseline import GrowthOnlyDataset, build_samples_from_manifest, normalize_manifest
from scripts.analyze_forecast_origin_predictability import parse_csv


def parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def evaluate_mask(input_mask: np.ndarray, target_mask: np.ndarray, growth_prob: np.ndarray, row: pd.Series, gate: float, scale: float, args) -> dict:
    locf_dice = float(dice_np(input_mask.astype(np.float32), target_mask.astype(np.float32)))
    true_growth = target_mask & ~input_mask
    true_loss = input_mask & ~target_mask
    pred_prob = float(row["pred_net_growth_prob"])
    gate_open = pred_prob >= gate
    base_k = budget_to_downsampled_k(
        row,
        input_volume_downsampled=int(input_mask.sum()),
        budget_col="pred_growth_budget_vox",
        projection=args.budget_projection,
        spatial_stride=args.spatial_stride,
    )
    k = int(round(max(0.0, scale) * base_k)) if gate_open else 0
    pred, added = topk_add(input_mask, growth_prob, k)
    pred_growth = pred & ~input_mask
    growth_tp = int((pred_growth & true_growth).sum())
    growth_fp = int((pred_growth & ~true_growth).sum())
    dice = float(dice_np(pred.astype(np.float32), target_mask.astype(np.float32)))
    net_direction = str(row["net_direction"])
    return {
        "gate_threshold": float(gate),
        "budget_scale": float(scale),
        "gate_open": int(gate_open),
        "pred_net_growth_prob": pred_prob,
        "pred_base_budget_vox_downsampled": int(base_k),
        "pred_scaled_budget_vox_downsampled": int(k),
        "added_growth_vox": int(added),
        "dice": dice,
        "locf_dice": locf_dice,
        "gap_vs_locf": dice - locf_dice,
        "net_direction": net_direction,
        "true_growth_volume_vox_downsampled": int(true_growth.sum()),
        "true_loss_volume_vox_downsampled": int(true_loss.sum()),
        "growth_tp_vox": growth_tp,
        "growth_fp_vox": growth_fp,
        "growth_precision": growth_tp / max(1, int(added)) if added else np.nan,
        "growth_recall": growth_tp / max(1, int(true_growth.sum())) if int(true_growth.sum()) else np.nan,
        "edit_on_shrinkage": int(gate_open and net_direction == "net_shrinkage"),
        "edit_on_growth": int(gate_open and net_direction == "net_growth"),
    }


def summarize(samples: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    group_cols_l = [c for c in group_cols if c in samples.columns]
    return (
        samples.groupby(group_cols_l, observed=True, dropna=False)
        .agg(
            n=("patient_id", "size"),
            n_patients=("patient_id", "nunique"),
            mean_dice=("dice", "mean"),
            std_dice=("dice", "std"),
            mean_locf_dice=("locf_dice", "mean"),
            mean_gap_vs_locf=("gap_vs_locf", "mean"),
            median_gap_vs_locf=("gap_vs_locf", "median"),
            win_rate_vs_locf=("gap_vs_locf", lambda x: float((x > 0).mean())),
            gate_open_rate=("gate_open", "mean"),
            edit_on_growth_rate=("edit_on_growth", "mean"),
            edit_on_shrinkage_rate=("edit_on_shrinkage", "mean"),
            added_growth_mean=("added_growth_vox", "mean"),
            scaled_budget_mean=("pred_scaled_budget_vox_downsampled", "mean"),
            true_growth_mean=("true_growth_volume_vox_downsampled", "mean"),
            growth_precision_mean=("growth_precision", "mean"),
            growth_recall_mean=("growth_recall", "mean"),
        )
        .reset_index()
        .sort_values(group_cols_l)
    )


def write_report(path: Path, validation_sweep: pd.DataFrame, selected_by_split: pd.DataFrame, selected_by_direction: pd.DataFrame) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Conservative Budgeted Growth Policy\n\n")
        f.write(
            "This audit tests whether the learned growth-field failure is primarily caused by over-editing and shrinkage leakage. "
            "It gates growth edits using forecast-origin net-growth probability and scales the predicted growth budget before top-k addition. "
            "The policy is selected on validation and then reported on validation/test.\n\n"
        )
        f.write("## Validation Sweep\n\n")
        f.write(validation_sweep.to_markdown(index=False))
        f.write("\n\n## Selected Policy By Split\n\n")
        f.write(selected_by_split.to_markdown(index=False))
        f.write("\n\n## Selected Policy By Direction\n\n")
        f.write(selected_by_direction.to_markdown(index=False))
        f.write(
            "\n\nReading guide: a useful conservative policy should reduce edit-on-shrinkage rate while retaining positive or neutral net-growth gains. "
            "If the selected policy collapses to zero edits, this confirms LOCF is safer than the current correction field under this split.\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep conservative gate/scale policies for budgeted learned growth-field correction.")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--manifest_csv", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--validation_split", type=str, default="val")
    parser.add_argument("--eval_splits", type=str, default="val,test")
    parser.add_argument("--feature_set", type=str, default="history_only")
    parser.add_argument("--budget_model", type=str, default="ridge_log")
    parser.add_argument("--budget_projection", type=str, default="input_fraction", choices=["input_fraction", "stride_volume"])
    parser.add_argument("--gate_thresholds", type=str, default="0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--budget_scales", type=str, default="0,0.1,0.25,0.5,0.75,1.0")
    parser.add_argument("--selection_objective", type=str, default="mean_dice", choices=["mean_dice", "mean_gap_vs_locf", "net_growth_gap", "shrinkage_safe_gap"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    try:
        import torch
        from torch.utils.data import DataLoader
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("PyTorch is required for conservative growth-policy evaluation.") from exc

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = normalize_manifest(pd.read_csv(args.manifest_csv))
    eval_splits = parse_csv(args.eval_splits)
    gate_thresholds = parse_float_list(args.gate_thresholds)
    budget_scales = parse_float_list(args.budget_scales)
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
    in_channels = int(ckpt["in_channels"])
    base_channels = int(ckpt.get("base_channels", 6))
    model_variant = str(ckpt.get("model_variant", "resunet"))
    input_mode = str(ckpt.get("input_mode", "image_mask"))
    spatial_stride = int(ckpt.get("spatial_stride", 1))
    args.spatial_stride = spatial_stride

    model = _build_torch_model(in_channels=in_channels, base_channels=base_channels, model_variant=model_variant, out_channels=1)
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
                    prob_j = probs[j, 0]
                    base = row.to_dict()
                    for gate in gate_thresholds:
                        for scale in budget_scales:
                            rec = evaluate_mask(input_j, target_j, prob_j, row, gate, scale, args)
                            rows.append({**base, **rec})

    all_samples = pd.DataFrame(rows)
    sweep = summarize(all_samples, ["split", "gate_threshold", "budget_scale"])
    val_sweep = sweep[sweep["split"].astype(str) == args.validation_split].copy()
    if val_sweep.empty:
        raise ValueError(f"No validation rows for split={args.validation_split}")
    val_direction = summarize(all_samples[all_samples["split"].astype(str) == args.validation_split], ["gate_threshold", "budget_scale", "net_direction"])
    net_growth_gap = val_direction[val_direction["net_direction"] == "net_growth"][["gate_threshold", "budget_scale", "mean_gap_vs_locf"]].rename(columns={"mean_gap_vs_locf": "net_growth_gap"})
    shrink_gap = val_direction[val_direction["net_direction"] == "net_shrinkage"][["gate_threshold", "budget_scale", "mean_gap_vs_locf"]].rename(columns={"mean_gap_vs_locf": "net_shrinkage_gap"})
    val_sweep = val_sweep.merge(net_growth_gap, on=["gate_threshold", "budget_scale"], how="left")
    val_sweep = val_sweep.merge(shrink_gap, on=["gate_threshold", "budget_scale"], how="left")
    val_sweep["shrinkage_safe_gap"] = val_sweep["mean_gap_vs_locf"] - val_sweep["edit_on_shrinkage_rate"].fillna(0.0) * 0.01

    selected_row = val_sweep.sort_values(args.selection_objective, ascending=False).iloc[0]
    selected_gate = float(selected_row["gate_threshold"])
    selected_scale = float(selected_row["budget_scale"])
    selected_samples = all_samples[
        (all_samples["gate_threshold"].astype(float) == selected_gate)
        & (all_samples["budget_scale"].astype(float) == selected_scale)
    ].copy()
    selected_by_split = summarize(selected_samples, ["split"])
    selected_by_direction = summarize(selected_samples, ["split", "net_direction"])

    all_samples.to_csv(output_dir / "conservative_growth_policy_all_samples.csv", index=False)
    sweep.to_csv(output_dir / "conservative_growth_policy_sweep_by_split.csv", index=False)
    val_sweep.to_csv(output_dir / "conservative_growth_policy_validation_sweep.csv", index=False)
    selected_samples.to_csv(output_dir / "conservative_growth_policy_selected_samples.csv", index=False)
    selected_by_split.to_csv(output_dir / "conservative_growth_policy_selected_by_split.csv", index=False)
    selected_by_direction.to_csv(output_dir / "conservative_growth_policy_selected_by_direction.csv", index=False)
    write_report(output_dir / "conservative_growth_policy_report.md", val_sweep, selected_by_split, selected_by_direction)

    summary = {
        "dataset_root": args.dataset_root,
        "manifest_csv": args.manifest_csv,
        "checkpoint": args.checkpoint,
        "train_split": args.train_split,
        "validation_split": args.validation_split,
        "eval_splits": eval_splits,
        "feature_set": args.feature_set,
        "budget_model": args.budget_model,
        "budget_projection": args.budget_projection,
        "gate_thresholds": gate_thresholds,
        "budget_scales": budget_scales,
        "selection_objective": args.selection_objective,
        "selected_gate_threshold": selected_gate,
        "selected_budget_scale": selected_scale,
        "selected_validation_row": selected_row.to_dict(),
        "model_variant": model_variant,
        "input_mode": input_mode,
        "in_channels": in_channels,
        "base_channels": base_channels,
        "spatial_stride": spatial_stride,
        "n_output_rows": int(all_samples.shape[0]),
        "output_dir": str(output_dir),
        "outputs": {
            "all_samples_csv": str(output_dir / "conservative_growth_policy_all_samples.csv"),
            "sweep_by_split_csv": str(output_dir / "conservative_growth_policy_sweep_by_split.csv"),
            "validation_sweep_csv": str(output_dir / "conservative_growth_policy_validation_sweep.csv"),
            "selected_samples_csv": str(output_dir / "conservative_growth_policy_selected_samples.csv"),
            "selected_by_split_csv": str(output_dir / "conservative_growth_policy_selected_by_split.csv"),
            "selected_by_direction_csv": str(output_dir / "conservative_growth_policy_selected_by_direction.csv"),
            "report_md": str(output_dir / "conservative_growth_policy_report.md"),
        },
    }
    with (output_dir / "conservative_growth_policy_run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
