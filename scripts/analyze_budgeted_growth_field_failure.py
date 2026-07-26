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

from baselines.metrics import dice_np
from baselines.unet import _build_torch_model
from scripts.evaluate_budgeted_distance_forecast import add_budget_predictions
from scripts.evaluate_budgeted_learned_growth_forecast import budget_to_downsampled_k, topk_add
from scripts.run_growth_only_manifest_baseline import GrowthOnlyDataset, build_samples_from_manifest, normalize_manifest
from scripts.analyze_forecast_origin_predictability import parse_csv


def safe_mean(values: np.ndarray, mask: np.ndarray) -> float:
    if int(mask.sum()) == 0:
        return float("nan")
    return float(values[mask].mean())


def safe_median(values: np.ndarray, mask: np.ndarray) -> float:
    if int(mask.sum()) == 0:
        return float("nan")
    return float(np.median(values[mask]))


def edit_stats(input_mask: np.ndarray, target_mask: np.ndarray, growth_prob: np.ndarray, k: int, policy_name: str) -> dict:
    try:
        from scipy.ndimage import binary_dilation, distance_transform_edt, label
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("scipy is required for spatial failure diagnostics.") from exc

    pred, added = topk_add(input_mask, growth_prob, k)
    true_growth = target_mask & ~input_mask
    pred_growth = pred & ~input_mask
    growth_tp = pred_growth & true_growth
    growth_fp = pred_growth & ~true_growth
    growth_fn = true_growth & ~pred_growth
    locf_dice = float(dice_np(input_mask.astype(np.float32), target_mask.astype(np.float32)))
    dice = float(dice_np(pred.astype(np.float32), target_mask.astype(np.float32)))

    # Distances are evaluated on the same downsampled grid as the model output.
    outside_distance_from_input = distance_transform_edt(~input_mask)
    if true_growth.any():
        distance_to_true_growth = distance_transform_edt(~true_growth)
    else:
        distance_to_true_growth = np.full(input_mask.shape, np.nan, dtype=np.float32)
    if pred_growth.any():
        distance_to_added = distance_transform_edt(~pred_growth)
    else:
        distance_to_added = np.full(input_mask.shape, np.nan, dtype=np.float32)

    boundary_shell = binary_dilation(input_mask, iterations=1) & ~input_mask
    labeled, n_components = label(pred_growth)
    if int(pred_growth.sum()) > 0 and n_components > 0:
        counts = np.bincount(labeled[pred_growth].reshape(-1))
        largest_component_fraction = float(counts[1:].max() / max(1, int(pred_growth.sum())))
    else:
        largest_component_fraction = float("nan")

    input_volume = int(input_mask.sum())
    target_volume = int(target_mask.sum())
    denominator = input_volume + target_volume
    # Adding a true-positive growth voxel increases the Dice numerator by 2 and denominator by 1;
    # adding a false-positive growth voxel only increases denominator by 1.
    dice_gain_numerator_vox = int(2 * growth_tp.sum())
    dice_cost_denominator_vox = int(pred_growth.sum())

    return {
        f"{policy_name}_dice": dice,
        f"{policy_name}_gap_vs_locf": dice - locf_dice,
        f"{policy_name}_added_vox": int(pred_growth.sum()),
        f"{policy_name}_growth_tp_vox": int(growth_tp.sum()),
        f"{policy_name}_growth_fp_vox": int(growth_fp.sum()),
        f"{policy_name}_growth_fn_vox": int(growth_fn.sum()),
        f"{policy_name}_growth_precision": int(growth_tp.sum()) / max(1, int(pred_growth.sum())),
        f"{policy_name}_growth_recall": int(growth_tp.sum()) / max(1, int(true_growth.sum())),
        f"{policy_name}_added_boundary_shell_fraction": int((pred_growth & boundary_shell).sum()) / max(1, int(pred_growth.sum())),
        f"{policy_name}_components": int(n_components),
        f"{policy_name}_largest_component_fraction": largest_component_fraction,
        f"{policy_name}_added_distance_from_input_mean": safe_mean(outside_distance_from_input, pred_growth),
        f"{policy_name}_added_distance_from_input_median": safe_median(outside_distance_from_input, pred_growth),
        f"{policy_name}_true_growth_distance_from_input_mean": safe_mean(outside_distance_from_input, true_growth),
        f"{policy_name}_added_distance_to_true_growth_mean": safe_mean(distance_to_true_growth, pred_growth),
        f"{policy_name}_true_growth_distance_to_added_mean": safe_mean(distance_to_added, true_growth),
        f"{policy_name}_dice_gain_numerator_vox": dice_gain_numerator_vox,
        f"{policy_name}_dice_cost_denominator_vox": dice_cost_denominator_vox,
        f"{policy_name}_dice_gain_cost_ratio": dice_gain_numerator_vox / max(1, dice_cost_denominator_vox),
    }


def classify_budget_ratio(ratio: float) -> str:
    if not np.isfinite(ratio):
        return "undefined"
    if ratio < 0.5:
        return "under_half"
    if ratio < 0.8:
        return "under_moderate"
    if ratio <= 1.25:
        return "near"
    if ratio <= 2.0:
        return "over_moderate"
    return "over_large"


def summarize(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    group_cols_l = [c for c in group_cols if c in df.columns]
    return (
        df.groupby(group_cols_l, observed=True, dropna=False)
        .agg(
            n=("patient_id", "size"),
            n_patients=("patient_id", "nunique"),
            locf_dice_mean=("locf_dice", "mean"),
            pred_gap_mean=("pred_budget_gap_vs_locf", "mean"),
            pred_gap_median=("pred_budget_gap_vs_locf", "median"),
            pred_win_rate=("pred_budget_gap_vs_locf", lambda x: float((x > 0).mean())),
            true_budget_gap_mean=("true_budget_gap_vs_locf", "mean"),
            true_budget_gap_median=("true_budget_gap_vs_locf", "median"),
            true_budget_win_rate=("true_budget_gap_vs_locf", lambda x: float((x > 0).mean())),
            pred_budget_ratio_mean=("pred_to_true_growth_budget_ratio", "mean"),
            direction_accuracy=("direction_correct", "mean"),
            pred_precision_mean=("pred_budget_growth_precision", "mean"),
            pred_recall_mean=("pred_budget_growth_recall", "mean"),
            true_precision_mean=("true_budget_growth_precision", "mean"),
            true_recall_mean=("true_budget_growth_recall", "mean"),
            pred_added_to_true_dist_mean=("pred_budget_added_distance_to_true_growth_mean", "mean"),
            true_added_to_true_dist_mean=("true_budget_added_distance_to_true_growth_mean", "mean"),
            pred_true_growth_to_added_dist_mean=("pred_budget_true_growth_distance_to_added_mean", "mean"),
            pred_components_mean=("pred_budget_components", "mean"),
            pred_largest_component_fraction_mean=("pred_budget_largest_component_fraction", "mean"),
            pred_boundary_shell_fraction_mean=("pred_budget_added_boundary_shell_fraction", "mean"),
            pred_dice_gain_cost_ratio_mean=("pred_budget_dice_gain_cost_ratio", "mean"),
            true_dice_gain_cost_ratio_mean=("true_budget_dice_gain_cost_ratio", "mean"),
            prob_true_growth_mean=("growth_prob_true_growth_mean", "mean"),
            prob_outside_non_growth_mean=("growth_prob_outside_non_growth_mean", "mean"),
        )
        .reset_index()
        .sort_values(group_cols_l)
    )


def build_evidence_map(overall: pd.DataFrame, by_direction: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split in sorted(overall["split"].astype(str).unique()):
        part = overall[overall["split"].astype(str) == split]
        if part.empty:
            continue
        row = part.iloc[0]
        rows.append(
            {
                "split": split,
                "question": "Does predicted budget + learned field beat LOCF?",
                "evidence": f"mean gap {row['pred_gap_mean']:.3f}, win rate {row['pred_win_rate']:.3f}",
                "interpretation": "No" if row["pred_gap_mean"] <= 0 else "Yes/weakly",
            }
        )
        rows.append(
            {
                "split": split,
                "question": "Does true budget + learned field have headroom?",
                "evidence": f"mean gap {row['true_budget_gap_mean']:.3f}, win rate {row['true_budget_win_rate']:.3f}",
                "interpretation": "Limited headroom" if row["true_budget_gap_mean"] > 0 else "No useful headroom",
            }
        )
        rows.append(
            {
                "split": split,
                "question": "Is the budget estimate near the true growth volume?",
                "evidence": f"mean predicted/true growth-budget ratio {row['pred_budget_ratio_mean']:.3f}",
                "interpretation": "Budget is not the only problem" if 0.5 <= row["pred_budget_ratio_mean"] <= 1.25 else "Budget mismatch is substantial",
            }
        )
        rows.append(
            {
                "split": split,
                "question": "Do selected voxels buy enough Dice numerator?",
                "evidence": f"predicted-budget Dice gain/cost ratio {row['pred_dice_gain_cost_ratio_mean']:.3f}",
                "interpretation": "Too many added voxels are low-value for Dice" if row["pred_dice_gain_cost_ratio_mean"] < 1.0 else "Voxel additions are efficient on average",
            }
        )
    for _, row in by_direction.iterrows():
        rows.append(
            {
                "split": str(row["split"]),
                "question": f"What happens in {row['net_direction']} cases?",
                "evidence": f"pred gap {row['pred_gap_mean']:.3f}; true-budget gap {row['true_budget_gap_mean']:.3f}; n={int(row['n'])}",
                "interpretation": "Use direction-aware handling" if str(row["net_direction"]) != "net_growth" else "Growth-only edits need better calibration/localization",
            }
        )
    return pd.DataFrame(rows)


def write_report(path: Path, overall: pd.DataFrame, by_direction: pd.DataFrame, by_budget_bin: pd.DataFrame, evidence: pd.DataFrame) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Budgeted Learned Growth-Field Failure Diagnosis\n\n")
        f.write(
            "This audit diagnoses why a learned growth-probability field can separate true-growth voxels from background "
            "but still hurt Dice after top-k LOCF correction. It decomposes the failure into budget mismatch, direction mismatch, "
            "spatial localization, edit fragmentation, and Dice gain/cost.\n\n"
        )
        f.write("## Overall\n\n")
        f.write(overall.to_markdown(index=False))
        f.write("\n\n## By Net Direction\n\n")
        f.write(by_direction.to_markdown(index=False))
        f.write("\n\n## By Budget Ratio Bin\n\n")
        f.write(by_budget_bin.to_markdown(index=False))
        f.write("\n\n## Evidence Map\n\n")
        f.write(evidence.to_markdown(index=False))
        f.write(
            "\n\nReading guide: a positive true-budget gap with a negative predicted-budget gap means the spatial field has some usable signal, "
            "but the operational policy is not calibrated. A low Dice gain/cost ratio means the selected voxels add more denominator "
            "than useful overlap. Strong shrinkage losses indicate that growth-only corrections should be gated or paired with a separate loss model.\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose failures in budgeted learned growth-field LOCF correction.")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--manifest_csv", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--eval_splits", type=str, default="val,test")
    parser.add_argument("--feature_set", type=str, default="history_only")
    parser.add_argument("--budget_model", type=str, default="ridge_log")
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
        raise RuntimeError("PyTorch is required for learned growth-field failure diagnostics.") from exc

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
                    true_growth = target_j & ~input_j
                    true_loss = input_j & ~target_j
                    locf_dice = float(dice_np(input_j.astype(np.float32), target_j.astype(np.float32)))
                    pred_k = budget_to_downsampled_k(row, int(input_j.sum()), "pred_growth_budget_vox", args.budget_projection, spatial_stride)
                    true_k = int(true_growth.sum())
                    pred_is_growth = bool(float(row["pred_net_growth_prob"]) >= float(args.direction_threshold))
                    true_is_growth = str(row["net_direction"]) == "net_growth"
                    pred_ratio = pred_k / max(1, true_k)
                    rec = {
                        **row.to_dict(),
                        "spatial_stride": int(spatial_stride),
                        "locf_dice": locf_dice,
                        "input_volume_vox_downsampled": int(input_j.sum()),
                        "target_volume_vox_downsampled": int(target_j.sum()),
                        "true_growth_volume_vox_downsampled": true_k,
                        "true_loss_volume_vox_downsampled": int(true_loss.sum()),
                        "pred_growth_budget_vox_downsampled": int(pred_k),
                        "pred_to_true_growth_budget_ratio": float(pred_ratio),
                        "pred_budget_ratio_bin": classify_budget_ratio(pred_ratio),
                        "predicted_net_growth": int(pred_is_growth),
                        "true_net_growth": int(true_is_growth),
                        "direction_correct": int(pred_is_growth == true_is_growth),
                        "growth_prob_true_growth_mean": safe_mean(prob_j, true_growth),
                        "growth_prob_outside_non_growth_mean": safe_mean(prob_j, (~input_j) & (~true_growth)),
                    }
                    rec.update(edit_stats(input_j, target_j, prob_j, pred_k, "pred_budget"))
                    rec.update(edit_stats(input_j, target_j, prob_j, true_k, "true_budget"))
                    rows.append(rec)

    sample_df = pd.DataFrame(rows)
    overall = summarize(sample_df, ["split"])
    by_direction = summarize(sample_df, ["split", "net_direction"])
    by_budget_bin = summarize(sample_df, ["split", "pred_budget_ratio_bin"])
    evidence = build_evidence_map(overall, by_direction)

    sample_df.to_csv(output_dir / "budgeted_growth_field_failure_samples.csv", index=False)
    overall.to_csv(output_dir / "budgeted_growth_field_failure_overall.csv", index=False)
    by_direction.to_csv(output_dir / "budgeted_growth_field_failure_by_direction.csv", index=False)
    by_budget_bin.to_csv(output_dir / "budgeted_growth_field_failure_by_budget_ratio.csv", index=False)
    evidence.to_csv(output_dir / "budgeted_growth_field_failure_evidence_map.csv", index=False)
    write_report(output_dir / "budgeted_growth_field_failure_report.md", overall, by_direction, by_budget_bin, evidence)

    summary = {
        "dataset_root": args.dataset_root,
        "manifest_csv": args.manifest_csv,
        "checkpoint": args.checkpoint,
        "train_split": args.train_split,
        "eval_splits": eval_splits,
        "feature_set": args.feature_set,
        "budget_model": args.budget_model,
        "budget_projection": args.budget_projection,
        "spatial_stride": int(spatial_stride),
        "n_samples": int(sample_df.shape[0]),
        "output_dir": str(output_dir),
        "outputs": {
            "samples_csv": str(output_dir / "budgeted_growth_field_failure_samples.csv"),
            "overall_csv": str(output_dir / "budgeted_growth_field_failure_overall.csv"),
            "by_direction_csv": str(output_dir / "budgeted_growth_field_failure_by_direction.csv"),
            "by_budget_ratio_csv": str(output_dir / "budgeted_growth_field_failure_by_budget_ratio.csv"),
            "evidence_map_csv": str(output_dir / "budgeted_growth_field_failure_evidence_map.csv"),
            "report_md": str(output_dir / "budgeted_growth_field_failure_report.md"),
        },
    }
    with (output_dir / "budgeted_growth_field_failure_run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
