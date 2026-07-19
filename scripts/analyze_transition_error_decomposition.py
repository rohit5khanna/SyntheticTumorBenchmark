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
from baselines.tasks import ForecastSample, patient_paths
from baselines.unet import _TorchForecastDataset, _build_torch_model


KEY_COLS = ["split", "patient_id", "input_idx", "target_idx", "horizon"]


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


def read_method(path: Path, name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[KEY_COLS + ["dice"]].rename(columns={"dice": f"dice_{name}"})


def safe_div(num: float, denom: float) -> float:
    return float(num / denom) if denom else np.nan


def extract_error_rows(
    dataset_root: Path,
    manifest: pd.DataFrame,
    checkpoint: Path,
    input_mode: str,
    model_variant: str,
    base_channels: int,
    eval_splits: List[str],
    batch_size: int,
    device: str,
    mask_threshold: float,
) -> pd.DataFrame:
    import torch
    from torch.utils.data import DataLoader

    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    ckpt = torch.load(checkpoint, map_location=dev, weights_only=False)
    in_channels = int(ckpt.get("in_channels", 4 if input_mode == "mask" else 5))
    base = int(ckpt.get("base_channels", base_channels))
    model = _build_torch_model(in_channels=in_channels, base_channels=base, model_variant=model_variant)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(dev)
    model.eval()

    rows = []
    label_cache: Dict[str, np.ndarray] = {}
    for split in eval_splits:
        samples = build_samples_from_manifest(manifest, split)
        ds = _TorchForecastDataset(dataset_root, samples, input_mode=input_mode)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(dev.type == "cuda"))
        with torch.no_grad():
            for x, _, idx in loader:
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
                    pred_mask = probs[j, 0] >= mask_threshold

                    true_growth = target_mask & ~input_mask
                    pred_growth = pred_mask & ~input_mask
                    true_loss = input_mask & ~target_mask
                    pred_loss = input_mask & ~pred_mask
                    stable_core = input_mask & target_mask

                    growth_tp = int((pred_growth & true_growth).sum())
                    growth_fp = int((pred_growth & ~true_growth).sum())
                    growth_fn = int((true_growth & ~pred_growth).sum())
                    loss_tp = int((pred_loss & true_loss).sum())
                    loss_fp = int((pred_loss & ~true_loss).sum())
                    loss_fn = int((true_loss & ~pred_loss).sum())
                    stable_core_missed = int((stable_core & ~pred_mask).sum())
                    stable_core_retained = int((stable_core & pred_mask).sum())

                    input_volume = int(input_mask.sum())
                    target_volume = int(target_mask.sum())
                    pred_volume = int(pred_mask.sum())
                    true_growth_volume = int(true_growth.sum())
                    pred_growth_volume = int(pred_growth.sum())
                    true_loss_volume = int(true_loss.sum())
                    pred_loss_volume = int(pred_loss.sum())

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
                            "pred_volume_vox": pred_volume,
                            "net_direction": "net_growth"
                            if target_volume > input_volume
                            else "net_shrinkage"
                            if target_volume < input_volume
                            else "net_stable",
                            "true_net_delta_vox": int(target_volume - input_volume),
                            "pred_net_delta_vox": int(pred_volume - input_volume),
                            "true_growth_volume_vox": true_growth_volume,
                            "pred_growth_volume_vox": pred_growth_volume,
                            "true_loss_volume_vox": true_loss_volume,
                            "pred_loss_volume_vox": pred_loss_volume,
                            "growth_tp_vox": growth_tp,
                            "growth_fp_vox": growth_fp,
                            "growth_fn_vox": growth_fn,
                            "loss_tp_vox": loss_tp,
                            "loss_fp_vox": loss_fp,
                            "loss_fn_vox": loss_fn,
                            "stable_core_retained_vox": stable_core_retained,
                            "stable_core_missed_vox": stable_core_missed,
                            "growth_precision": safe_div(growth_tp, pred_growth_volume),
                            "growth_recall": safe_div(growth_tp, true_growth_volume),
                            "loss_precision": safe_div(loss_tp, pred_loss_volume),
                            "loss_recall": safe_div(loss_tp, true_loss_volume),
                            "stable_core_recall": safe_div(stable_core_retained, int(stable_core.sum())),
                            "growth_fp_per_input": safe_div(growth_fp, input_volume),
                            "growth_fn_per_input": safe_div(growth_fn, input_volume),
                            "loss_fp_per_input": safe_div(loss_fp, input_volume),
                            "loss_fn_per_input": safe_div(loss_fn, input_volume),
                            "dice_model_recomputed": dice_np(pred_mask.astype(np.float32), target_mask.astype(np.float32)),
                            "dice_locf_recomputed": dice_np(input_mask.astype(np.float32), target_mask.astype(np.float32)),
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
            locf_mean=("dice_locf", "mean"),
            model_mean=("dice_model", "mean"),
            mean_gap=("dice_gap", "mean"),
            true_growth_volume_mean=("true_growth_volume_vox", "mean"),
            pred_growth_volume_mean=("pred_growth_volume_vox", "mean"),
            growth_precision_mean=("growth_precision", "mean"),
            growth_recall_mean=("growth_recall", "mean"),
            growth_fp_per_input_mean=("growth_fp_per_input", "mean"),
            growth_fn_per_input_mean=("growth_fn_per_input", "mean"),
            true_loss_volume_mean=("true_loss_volume_vox", "mean"),
            pred_loss_volume_mean=("pred_loss_volume_vox", "mean"),
            loss_precision_mean=("loss_precision", "mean"),
            loss_recall_mean=("loss_recall", "mean"),
            loss_fp_per_input_mean=("loss_fp_per_input", "mean"),
            loss_fn_per_input_mean=("loss_fn_per_input", "mean"),
            stable_core_recall_mean=("stable_core_recall", "mean"),
        )
        .reset_index()
    )
    if "_overall" in out.columns:
        out = out.drop(columns=["_overall"])
    return out


def write_report(path: Path, overall: pd.DataFrame, by_direction: pd.DataFrame) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Transition Error Decomposition\n\n")
        f.write(
            "This diagnostic decomposes direct model errors into growth and shrinkage components "
            "relative to the input mask. Growth is target/prediction outside the input mask; loss is "
            "input-mask voxels absent from the target/prediction.\n\n"
        )
        f.write("## Overall\n\n")
        f.write(overall.to_markdown(index=False))
        f.write("\n\n## By Net Direction\n\n")
        f.write(by_direction.to_markdown(index=False))
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Decompose direct forecast errors into growth/loss components.")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--manifest_csv", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--locf_csv", type=str, required=True)
    parser.add_argument("--model_csv", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="model")
    parser.add_argument("--model_variant", type=str, default="resunet")
    parser.add_argument("--input_mode", type=str, default="image_mask")
    parser.add_argument("--base_channels", type=int, default=12)
    parser.add_argument("--eval_splits", type=str, default="val,test")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_splits = [s.strip() for s in args.eval_splits.split(",") if s.strip()]
    manifest = normalize_manifest(pd.read_csv(args.manifest_csv))

    errors = extract_error_rows(
        dataset_root=Path(args.dataset_root),
        manifest=manifest,
        checkpoint=Path(args.checkpoint),
        input_mode=args.input_mode,
        model_variant=args.model_variant,
        base_channels=args.base_channels,
        eval_splits=eval_splits,
        batch_size=args.batch_size,
        device=args.device,
        mask_threshold=args.mask_threshold,
    )
    paired = (
        errors.merge(read_method(Path(args.locf_csv), "locf"), on=KEY_COLS, how="inner")
        .merge(read_method(Path(args.model_csv), "model"), on=KEY_COLS, how="inner")
    )
    paired["dice_gap"] = paired["dice_model"] - paired["dice_locf"]
    paired["dice_model_recompute_delta"] = paired["dice_model_recomputed"] - paired["dice_model"]
    paired["dice_locf_recompute_delta"] = paired["dice_locf_recomputed"] - paired["dice_locf"]

    overall = summarize(paired, ["split"])
    by_direction = summarize(paired, ["split", "net_direction"])
    by_patient = summarize(paired, ["split", "patient_id"])

    paired.to_csv(out_dir / "transition_error_decomposition_samples.csv", index=False)
    overall.to_csv(out_dir / "transition_error_decomposition_summary.csv", index=False)
    by_direction.to_csv(out_dir / "transition_error_decomposition_by_net_direction.csv", index=False)
    by_patient.to_csv(out_dir / "transition_error_decomposition_by_patient.csv", index=False)
    write_report(out_dir / "transition_error_decomposition_report.md", overall, by_direction)
    with (out_dir / "transition_error_decomposition_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_root": args.dataset_root,
                "manifest_csv": args.manifest_csv,
                "checkpoint": args.checkpoint,
                "eval_splits": eval_splits,
                "output_dir": str(out_dir),
            },
            f,
            indent=2,
        )

    print(
        json.dumps(
            {
                "samples_csv": str(out_dir / "transition_error_decomposition_samples.csv"),
                "summary_csv": str(out_dir / "transition_error_decomposition_summary.csv"),
                "by_direction_csv": str(out_dir / "transition_error_decomposition_by_net_direction.csv"),
                "output_dir": str(out_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
