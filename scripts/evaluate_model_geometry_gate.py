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


def build_samples_from_manifest(manifest: pd.DataFrame, split: str) -> List[ForecastSample]:
    rows = manifest[manifest["split"] == split].copy()
    if rows.empty:
        raise ValueError(f"No rows found for split='{split}' in manifest.")
    samples = []
    for _, row in rows.iterrows():
        input_idx = int(row["input_end_idx"] if "input_end_idx" in row else row["input_idx"])
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


def normalize_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    out = manifest.copy()
    if "input_idx" not in out.columns and "input_end_idx" in out.columns:
        out["input_idx"] = out["input_end_idx"]
    out["future_is_net_growth"] = (out["net_direction"] == "net_growth").astype(int)
    return out


def read_method_csv(path: Path, method_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[KEY_COLS + ["dice"]].rename(columns={"dice": f"dice_{method_name}"})


def threshold_grid(values: pd.Series) -> List[float]:
    vals = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size == 0:
        return [0.0]
    qs = np.quantile(vals, np.linspace(0.05, 0.95, 19)).tolist()
    fixed = [-0.50, -0.25, -0.10, -0.05, 0.0, 0.05, 0.10, 0.25, 0.50]
    return sorted(set(float(x) for x in qs + fixed))


def summarize_policy(df: pd.DataFrame, score_col: str, threshold: float, model_name: str) -> dict:
    pred_use_model = df[score_col] >= threshold
    policy_dice = np.where(pred_use_model, df[f"dice_{model_name}"], df["dice_locf"])
    true_growth = df["future_is_net_growth"].to_numpy(dtype=bool)
    true_shrink = ~true_growth
    return {
        "score_col": score_col,
        "threshold": float(threshold),
        "n": int(len(df)),
        "n_patients": int(df["patient_id"].nunique()),
        "locf_mean": float(df["dice_locf"].mean()),
        "model_mean": float(df[f"dice_{model_name}"].mean()),
        "policy_mean": float(np.mean(policy_dice)),
        "gap_vs_locf": float(np.mean(policy_dice - df["dice_locf"].to_numpy())),
        "gap_vs_model": float(np.mean(policy_dice - df[f"dice_{model_name}"].to_numpy())),
        "predicted_model_rate": float(pred_use_model.mean()),
        "growth_recall": float((pred_use_model.to_numpy() & true_growth).sum() / max(1, true_growth.sum())),
        "shrinkage_false_model_rate": float((pred_use_model.to_numpy() & true_shrink).sum() / max(1, true_shrink.sum())),
    }


def apply_policy(df: pd.DataFrame, score_col: str, threshold: float, model_name: str) -> pd.DataFrame:
    out = df.copy()
    out["geometry_score_col"] = score_col
    out["geometry_threshold"] = float(threshold)
    out["geometry_use_model"] = out[score_col] >= threshold
    out["geometry_policy_dice"] = np.where(out["geometry_use_model"], out[f"dice_{model_name}"], out["dice_locf"])
    out["oracle_direction_dice"] = np.where(out["future_is_net_growth"] == 1, out[f"dice_{model_name}"], out["dice_locf"])
    out["gap_geometry_vs_locf"] = out["geometry_policy_dice"] - out["dice_locf"]
    out["gap_geometry_vs_model"] = out["geometry_policy_dice"] - out[f"dice_{model_name}"]
    return out


def bootstrap_patient_ci(df: pd.DataFrame, n_bootstrap: int, seed: int) -> pd.DataFrame:
    if n_bootstrap <= 0:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    rows = []
    for split, part in df.groupby("split", observed=True):
        patients = np.asarray(sorted(part["patient_id"].unique()))
        boot_gaps = []
        for _ in range(n_bootstrap):
            sampled = rng.choice(patients, size=len(patients), replace=True)
            boot = pd.concat([part[part["patient_id"] == pid] for pid in sampled], ignore_index=True)
            boot_gaps.append(float(boot["gap_geometry_vs_locf"].mean()))
        lo, hi = np.percentile(boot_gaps, [2.5, 97.5])
        rows.append(
            {
                "split": split,
                "metric": "gap_geometry_vs_locf",
                "mean": float(part["gap_geometry_vs_locf"].mean()),
                "ci_low": float(lo),
                "ci_high": float(hi),
                "n_patients": int(len(patients)),
                "n_bootstrap": int(n_bootstrap),
            }
        )
    return pd.DataFrame(rows)


def extract_geometry(
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
            for x, y, idx in loader:
                x = x.to(dev, non_blocking=True)
                logits = model(x)
                probs = torch.sigmoid(logits).detach().cpu().numpy()
                for j in range(probs.shape[0]):
                    sample = samples[int(idx[j])]
                    if sample.patient_id not in label_cache:
                        labels = np.load(patient_paths(dataset_root, sample.patient_id)["label"])
                        label_cache[sample.patient_id] = _TorchForecastDataset._standardize_label_sessions(labels)
                    labels = label_cache[sample.patient_id] > 0
                    input_mask = labels[sample.input_idx, 0]
                    target_mask = labels[sample.target_idx, 0]
                    prob = probs[j, 0]
                    pred_mask = prob >= mask_threshold

                    input_volume = int(input_mask.sum())
                    target_volume = int(target_mask.sum())
                    pred_volume = int(pred_mask.sum())
                    true_growth = target_mask & ~input_mask
                    true_loss = input_mask & ~target_mask
                    pred_growth = pred_mask & ~input_mask
                    pred_loss = input_mask & ~pred_mask
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
                            "true_net_delta_vox": int(target_volume - input_volume),
                            "pred_net_delta_vox": int(pred_volume - input_volume),
                            "true_relative_net_delta": float((target_volume - input_volume) / max(1, input_volume)),
                            "pred_relative_net_delta": float((pred_volume - input_volume) / max(1, input_volume)),
                            "true_growth_volume_vox": int(true_growth.sum()),
                            "true_loss_volume_vox": int(true_loss.sum()),
                            "pred_growth_volume_vox": int(pred_growth.sum()),
                            "pred_loss_volume_vox": int(pred_loss.sum()),
                            "pred_relative_growth": float(pred_growth.sum() / max(1, input_volume)),
                            "pred_relative_loss": float(pred_loss.sum() / max(1, input_volume)),
                            "prob_mass": float(prob.sum()),
                            "prob_mass_delta": float(prob.sum() - input_volume),
                            "prob_relative_delta": float((prob.sum() - input_volume) / max(1, input_volume)),
                            "model_pred_dice_recomputed": dice_np(pred_mask.astype(np.float32), target_mask.astype(np.float32)),
                        }
                    )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model-derived geometry as a LOCF/model gate.")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--manifest_csv", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--locf_csv", type=str, required=True)
    parser.add_argument("--model_csv", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="resunet")
    parser.add_argument("--model_variant", type=str, default="resunet")
    parser.add_argument("--input_mode", type=str, default="image_mask")
    parser.add_argument("--base_channels", type=int, default=12)
    parser.add_argument("--eval_splits", type=str, default="val,test")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    parser.add_argument("--n_bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(args.dataset_root)
    manifest = normalize_manifest(pd.read_csv(args.manifest_csv))
    eval_splits = [s.strip() for s in args.eval_splits.split(",") if s.strip()]

    geometry = extract_geometry(
        dataset_root=dataset_root,
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
    base_cols = KEY_COLS + ["future_is_net_growth", "net_direction", "absolute_growth_bin", "relative_new_growth"]
    paired = (
        manifest[base_cols]
        .merge(read_method_csv(Path(args.locf_csv), "locf"), on=KEY_COLS, how="inner")
        .merge(read_method_csv(Path(args.model_csv), args.model_name), on=KEY_COLS, how="inner")
        .merge(geometry, on=KEY_COLS, how="inner")
    )

    score_cols = [
        "pred_relative_net_delta",
        "pred_net_delta_vox",
        "pred_relative_growth",
        "prob_relative_delta",
        "prob_mass_delta",
    ]
    val = paired[paired["split"] == "val"].copy()
    test = paired[paired["split"] == "test"].copy()
    if val.empty or test.empty:
        raise ValueError("Need non-empty val and test splits.")

    sweep_rows = []
    for score_col in score_cols:
        for threshold in threshold_grid(val[score_col]):
            for split, part in paired.groupby("split", observed=True):
                if split not in eval_splits:
                    continue
                row = summarize_policy(part, score_col, threshold, args.model_name)
                row["split"] = split
                sweep_rows.append(row)
    sweep = pd.DataFrame(sweep_rows)
    val_sweep = sweep[sweep["split"] == "val"].sort_values(
        ["policy_mean", "gap_vs_locf", "growth_recall"], ascending=False
    )
    selected = val_sweep.iloc[0]
    selected_score_col = str(selected["score_col"])
    selected_threshold = float(selected["threshold"])
    policy = apply_policy(paired, selected_score_col, selected_threshold, args.model_name)

    summary = (
        policy.groupby("split", observed=True)
        .agg(
            n=("patient_id", "size"),
            n_patients=("patient_id", "nunique"),
            locf_mean=("dice_locf", "mean"),
            model_mean=(f"dice_{args.model_name}", "mean"),
            geometry_policy_mean=("geometry_policy_dice", "mean"),
            oracle_mean=("oracle_direction_dice", "mean"),
            gap_geometry_vs_locf=("gap_geometry_vs_locf", "mean"),
            gap_geometry_vs_model=("gap_geometry_vs_model", "mean"),
            use_model_rate=("geometry_use_model", "mean"),
            true_growth_rate=("future_is_net_growth", "mean"),
        )
        .reset_index()
    )
    by_direction = (
        policy.groupby(["split", "net_direction"], observed=True)
        .agg(
            n=("patient_id", "size"),
            locf_mean=("dice_locf", "mean"),
            model_mean=(f"dice_{args.model_name}", "mean"),
            geometry_policy_mean=("geometry_policy_dice", "mean"),
            gap_geometry_vs_locf=("gap_geometry_vs_locf", "mean"),
            use_model_rate=("geometry_use_model", "mean"),
        )
        .reset_index()
    )
    boot = bootstrap_patient_ci(policy, args.n_bootstrap, args.seed)

    geometry.to_csv(output_dir / "model_prediction_geometry.csv", index=False)
    paired.to_csv(output_dir / "model_geometry_paired_samples.csv", index=False)
    sweep.to_csv(output_dir / "geometry_gate_threshold_sweep.csv", index=False)
    policy.to_csv(output_dir / "geometry_gated_policy_samples.csv", index=False)
    summary.to_csv(output_dir / "geometry_gated_policy_summary.csv", index=False)
    by_direction.to_csv(output_dir / "geometry_gated_policy_by_net_direction.csv", index=False)
    boot.to_csv(output_dir / "geometry_gated_policy_patient_bootstrap.csv", index=False)
    with (output_dir / "geometry_gated_policy_report.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "selected_score_col": selected_score_col,
                "selected_threshold": selected_threshold,
                "mask_threshold": args.mask_threshold,
                "eval_splits": eval_splits,
                "output_dir": str(output_dir),
            },
            f,
            indent=2,
        )

    print(
        json.dumps(
            {
                "selected_score_col": selected_score_col,
                "selected_threshold": selected_threshold,
                "summary_csv": str(output_dir / "geometry_gated_policy_summary.csv"),
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
