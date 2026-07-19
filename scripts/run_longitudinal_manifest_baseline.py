#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.metrics import dice_np
from baselines.tasks import ForecastSample, patient_paths
from baselines.unet import (
    _TorchForecastDataset,
    _build_torch_model,
    _dice_from_logits,
    _dice_loss_soft,
    _set_seed,
)


def build_samples_from_manifest(manifest: pd.DataFrame, split: str) -> List[ForecastSample]:
    rows = manifest[manifest["split"] == split].copy()
    if rows.empty:
        raise ValueError(f"No rows found for split='{split}' in manifest.")

    samples: List[ForecastSample] = []
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


def sample_key(sample: ForecastSample) -> tuple:
    return (sample.patient_id, int(sample.input_idx), int(sample.target_idx), int(sample.horizon))


def manifest_lookup(manifest: pd.DataFrame) -> Dict[tuple, dict]:
    out: Dict[tuple, dict] = {}
    for _, row in manifest.iterrows():
        input_idx = int(row["input_end_idx"] if "input_end_idx" in row else row["input_idx"])
        horizon = int(row.get("horizon", int(row["target_idx"]) - input_idx))
        out[(str(row["patient_id"]), input_idx, int(row["target_idx"]), horizon)] = row.to_dict()
    return out


def run_locf_manifest(dataset_root: Path, manifest: pd.DataFrame, splits: Iterable[str], output_dir: Path) -> dict:
    lookup = manifest_lookup(manifest)
    rows = []
    arr_cache: Dict[str, np.ndarray] = {}

    for split in splits:
        for sample in build_samples_from_manifest(manifest, split):
            if sample.patient_id not in arr_cache:
                p = patient_paths(dataset_root, sample.patient_id)
                arr_cache[sample.patient_id] = _TorchForecastDataset._standardize_label_sessions(np.load(p["label"]))
            labels = arr_cache[sample.patient_id]
            pred = labels[sample.input_idx]
            target = labels[sample.target_idx]
            d = dice_np(pred, target)
            meta = lookup.get(sample_key(sample), {})
            rows.append(
                {
                    "split": split,
                    "patient_id": sample.patient_id,
                    "input_idx": int(sample.input_idx),
                    "target_idx": int(sample.target_idx),
                    "horizon": int(sample.horizon),
                    "delta_days": float(sample.delta_days),
                    "dice": float(d),
                    "method": "locf",
                    "net_direction": meta.get("net_direction"),
                    "absolute_growth_bin": meta.get("absolute_growth_bin"),
                    "growth_volume_vox": meta.get("growth_volume_vox"),
                    "relative_new_growth": meta.get("relative_new_growth"),
                }
            )

    per_sample = pd.DataFrame(rows)
    summary_by_split = (
        per_sample.groupby("split", observed=True)
        .agg(n_samples=("dice", "size"), mean_dice=("dice", "mean"), std_dice=("dice", "std"))
        .reset_index()
    )
    summary = {
        "baseline": "locf",
        "dataset_root": str(dataset_root.resolve()),
        "n_samples": int(len(per_sample)),
        "by_split": summary_by_split.to_dict(orient="records"),
    }
    per_sample.to_json(output_dir / "locf_per_sample.json", orient="records", indent=2)
    per_sample.to_csv(output_dir / "locf_per_sample.csv", index=False)
    summary_by_split.to_csv(output_dir / "locf_summary_by_split.csv", index=False)
    with (output_dir / "locf_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def evaluate_model(
    model,
    dataset,
    samples: List[ForecastSample],
    loader,
    device,
    split: str,
    method: str,
    lookup: Dict[tuple, dict],
) -> List[dict]:
    import torch

    rows = []
    model.eval()
    with torch.no_grad():
        for x, y, idx in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            batch_dice = _dice_from_logits(logits, y).detach().cpu().numpy().tolist()
            for j, d in enumerate(batch_dice):
                sample = samples[int(idx[j])]
                meta = lookup.get(sample_key(sample), {})
                rows.append(
                    {
                        "split": split,
                        "patient_id": sample.patient_id,
                        "input_idx": int(sample.input_idx),
                        "target_idx": int(sample.target_idx),
                        "horizon": int(sample.horizon),
                        "delta_days": float(sample.delta_days),
                        "dice": float(d),
                        "method": method,
                        "net_direction": meta.get("net_direction"),
                        "absolute_growth_bin": meta.get("absolute_growth_bin"),
                        "growth_volume_vox": meta.get("growth_volume_vox"),
                        "relative_new_growth": meta.get("relative_new_growth"),
                    }
                )
    return rows


def run_model_manifest(
    dataset_root: Path,
    manifest: pd.DataFrame,
    train_split: str,
    val_split: str,
    eval_splits: List[str],
    input_mode: str,
    model_variant: str,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    num_workers: int,
    base_channels: int,
    seed: int,
    device: str,
) -> dict:
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
    except Exception as e:
        raise RuntimeError("PyTorch is required for manifest model baselines.") from e

    if input_mode not in {"mask", "image_mask"}:
        raise ValueError("input_mode must be one of: mask, image_mask.")
    if model_variant not in {"unet", "resunet", "plain_cnn", "unetr"}:
        raise ValueError("model_variant must be one of: unet, resunet, plain_cnn, unetr.")

    _set_seed(seed)
    random.seed(seed)
    lookup = manifest_lookup(manifest)

    train_samples = build_samples_from_manifest(manifest, train_split)
    val_samples = build_samples_from_manifest(manifest, val_split)
    train_ds = _TorchForecastDataset(dataset_root, train_samples, input_mode=input_mode)
    val_ds = _TorchForecastDataset(dataset_root, val_samples, input_mode=input_mode)

    sample_x, _, _ = train_ds[0]
    in_channels = int(sample_x.shape[0])
    model = _build_torch_model(in_channels=in_channels, base_channels=base_channels, model_variant=model_variant)

    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)
    model.to(dev)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(dev.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(dev.type == "cuda"),
    )

    bce = nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    prefix = f"{model_variant}_{input_mode}"
    best_ckpt = output_dir / f"model_best_{prefix}.pt"
    best_val_dice = -1.0
    history: List[dict] = []

    for ep in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_dice_sum = 0.0
        train_count = 0
        for x, y, _ in train_loader:
            x = x.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)
            logits = model(x)
            loss = bce(logits, y) + _dice_loss_soft(logits, y)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            bs = int(x.shape[0])
            train_loss_sum += float(loss.item()) * bs
            train_dice_sum += float(_dice_from_logits(logits, y).mean().item()) * bs
            train_count += bs

        model.eval()
        val_loss_sum = 0.0
        val_dice_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for x, y, _ in val_loader:
                x = x.to(dev, non_blocking=True)
                y = y.to(dev, non_blocking=True)
                logits = model(x)
                loss = bce(logits, y) + _dice_loss_soft(logits, y)
                bs = int(x.shape[0])
                val_loss_sum += float(loss.item()) * bs
                val_dice_sum += float(_dice_from_logits(logits, y).mean().item()) * bs
                val_count += bs

        row = {
            "epoch": ep,
            "train_loss": train_loss_sum / max(1, train_count),
            "train_dice": train_dice_sum / max(1, train_count),
            "val_loss": val_loss_sum / max(1, val_count),
            "val_dice": val_dice_sum / max(1, val_count),
        }
        history.append(row)
        print(
            f"[Epoch {ep:03d}] train_loss={row['train_loss']:.4f} train_dice={row['train_dice']:.4f} "
            f"val_loss={row['val_loss']:.4f} val_dice={row['val_dice']:.4f}"
        )
        if row["val_dice"] > best_val_dice:
            best_val_dice = float(row["val_dice"])
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "in_channels": in_channels,
                    "base_channels": base_channels,
                    "input_mode": input_mode,
                    "model_variant": model_variant,
                    "seed": seed,
                },
                best_ckpt,
            )

    ckpt = torch.load(best_ckpt, map_location=dev, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    eval_rows = []
    for split in eval_splits:
        samples = build_samples_from_manifest(manifest, split)
        ds = _TorchForecastDataset(dataset_root, samples, input_mode=input_mode)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(dev.type == "cuda"))
        eval_rows.extend(evaluate_model(model, ds, samples, loader, dev, split, prefix, lookup))

    per_sample = pd.DataFrame(eval_rows)
    summary_by_split = (
        per_sample.groupby("split", observed=True)
        .agg(n_samples=("dice", "size"), mean_dice=("dice", "mean"), std_dice=("dice", "std"))
        .reset_index()
    )
    summary = {
        "baseline": prefix,
        "dataset_root": str(dataset_root.resolve()),
        "manifest_rows": int(len(manifest)),
        "train_split": train_split,
        "val_split": val_split,
        "eval_splits": eval_splits,
        "input_mode": input_mode,
        "model_variant": model_variant,
        "n_train_samples": len(train_samples),
        "n_val_samples": len(val_samples),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "seed": int(seed),
        "best_val_dice": float(best_val_dice),
        "checkpoint": str(best_ckpt),
        "by_split": summary_by_split.to_dict(orient="records"),
    }
    with (output_dir / f"{prefix}_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    per_sample.to_json(output_dir / f"{prefix}_per_sample.json", orient="records", indent=2)
    per_sample.to_csv(output_dir / f"{prefix}_per_sample.csv", index=False)
    summary_by_split.to_csv(output_dir / f"{prefix}_summary_by_split.csv", index=False)
    with (output_dir / f"{prefix}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (output_dir / f"{prefix}_train_samples.json").open("w", encoding="utf-8") as f:
        json.dump([asdict(s) for s in train_samples], f, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LOCF/CNN baselines on a patient-level longitudinal window manifest.")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--manifest_csv", type=str, required=True)
    parser.add_argument("--model_variant", type=str, choices=["locf", "unet", "resunet", "plain_cnn", "unetr"], default="locf")
    parser.add_argument("--input_mode", type=str, choices=["mask", "image_mask"], default="image_mask")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="val")
    parser.add_argument("--eval_splits", type=str, default="val,test")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--base_channels", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    manifest = pd.read_csv(args.manifest_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_splits = [s.strip() for s in args.eval_splits.split(",") if s.strip()]

    if args.model_variant == "locf":
        summary = run_locf_manifest(dataset_root, manifest, eval_splits, output_dir)
    else:
        summary = run_model_manifest(
            dataset_root=dataset_root,
            manifest=manifest,
            train_split=args.train_split,
            val_split=args.val_split,
            eval_splits=eval_splits,
            input_mode=args.input_mode,
            model_variant=args.model_variant,
            output_dir=output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_workers=args.num_workers,
            base_channels=args.base_channels,
            seed=args.seed,
            device=args.device,
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
