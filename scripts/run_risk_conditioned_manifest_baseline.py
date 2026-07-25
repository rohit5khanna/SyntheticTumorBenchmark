#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.metrics import dice_np
from baselines.tasks import ForecastSample, patient_paths
from baselines.unet import _TorchForecastDataset, _build_torch_model, _dice_from_logits, _dice_loss_soft, _set_seed
from scripts.run_longitudinal_manifest_baseline import build_samples_from_manifest, manifest_lookup, sample_key  # noqa: E402


def parse_csv(payload: str) -> List[str]:
    return [x.strip() for x in payload.split(",") if x.strip()]


def normalize_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    out = manifest.copy()
    if "input_idx" not in out.columns and "input_end_idx" in out.columns:
        out["input_idx"] = out["input_end_idx"]
    for col in ["input_idx", "target_idx", "horizon"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    return out


def build_risk_lookup(manifest: pd.DataFrame, risk_columns: List[str], fill_value: float) -> Dict[tuple, np.ndarray]:
    lookup: Dict[tuple, np.ndarray] = {}
    for _, row in manifest.iterrows():
        input_idx = int(row["input_idx"] if "input_idx" in row else row["input_end_idx"])
        horizon = int(row.get("horizon", int(row["target_idx"]) - input_idx))
        key = (str(row["patient_id"]), input_idx, int(row["target_idx"]), horizon)
        vals = []
        for col in risk_columns:
            val = pd.to_numeric(row.get(col, np.nan), errors="coerce")
            if pd.isna(val):
                val = fill_value
            vals.append(float(np.clip(val, 0.0, 1.0)))
        lookup[key] = np.asarray(vals, dtype=np.float32)
    return lookup


class RiskConditionedForecastDataset:
    def __init__(
        self,
        dataset_root: str | Path,
        samples: List[ForecastSample],
        input_mode: str,
        risk_lookup: Dict[tuple, np.ndarray],
        risk_columns: List[str],
        risk_fill_value: float = 0.5,
        cache_arrays: bool = True,
    ) -> None:
        self.base = _TorchForecastDataset(
            dataset_root=dataset_root,
            samples=samples,
            input_mode=input_mode,
            cache_arrays=cache_arrays,
        )
        self.samples = samples
        self.risk_lookup = risk_lookup
        self.risk_columns = risk_columns
        self.risk_fill_value = float(risk_fill_value)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        import torch

        x, y, _ = self.base[idx]
        sample = self.samples[idx]
        vals = self.risk_lookup.get(sample_key(sample))
        if vals is None:
            vals = np.full((len(self.risk_columns),), self.risk_fill_value, dtype=np.float32)
        h, w, d = x.shape[-3:]
        risk_channels = np.asarray(vals, dtype=np.float32)[:, None, None, None]
        risk_channels = np.broadcast_to(risk_channels, (len(self.risk_columns), h, w, d)).copy()
        x_aug = torch.cat([x, torch.from_numpy(risk_channels)], dim=0)
        return x_aug, y, idx


def evaluate_model(model, samples: List[ForecastSample], loader, device, split: str, method: str, lookup: Dict[tuple, dict]) -> List[dict]:
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


def summarize_by_split(per_sample: pd.DataFrame) -> pd.DataFrame:
    return (
        per_sample.groupby("split", observed=True)
        .agg(n_samples=("dice", "size"), mean_dice=("dice", "mean"), std_dice=("dice", "std"))
        .reset_index()
    )


def run_risk_conditioned_model(
    dataset_root: Path,
    manifest: pd.DataFrame,
    train_split: str,
    val_split: str,
    eval_splits: List[str],
    input_mode: str,
    model_variant: str,
    risk_columns: List[str],
    output_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    num_workers: int,
    base_channels: int,
    seed: int,
    device: str,
    risk_fill_value: float,
) -> dict:
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
    except Exception as e:
        raise RuntimeError("PyTorch is required for risk-conditioned manifest baselines.") from e

    if input_mode not in {"mask", "image_mask"}:
        raise ValueError("input_mode must be one of: mask, image_mask.")
    if model_variant not in {"unet", "resunet", "plain_cnn"}:
        raise ValueError("model_variant must be one of: unet, resunet, plain_cnn.")
    missing = [c for c in risk_columns if c not in manifest.columns]
    if missing:
        raise ValueError(f"Missing risk columns in manifest: {missing}")

    _set_seed(seed)
    random.seed(seed)
    lookup = manifest_lookup(manifest)
    risk_lookup = build_risk_lookup(manifest, risk_columns, fill_value=risk_fill_value)

    train_samples = build_samples_from_manifest(manifest, train_split)
    val_samples = build_samples_from_manifest(manifest, val_split)
    train_ds = RiskConditionedForecastDataset(dataset_root, train_samples, input_mode, risk_lookup, risk_columns, risk_fill_value)
    val_ds = RiskConditionedForecastDataset(dataset_root, val_samples, input_mode, risk_lookup, risk_columns, risk_fill_value)

    sample_x, _, _ = train_ds[0]
    in_channels = int(sample_x.shape[0])
    model = _build_torch_model(in_channels=in_channels, base_channels=base_channels, model_variant=model_variant)
    dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else device if device != "auto" else "cpu")
    model.to(dev)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(dev.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(dev.type == "cuda"))

    bce = nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    prefix = f"risk_conditioned_{model_variant}_{input_mode}"
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
            f"val_loss={row['val_loss']:.4f} val_dice={row['val_dice']:.4f}",
            flush=True,
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
                    "risk_columns": risk_columns,
                    "risk_fill_value": risk_fill_value,
                    "seed": seed,
                },
                best_ckpt,
            )

    ckpt = torch.load(best_ckpt, map_location=dev, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    eval_rows = []
    for split in eval_splits:
        samples = build_samples_from_manifest(manifest, split)
        ds = RiskConditionedForecastDataset(dataset_root, samples, input_mode, risk_lookup, risk_columns, risk_fill_value)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(dev.type == "cuda"))
        eval_rows.extend(evaluate_model(model, samples, loader, dev, split, prefix, lookup))

    per_sample = pd.DataFrame(eval_rows)
    summary_by_split = summarize_by_split(per_sample)
    summary = {
        "baseline": prefix,
        "dataset_root": str(dataset_root.resolve()),
        "manifest_rows": int(len(manifest)),
        "train_split": train_split,
        "val_split": val_split,
        "eval_splits": eval_splits,
        "input_mode": input_mode,
        "model_variant": model_variant,
        "risk_columns": risk_columns,
        "risk_fill_value": float(risk_fill_value),
        "n_train_samples": len(train_samples),
        "n_val_samples": len(val_samples),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "base_channels": int(base_channels),
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
    parser = argparse.ArgumentParser(description="Run a risk-conditioned CNN baseline on a longitudinal manifest.")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--manifest_csv", type=str, required=True)
    parser.add_argument("--risk_columns", type=str, required=True)
    parser.add_argument("--model_variant", type=str, choices=["unet", "resunet", "plain_cnn"], default="resunet")
    parser.add_argument("--input_mode", type=str, choices=["mask", "image_mask"], default="image_mask")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="val")
    parser.add_argument("--eval_splits", type=str, default="val,test")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--base_channels", type=int, default=6)
    parser.add_argument("--risk_fill_value", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    manifest = normalize_manifest(pd.read_csv(args.manifest_csv))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = run_risk_conditioned_model(
        dataset_root=Path(args.dataset_root),
        manifest=manifest,
        train_split=args.train_split,
        val_split=args.val_split,
        eval_splits=parse_csv(args.eval_splits),
        input_mode=args.input_mode,
        model_variant=args.model_variant,
        risk_columns=parse_csv(args.risk_columns),
        output_dir=output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        base_channels=args.base_channels,
        seed=args.seed,
        device=args.device,
        risk_fill_value=args.risk_fill_value,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
