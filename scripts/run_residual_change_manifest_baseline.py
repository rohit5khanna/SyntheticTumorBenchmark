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
from baselines.unet import _TorchForecastDataset, _build_torch_model, _set_seed


KEY_COLS = ["split", "patient_id", "input_idx", "target_idx", "horizon"]


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


def normalize_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    out = manifest.copy()
    if "input_idx" not in out.columns and "input_end_idx" in out.columns:
        out["input_idx"] = out["input_end_idx"]
    return out


class ResidualChangeDataset:
    def __init__(
        self,
        dataset_root: str | Path,
        samples: List[ForecastSample],
        input_mode: str,
        spatial_stride: int = 1,
        cache_arrays: bool = True,
    ) -> None:
        if spatial_stride < 1:
            raise ValueError("spatial_stride must be >= 1.")
        self.base = _TorchForecastDataset(
            dataset_root=dataset_root,
            samples=samples,
            input_mode=input_mode,
            cache_arrays=cache_arrays,
        )
        self.samples = samples
        self.spatial_stride = int(spatial_stride)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        import torch

        x, target, _ = self.base[idx]
        sample = self.samples[idx]
        arrs = self.base._load_pid(sample.patient_id)
        input_mask = arrs["label"][sample.input_idx] > 0
        target_mask = arrs["label"][sample.target_idx] > 0

        growth = (target_mask & ~input_mask).astype(np.float32)
        loss = (input_mask & ~target_mask).astype(np.float32)
        y_change = np.concatenate([growth, loss], axis=0).astype(np.float32)
        if self.spatial_stride > 1:
            s = self.spatial_stride
            x = x[:, ::s, ::s, ::s]
            target = target[:, ::s, ::s, ::s]
            y_change = y_change[:, ::s, ::s, ::s]
            input_mask = input_mask[:, ::s, ::s, ::s]
        return x, torch.from_numpy(y_change), target, torch.from_numpy(input_mask.astype(np.float32)), idx


def soft_dice_loss(logits, target, eps: float = 1e-6):
    import torch

    probs = torch.sigmoid(logits)
    inter = (probs * target).sum(dim=(2, 3, 4))
    denom = probs.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def reconstruct(input_mask: np.ndarray, growth_prob: np.ndarray, loss_prob: np.ndarray, growth_thr: float, loss_thr: float):
    growth = growth_prob >= growth_thr
    loss = loss_prob >= loss_thr
    return ((input_mask > 0) & ~loss) | growth


def evaluate_model(model, loader, samples: List[ForecastSample], device, growth_thr: float, loss_thr: float) -> pd.DataFrame:
    import torch

    rows = []
    model.eval()
    with torch.no_grad():
        for x, _, target, input_mask, idx in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            target_np = target.numpy()
            input_np = input_mask.numpy()
            for j in range(probs.shape[0]):
                sample = samples[int(idx[j])]
                pred = reconstruct(input_np[j, 0], probs[j, 0], probs[j, 1], growth_thr, loss_thr)
                target_mask = target_np[j, 0] > 0
                input_mask_j = input_np[j, 0] > 0
                true_growth = target_mask & ~input_mask_j
                true_loss = input_mask_j & ~target_mask
                pred_growth = pred & ~input_mask_j
                pred_loss = input_mask_j & ~pred
                rows.append(
                    {
                        "patient_id": sample.patient_id,
                        "input_idx": int(sample.input_idx),
                        "target_idx": int(sample.target_idx),
                        "horizon": int(sample.horizon),
                        "delta_days": float(sample.delta_days),
                        "dice": float(dice_np(pred.astype(np.float32), target_mask.astype(np.float32))),
                        "locf_dice_recomputed": float(dice_np(input_mask_j.astype(np.float32), target_mask.astype(np.float32))),
                        "input_volume_vox": int(input_mask_j.sum()),
                        "target_volume_vox": int(target_mask.sum()),
                        "pred_volume_vox": int(pred.sum()),
                        "net_direction": "net_growth"
                        if int(target_mask.sum()) > int(input_mask_j.sum())
                        else "net_shrinkage"
                        if int(target_mask.sum()) < int(input_mask_j.sum())
                        else "net_stable",
                        "true_growth_volume_vox": int(true_growth.sum()),
                        "pred_growth_volume_vox": int(pred_growth.sum()),
                        "true_loss_volume_vox": int(true_loss.sum()),
                        "pred_loss_volume_vox": int(pred_loss.sum()),
                        "growth_threshold": float(growth_thr),
                        "loss_threshold": float(loss_thr),
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
            std_dice=("dice", "std"),
            locf_recomputed_mean=("locf_dice_recomputed", "mean"),
            pred_growth_volume_mean=("pred_growth_volume_vox", "mean"),
            pred_loss_volume_mean=("pred_loss_volume_vox", "mean"),
            true_growth_volume_mean=("true_growth_volume_vox", "mean"),
            true_loss_volume_mean=("true_loss_volume_vox", "mean"),
        )
        .reset_index()
    )
    if "_overall" in out.columns:
        out = out.drop(columns=["_overall"])
    return out


def limit_samples(samples: List[ForecastSample], max_samples: int | None) -> List[ForecastSample]:
    if max_samples is None or max_samples <= 0 or max_samples >= len(samples):
        return samples
    return samples[:max_samples]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train residual growth/loss forecasting baseline on a longitudinal manifest.")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--manifest_csv", type=str, required=True)
    parser.add_argument("--model_variant", type=str, choices=["unet", "resunet", "plain_cnn"], default="resunet")
    parser.add_argument("--input_mode", type=str, choices=["mask", "image_mask"], default="image_mask")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="val")
    parser.add_argument("--eval_splits", type=str, default="val,test")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--base_channels", type=int, default=12)
    parser.add_argument("--spatial_stride", type=int, default=1)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_val_samples", type=int, default=0)
    parser.add_argument("--max_eval_samples_per_split", type=int, default=0)
    parser.add_argument("--progress_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--thresholds", type=str, default="0.20,0.30,0.40,0.50,0.60,0.70,0.80")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
    except Exception as e:
        raise RuntimeError("PyTorch is required for residual-change baseline.") from e

    _set_seed(args.seed)
    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(args.dataset_root)
    manifest = normalize_manifest(pd.read_csv(args.manifest_csv))
    eval_splits = [s.strip() for s in args.eval_splits.split(",") if s.strip()]
    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]

    train_samples_all = build_samples_from_manifest(manifest, args.train_split)
    val_samples_all = build_samples_from_manifest(manifest, args.val_split)
    train_samples = limit_samples(train_samples_all, args.max_train_samples)
    val_samples = limit_samples(val_samples_all, args.max_val_samples)
    print(
        "[INFO] Residual-change setup | "
        f"train={len(train_samples)}/{len(train_samples_all)} "
        f"val={len(val_samples)}/{len(val_samples_all)} "
        f"input_mode={args.input_mode} spatial_stride={args.spatial_stride}",
        flush=True,
    )
    train_ds = ResidualChangeDataset(
        dataset_root,
        train_samples,
        args.input_mode,
        spatial_stride=args.spatial_stride,
    )
    val_ds = ResidualChangeDataset(
        dataset_root,
        val_samples,
        args.input_mode,
        spatial_stride=args.spatial_stride,
    )
    print("[INFO] Loading first training sample to infer model shape...", flush=True)
    sample_x, _, _, _, _ = train_ds[0]
    in_channels = int(sample_x.shape[0])
    print(f"[INFO] First input tensor shape: {tuple(sample_x.shape)}", flush=True)

    model = _build_torch_model(
        in_channels=in_channels,
        base_channels=args.base_channels,
        model_variant=args.model_variant,
        out_channels=2,
    )
    if args.device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(args.device)
    model.to(dev)
    print(f"[INFO] Device: {dev} | model={args.model_variant} base_channels={args.base_channels}", flush=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(dev.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(dev.type == "cuda"),
    )
    bce = nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_dice = -1.0
    best_ckpt = output_dir / f"model_best_residual_change_{args.model_variant}_{args.input_mode}.pt"
    history = []
    for ep in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for batch_i, (x, y_change, _, _, _) in enumerate(train_loader, start=1):
            x = x.to(dev, non_blocking=True)
            y_change = y_change.to(dev, non_blocking=True)
            logits = model(x)
            loss = bce(logits, y_change) + soft_dice_loss(logits, y_change)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            bs = int(x.shape[0])
            train_loss_sum += float(loss.item()) * bs
            train_count += bs
            if args.progress_every > 0 and (batch_i == 1 or batch_i % args.progress_every == 0):
                print(
                    f"[Epoch {ep:03d}] batch={batch_i}/{len(train_loader)} "
                    f"loss={float(loss.item()):.4f}",
                    flush=True,
                )

        print(f"[Epoch {ep:03d}] evaluating validation reconstruction...", flush=True)
        val_eval = evaluate_model(model, val_loader, val_samples, dev, 0.5, 0.5)
        val_dice = float(val_eval["dice"].mean())
        row = {
            "epoch": ep,
            "train_loss": train_loss_sum / max(1, train_count),
            "val_reconstruct_dice": val_dice,
        }
        history.append(row)
        print(
            f"[Epoch {ep:03d}] train_loss={row['train_loss']:.4f} "
            f"val_reconstruct_dice={val_dice:.4f}",
            flush=True,
        )
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "in_channels": in_channels,
                    "base_channels": args.base_channels,
                    "input_mode": args.input_mode,
                    "model_variant": args.model_variant,
                    "out_channels": 2,
                    "seed": args.seed,
                },
                best_ckpt,
            )

    ckpt = torch.load(best_ckpt, map_location=dev, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    val_threshold_rows = []
    for gthr in thresholds:
        for lthr in thresholds:
            val_eval = evaluate_model(model, val_loader, val_samples, dev, gthr, lthr)
            val_threshold_rows.append(
                {
                    "growth_threshold": gthr,
                    "loss_threshold": lthr,
                    "val_mean_dice": float(val_eval["dice"].mean()),
                    "val_locf_recomputed_mean": float(val_eval["locf_dice_recomputed"].mean()),
                }
            )
    val_thresholds = pd.DataFrame(val_threshold_rows).sort_values("val_mean_dice", ascending=False)
    best_growth_thr = float(val_thresholds.iloc[0]["growth_threshold"])
    best_loss_thr = float(val_thresholds.iloc[0]["loss_threshold"])

    eval_rows = []
    for split in eval_splits:
        samples_all = build_samples_from_manifest(manifest, split)
        samples = limit_samples(samples_all, args.max_eval_samples_per_split)
        print(
            f"[INFO] Evaluating split={split} samples={len(samples)}/{len(samples_all)}",
            flush=True,
        )
        ds = ResidualChangeDataset(
            dataset_root,
            samples,
            args.input_mode,
            spatial_stride=args.spatial_stride,
        )
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(dev.type == "cuda"))
        part = evaluate_model(model, loader, samples, dev, best_growth_thr, best_loss_thr)
        part["split"] = split
        eval_rows.append(part)
    per_sample = pd.concat(eval_rows, ignore_index=True)
    per_sample["method"] = f"residual_change_{args.model_variant}_{args.input_mode}"

    summary_by_split = summarize(per_sample, ["split"])
    summary_by_direction = summarize(per_sample, ["split", "net_direction"])
    summary = {
        "baseline": f"residual_change_{args.model_variant}_{args.input_mode}",
        "dataset_root": str(dataset_root.resolve()),
        "manifest_rows": int(len(manifest)),
        "train_split": args.train_split,
        "val_split": args.val_split,
        "eval_splits": eval_splits,
        "n_train_samples": len(train_samples),
        "n_train_samples_full": len(train_samples_all),
        "n_val_samples": len(val_samples),
        "n_val_samples_full": len(val_samples_all),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "spatial_stride": int(args.spatial_stride),
        "max_train_samples": int(args.max_train_samples),
        "max_val_samples": int(args.max_val_samples),
        "max_eval_samples_per_split": int(args.max_eval_samples_per_split),
        "seed": int(args.seed),
        "best_val_reconstruct_dice_at_0p5": float(best_val_dice),
        "selected_growth_threshold": best_growth_thr,
        "selected_loss_threshold": best_loss_thr,
        "checkpoint": str(best_ckpt),
        "by_split": summary_by_split.to_dict(orient="records"),
    }

    prefix = f"residual_change_{args.model_variant}_{args.input_mode}"
    pd.DataFrame(history).to_csv(output_dir / f"{prefix}_history.csv", index=False)
    val_thresholds.to_csv(output_dir / f"{prefix}_threshold_selection.csv", index=False)
    per_sample.to_csv(output_dir / f"{prefix}_per_sample.csv", index=False)
    summary_by_split.to_csv(output_dir / f"{prefix}_summary_by_split.csv", index=False)
    summary_by_direction.to_csv(output_dir / f"{prefix}_summary_by_net_direction.csv", index=False)
    with (output_dir / f"{prefix}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (output_dir / f"{prefix}_train_samples.json").open("w", encoding="utf-8") as f:
        json.dump([asdict(s) for s in train_samples], f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
