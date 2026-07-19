#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
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


def limit_samples(samples: List[ForecastSample], max_samples: int | None) -> List[ForecastSample]:
    if max_samples is None or max_samples <= 0 or max_samples >= len(samples):
        return samples
    return samples[:max_samples]


class GrowthOnlyDataset:
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
        candidate = (~input_mask).astype(np.float32)

        if self.spatial_stride > 1:
            s = self.spatial_stride
            x = x[:, ::s, ::s, ::s]
            target = target[:, ::s, ::s, ::s]
            input_mask = input_mask[:, ::s, ::s, ::s]
            growth = growth[:, ::s, ::s, ::s]
            candidate = candidate[:, ::s, ::s, ::s]

        return (
            x,
            torch.from_numpy(growth.astype(np.float32)),
            target,
            torch.from_numpy(input_mask.astype(np.float32)),
            torch.from_numpy(candidate.astype(np.float32)),
            idx,
        )


def masked_bce_with_logits(logits, target, mask, max_pos_weight: float):
    import torch
    import torch.nn.functional as F

    pos = (target * mask).sum()
    neg = ((1.0 - target) * mask).sum()
    pos_weight = torch.clamp(neg / torch.clamp(pos, min=1.0), min=1.0, max=max_pos_weight)
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none", pos_weight=pos_weight)
    return (loss * mask).sum() / torch.clamp(mask.sum(), min=1.0)


def masked_soft_dice_loss(logits, target, mask, eps: float = 1e-6):
    import torch

    probs = torch.sigmoid(logits) * mask
    target_m = target * mask
    inter = (probs * target_m).sum(dim=(1, 2, 3, 4))
    denom = probs.sum(dim=(1, 2, 3, 4)) + target_m.sum(dim=(1, 2, 3, 4))
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


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


def evaluate_model(model, loader, samples: List[ForecastSample], device, policy_mode: str, policy_value: float) -> pd.DataFrame:
    import torch

    rows = []
    model.eval()
    with torch.no_grad():
        for x, _, target, input_mask, _, idx in loader:
            x = x.to(device, non_blocking=True)
            growth_prob = torch.sigmoid(model(x)).detach().cpu().numpy()
            target_np = target.numpy()
            input_np = input_mask.numpy()
            for j in range(growth_prob.shape[0]):
                sample = samples[int(idx[j])]
                prob = growth_prob[j, 0]
                input_j = input_np[j, 0] > 0
                target_j = target_np[j, 0] > 0
                if policy_mode == "threshold":
                    pred, added = add_by_threshold(input_j, prob, policy_value)
                elif policy_mode == "budget":
                    pred, added = add_by_budget(input_j, prob, policy_value)
                else:
                    raise ValueError(f"Unknown policy mode: {policy_mode}")

                true_growth = target_j & ~input_j
                pred_growth = pred & ~input_j
                growth_tp = int((pred_growth & true_growth).sum())
                growth_fp = int((pred_growth & ~true_growth).sum())
                input_volume = int(input_j.sum())
                target_volume = int(target_j.sum())
                locf_dice = float(dice_np(input_j.astype(np.float32), target_j.astype(np.float32)))
                dice = float(dice_np(pred.astype(np.float32), target_j.astype(np.float32)))
                rows.append(
                    {
                        "patient_id": sample.patient_id,
                        "input_idx": int(sample.input_idx),
                        "target_idx": int(sample.target_idx),
                        "horizon": int(sample.horizon),
                        "delta_days": float(sample.delta_days),
                        "policy_mode": policy_mode,
                        "policy_value": float(policy_value),
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
            median_gap_vs_locf=("gap_vs_locf", "median"),
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


def parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train growth-only outside-input forecasting baseline on a longitudinal manifest.")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--manifest_csv", type=str, required=True)
    parser.add_argument("--model_variant", type=str, choices=["unet", "resunet", "plain_cnn"], default="resunet")
    parser.add_argument("--input_mode", type=str, choices=["mask", "image_mask"], default="image_mask")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="val")
    parser.add_argument("--eval_splits", type=str, default="val,test")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--base_channels", type=int, default=6)
    parser.add_argument("--spatial_stride", type=int, default=2)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_val_samples", type=int, default=0)
    parser.add_argument("--max_eval_samples_per_split", type=int, default=0)
    parser.add_argument("--max_pos_weight", type=float, default=100.0)
    parser.add_argument("--dice_loss_weight", type=float, default=1.0)
    parser.add_argument("--thresholds", type=str, default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--budget_fractions", type=str, default="0,0.005,0.01,0.02,0.05,0.1,0.2")
    parser.add_argument("--selection_objective", type=str, choices=["mean_dice", "mean_gap_vs_locf", "net_growth_gap"], default="mean_dice")
    parser.add_argument("--progress_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    try:
        import torch
        from torch.utils.data import DataLoader
    except Exception as e:
        raise RuntimeError("PyTorch is required for growth-only baseline.") from e

    _set_seed(args.seed)
    random.seed(args.seed)
    dataset_root = Path(args.dataset_root)
    manifest = normalize_manifest(pd.read_csv(args.manifest_csv))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_splits = [s.strip() for s in args.eval_splits.split(",") if s.strip()]

    train_samples_full = build_samples_from_manifest(manifest, args.train_split)
    val_samples_full = build_samples_from_manifest(manifest, args.val_split)
    train_samples = limit_samples(train_samples_full, args.max_train_samples)
    val_samples = limit_samples(val_samples_full, args.max_val_samples)

    print(
        "[INFO] Growth-only setup | "
        f"train={len(train_samples)}/{len(train_samples_full)} "
        f"val={len(val_samples)}/{len(val_samples_full)} "
        f"input_mode={args.input_mode} spatial_stride={args.spatial_stride}",
        flush=True,
    )
    train_ds = GrowthOnlyDataset(dataset_root, train_samples, args.input_mode, spatial_stride=args.spatial_stride)
    val_ds = GrowthOnlyDataset(dataset_root, val_samples, args.input_mode, spatial_stride=args.spatial_stride)
    print("[INFO] Loading first training sample to infer model shape...", flush=True)
    sample_x, _, _, _, _, _ = train_ds[0]
    in_channels = int(sample_x.shape[0])
    print(f"[INFO] First input tensor shape: {tuple(sample_x.shape)}", flush=True)

    model = _build_torch_model(
        in_channels=in_channels,
        base_channels=args.base_channels,
        model_variant=args.model_variant,
        out_channels=1,
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
    optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_val_dice = -1.0
    best_ckpt = output_dir / f"model_best_growth_only_{args.model_variant}_{args.input_mode}.pt"
    history = []

    for ep in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for batch_i, (x, growth, _, _, candidate, _) in enumerate(train_loader, start=1):
            x = x.to(dev, non_blocking=True)
            growth = growth.to(dev, non_blocking=True)
            candidate = candidate.to(dev, non_blocking=True)
            logits = model(x)
            bce_loss = masked_bce_with_logits(logits, growth, candidate, max_pos_weight=args.max_pos_weight)
            dice_loss = masked_soft_dice_loss(logits, growth, candidate)
            loss = bce_loss + args.dice_loss_weight * dice_loss
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            bs = int(x.shape[0])
            train_loss_sum += float(loss.item()) * bs
            train_count += bs
            if args.progress_every > 0 and (batch_i == 1 or batch_i % args.progress_every == 0):
                print(
                    f"[Epoch {ep:03d}] batch={batch_i}/{len(train_loader)} "
                    f"loss={float(loss.item()):.4f} bce={float(bce_loss.item()):.4f} dice={float(dice_loss.item()):.4f}",
                    flush=True,
                )

        val_eval = evaluate_model(model, val_loader, val_samples, dev, "budget", 0.05)
        val_dice = float(val_eval["dice"].mean())
        train_loss = train_loss_sum / max(1, train_count)
        row = {"epoch": ep, "train_loss": train_loss, "val_growth_only_dice_budget_0p05": val_dice}
        history.append(row)
        print(f"[Epoch {ep:03d}] train_loss={train_loss:.4f} val_budget_0p05_dice={val_dice:.4f}", flush=True)
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "in_channels": in_channels,
                    "base_channels": args.base_channels,
                    "input_mode": args.input_mode,
                    "model_variant": args.model_variant,
                    "out_channels": 1,
                    "seed": args.seed,
                    "spatial_stride": args.spatial_stride,
                    "target": "growth_only_outside_input",
                },
                best_ckpt,
            )

    ckpt = torch.load(best_ckpt, map_location=dev, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    candidates = [("threshold", v) for v in parse_float_list(args.thresholds)]
    candidates += [("budget", v) for v in parse_float_list(args.budget_fractions)]
    val_rows = []
    val_summaries = []
    for mode, value in candidates:
        part = evaluate_model(model, val_loader, val_samples, dev, mode, value)
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

    eval_rows = []
    for split in eval_splits:
        samples_full = build_samples_from_manifest(manifest, split)
        samples = limit_samples(samples_full, args.max_eval_samples_per_split)
        ds = GrowthOnlyDataset(dataset_root, samples, args.input_mode, spatial_stride=args.spatial_stride)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(dev.type == "cuda"))
        part = evaluate_model(model, loader, samples, dev, selected_mode, selected_value)
        part["split"] = split
        eval_rows.append(part)
    per_sample = pd.concat(eval_rows, ignore_index=True)
    by_split = summarize(per_sample, ["split"])
    by_direction = summarize(per_sample, ["split", "net_direction"])

    prefix = f"growth_only_{args.model_variant}_{args.input_mode}"
    pd.DataFrame(history).to_csv(output_dir / f"{prefix}_history.csv", index=False)
    pd.concat(val_rows, ignore_index=True).to_csv(output_dir / f"{prefix}_validation_samples_all.csv", index=False)
    val_selection.to_csv(output_dir / f"{prefix}_validation_sweep.csv", index=False)
    per_sample.to_csv(output_dir / f"{prefix}_selected_samples.csv", index=False)
    by_split.to_csv(output_dir / f"{prefix}_selected_by_split.csv", index=False)
    by_direction.to_csv(output_dir / f"{prefix}_selected_by_net_direction.csv", index=False)
    with (output_dir / f"{prefix}_train_samples.json").open("w", encoding="utf-8") as f:
        json.dump([asdict(s) for s in train_samples], f, indent=2)

    summary = {
        "baseline": prefix,
        "dataset_root": str(dataset_root.resolve()),
        "manifest_rows": int(len(manifest)),
        "train_split": args.train_split,
        "val_split": args.val_split,
        "eval_splits": eval_splits,
        "n_train_samples": len(train_samples),
        "n_train_samples_full": len(train_samples_full),
        "n_val_samples": len(val_samples),
        "n_val_samples_full": len(val_samples_full),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "base_channels": int(args.base_channels),
        "spatial_stride": int(args.spatial_stride),
        "max_pos_weight": float(args.max_pos_weight),
        "dice_loss_weight": float(args.dice_loss_weight),
        "seed": int(args.seed),
        "best_val_growth_only_dice_budget_0p05": float(best_val_dice),
        "selection_objective": args.selection_objective,
        "selected_policy_mode": selected_mode,
        "selected_policy_value": selected_value,
        "selected_validation_row": selected_row.to_dict(),
        "checkpoint": str(best_ckpt),
        "by_split": by_split.to_dict(orient="records"),
    }
    with (output_dir / f"{prefix}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
