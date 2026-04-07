from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .tasks import ForecastSample, build_samples_for_split, patient_paths


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


class _TorchForecastDataset:
    def __init__(
        self,
        dataset_root: str | Path,
        samples: List[ForecastSample],
        input_mode: str = "mask",
        delta_days_norm: float = 180.0,
        cache_arrays: bool = True,
    ) -> None:
        self.root = Path(dataset_root)
        self.samples = samples
        self.input_mode = input_mode
        self.delta_days_norm = float(delta_days_norm)
        self.cache_arrays = cache_arrays
        self._cache: Dict[str, Dict[str, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def _load_pid(self, patient_id: str) -> Dict[str, np.ndarray]:
        if self.cache_arrays and patient_id in self._cache:
            return self._cache[patient_id]

        p = patient_paths(self.root, patient_id)
        arrs = {
            "label": np.load(p["label"]).astype(np.float32),  # [S,1,H,W,D]
            "days": np.load(p["days"]).astype(np.float32),
            "treatment": np.load(p["treatment"]).astype(np.float32),
        }
        if self.input_mode == "image_mask":
            arrs["image"] = np.load(p["image"]).astype(np.float32)  # [S,C,H,W,D]

        if self.cache_arrays:
            self._cache[patient_id] = arrs
        return arrs

    def __getitem__(self, idx: int):
        import torch

        s = self.samples[idx]
        arrs = self._load_pid(s.patient_id)

        mask_in = arrs["label"][s.input_idx]  # [1,H,W,D]
        mask_target = arrs["label"][s.target_idx]  # [1,H,W,D]
        h, w, d = mask_in.shape[-3:]

        delta_scale = float(s.delta_days) / self.delta_days_norm
        delta_chan = np.full((1, h, w, d), fill_value=delta_scale, dtype=np.float32)
        cur_treat_chan = np.full((1, h, w, d), fill_value=float(s.current_treatment), dtype=np.float32)
        tgt_treat_chan = np.full((1, h, w, d), fill_value=float(s.target_treatment), dtype=np.float32)

        feats = []
        if self.input_mode == "image_mask":
            feats.append(arrs["image"][s.input_idx])  # [C,H,W,D]
        feats.extend([mask_in, delta_chan, cur_treat_chan, tgt_treat_chan])
        x = np.concatenate(feats, axis=0).astype(np.float32)  # [Cin,H,W,D]
        y = mask_target.astype(np.float32)

        return torch.from_numpy(x), torch.from_numpy(y), idx


def _build_torch_model(in_channels: int, base_channels: int):
    import torch
    import torch.nn as nn

    class ConvBlock(nn.Module):
        def __init__(self, c_in: int, c_out: int):
            super().__init__()
            self.seq = nn.Sequential(
                nn.Conv3d(c_in, c_out, kernel_size=3, padding=1),
                nn.InstanceNorm3d(c_out),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv3d(c_out, c_out, kernel_size=3, padding=1),
                nn.InstanceNorm3d(c_out),
                nn.LeakyReLU(0.1, inplace=True),
            )

        def forward(self, x):
            return self.seq(x)

    class UNet3D(nn.Module):
        def __init__(self, cin: int, base: int):
            super().__init__()
            self.e1 = ConvBlock(cin, base)
            self.p1 = nn.MaxPool3d(2)
            self.e2 = ConvBlock(base, base * 2)
            self.p2 = nn.MaxPool3d(2)
            self.b = ConvBlock(base * 2, base * 4)
            self.u2 = nn.ConvTranspose3d(base * 4, base * 2, kernel_size=2, stride=2)
            self.d2 = ConvBlock(base * 4, base * 2)
            self.u1 = nn.ConvTranspose3d(base * 2, base, kernel_size=2, stride=2)
            self.d1 = ConvBlock(base * 2, base)
            self.head = nn.Conv3d(base, 1, kernel_size=1)

        def forward(self, x):
            e1 = self.e1(x)
            e2 = self.e2(self.p1(e1))
            b = self.b(self.p2(e2))
            u2 = self.u2(b)
            d2 = self.d2(torch.cat([u2, e2], dim=1))
            u1 = self.u1(d2)
            d1 = self.d1(torch.cat([u1, e1], dim=1))
            return self.head(d1)

    return UNet3D(in_channels, base_channels)


def _dice_from_logits(logits, target, eps: float = 1e-6):
    import torch

    probs = torch.sigmoid(logits)
    pred = (probs > 0.5).float()
    inter = (pred * target).sum(dim=(1, 2, 3, 4))
    denom = pred.sum(dim=(1, 2, 3, 4)) + target.sum(dim=(1, 2, 3, 4))
    return (2.0 * inter + eps) / (denom + eps)


def _dice_loss_soft(logits, target, eps: float = 1e-6):
    import torch

    probs = torch.sigmoid(logits)
    inter = (probs * target).sum(dim=(1, 2, 3, 4))
    denom = probs.sum(dim=(1, 2, 3, 4)) + target.sum(dim=(1, 2, 3, 4))
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def run_unet_baseline(
    dataset_root: str | Path,
    train_split: str,
    eval_split: str,
    fit_sessions: int,
    horizons: Iterable[int] | str,
    input_mode: str,
    output_dir: str | Path,
    epochs: int = 12,
    batch_size: int = 2,
    learning_rate: float = 2e-4,
    num_workers: int = 0,
    base_channels: int = 12,
    seed: int = 42,
    device: str = "auto",
) -> Dict:
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
    except Exception as e:
        raise RuntimeError(
            "PyTorch is required for UNet baselines. Install torch in this environment."
        ) from e

    if input_mode not in {"mask", "image_mask"}:
        raise ValueError("input_mode must be one of: mask, image_mask.")

    _set_seed(seed)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_samples = build_samples_for_split(dataset_root, train_split, fit_sessions, horizons)
    eval_samples = build_samples_for_split(dataset_root, eval_split, fit_sessions, horizons)

    train_ds = _TorchForecastDataset(dataset_root, train_samples, input_mode=input_mode)
    eval_ds = _TorchForecastDataset(dataset_root, eval_samples, input_mode=input_mode)

    sample_x, _, _ = train_ds[0]
    in_channels = int(sample_x.shape[0])
    model = _build_torch_model(in_channels=in_channels, base_channels=base_channels)

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
    eval_loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(dev.type == "cuda"),
    )

    bce = nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_eval_dice = -1.0
    best_ckpt = out_dir / f"model_best_{input_mode}.pt"
    history: List[Dict] = []

    for ep in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_dice_sum = 0.0
        train_count = 0

        for x, y, _ in train_loader:
            x = x.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)

            logits = model(x)
            seg_loss = bce(logits, y)
            dloss = _dice_loss_soft(logits, y)
            loss = seg_loss + dloss

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            bs = int(x.shape[0])
            train_loss_sum += float(loss.item()) * bs
            train_dice_sum += float(_dice_from_logits(logits, y).mean().item()) * bs
            train_count += bs

        model.eval()
        eval_loss_sum = 0.0
        eval_dice_sum = 0.0
        eval_count = 0
        with torch.no_grad():
            for x, y, _ in eval_loader:
                x = x.to(dev, non_blocking=True)
                y = y.to(dev, non_blocking=True)
                logits = model(x)
                seg_loss = bce(logits, y)
                dloss = _dice_loss_soft(logits, y)
                loss = seg_loss + dloss
                bs = int(x.shape[0])
                eval_loss_sum += float(loss.item()) * bs
                eval_dice_sum += float(_dice_from_logits(logits, y).mean().item()) * bs
                eval_count += bs

        train_loss = train_loss_sum / max(1, train_count)
        train_dice = train_dice_sum / max(1, train_count)
        eval_loss = eval_loss_sum / max(1, eval_count)
        eval_dice = eval_dice_sum / max(1, eval_count)
        history.append(
            {
                "epoch": ep,
                "train_loss": train_loss,
                "train_dice": train_dice,
                "eval_loss": eval_loss,
                "eval_dice": eval_dice,
            }
        )
        print(
            f"[Epoch {ep:03d}] "
            f"train_loss={train_loss:.4f} train_dice={train_dice:.4f} "
            f"eval_loss={eval_loss:.4f} eval_dice={eval_dice:.4f}"
        )

        if eval_dice > best_eval_dice:
            best_eval_dice = float(eval_dice)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "in_channels": in_channels,
                    "base_channels": base_channels,
                    "input_mode": input_mode,
                    "seed": seed,
                },
                best_ckpt,
            )

    ckpt = torch.load(best_ckpt, map_location=dev, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    rows = []
    dices = []
    with torch.no_grad():
        for x, y, idx in eval_loader:
            x = x.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)
            logits = model(x)
            batch_dice = _dice_from_logits(logits, y).detach().cpu().numpy().tolist()
            for j, d in enumerate(batch_dice):
                s = eval_samples[int(idx[j])]
                rows.append(
                    {
                        "patient_id": s.patient_id,
                        "input_idx": s.input_idx,
                        "target_idx": s.target_idx,
                        "horizon": s.horizon,
                        "delta_days": s.delta_days,
                        "dice": float(d),
                    }
                )
                dices.append(float(d))

    summary = {
        "baseline": f"unet_{input_mode}",
        "dataset_root": str(Path(dataset_root).resolve()),
        "train_split": train_split,
        "eval_split": eval_split,
        "fit_sessions": int(fit_sessions),
        "input_mode": input_mode,
        "n_train_samples": len(train_samples),
        "n_eval_samples": len(eval_samples),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "seed": int(seed),
        "best_eval_dice": float(best_eval_dice),
        "mean_eval_dice": float(np.mean(dices)),
        "std_eval_dice": float(np.std(dices)),
        "checkpoint": str(best_ckpt),
    }

    with (out_dir / f"unet_{input_mode}_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with (out_dir / f"unet_{input_mode}_per_sample.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    with (out_dir / f"unet_{input_mode}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (out_dir / f"unet_{input_mode}_train_samples.json").open("w", encoding="utf-8") as f:
        json.dump([asdict(s) for s in train_samples], f, indent=2)
    with (out_dir / f"unet_{input_mode}_eval_samples.json").open("w", encoding="utf-8") as f:
        json.dump([asdict(s) for s in eval_samples], f, indent=2)

    return summary
