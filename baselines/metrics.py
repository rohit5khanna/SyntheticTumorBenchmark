from __future__ import annotations

import numpy as np


def dice_np(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    pred_bin = (pred > 0.5).astype(np.float32)
    tgt_bin = (target > 0.5).astype(np.float32)
    inter = float((pred_bin * tgt_bin).sum())
    denom = float(pred_bin.sum() + tgt_bin.sum())
    return float((2.0 * inter + eps) / (denom + eps))

