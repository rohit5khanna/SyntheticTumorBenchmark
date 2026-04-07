from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from .metrics import dice_np
from .tasks import build_samples_for_split, patient_paths


def run_locf_baseline(
    dataset_root: str | Path,
    split: str,
    fit_sessions: int,
    horizons: Iterable[int] | str,
    output_dir: str | Path,
) -> Dict:
    root = Path(dataset_root)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = build_samples_for_split(
        dataset_root=root,
        split=split,
        fit_sessions=fit_sessions,
        horizons=horizons,
    )

    rows: List[Dict] = []
    dices: List[float] = []

    for s in samples:
        p = patient_paths(root, s.patient_id)
        labels = np.load(p["label"]).astype(np.float32)  # [S,1,H,W,D]
        pred = labels[s.input_idx]
        target = labels[s.target_idx]
        d = dice_np(pred, target)
        dices.append(d)
        rows.append(
            {
                "patient_id": s.patient_id,
                "input_idx": s.input_idx,
                "target_idx": s.target_idx,
                "horizon": s.horizon,
                "delta_days": s.delta_days,
                "dice": d,
            }
        )

    summary = {
        "baseline": "locf",
        "dataset_root": str(root.resolve()),
        "split": split,
        "fit_sessions": int(fit_sessions),
        "n_samples": len(rows),
        "mean_dice": float(np.mean(dices)),
        "std_dice": float(np.std(dices)),
    }

    with (out_dir / "locf_per_sample.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    with (out_dir / "locf_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary

