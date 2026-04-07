from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import json
import numpy as np


@dataclass(frozen=True)
class ForecastSample:
    patient_id: str
    input_idx: int
    target_idx: int
    horizon: int
    delta_days: float
    current_treatment: float
    target_treatment: float


def parse_horizons(horizons: str | Iterable[int]) -> List[int]:
    if isinstance(horizons, str):
        out = [int(x.strip()) for x in horizons.split(",") if x.strip()]
    else:
        out = [int(x) for x in horizons]
    out = [h for h in out if h >= 1]
    if not out:
        raise ValueError("Need at least one horizon >= 1.")
    return sorted(set(out))


def load_splits(dataset_root: str | Path) -> Dict[str, List[str]]:
    root = Path(dataset_root)
    split_path = root / "splits" / "splits.json"
    with split_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def patient_paths(dataset_root: str | Path, patient_id: str) -> Dict[str, Path]:
    pdir = Path(dataset_root) / "patients"
    return {
        "image": pdir / f"{patient_id}_image.npy",
        "label": pdir / f"{patient_id}_label.npy",
        "days": pdir / f"{patient_id}_days.npy",
        "treatment": pdir / f"{patient_id}_treatment.npy",
    }


def build_samples_for_split(
    dataset_root: str | Path,
    split: str,
    fit_sessions: int,
    horizons: Iterable[int] | str,
) -> List[ForecastSample]:
    if fit_sessions < 1:
        raise ValueError("fit_sessions must be >= 1.")

    horizons_l = parse_horizons(horizons)
    splits = load_splits(dataset_root)
    patient_ids = list(splits.get(split, []))
    if not patient_ids:
        raise ValueError(f"No patients found for split '{split}'.")

    out: List[ForecastSample] = []
    for pid in patient_ids:
        p = patient_paths(dataset_root, pid)
        days = np.load(p["days"]).astype(np.float32)
        treatment = np.load(p["treatment"]).astype(np.float32)
        n_sessions = int(days.shape[0])
        input_idx = fit_sessions - 1
        if input_idx >= n_sessions:
            continue
        for h in horizons_l:
            target_idx = input_idx + h
            if target_idx >= n_sessions:
                continue
            out.append(
                ForecastSample(
                    patient_id=pid,
                    input_idx=input_idx,
                    target_idx=target_idx,
                    horizon=h,
                    delta_days=float(days[target_idx] - days[input_idx]),
                    current_treatment=float(treatment[input_idx]),
                    target_treatment=float(treatment[target_idx]),
                )
            )

    if not out:
        raise ValueError(
            f"No valid samples for split={split}, fit_sessions={fit_sessions}, horizons={horizons_l}."
        )
    return out

