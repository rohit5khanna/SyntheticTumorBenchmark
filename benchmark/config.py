from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "dataset": {
        "name": "synthetic_tumor_benchmark_v1",
        "seed": 42,
        "output_root": "./data/synthetic_tumor_benchmark_v1",
        "volume_shape": [96, 96, 64],
        "modalities": ["t1", "t1ce", "t2", "flair"],
        "save_concentration": True,
        "overwrite": False,
        "patient_id_prefix": "syn",
    },
    "schedule": {
        "n_sessions_min": 4,
        "n_sessions_max": 7,
        "days_interval_min": 20,
        "days_interval_max": 60,
        "treatment_patient_prob": 0.55,
        "treatment_start_session_min": 1,
    },
    "split": {
        "train_frac": 0.7,
        "val_frac": 0.15,
        "test_frac": 0.15,
        "seed": 42,
    },
    "labeling": {
        "mask_threshold": 0.35,
    },
    "simulation": {
        "steps_per_day": 2,
        "dt": 0.10,
        "rho_range": [0.025, 0.075],
        "dw_range": [0.05, 0.20],
        "treatment_effect_range": [0.05, 0.20],
        "init_foci_min": 1,
        "init_foci_max": 3,
        "init_sigma_vox_range": [1.8, 4.5],
        "init_amp_range": [0.7, 1.2],
    },
    "tiers": {
        "A": {
            "enabled": True,
            "n_patients": 60,
            "description": "Simple procedural geometric growth",
        },
        "B": {
            "enabled": True,
            "n_patients": 60,
            "description": "Isotropic reaction-diffusion growth",
        },
        "C": {
            "enabled": True,
            "n_patients": 60,
            "description": "Anisotropic + heterogeneous reaction-diffusion growth",
        },
    },
    "image_synthesis": {
        "noise_std": 0.025,
        "bias_amp": 0.12,
        "bias_smooth_steps": 8,
        "core_threshold": 0.70,
        "edema_threshold": 0.20,
        "t1_core_boost": 0.20,
        "t1ce_core_boost": 0.55,
        "t2_edema_boost": 0.45,
        "flair_edema_boost": 0.60,
    },
}


def _deep_update(base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(base)
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def _validate_cfg(cfg: Dict[str, Any]) -> None:
    shape = cfg["dataset"]["volume_shape"]
    if len(shape) != 3:
        raise ValueError("dataset.volume_shape must have 3 entries: [H, W, D].")
    if any(int(v) <= 0 for v in shape):
        raise ValueError("dataset.volume_shape entries must be positive.")

    split = cfg["split"]
    frac_sum = float(split["train_frac"]) + float(split["val_frac"]) + float(split["test_frac"])
    if abs(frac_sum - 1.0) > 1e-6:
        raise ValueError(f"split fractions must sum to 1.0. Got {frac_sum:.6f}.")

    schedule = cfg["schedule"]
    if int(schedule["n_sessions_min"]) < 3:
        raise ValueError("schedule.n_sessions_min must be >= 3.")
    if int(schedule["n_sessions_max"]) < int(schedule["n_sessions_min"]):
        raise ValueError("schedule.n_sessions_max must be >= schedule.n_sessions_min.")

    tiers = cfg["tiers"]
    enabled_tiers = [k for k, v in tiers.items() if bool(v.get("enabled", False))]
    if not enabled_tiers:
        raise ValueError("At least one tier must be enabled in tiers.{A,B,C}.")
    for k in enabled_tiers:
        if int(tiers[k].get("n_patients", 0)) <= 0:
            raise ValueError(f"tiers.{k}.n_patients must be > 0 when enabled.")


def load_config(config_path: str | Path) -> Dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg = _deep_update(DEFAULT_CONFIG, raw)
    _validate_cfg(cfg)
    return cfg

