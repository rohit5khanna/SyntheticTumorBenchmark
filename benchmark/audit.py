from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


@dataclass(frozen=True)
class AuditBundle:
    patient_id: str
    tier: str
    split: str
    label: np.ndarray  # [S,H,W,D] binary
    days: np.ndarray  # [S]
    treatment: np.ndarray  # [S]


def detect_dataset_kind(dataset_root: str | Path) -> str:
    root = Path(dataset_root)
    if (root / "patients").exists():
        return "synthetic_benchmark"
    return "plain_npy"


def _patient_dir(dataset_root: Path, kind: str) -> Path:
    return dataset_root / "patients" if kind == "synthetic_benchmark" else dataset_root


def list_patient_ids(dataset_root: str | Path, kind: str | None = None) -> List[str]:
    root = Path(dataset_root)
    kind = kind or detect_dataset_kind(root)
    pdir = _patient_dir(root, kind)
    patient_ids = []
    for p in sorted(pdir.glob("*_label.npy")):
        patient_ids.append(p.name[: -len("_label.npy")])
    return patient_ids


def load_split_map(dataset_root: str | Path, kind: str | None = None) -> Dict[str, str]:
    root = Path(dataset_root)
    kind = kind or detect_dataset_kind(root)
    if kind != "synthetic_benchmark":
        return {}
    split_path = root / "splits" / "splits.json"
    if not split_path.exists():
        return {}
    with split_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    split_map: Dict[str, str] = {}
    for split_name, ids in payload.items():
        for pid in ids:
            split_map[str(pid)] = str(split_name)
    return split_map


def infer_tier(patient_id: str, default_tier: str = "REAL") -> str:
    parts = patient_id.split("-")
    if len(parts) >= 2 and parts[1] in {"A", "B", "C"}:
        return parts[1]
    return default_tier


def _standardize_label(label: np.ndarray) -> np.ndarray:
    arr = np.asarray(label)
    if arr.ndim == 5 and arr.shape[1] == 1:
        arr = arr[:, 0]
    elif arr.ndim != 4:
        raise ValueError(f"Unsupported label shape: {arr.shape}")
    return (arr > 0).astype(np.uint8)


def load_audit_bundle(
    dataset_root: str | Path,
    patient_id: str,
    kind: str | None = None,
    split_map: Dict[str, str] | None = None,
    real_tier_name: str = "REAL",
) -> AuditBundle:
    root = Path(dataset_root)
    kind = kind or detect_dataset_kind(root)
    pdir = _patient_dir(root, kind)

    label = _standardize_label(np.load(pdir / f"{patient_id}_label.npy"))
    days = np.asarray(np.load(pdir / f"{patient_id}_days.npy"), dtype=np.float32)
    treatment = np.asarray(np.load(pdir / f"{patient_id}_treatment.npy"), dtype=np.float32)

    if label.shape[0] != len(days) or label.shape[0] != len(treatment):
        raise ValueError(
            f"Session mismatch for {patient_id}: label={label.shape[0]}, days={len(days)}, treatment={len(treatment)}"
        )

    split = "unknown"
    if split_map is not None:
        split = split_map.get(patient_id, "unknown")

    return AuditBundle(
        patient_id=patient_id,
        tier=infer_tier(patient_id, default_tier=real_tier_name),
        split=split,
        label=label,
        days=days,
        treatment=treatment,
    )


def bbox_dims(mask: np.ndarray) -> Tuple[int, int, int]:
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return (0, 0, 0)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    dims = maxs - mins + 1
    return int(dims[0]), int(dims[1]), int(dims[2])


def session_shape_metrics(mask: np.ndarray) -> Dict[str, float]:
    vol = float(mask.sum())
    bx, by, bz = bbox_dims(mask)
    bbox_vol = float(max(1, bx * by * bz))
    nonzero_dims = [d for d in (bx, by, bz) if d > 0]
    elongation = float(max(nonzero_dims) / min(nonzero_dims)) if len(nonzero_dims) == 3 else 0.0
    compactness_proxy = float(vol / bbox_vol) if bbox_vol > 0 else 0.0
    return {
        "volume_vox": vol,
        "bbox_x": float(bx),
        "bbox_y": float(by),
        "bbox_z": float(bz),
        "bbox_volume_vox": bbox_vol,
        "elongation_ratio": elongation,
        "compactness_proxy": compactness_proxy,
    }


def build_audit_tables(
    dataset_root: str | Path,
    dataset_name: str,
    kind: str | None = None,
    real_tier_name: str = "REAL",
) -> Dict[str, List[Dict]]:
    root = Path(dataset_root)
    kind = kind or detect_dataset_kind(root)
    split_map = load_split_map(root, kind=kind)
    patient_ids = list_patient_ids(root, kind=kind)

    patient_rows: List[Dict] = []
    session_rows: List[Dict] = []
    transition_rows: List[Dict] = []

    for pid in patient_ids:
        bundle = load_audit_bundle(
            root,
            pid,
            kind=kind,
            split_map=split_map,
            real_tier_name=real_tier_name,
        )
        n_sessions = int(bundle.label.shape[0])
        intervals = np.diff(bundle.days)

        patient_rows.append(
            {
                "dataset_name": dataset_name,
                "dataset_kind": kind,
                "patient_id": bundle.patient_id,
                "tier": bundle.tier,
                "split": bundle.split,
                "n_sessions": n_sessions,
                "followup_days": float(bundle.days[-1] - bundle.days[0]) if n_sessions > 0 else 0.0,
                "treatment_on_any": int(np.any(bundle.treatment > 0)),
                "treatment_start_session": int(np.argmax(bundle.treatment > 0)) if np.any(bundle.treatment > 0) else -1,
                "mean_interval_days": float(np.mean(intervals)) if len(intervals) else 0.0,
            }
        )

        prev_vol = None
        for s in range(n_sessions):
            mask = bundle.label[s]
            metrics = session_shape_metrics(mask)
            row = {
                "dataset_name": dataset_name,
                "dataset_kind": kind,
                "patient_id": bundle.patient_id,
                "tier": bundle.tier,
                "split": bundle.split,
                "session_idx": s,
                "day": float(bundle.days[s]),
                "treatment": float(bundle.treatment[s]),
                **metrics,
            }
            session_rows.append(row)

            cur_vol = metrics["volume_vox"]
            if prev_vol is not None:
                delta = cur_vol - prev_vol
                rel = delta / max(prev_vol, 1.0)
                transition_rows.append(
                    {
                        "dataset_name": dataset_name,
                        "dataset_kind": kind,
                        "patient_id": bundle.patient_id,
                        "tier": bundle.tier,
                        "split": bundle.split,
                        "from_session_idx": s - 1,
                        "to_session_idx": s,
                        "from_day": float(bundle.days[s - 1]),
                        "to_day": float(bundle.days[s]),
                        "delta_days": float(bundle.days[s] - bundle.days[s - 1]),
                        "from_volume_vox": float(prev_vol),
                        "to_volume_vox": float(cur_vol),
                        "delta_volume_vox": float(delta),
                        "relative_growth_rate": float(rel),
                    }
                )
            prev_vol = cur_vol

    return {
        "patients": patient_rows,
        "sessions": session_rows,
        "transitions": transition_rows,
    }


def summarize_rows(rows: List[Dict], value_keys: Iterable[str], group_keys: Iterable[str]) -> List[Dict]:
    if not rows:
        return []

    group_keys = list(group_keys)
    value_keys = list(value_keys)
    grouped: Dict[Tuple, List[Dict]] = {}
    for row in rows:
        key = tuple(row[g] for g in group_keys)
        grouped.setdefault(key, []).append(row)

    out: List[Dict] = []
    for key, grows in grouped.items():
        base = {g: k for g, k in zip(group_keys, key)}
        base["count"] = len(grows)
        for vk in value_keys:
            vals = [float(r[vk]) for r in grows if vk in r]
            if not vals:
                continue
            arr = np.asarray(vals, dtype=np.float64)
            base[f"{vk}_mean"] = float(arr.mean())
            base[f"{vk}_std"] = float(arr.std())
            base[f"{vk}_min"] = float(arr.min())
            base[f"{vk}_max"] = float(arr.max())
        out.append(base)
    return out
