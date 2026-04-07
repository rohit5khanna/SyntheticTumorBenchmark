from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

from .images import make_session_modalities
from .simulator import simulate_patient


def _prepare_dirs(output_root: Path, overwrite: bool) -> Dict[str, Path]:
    if output_root.exists() and overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    out = {
        "root": output_root,
        "patients": output_root / "patients",
        "metadata": output_root / "metadata",
        "manifests": output_root / "manifests",
        "splits": output_root / "splits",
    }
    for p in out.values():
        p.mkdir(parents=True, exist_ok=True)
    return out


def _assign_splits(patient_ids: List[str], split_cfg: Dict) -> Dict[str, str]:
    rng = np.random.default_rng(int(split_cfg["seed"]))
    ids = list(patient_ids)
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(round(float(split_cfg["train_frac"]) * n))
    n_val = int(round(float(split_cfg["val_frac"]) * n))
    if n_train + n_val > n:
        n_val = max(0, n - n_train)
    n_test = n - n_train - n_val
    if n_test < 0:
        n_test = 0

    split_map = {}
    for pid in ids[:n_train]:
        split_map[pid] = "train"
    for pid in ids[n_train : n_train + n_val]:
        split_map[pid] = "val"
    for pid in ids[n_train + n_val :]:
        split_map[pid] = "test"
    return split_map


def _write_manifest_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def generate_benchmark_dataset(cfg: Dict) -> Dict:
    dcfg = cfg["dataset"]
    output_root = Path(dcfg["output_root"]).expanduser().resolve()
    dirs = _prepare_dirs(output_root=output_root, overwrite=bool(dcfg["overwrite"]))

    seed = int(dcfg["seed"])
    rng = np.random.default_rng(seed)
    modalities = [str(m) for m in dcfg["modalities"]]
    image_cfg = cfg["image_synthesis"]

    manifest_rows: List[Dict] = []
    patient_ids: List[str] = []
    counter = 1

    tiers = cfg["tiers"]
    for tier_name in ("A", "B", "C"):
        tcfg = tiers.get(tier_name, {})
        if not bool(tcfg.get("enabled", False)):
            continue
        n_patients = int(tcfg.get("n_patients", 0))
        for _ in range(n_patients):
            patient_id = f"{dcfg['patient_id_prefix']}-{tier_name}-{counter:05d}"
            counter += 1
            sample = simulate_patient(
                rng=rng,
                patient_id=patient_id,
                tier=tier_name,
                cfg=cfg,
            )

            # Build session images.
            concentration = sample["concentration"]  # [S,H,W,D]
            n_sessions = concentration.shape[0]
            session_images = []
            for s in range(n_sessions):
                img = make_session_modalities(
                    concentration=concentration[s],
                    brain_mask=sample["brain_mask"],
                    tissue_maps=sample["tissues"],
                    image_cfg=image_cfg,
                    modalities=modalities,
                    rng=rng,
                )
                session_images.append(img)
            image = np.stack(session_images, axis=0).astype(np.float32)  # [S,C,H,W,D]

            # Save patient tensors in a consistent format.
            np.save(dirs["patients"] / f"{patient_id}_image.npy", image)
            np.save(dirs["patients"] / f"{patient_id}_label.npy", sample["labels"].astype(np.uint8))
            np.save(dirs["patients"] / f"{patient_id}_days.npy", sample["days"].astype(np.float32))
            np.save(dirs["patients"] / f"{patient_id}_treatment.npy", sample["treatment"].astype(np.float32))
            np.save(dirs["patients"] / f"{patient_id}_brainmask.npy", sample["brain_mask"].astype(np.float32))
            if bool(dcfg.get("save_concentration", True)):
                np.save(dirs["patients"] / f"{patient_id}_concentration.npy", concentration.astype(np.float32))

            meta = {
                "patient_id": patient_id,
                "tier": tier_name,
                "n_sessions": int(n_sessions),
                "days_total": float(sample["days"][-1]),
                "treatment_on_any": bool(np.any(sample["treatment"] > 0)),
                "modalities": modalities,
                "volume_shape": list(image.shape[-3:]),
                "sim_meta": sample["meta"],
            }
            with (dirs["metadata"] / f"{patient_id}.json").open("w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            patient_ids.append(patient_id)
            manifest_rows.append(
                {
                    "patient_id": patient_id,
                    "tier": tier_name,
                    "n_sessions": int(n_sessions),
                    "days_total": float(sample["days"][-1]),
                    "treatment_on_any": int(np.any(sample["treatment"] > 0)),
                    "rho": sample["meta"].get("rho"),
                    "Dw": sample["meta"].get("Dw"),
                    "treatment_effect": sample["meta"].get("treatment_effect"),
                    "mode": sample["meta"].get("mode"),
                }
            )

    split_map = _assign_splits(patient_ids, cfg["split"])
    for row in manifest_rows:
        row["split"] = split_map[row["patient_id"]]

    _write_manifest_csv(dirs["manifests"] / "manifest.csv", manifest_rows)
    with (dirs["manifests"] / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for r in manifest_rows:
            f.write(json.dumps(r) + "\n")

    split_payload = {
        "train": [pid for pid in patient_ids if split_map[pid] == "train"],
        "val": [pid for pid in patient_ids if split_map[pid] == "val"],
        "test": [pid for pid in patient_ids if split_map[pid] == "test"],
    }
    with (dirs["splits"] / "splits.json").open("w", encoding="utf-8") as f:
        json.dump(split_payload, f, indent=2)

    info = {
        "dataset_name": dcfg["name"],
        "created_at": datetime.utcnow().isoformat() + "Z",
        "seed": seed,
        "output_root": str(output_root),
        "n_patients": len(patient_ids),
        "n_modalities": len(modalities),
        "modalities": modalities,
        "tiers": {k: {"enabled": bool(v.get("enabled", False)), "n_patients": int(v.get("n_patients", 0))} for k, v in cfg["tiers"].items()},
        "format": {
            "image": "[S,C,H,W,D] float32",
            "label": "[S,1,H,W,D] uint8",
            "days": "[S] float32 (absolute days from baseline)",
            "treatment": "[S] float32 (0/1)",
        },
    }
    with (dirs["root"] / "dataset_info.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    return {
        "output_root": str(output_root),
        "n_patients": len(patient_ids),
        "manifest_csv": str(dirs["manifests"] / "manifest.csv"),
        "splits_json": str(dirs["splits"] / "splits.json"),
    }

