#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.tasks import list_patient_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Create train/val/test splits for a plain NPY longitudinal dataset.")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--train_frac", type=float, default=0.70)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--test_frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frac_sum = args.train_frac + args.val_frac + args.test_frac
    if abs(frac_sum - 1.0) > 1e-6:
        raise ValueError(f"Fractions must sum to 1.0, got {frac_sum:.6f}")

    root = Path(args.dataset_root)
    patient_ids = list_patient_ids(root)
    if not patient_ids:
        raise ValueError(f"No patient *_label.npy files found in {root}")

    rng = random.Random(args.seed)
    ids = patient_ids[:]
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(round(n * args.train_frac))
    n_val = int(round(n * args.val_frac))
    n_train = min(max(n_train, 1), n)
    n_val = min(max(n_val, 0), max(0, n - n_train))
    n_test = n - n_train - n_val
    if n_test <= 0 and n >= 3:
        n_test = 1
        if n_val > 0:
            n_val -= 1
        else:
            n_train -= 1

    splits = {
        "train": ids[:n_train],
        "val": ids[n_train : n_train + n_val],
        "test": ids[n_train + n_val :],
    }

    split_dir = root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    out_path = split_dir / "splits.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    print(json.dumps({"dataset_root": str(root.resolve()), "n_patients": n, "splits": {k: len(v) for k, v in splits.items()}, "output": str(out_path.resolve())}, indent=2))


if __name__ == "__main__":
    main()
