#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.locf import run_locf_baseline
from baselines.unet import run_unet_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LOCF + UNet baselines.")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--fit_sessions", type=int, default=3)
    parser.add_argument("--horizons", type=str, default="1,2")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--eval_split", type=str, default="test")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--base_channels", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, default="outputs/baselines")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    summary["locf"] = run_locf_baseline(
        dataset_root=args.dataset_root,
        split=args.eval_split,
        fit_sessions=args.fit_sessions,
        horizons=args.horizons,
        output_dir=out_dir,
    )

    for mode in ("mask", "image_mask"):
        key = f"unet_{mode}"
        try:
            summary[key] = run_unet_baseline(
                dataset_root=args.dataset_root,
                train_split=args.train_split,
                eval_split=args.eval_split,
                fit_sessions=args.fit_sessions,
                horizons=args.horizons,
                input_mode=mode,
                output_dir=out_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                num_workers=args.num_workers,
                base_channels=args.base_channels,
                seed=args.seed,
                device=args.device,
            )
        except RuntimeError as e:
            summary[key] = {"status": "skipped", "reason": str(e)}

    with (out_dir / "all_baselines_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
