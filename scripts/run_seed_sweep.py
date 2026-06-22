#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.locf import run_locf_baseline
from baselines.unet import run_unet_baseline


def parse_seeds(seed_text: str) -> list[int]:
    seeds = [int(x.strip()) for x in seed_text.split(",") if x.strip()]
    if not seeds:
        raise ValueError("Need at least one seed.")
    return seeds


def build_seed_dirs(output_root: Path, seed: int) -> Path:
    out_dir = output_root / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def aggregate_seed_metrics(seed_summaries: list[dict], key: str) -> dict:
    values = [float(s[key]) for s in seed_summaries]
    return {
        "count": len(values),
        "mean": mean(values),
        "std": 0.0 if len(values) == 1 else pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a multi-seed baseline sweep.")
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
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--seeds",
        type=str,
        default="42",
        help="Comma-separated seed list, e.g. '7,21,42,123,999'.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Directory where per-seed runs and aggregate summaries will be stored.",
    )
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # LOCF is deterministic for a frozen dataset, so compute it once and reuse it.
    locf_dir = output_root / "shared_locf"
    locf_dir.mkdir(parents=True, exist_ok=True)
    locf_summary = run_locf_baseline(
        dataset_root=args.dataset_root,
        split=args.eval_split,
        fit_sessions=args.fit_sessions,
        horizons=args.horizons,
        output_dir=locf_dir,
    )

    sweep_rows = []
    per_seed_payload = {}

    for seed in seeds:
        seed_dir = build_seed_dirs(output_root, seed)
        seed_summary = {"locf": locf_summary}

        for mode in ("mask", "image_mask"):
            key = f"unet_{mode}"
            try:
                seed_summary[key] = run_unet_baseline(
                    dataset_root=args.dataset_root,
                    train_split=args.train_split,
                    eval_split=args.eval_split,
                    fit_sessions=args.fit_sessions,
                    horizons=args.horizons,
                    input_mode=mode,
                    output_dir=seed_dir,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    num_workers=args.num_workers,
                    base_channels=args.base_channels,
                    seed=seed,
                    device=args.device,
                )
            except RuntimeError as e:
                seed_summary[key] = {"status": "skipped", "reason": str(e), "seed": seed}

        with (seed_dir / "all_baselines_summary.json").open("w", encoding="utf-8") as f:
            json.dump(seed_summary, f, indent=2)

        per_seed_payload[str(seed)] = seed_summary

        row = {"seed": seed, "locf_mean_dice": float(locf_summary["mean_dice"])}
        for key in ("unet_mask", "unet_image_mask"):
            metric_key = "mean_eval_dice"
            row[f"{key}_mean_dice"] = (
                float(seed_summary[key][metric_key]) if metric_key in seed_summary[key] else None
            )
        sweep_rows.append(row)

    aggregate = {
        "dataset_root": args.dataset_root,
        "fit_sessions": int(args.fit_sessions),
        "horizons": args.horizons,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "seeds": seeds,
        "locf": locf_summary,
        "unet_mask": aggregate_seed_metrics(
            [per_seed_payload[str(seed)]["unet_mask"] for seed in seeds if "mean_eval_dice" in per_seed_payload[str(seed)]["unet_mask"]],
            "mean_eval_dice",
        ),
        "unet_image_mask": aggregate_seed_metrics(
            [per_seed_payload[str(seed)]["unet_image_mask"] for seed in seeds if "mean_eval_dice" in per_seed_payload[str(seed)]["unet_image_mask"]],
            "mean_eval_dice",
        ),
        "per_seed_table": sweep_rows,
    }

    with (output_root / "seed_sweep_summary.json").open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)
    with (output_root / "per_seed_summaries.json").open("w", encoding="utf-8") as f:
        json.dump(per_seed_payload, f, indent=2)

    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
