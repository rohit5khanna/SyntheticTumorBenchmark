#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.locf import run_locf_baseline
from baselines.tasks import parse_tiers
from baselines.unet import run_unet_baseline


def parse_csv_list(value: str | None) -> List[str]:
    if value is None:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run cross-regime transfer experiments for longitudinal tumor forecasting models."
    )
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--eval_split", type=str, default="test")
    parser.add_argument("--fit_sessions", type=int, default=3)
    parser.add_argument("--horizons", type=str, default="1,2,3")
    parser.add_argument("--input_mode", type=str, choices=["mask", "image_mask"], default="image_mask")
    parser.add_argument(
        "--models",
        type=str,
        default="resunet,unetr",
        help="Comma-separated model variants from {unet,resunet,plain_cnn,unetr}.",
    )
    parser.add_argument(
        "--train_tiers",
        type=str,
        default="A,B,C",
        help="Comma-separated training tiers to sweep.",
    )
    parser.add_argument(
        "--eval_tiers",
        type=str,
        default="A,B,C",
        help="Comma-separated evaluation tiers to sweep.",
    )
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--base_channels", type=int, default=12)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--include_locf",
        action="store_true",
        help="Also run LOCF once per evaluation tier as a reference.",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    dataset_root = args.dataset_root
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = parse_csv_list(args.models)
    train_tiers = parse_tiers(args.train_tiers)
    eval_tiers = parse_tiers(args.eval_tiers)
    if train_tiers is None or eval_tiers is None:
        raise ValueError("train_tiers and eval_tiers must each contain at least one tier.")
    seeds = [int(x) for x in parse_csv_list(args.seeds)]
    if not seeds:
        raise ValueError("Need at least one seed.")

    overall_rows: List[Dict] = []
    locf_rows: List[Dict] = []
    failures: List[Dict] = []

    if args.include_locf:
        for eval_tier in eval_tiers:
            locf_dir = out_dir / f"locf_eval_{eval_tier}"
            try:
                locf_summary = run_locf_baseline(
                    dataset_root=dataset_root,
                    split=args.eval_split,
                    fit_sessions=args.fit_sessions,
                    horizons=args.horizons,
                    output_dir=locf_dir,
                    allowed_tiers=[eval_tier],
                )
                locf_rows.append(
                    {
                        "model_variant": "locf",
                        "input_mode": "mask_copy",
                        "train_tier": "NA",
                        "eval_tier": eval_tier,
                        "seed": "NA",
                        "mean_eval_dice": locf_summary["mean_dice"],
                        "std_eval_dice": locf_summary["std_dice"],
                        "n_eval_samples": locf_summary["n_samples"],
                        "status": "ok",
                        "run_dir": str(locf_dir.resolve()),
                    }
                )
            except Exception as e:
                failures.append(
                    {
                        "model_variant": "locf",
                        "train_tier": "NA",
                        "eval_tier": eval_tier,
                        "seed": "NA",
                        "status": "failed",
                        "reason": str(e),
                    }
                )

    for model_variant in models:
        for train_tier in train_tiers:
            for eval_tier in eval_tiers:
                for seed in seeds:
                    run_name = f"{model_variant}_{args.input_mode}_train{train_tier}_eval{eval_tier}_s{seed}"
                    run_dir = out_dir / run_name
                    try:
                        summary = run_unet_baseline(
                            dataset_root=dataset_root,
                            train_split=args.train_split,
                            eval_split=args.eval_split,
                            fit_sessions=args.fit_sessions,
                            horizons=args.horizons,
                            input_mode=args.input_mode,
                            output_dir=run_dir,
                            train_tiers=[train_tier],
                            eval_tiers=[eval_tier],
                            model_variant=model_variant,
                            epochs=args.epochs,
                            batch_size=args.batch_size,
                            learning_rate=args.learning_rate,
                            num_workers=args.num_workers,
                            base_channels=args.base_channels,
                            seed=seed,
                            device=args.device,
                        )
                        overall_rows.append(
                            {
                                "model_variant": model_variant,
                                "input_mode": args.input_mode,
                                "train_tier": train_tier,
                                "eval_tier": eval_tier,
                                "seed": seed,
                                "mean_eval_dice": summary["mean_eval_dice"],
                                "std_eval_dice": summary["std_eval_dice"],
                                "n_train_samples": summary["n_train_samples"],
                                "n_eval_samples": summary["n_eval_samples"],
                                "epochs": summary["epochs"],
                                "status": "ok",
                                "run_dir": str(run_dir.resolve()),
                            }
                        )
                    except Exception as e:
                        failures.append(
                            {
                                "model_variant": model_variant,
                                "train_tier": train_tier,
                                "eval_tier": eval_tier,
                                "seed": seed,
                                "status": "failed",
                                "reason": str(e),
                            }
                        )

    payload = {
        "dataset_root": str(Path(dataset_root).resolve()),
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "fit_sessions": args.fit_sessions,
        "horizons": args.horizons,
        "input_mode": args.input_mode,
        "models": models,
        "train_tiers": train_tiers,
        "eval_tiers": eval_tiers,
        "seeds": seeds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "overall_rows": overall_rows,
        "locf_rows": locf_rows,
        "failures": failures,
    }

    with (out_dir / "cross_regime_transfer_summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    write_csv(out_dir / "cross_regime_transfer_overall.csv", overall_rows)
    write_csv(out_dir / "cross_regime_transfer_locf.csv", locf_rows)
    write_csv(out_dir / "cross_regime_transfer_failures.csv", failures)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
