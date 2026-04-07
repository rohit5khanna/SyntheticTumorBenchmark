#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark import generate_benchmark_dataset, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Synthetic Tumor Benchmark dataset.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/benchmark_v1.yaml",
        help="Path to benchmark config YAML.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Optional override for dataset.output_root in config.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.output_root:
        cfg["dataset"]["output_root"] = args.output_root

    summary = generate_benchmark_dataset(cfg)
    print("[DONE] Dataset generation complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

