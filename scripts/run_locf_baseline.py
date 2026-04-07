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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LOCF baseline on SyntheticTumorBenchmark.")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--fit_sessions", type=int, default=3)
    parser.add_argument("--horizons", type=str, default="1,2")
    parser.add_argument("--output_dir", type=str, default="outputs/baselines")
    args = parser.parse_args()

    summary = run_locf_baseline(
        dataset_root=args.dataset_root,
        split=args.split,
        fit_sessions=args.fit_sessions,
        horizons=args.horizons,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
