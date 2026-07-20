#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_transition_taxonomy.py"


def parse_radii(payload: str) -> List[float]:
    vals = []
    for item in payload.split(","):
        item = item.strip()
        if item:
            vals.append(float(item))
    if not vals:
        raise ValueError("Need at least one radius.")
    return vals


def radius_label(radius: float) -> str:
    return f"r{radius:g}".replace(".", "p")


def read_with_radius(path: Path, radius: float) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.insert(0, "boundary_radius_vox", radius)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run transition taxonomy across boundary radii and collate summaries.")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--manifest_csv", type=str, default=None)
    parser.add_argument("--splits", type=str, default="train,val,test")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--fit_sessions", type=int, default=3)
    parser.add_argument("--horizons", type=str, default="1,2,3")
    parser.add_argument("--allowed_tiers", type=str, default=None)
    parser.add_argument("--interval_bins", type=str, default="0,30,60,90,180,365,inf")
    parser.add_argument("--radii", type=str, default="1,3,5,7")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--no_plots", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    radii = parse_radii(args.radii)

    run_dirs = []
    for radius in radii:
        run_dir = out_dir / radius_label(radius)
        cmd = [
            sys.executable,
            str(SCRIPT),
            "--dataset_root",
            args.dataset_root,
            "--spatial_mode",
            "recompute",
            "--boundary_radius_vox",
            str(radius),
            "--interval_bins",
            args.interval_bins,
            "--output_dir",
            str(run_dir),
        ]
        if args.manifest_csv:
            cmd.extend(["--manifest_csv", args.manifest_csv, "--splits", args.splits])
        else:
            cmd.extend([
                "--split",
                args.split,
                "--fit_sessions",
                str(args.fit_sessions),
                "--horizons",
                args.horizons,
            ])
            if args.allowed_tiers:
                cmd.extend(["--allowed_tiers", args.allowed_tiers])
        if args.no_plots:
            cmd.append("--no_plots")
        print("[RUN]", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        run_dirs.append(run_dir)

    combined = {}
    targets = {
        "overall": "transition_taxonomy_overall.csv",
        "by_split": "transition_taxonomy_by_split.csv",
        "by_horizon": "transition_taxonomy_by_horizon.csv",
        "by_net_direction": "transition_taxonomy_by_net_direction.csv",
        "by_transition_type": "transition_taxonomy_by_transition_type.csv",
        "by_relative_absolute_change_bin": "transition_taxonomy_by_relative_absolute_change_bin.csv",
    }
    for key, filename in targets.items():
        parts = [read_with_radius(run_dir / filename, radius) for run_dir, radius in zip(run_dirs, radii)]
        parts = [p for p in parts if not p.empty]
        if parts:
            df = pd.concat(parts, ignore_index=True)
            out_path = out_dir / f"radius_sensitivity_{key}.csv"
            df.to_csv(out_path, index=False)
            combined[key] = str(out_path)

    stability_rows = []
    for run_dir, radius in zip(run_dirs, radii):
        sample_path = run_dir / "transition_taxonomy_samples.csv"
        if not sample_path.exists():
            continue
        samples = pd.read_csv(sample_path)
        stability_rows.append(
            {
                "boundary_radius_vox": radius,
                "n_transitions": int(len(samples)),
                "n_patients": int(samples["patient_id"].nunique()),
                "mean_locf_dice": float(samples["locf_dice"].mean()),
                "mean_boundary_growth_fraction": float(samples["boundary_growth_fraction"].mean()),
                "mean_distant_growth_fraction": float(samples["distant_growth_fraction"].mean()),
                "distant_growth_rate": float(samples["has_distant_growth"].mean()),
                "mean_boundary_loss_fraction": float(samples["boundary_loss_fraction"].mean()),
                "mean_core_loss_fraction": float(samples["core_loss_fraction"].mean()),
                "core_loss_rate": float(samples["has_core_loss"].mean()),
                "n_transition_types": int(samples["transition_type"].nunique()),
            }
        )
    stability = pd.DataFrame(stability_rows)
    stability_path = out_dir / "radius_sensitivity_stability_summary.csv"
    stability.to_csv(stability_path, index=False)
    combined["stability_summary"] = str(stability_path)

    report_path = out_dir / "radius_sensitivity_report.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Transition Taxonomy Radius Sensitivity\n\n")
        f.write("This analysis reruns the spatial transition taxonomy over multiple boundary radii. ")
        f.write("It checks whether boundary/distant growth and boundary/core loss conclusions depend strongly on one radius choice.\n\n")
        f.write(f"Radii: `{args.radii}` voxels.\n\n")
        f.write("## Stability Summary\n\n")
        f.write(stability.to_markdown(index=False) if not stability.empty else "No rows.")
        f.write("\n\n")
        f.write("## Outputs\n\n")
        for name, path in combined.items():
            f.write(f"- {name}: `{path}`\n")

    payload = {
        "dataset_root": args.dataset_root,
        "manifest_csv": args.manifest_csv,
        "radii": radii,
        "output_dir": str(out_dir),
        "run_dirs": [str(p) for p in run_dirs],
        "outputs": {**combined, "report_md": str(report_path)},
    }
    with (out_dir / "radius_sensitivity_summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
