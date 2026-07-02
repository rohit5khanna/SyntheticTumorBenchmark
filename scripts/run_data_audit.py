#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.audit import build_audit_tables, detect_dataset_kind, summarize_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8") as f:
            f.write("")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dataset audit for SyntheticTumorBenchmark or TaDiff-style NPY data.")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--dataset_kind", type=str, choices=["auto", "synthetic_benchmark", "plain_npy"], default="auto")
    parser.add_argument("--real_tier_name", type=str, default="REAL")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    kind = detect_dataset_kind(args.dataset_root) if args.dataset_kind == "auto" else args.dataset_kind
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = build_audit_tables(
        dataset_root=args.dataset_root,
        dataset_name=args.dataset_name,
        kind=kind,
        real_tier_name=args.real_tier_name,
    )

    patient_rows = tables["patients"]
    session_rows = tables["sessions"]
    transition_rows = tables["transitions"]

    patient_summary_tier = summarize_rows(
        patient_rows,
        value_keys=["n_sessions", "followup_days", "treatment_on_any", "mean_interval_days"],
        group_keys=["dataset_name", "tier", "split"],
    )
    session_summary_tier = summarize_rows(
        session_rows,
        value_keys=[
            "volume_vox",
            "elongation_ratio",
            "compactness_proxy",
            "connected_component_count",
            "bbox_x",
            "bbox_y",
            "bbox_z",
        ],
        group_keys=["dataset_name", "tier", "split"],
    )
    transition_summary_tier = summarize_rows(
        transition_rows,
        value_keys=["delta_days", "delta_volume_vox", "relative_growth_rate"],
        group_keys=["dataset_name", "tier", "split"],
    )

    patient_summary_overall = summarize_rows(
        patient_rows,
        value_keys=["n_sessions", "followup_days", "treatment_on_any", "mean_interval_days"],
        group_keys=["dataset_name"],
    )
    session_summary_overall = summarize_rows(
        session_rows,
        value_keys=[
            "volume_vox",
            "elongation_ratio",
            "compactness_proxy",
            "connected_component_count",
            "bbox_x",
            "bbox_y",
            "bbox_z",
        ],
        group_keys=["dataset_name"],
    )
    transition_summary_overall = summarize_rows(
        transition_rows,
        value_keys=["delta_days", "delta_volume_vox", "relative_growth_rate"],
        group_keys=["dataset_name"],
    )

    write_csv(out_dir / "patients.csv", patient_rows)
    write_csv(out_dir / "sessions.csv", session_rows)
    write_csv(out_dir / "transitions.csv", transition_rows)
    write_csv(out_dir / "patient_summary_by_tier.csv", patient_summary_tier)
    write_csv(out_dir / "session_summary_by_tier.csv", session_summary_tier)
    write_csv(out_dir / "transition_summary_by_tier.csv", transition_summary_tier)
    write_csv(out_dir / "patient_summary_overall.csv", patient_summary_overall)
    write_csv(out_dir / "session_summary_overall.csv", session_summary_overall)
    write_csv(out_dir / "transition_summary_overall.csv", transition_summary_overall)

    payload = {
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "dataset_name": args.dataset_name,
        "dataset_kind": kind,
        "n_patients": len(patient_rows),
        "n_sessions": len(session_rows),
        "n_transitions": len(transition_rows),
        "outputs": {
            "patients_csv": str((out_dir / "patients.csv").resolve()),
            "sessions_csv": str((out_dir / "sessions.csv").resolve()),
            "transitions_csv": str((out_dir / "transitions.csv").resolve()),
            "patient_summary_by_tier_csv": str((out_dir / "patient_summary_by_tier.csv").resolve()),
            "session_summary_by_tier_csv": str((out_dir / "session_summary_by_tier.csv").resolve()),
            "transition_summary_by_tier_csv": str((out_dir / "transition_summary_by_tier.csv").resolve()),
        },
    }
    write_json(out_dir / "audit_summary.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
