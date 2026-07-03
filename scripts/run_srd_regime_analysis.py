#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_step(cmd: list[str], label: str, env: dict[str, str] | None = None) -> None:
    print(f"\n[RUN] {label}")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_report(path: Path, payload: dict) -> None:
    lines = [
        "# SRD Regime Analysis Bundle",
        "",
        "## Inputs",
        "",
        f"- Dataset root: `{payload['dataset_root']}`",
        f"- Dataset name: `{payload['dataset_name']}`",
        f"- Baseline output dir: `{payload['baseline_output_dir']}`",
        f"- Baseline method: `{payload['baseline_method']}`",
        f"- Target method: `{payload['target_method']}`",
        "",
        "## Outputs",
        "",
        f"- Audit: `{payload['outputs']['audit_dir']}`",
        f"- Regime analysis: `{payload['outputs']['regime_analysis_dir']}`",
        f"- Case types: `{payload['outputs']['case_type_dir']}`",
        f"- Morphology/treatment: `{payload['outputs']['morphology_treatment_dir']}`",
        f"- Exception cases: `{payload['outputs']['exception_case_dir']}`",
        f"- Figures: `{payload['outputs']['figure_dir']}`",
        "",
        "## Notes",
        "",
        "- This bundle is intended to freeze one complete SRD-facing analysis pass.",
        "- The goal is to keep the final synthetic analysis reproducible and consolidated rather than scattered across ad hoc outputs.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a consolidated SRD regime-analysis workflow: audit, pairwise driver analysis, case typing, morphology/treatment summaries, and figure export."
    )
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="SRD")
    parser.add_argument(
        "--dataset_kind",
        type=str,
        choices=["auto", "synthetic_benchmark", "plain_npy"],
        default="auto",
    )
    parser.add_argument("--baseline_output_dir", type=str, required=True)
    parser.add_argument("--baseline_method", type=str, default="locf")
    parser.add_argument("--target_method", type=str, default="resunet_image_mask")
    parser.add_argument("--real_tier_name", type=str, default="REAL")
    parser.add_argument("--gap_margin", type=float, default=0.05)
    parser.add_argument("--high_dice", type=float, default=0.85)
    parser.add_argument("--low_dice", type=float, default=0.70)
    parser.add_argument("--skip_descriptor_probe", action="store_true")
    parser.add_argument("--skip_regime_map", action="store_true")
    parser.add_argument("--skip_exception_cases", action="store_true")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    mplconfig = out_root / ".mplconfig"
    mplconfig.mkdir(parents=True, exist_ok=True)
    common_env = dict(os.environ)
    common_env["MPLCONFIGDIR"] = str(mplconfig.resolve())

    audit_dir = out_root / "audit"
    regime_dir = out_root / "regime_analysis"
    case_dir = out_root / "case_types"
    morph_dir = out_root / "morphology_treatment"
    descriptor_dir = out_root / "descriptor_signal"
    regime_map_dir = out_root / "regime_map"
    exception_dir = out_root / "exception_cases"
    fig_dir = out_root / "figures"

    python = sys.executable

    run_step(
        [
            python,
            str(ROOT / "scripts" / "run_data_audit.py"),
            "--dataset_root",
            args.dataset_root,
            "--dataset_name",
            args.dataset_name,
            "--dataset_kind",
            args.dataset_kind,
            "--real_tier_name",
            args.real_tier_name,
            "--output_dir",
            str(audit_dir),
        ],
        "dataset audit",
        env=common_env,
    )

    run_step(
        [
            python,
            str(ROOT / "scripts" / "analyze_regime_drivers.py"),
            "--dataset_root",
            args.dataset_root,
            "--audit_root",
            str(audit_dir),
            "--baseline_output_dir",
            args.baseline_output_dir,
            "--baseline_method",
            args.baseline_method,
            "--target_method",
            args.target_method,
            "--output_dir",
            str(regime_dir),
        ],
        "pairwise regime-driver analysis",
        env=common_env,
    )

    pairwise_csv = regime_dir / f"{args.baseline_method}_vs_{args.target_method}_pairwise.csv"
    run_step(
        [
            python,
            str(ROOT / "scripts" / "analyze_case_types.py"),
            "--pairwise_csv",
            str(pairwise_csv),
            "--output_dir",
            str(case_dir),
            "--gap_margin",
            str(args.gap_margin),
            "--high_dice",
            str(args.high_dice),
            "--low_dice",
            str(args.low_dice),
        ],
        "case-type analysis",
        env=common_env,
    )

    run_step(
        [
            python,
            str(ROOT / "scripts" / "analyze_morphology_treatment.py"),
            "--case_type_csv",
            str(case_dir / "case_type_samples.csv"),
            "--output_dir",
            str(morph_dir),
        ],
        "morphology and treatment analysis",
        env=common_env,
    )

    if not args.skip_descriptor_probe:
        run_step(
            [
                python,
                str(ROOT / "scripts" / "analyze_descriptor_signal.py"),
                "--case_type_csv",
                str(case_dir / "case_type_samples.csv"),
                "--output_dir",
                str(descriptor_dir),
            ],
            "descriptor signal probe",
            env=common_env,
        )

    if not args.skip_regime_map:
        run_step(
            [
                python,
                str(ROOT / "scripts" / "analyze_regime_map.py"),
                "--case_type_csv",
                str(case_dir / "case_type_samples.csv"),
                "--output_dir",
                str(regime_map_dir),
            ],
            "two-axis regime map",
            env=common_env,
        )

    if not args.skip_exception_cases:
        run_step(
            [
                python,
                str(ROOT / "scripts" / "analyze_exception_cases.py"),
                "--case_type_csv",
                str(case_dir / "case_type_samples.csv"),
                "--output_dir",
                str(exception_dir),
            ],
            "exception-case audit",
            env=common_env,
        )

    run_step(
        [
            python,
            str(ROOT / "scripts" / "export_regime_figures.py"),
            "--analysis_root",
            str(regime_dir),
            "--baseline_output_dir",
            args.baseline_output_dir,
            "--output_dir",
            str(fig_dir),
        ],
        "figure export",
        env=common_env,
    )

    summary = {
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "dataset_name": args.dataset_name,
        "dataset_kind": args.dataset_kind,
        "baseline_output_dir": str(Path(args.baseline_output_dir).resolve()),
        "baseline_method": args.baseline_method,
        "target_method": args.target_method,
        "gap_margin": args.gap_margin,
        "high_dice": args.high_dice,
        "low_dice": args.low_dice,
        "outputs": {
            "audit_dir": str(audit_dir.resolve()),
            "regime_analysis_dir": str(regime_dir.resolve()),
            "case_type_dir": str(case_dir.resolve()),
            "morphology_treatment_dir": str(morph_dir.resolve()),
            "descriptor_signal_dir": str(descriptor_dir.resolve()) if not args.skip_descriptor_probe else None,
            "regime_map_dir": str(regime_map_dir.resolve()) if not args.skip_regime_map else None,
            "exception_case_dir": str(exception_dir.resolve()) if not args.skip_exception_cases else None,
            "figure_dir": str(fig_dir.resolve()),
        },
    }

    write_manifest(out_root / "analysis_bundle_summary.json", summary)
    write_report(out_root / "analysis_bundle_report.md", summary)
    print("\n[OK] SRD regime-analysis bundle complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
