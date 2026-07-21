#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_SCRIPT = ROOT / "scripts" / "analyze_forecast_origin_predictability.py"

FEATURE_SETS: Dict[str, List[str]] = {
    "full_origin": [
        "log_delta_days",
        "log_input_span_days",
        "log_input_volume_vox",
        "current_treatment",
        "treatment_changed_in_input",
        "log_previous_growth_volume_vox",
        "log_previous_loss_volume_vox",
        "previous_growth_ratio",
    ],
    "no_interval": [
        "log_input_volume_vox",
        "current_treatment",
        "treatment_changed_in_input",
        "log_previous_growth_volume_vox",
        "log_previous_loss_volume_vox",
        "previous_growth_ratio",
    ],
    "time_only": [
        "log_delta_days",
        "log_input_span_days",
    ],
    "history_only": [
        "log_input_volume_vox",
        "log_previous_growth_volume_vox",
        "log_previous_loss_volume_vox",
        "previous_growth_ratio",
    ],
    "treatment_only": [
        "current_treatment",
        "treatment_changed_in_input",
    ],
}


def parse_csv(payload: str | None) -> List[str]:
    if payload is None:
        return []
    return [x.strip() for x in payload.split(",") if x.strip()]


def run_feature_set(args: argparse.Namespace, name: str, features: List[str], out_dir: Path) -> None:
    cmd = [
        sys.executable,
        str(ANALYSIS_SCRIPT),
        "--train_split",
        args.train_split,
        "--eval_splits",
        args.eval_splits,
        "--features",
        ",".join(features),
        "--targets",
        args.targets,
        "--growth_loss_threshold",
        str(args.growth_loss_threshold),
        "--distant_growth_threshold",
        str(args.distant_growth_threshold),
        "--high_burden_quantile",
        str(args.high_burden_quantile),
        "--high_change_rate_quantile",
        str(args.high_change_rate_quantile),
        "--locf_breakdown_threshold",
        str(args.locf_breakdown_threshold),
        "--n_bootstrap",
        str(args.n_bootstrap),
        "--seed",
        str(args.seed),
        "--output_dir",
        str(out_dir),
    ]
    if args.samples_csv:
        cmd.extend(["--samples_csv", args.samples_csv])
    if args.taxonomy_dir:
        cmd.extend(["--taxonomy_dir", args.taxonomy_dir])
    if args.manifest_csv:
        cmd.extend(["--manifest_csv", args.manifest_csv])
    if args.verbose:
        print("\n[RUN]", name)
        print(" ".join(cmd))
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if args.verbose and result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        raise subprocess.CalledProcessError(result.returncode, cmd)


def read_with_feature_set(path: Path, feature_set: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.insert(0, "feature_set", feature_set)
    return df


def collect_outputs(output_dir: Path, feature_sets: List[str]) -> Dict[str, pd.DataFrame]:
    collected: Dict[str, List[pd.DataFrame]] = {
        "summary": [],
        "bootstrap_summary": [],
        "feature_weights": [],
        "prevalence": [],
    }
    filenames = {
        "summary": "forecast_origin_predictability_summary.csv",
        "bootstrap_summary": "forecast_origin_predictability_patient_bootstrap_summary.csv",
        "feature_weights": "forecast_origin_predictability_feature_weights.csv",
        "prevalence": "forecast_origin_predictability_prevalence.csv",
    }
    for feature_set in feature_sets:
        subdir = output_dir / feature_set
        for key, filename in filenames.items():
            df = read_with_feature_set(subdir / filename, feature_set)
            if not df.empty:
                collected[key].append(df)
    return {key: pd.concat(parts, ignore_index=True) if parts else pd.DataFrame() for key, parts in collected.items()}


def build_delta_tables(summary: pd.DataFrame, baseline_feature_set: str) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    key_cols = ["target", "model", "split"]
    metrics = ["balanced_accuracy", "roc_auc", "average_precision", "precision", "recall", "false_positive_rate", "false_negative_rate"]
    baseline = summary[summary["feature_set"] == baseline_feature_set][key_cols + metrics].copy()
    baseline = baseline.rename(columns={m: f"{m}_{baseline_feature_set}" for m in metrics})
    merged = summary.merge(baseline, on=key_cols, how="left")
    for metric in metrics:
        merged[f"delta_{metric}_vs_{baseline_feature_set}"] = merged[metric] - merged[f"{metric}_{baseline_feature_set}"]
    return merged


def write_report(
    path: Path,
    feature_sets: Dict[str, List[str]],
    combined_summary: pd.DataFrame,
    delta_summary: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Forecast-Origin Feature Ablation Report\n\n")
        f.write(
            "This report compares whether transition-state predictability comes from interval information, "
            "input/history descriptors, treatment indicators, or the full origin-known feature set. "
            "It is designed to detect definitional coupling before we make claims about biological or temporal predictability.\n\n"
        )
        f.write("## Feature Sets\n\n")
        for name, features in feature_sets.items():
            f.write(f"- `{name}`: " + ", ".join(f"`{x}`" for x in features) + "\n")
        f.write("\n## Combined Summary\n\n")
        f.write(combined_summary.to_markdown(index=False) if not combined_summary.empty else "No summary rows.")
        if not delta_summary.empty:
            focus_cols = [
                "feature_set",
                "target",
                "model",
                "split",
                "roc_auc",
                "average_precision",
                "balanced_accuracy",
                "delta_roc_auc_vs_full_origin",
                "delta_average_precision_vs_full_origin",
                "delta_balanced_accuracy_vs_full_origin",
            ]
            cols = [c for c in focus_cols if c in delta_summary.columns]
            f.write("\n\n## Delta From Full-Origin Features\n\n")
            f.write(delta_summary[cols].to_markdown(index=False))
        if not bootstrap_summary.empty:
            f.write("\n\n## Patient Bootstrap Summary\n\n")
            cols = [
                "feature_set",
                "target",
                "model",
                "split",
                "balanced_accuracy_mean",
                "balanced_accuracy_ci_low",
                "balanced_accuracy_ci_high",
                "roc_auc_mean",
                "roc_auc_ci_low",
                "roc_auc_ci_high",
                "average_precision_mean",
                "average_precision_ci_low",
                "average_precision_ci_high",
            ]
            cols = [c for c in cols if c in bootstrap_summary.columns]
            f.write(bootstrap_summary[cols].to_markdown(index=False))
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run feature-group ablations for forecast-origin transition predictability.")
    parser.add_argument("--samples_csv", type=str, default=None)
    parser.add_argument("--taxonomy_dir", type=str, default=None)
    parser.add_argument("--manifest_csv", type=str, default=None)
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--eval_splits", type=str, default="val,test")
    parser.add_argument("--feature_sets", type=str, default="full_origin,no_interval,time_only,history_only,treatment_only")
    parser.add_argument("--targets", type=str, default="mixed_growth_loss,distant_growth_present,high_transition_burden,locf_breakdown,high_change_rate")
    parser.add_argument("--growth_loss_threshold", type=float, default=0.2)
    parser.add_argument("--distant_growth_threshold", type=float, default=0.2)
    parser.add_argument("--high_burden_quantile", type=float, default=0.75)
    parser.add_argument("--high_change_rate_quantile", type=float, default=0.75)
    parser.add_argument("--locf_breakdown_threshold", type=float, default=0.5)
    parser.add_argument("--n_bootstrap", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    selected = parse_csv(args.feature_sets)
    unknown = [name for name in selected if name not in FEATURE_SETS]
    if unknown:
        raise ValueError(f"Unknown feature set(s): {unknown}. Available: {sorted(FEATURE_SETS)}")
    if not args.samples_csv and not args.taxonomy_dir:
        raise ValueError("Provide either --samples_csv or --taxonomy_dir.")
    if not ANALYSIS_SCRIPT.exists():
        raise FileNotFoundError(f"Missing analysis script: {ANALYSIS_SCRIPT}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_features = {name: FEATURE_SETS[name] for name in selected}

    for name, features in selected_features.items():
        run_feature_set(args, name=name, features=features, out_dir=output_dir / name)

    outputs = collect_outputs(output_dir, selected)
    combined_summary = outputs["summary"]
    bootstrap_summary = outputs["bootstrap_summary"]
    feature_weights = outputs["feature_weights"]
    prevalence = outputs["prevalence"]
    delta_summary = build_delta_tables(combined_summary, baseline_feature_set="full_origin")

    combined_summary.to_csv(output_dir / "forecast_origin_feature_ablation_summary.csv", index=False)
    delta_summary.to_csv(output_dir / "forecast_origin_feature_ablation_delta_vs_full_origin.csv", index=False)
    bootstrap_summary.to_csv(output_dir / "forecast_origin_feature_ablation_patient_bootstrap_summary.csv", index=False)
    feature_weights.to_csv(output_dir / "forecast_origin_feature_ablation_feature_weights.csv", index=False)
    prevalence.to_csv(output_dir / "forecast_origin_feature_ablation_prevalence.csv", index=False)
    write_report(
        output_dir / "forecast_origin_feature_ablation_report.md",
        feature_sets=selected_features,
        combined_summary=combined_summary,
        delta_summary=delta_summary,
        bootstrap_summary=bootstrap_summary,
    )

    payload = {
        "samples_csv": args.samples_csv,
        "taxonomy_dir": args.taxonomy_dir,
        "manifest_csv": args.manifest_csv,
        "feature_sets": selected_features,
        "train_split": args.train_split,
        "eval_splits": parse_csv(args.eval_splits),
        "targets": parse_csv(args.targets),
        "n_bootstrap": int(args.n_bootstrap),
        "output_dir": str(output_dir),
        "outputs": {
            "summary_csv": str(output_dir / "forecast_origin_feature_ablation_summary.csv"),
            "delta_vs_full_origin_csv": str(output_dir / "forecast_origin_feature_ablation_delta_vs_full_origin.csv"),
            "patient_bootstrap_summary_csv": str(output_dir / "forecast_origin_feature_ablation_patient_bootstrap_summary.csv"),
            "feature_weights_csv": str(output_dir / "forecast_origin_feature_ablation_feature_weights.csv"),
            "prevalence_csv": str(output_dir / "forecast_origin_feature_ablation_prevalence.csv"),
            "report_md": str(output_dir / "forecast_origin_feature_ablation_report.md"),
        },
    }
    with (output_dir / "forecast_origin_feature_ablation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
