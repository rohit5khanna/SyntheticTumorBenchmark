#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_per_sample_tables(output_dir: Path) -> pd.DataFrame:
    files = {
        "locf": output_dir / "locf_per_sample.json",
        "unet_mask": output_dir / "unet_mask_per_sample.json",
        "unet_image_mask": output_dir / "unet_image_mask_per_sample.json",
        "resunet_image_mask": output_dir / "resunet_image_mask_per_sample.json",
        "plain_cnn_image_mask": output_dir / "plain_cnn_image_mask_per_sample.json",
    }

    rows = []
    for method, path in files.items():
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        for row in payload:
            out = dict(row)
            out["method"] = method
            rows.append(out)

    if not rows:
        raise FileNotFoundError(f"No per-sample baseline JSON files found in {output_dir}")
    return pd.DataFrame(rows)


def _rename_with_prefix(df: pd.DataFrame, prefix: str, key_cols: list[str]) -> pd.DataFrame:
    rename = {c: f"{prefix}{c}" for c in df.columns if c not in key_cols}
    return df.rename(columns=rename)


def add_feature_bins(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def qbin(series: pd.Series, labels: list[str]) -> pd.Series:
        valid = series.dropna()
        if valid.nunique() < len(labels):
            return pd.Series(["all"] * len(series), index=series.index)
        return pd.qcut(series, q=len(labels), labels=labels, duplicates="drop")

    out["input_volume_bin"] = qbin(out["input_volume_vox"], ["small", "medium", "large"])
    out["future_growth_bin"] = qbin(out["future_relative_growth"], ["low", "medium", "high"])
    out["delta_volume_bin"] = qbin(out["future_delta_volume_vox"], ["small", "medium", "large"])
    out["recent_growth_bin"] = qbin(out["recent_relative_growth"], ["low", "medium", "high"])
    out["delta_days_bin"] = qbin(out["delta_days"], ["short", "medium", "long"])
    return out


def enrich_with_audit_features(df: pd.DataFrame, audit_root: Path, dataset_root: Path) -> pd.DataFrame:
    sessions = pd.read_csv(audit_root / "sessions.csv")
    patients = pd.read_csv(audit_root / "patients.csv")
    manifest = pd.read_csv(dataset_root / "manifests" / "manifest.csv")

    # Ensure tier comes from manifest used in baseline generation.
    tier_map = manifest[["patient_id", "tier", "split"]].drop_duplicates()
    out = df.merge(tier_map, on="patient_id", how="left", suffixes=("", "_manifest"))

    input_sessions = sessions.rename(columns={"session_idx": "input_idx"})
    out = out.merge(
        _rename_with_prefix(
            input_sessions[
                [
                    "patient_id",
                    "input_idx",
                    "day",
                    "treatment",
                    "volume_vox",
                    "elongation_ratio",
                    "compactness_proxy",
                    "connected_component_count",
                    "bbox_x",
                    "bbox_y",
                    "bbox_z",
                ]
            ],
            "input_",
            ["patient_id", "input_idx"],
        ),
        on=["patient_id", "input_idx"],
        how="left",
    )

    target_sessions = sessions.rename(columns={"session_idx": "target_idx"})
    out = out.merge(
        _rename_with_prefix(
            target_sessions[
                [
                    "patient_id",
                    "target_idx",
                    "day",
                    "treatment",
                    "volume_vox",
                    "elongation_ratio",
                    "compactness_proxy",
                    "connected_component_count",
                    "bbox_x",
                    "bbox_y",
                    "bbox_z",
                ]
            ],
            "target_",
            ["patient_id", "target_idx"],
        ),
        on=["patient_id", "target_idx"],
        how="left",
    )

    # Previous session for recent trend at forecast start.
    prev_rows = sessions.copy()
    prev_rows["input_idx"] = prev_rows["session_idx"] + 1
    prev_rows = prev_rows[
        ["patient_id", "input_idx", "day", "volume_vox", "treatment"]
    ]
    prev_rows = prev_rows.rename(
        columns={
            "day": "prev_day",
            "volume_vox": "prev_volume_vox",
            "treatment": "prev_treatment",
        }
    )
    out = out.merge(prev_rows, on=["patient_id", "input_idx"], how="left")

    out["future_delta_volume_vox"] = out["target_volume_vox"] - out["input_volume_vox"]
    out["future_relative_growth"] = out["future_delta_volume_vox"] / out["input_volume_vox"].clip(lower=1.0)
    out["recent_delta_days"] = out["input_day"] - out["prev_day"]
    out["recent_delta_volume_vox"] = out["input_volume_vox"] - out["prev_volume_vox"]
    out["recent_relative_growth"] = out["recent_delta_volume_vox"] / out["prev_volume_vox"].clip(lower=1.0)

    # Treatment / patient-level helper columns.
    out = out.merge(
        patients[
            [
                "patient_id",
                "n_sessions",
                "followup_days",
                "treatment_on_any",
                "treatment_start_session",
                "mean_interval_days",
            ]
        ],
        on="patient_id",
        how="left",
    )
    out["treated_at_input"] = (out["input_treatment"].fillna(0) > 0).astype(int)
    out["treated_at_target"] = (out["target_treatment"].fillna(0) > 0).astype(int)
    out["treatment_started_before_input"] = (
        out["treatment_start_session"].fillna(-1) >= 0
    ) & (out["treatment_start_session"].fillna(-1) <= out["input_idx"])
    out["treatment_started_before_input"] = out["treatment_started_before_input"].astype(int)

    return add_feature_bins(out)


def summarize_group(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if not group_cols:
        return pd.DataFrame(
            [
                {
                    "count": int(len(df)),
                    "win_rate": float(df["win_flag"].mean()),
                    "mean_baseline_dice": float(df["baseline_dice"].mean()),
                    "mean_target_dice": float(df["target_dice"].mean()),
                    "mean_gap": float(df["dice_gap"].mean()),
                    "median_gap": float(df["dice_gap"].median()),
                }
            ]
        )
    return (
        df.groupby(group_cols)
        .agg(
            count=("win_flag", "size"),
            win_rate=("win_flag", "mean"),
            mean_baseline_dice=("baseline_dice", "mean"),
            mean_target_dice=("target_dice", "mean"),
            mean_gap=("dice_gap", "mean"),
            median_gap=("dice_gap", "median"),
        )
        .reset_index()
        .sort_values(group_cols)
    )


def build_pairwise_comparison(samples: pd.DataFrame, baseline_method: str, target_method: str) -> pd.DataFrame:
    key_cols = ["patient_id", "input_idx", "target_idx", "horizon", "delta_days", "tier", "split"]
    feature_cols = [
        "input_day",
        "input_treatment",
        "input_volume_vox",
        "input_elongation_ratio",
        "input_compactness_proxy",
        "input_bbox_x",
        "input_bbox_y",
        "input_bbox_z",
        "input_connected_component_count",
        "target_day",
        "target_treatment",
        "target_volume_vox",
        "target_elongation_ratio",
        "target_compactness_proxy",
        "target_bbox_x",
        "target_bbox_y",
        "target_bbox_z",
        "target_connected_component_count",
        "future_delta_volume_vox",
        "future_relative_growth",
        "prev_day",
        "prev_volume_vox",
        "recent_delta_days",
        "recent_delta_volume_vox",
        "recent_relative_growth",
        "n_sessions",
        "followup_days",
        "treatment_on_any",
        "treatment_start_session",
        "mean_interval_days",
        "treated_at_input",
        "treated_at_target",
        "treatment_started_before_input",
        "input_volume_bin",
        "future_growth_bin",
        "delta_volume_bin",
        "recent_growth_bin",
        "delta_days_bin",
    ]
    all_cols = key_cols + [c for c in feature_cols if c in samples.columns]

    base = samples[samples["method"] == baseline_method][all_cols + ["dice"]].rename(columns={"dice": "baseline_dice"})
    target = samples[samples["method"] == target_method][key_cols + ["dice"]].rename(columns={"dice": "target_dice"})

    pair = base.merge(target, on=key_cols, how="inner")
    pair["dice_gap"] = pair["target_dice"] - pair["baseline_dice"]
    pair["win_flag"] = (pair["dice_gap"] > 0).astype(int)
    pair["tie_flag"] = (pair["dice_gap"] == 0).astype(int)
    return pair.sort_values(["tier", "horizon", "dice_gap"], ascending=[True, True, False])


def write_report(path: Path, overall: pd.DataFrame, by_horizon: pd.DataFrame, by_tier: pd.DataFrame, by_tier_h: pd.DataFrame) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Regime Driver Report\n\n")
        f.write("## Overall\n\n")
        f.write(overall.to_markdown(index=False))
        f.write("\n\n## By Horizon\n\n")
        f.write(by_horizon.to_markdown(index=False))
        f.write("\n\n## By Tier\n\n")
        f.write(by_tier.to_markdown(index=False))
        f.write("\n\n## By Tier and Horizon\n\n")
        f.write(by_tier_h.to_markdown(index=False))
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze which tumor/trajectory regimes favor different forecasting methods.")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--audit_root", type=str, required=True)
    parser.add_argument("--baseline_output_dir", type=str, required=True)
    parser.add_argument("--baseline_method", type=str, default="locf")
    parser.add_argument("--target_method", type=str, default="resunet_image_mask")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    audit_root = Path(args.audit_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_per_sample_tables(Path(args.baseline_output_dir))
    enriched = enrich_with_audit_features(samples, audit_root=audit_root, dataset_root=dataset_root)
    pair = build_pairwise_comparison(enriched, baseline_method=args.baseline_method, target_method=args.target_method)

    overall = summarize_group(pair, [])
    by_horizon = summarize_group(pair, ["horizon"])
    by_tier = summarize_group(pair, ["tier"])
    by_tier_h = summarize_group(pair, ["tier", "horizon"])
    by_input_volume = summarize_group(pair, ["input_volume_bin"])
    by_future_growth = summarize_group(pair, ["future_growth_bin"])
    by_recent_growth = summarize_group(pair, ["recent_growth_bin"])
    by_treatment = summarize_group(pair, ["treated_at_input"])

    enriched.to_csv(output_dir / "all_methods_enriched_samples.csv", index=False)
    pair.to_csv(output_dir / f"{args.baseline_method}_vs_{args.target_method}_pairwise.csv", index=False)
    overall.to_csv(output_dir / "pairwise_overall.csv", index=False)
    by_horizon.to_csv(output_dir / "pairwise_by_horizon.csv", index=False)
    by_tier.to_csv(output_dir / "pairwise_by_tier.csv", index=False)
    by_tier_h.to_csv(output_dir / "pairwise_by_tier_horizon.csv", index=False)
    by_input_volume.to_csv(output_dir / "pairwise_by_input_volume_bin.csv", index=False)
    by_future_growth.to_csv(output_dir / "pairwise_by_future_growth_bin.csv", index=False)
    by_recent_growth.to_csv(output_dir / "pairwise_by_recent_growth_bin.csv", index=False)
    by_treatment.to_csv(output_dir / "pairwise_by_treated_at_input.csv", index=False)

    write_report(output_dir / "regime_driver_report.md", overall, by_horizon, by_tier, by_tier_h)

    summary = {
        "dataset_root": str(dataset_root.resolve()),
        "audit_root": str(audit_root.resolve()),
        "baseline_output_dir": str(Path(args.baseline_output_dir).resolve()),
        "baseline_method": args.baseline_method,
        "target_method": args.target_method,
        "n_all_method_rows": int(len(enriched)),
        "n_pairwise_rows": int(len(pair)),
        "outputs": {
            "all_methods_enriched_samples_csv": str((output_dir / "all_methods_enriched_samples.csv").resolve()),
            "pairwise_csv": str((output_dir / f"{args.baseline_method}_vs_{args.target_method}_pairwise.csv").resolve()),
            "report_md": str((output_dir / "regime_driver_report.md").resolve()),
        },
    }
    with (output_dir / "analysis_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
