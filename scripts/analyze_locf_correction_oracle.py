#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.metrics import dice_np
from baselines.tasks import ForecastSample, infer_tier_from_patient_id, patient_paths
from scripts.run_growth_only_manifest_baseline import build_samples_from_manifest, normalize_manifest
from baselines.tasks import build_samples_for_split


def parse_csv(payload: str | None) -> List[str]:
    if payload is None:
        return []
    return [x.strip() for x in payload.split(",") if x.strip()]


def parse_float_csv(payload: str) -> List[float]:
    return [float(x.strip()) for x in payload.split(",") if x.strip()]


def standardize_label(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 5 and arr.shape[1] == 1:
        return arr > 0
    if arr.ndim == 4:
        return arr[:, None, ...] > 0
    raise ValueError(f"Unsupported label shape: {arr.shape}")


def load_samples(args: argparse.Namespace) -> List[ForecastSample]:
    if args.manifest_csv:
        manifest = normalize_manifest(pd.read_csv(args.manifest_csv))
        splits = set(parse_csv(args.splits))
        if splits:
            manifest = manifest[manifest["split"].isin(splits)].copy()
        samples: List[ForecastSample] = []
        for split in sorted(manifest["split"].dropna().unique()):
            samples.extend(build_samples_from_manifest(manifest, split))
        return samples

    samples = []
    for split in parse_csv(args.splits):
        samples.extend(
            build_samples_for_split(
                args.dataset_root,
                split=split,
                fit_sessions=args.fit_sessions,
                horizons=args.horizons,
                allowed_tiers=args.allowed_tiers,
            )
        )
    return samples


def sample_key(sample: ForecastSample) -> tuple:
    return (sample.patient_id, int(sample.input_idx), int(sample.target_idx), int(sample.horizon))


def top_true_subset(base_mask: np.ndarray, candidate_mask: np.ndarray, k: int, add: bool) -> np.ndarray:
    pred = base_mask.copy()
    idx = np.flatnonzero(candidate_mask.reshape(-1))
    if len(idx) == 0 or k <= 0:
        return pred
    chosen = idx[: min(int(k), len(idx))]
    flat = pred.reshape(-1)
    flat[chosen] = bool(add)
    return flat.reshape(pred.shape)


def summarize(df: pd.DataFrame, group_cols: Iterable[str]) -> pd.DataFrame:
    group_cols_l = [c for c in group_cols if c in df.columns]
    if not group_cols_l:
        return pd.DataFrame()
    value_cols = {
        "n": ("patient_id", "size"),
        "locf_mean": ("locf_dice", "mean"),
        "growth_only_oracle_mean": ("growth_only_oracle_dice", "mean"),
        "loss_only_oracle_mean": ("loss_only_oracle_dice", "mean"),
        "directional_oracle_mean": ("directional_oracle_dice", "mean"),
        "growth_only_gap_mean": ("growth_only_oracle_gap_vs_locf", "mean"),
        "loss_only_gap_mean": ("loss_only_oracle_gap_vs_locf", "mean"),
        "directional_gap_mean": ("directional_oracle_gap_vs_locf", "mean"),
        "input_volume_mean": ("input_volume_vox", "mean"),
        "growth_volume_mean": ("true_growth_volume_vox", "mean"),
        "loss_volume_mean": ("true_loss_volume_vox", "mean"),
        "relative_growth_mean": ("relative_new_growth", "mean"),
        "relative_loss_mean": ("relative_loss", "mean"),
    }
    return (
        df.groupby(group_cols_l, observed=True, dropna=False)
        .agg(**value_cols)
        .reset_index()
        .sort_values(group_cols_l)
    )


def summarize_budget(df: pd.DataFrame, group_cols: Iterable[str]) -> pd.DataFrame:
    group_cols_l = [c for c in group_cols if c in df.columns]
    if not group_cols_l:
        return pd.DataFrame()
    return (
        df.groupby(group_cols_l, observed=True, dropna=False)
        .agg(
            n=("patient_id", "size"),
            mean_dice=("dice", "mean"),
            mean_gap_vs_locf=("gap_vs_locf", "mean"),
            win_rate_vs_locf=("gap_vs_locf", lambda x: float((x > 0).mean())),
            mean_added_growth_vox=("added_growth_vox", "mean"),
            mean_true_growth_recall=("true_growth_recall", "mean"),
            mean_relative_growth=("relative_new_growth", "mean"),
            mean_relative_loss=("relative_loss", "mean"),
        )
        .reset_index()
        .sort_values(group_cols_l)
    )


def compute_oracles(samples: List[ForecastSample], dataset_root: Path, budget_fractions: List[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    label_cache: Dict[str, np.ndarray] = {}
    rows = []
    budget_rows = []

    def labels_for(pid: str) -> np.ndarray:
        if pid not in label_cache:
            label_cache[pid] = standardize_label(np.load(patient_paths(dataset_root, pid)["label"]))
        return label_cache[pid]

    for sample in samples:
        labels = labels_for(sample.patient_id)
        input_mask = labels[sample.input_idx, 0].astype(bool)
        target_mask = labels[sample.target_idx, 0].astype(bool)
        true_growth = target_mask & ~input_mask
        true_loss = input_mask & ~target_mask
        input_volume = int(input_mask.sum())
        target_volume = int(target_mask.sum())
        growth_volume = int(true_growth.sum())
        loss_volume = int(true_loss.sum())
        locf_dice = float(dice_np(input_mask, target_mask))

        growth_only_pred = input_mask | true_growth
        loss_only_pred = input_mask & ~true_loss
        growth_only_dice = float(dice_np(growth_only_pred, target_mask))
        loss_only_dice = float(dice_np(loss_only_pred, target_mask))

        if target_volume > input_volume:
            directional_pred = growth_only_pred
            directional_policy = "growth_only"
        elif target_volume < input_volume:
            directional_pred = loss_only_pred
            directional_policy = "loss_only"
        else:
            directional_pred = input_mask
            directional_policy = "locf"
        directional_dice = float(dice_np(directional_pred, target_mask))

        net_direction = (
            "net_growth"
            if target_volume > input_volume
            else "net_shrinkage"
            if target_volume < input_volume
            else "net_stable"
        )
        base = {
            "patient_id": sample.patient_id,
            "input_idx": int(sample.input_idx),
            "target_idx": int(sample.target_idx),
            "horizon": int(sample.horizon),
            "delta_days": float(sample.delta_days),
            "tier": infer_tier_from_patient_id(sample.patient_id),
            "input_volume_vox": input_volume,
            "target_volume_vox": target_volume,
            "true_growth_volume_vox": growth_volume,
            "true_loss_volume_vox": loss_volume,
            "relative_new_growth": float(growth_volume / max(1, input_volume)),
            "relative_loss": float(loss_volume / max(1, input_volume)),
            "net_direction": net_direction,
            "locf_dice": locf_dice,
            "growth_only_oracle_dice": growth_only_dice,
            "loss_only_oracle_dice": loss_only_dice,
            "directional_oracle_policy": directional_policy,
            "directional_oracle_dice": directional_dice,
            "growth_only_oracle_gap_vs_locf": growth_only_dice - locf_dice,
            "loss_only_oracle_gap_vs_locf": loss_only_dice - locf_dice,
            "directional_oracle_gap_vs_locf": directional_dice - locf_dice,
        }
        rows.append(base)

        for frac in budget_fractions:
            k = int(round(max(0.0, frac) * max(1, input_volume)))
            pred = top_true_subset(input_mask, true_growth, k=k, add=True)
            d = float(dice_np(pred, target_mask))
            added = int((pred & ~input_mask).sum())
            budget_rows.append(
                {
                    **base,
                    "budget_fraction_of_input": float(frac),
                    "growth_budget_vox": int(k),
                    "added_growth_vox": added,
                    "true_growth_recall": float(added / max(1, growth_volume)),
                    "dice": d,
                    "gap_vs_locf": d - locf_dice,
                }
            )

        pred_all_growth = input_mask | true_growth
        budget_rows.append(
            {
                **base,
                "budget_fraction_of_input": np.nan,
                "growth_budget_vox": growth_volume,
                "added_growth_vox": growth_volume,
                "true_growth_recall": 1.0 if growth_volume > 0 else 0.0,
                "dice": growth_only_dice,
                "gap_vs_locf": growth_only_dice - locf_dice,
                "budget_policy": "oracle_true_growth_volume",
            }
        )

    budget = pd.DataFrame(budget_rows)
    if "budget_policy" not in budget.columns:
        budget["budget_policy"] = ""
    budget["budget_policy"] = budget["budget_policy"].fillna(
        budget["budget_fraction_of_input"].apply(lambda x: f"input_frac_{x:g}" if pd.notna(x) else "unknown")
    )
    budget.loc[budget["budget_policy"] == "", "budget_policy"] = budget.loc[
        budget["budget_policy"] == "", "budget_fraction_of_input"
    ].apply(lambda x: f"input_frac_{x:g}" if pd.notna(x) else "unknown")
    return pd.DataFrame(rows), budget


def write_report(output_dir: Path, summary: dict, overall: pd.DataFrame, by_direction: pd.DataFrame, budget_overall: pd.DataFrame) -> None:
    with (output_dir / "locf_correction_oracle_report.md").open("w", encoding="utf-8") as f:
        f.write("# LOCF-Correction Oracle Analysis\n\n")
        f.write("This analysis estimates non-trivial ceilings for persistence-anchored forecasting. ")
        f.write("The model is not allowed to redraw the whole tumor; it starts from LOCF and applies constrained corrections.\n\n")
        f.write("## Run Summary\n\n")
        f.write(f"- samples: `{summary['n_samples']}`\n")
        f.write(f"- patients: `{summary['n_patients']}`\n")
        f.write(f"- budget fractions: `{summary['budget_fractions']}`\n\n")
        f.write("## Overall Oracle Summary\n\n")
        f.write(overall.to_markdown(index=False))
        f.write("\n\n## By Net Direction\n\n")
        f.write(by_direction.to_markdown(index=False))
        f.write("\n\n## Growth-Budget Ceiling\n\n")
        f.write(budget_overall.to_markdown(index=False))
        f.write("\n\nInterpretation rule: the full growth+loss target is deliberately not used as the main ceiling because it is trivially perfect. ")
        f.write("The useful question is whether growth-only or small-budget growth corrections leave enough headroom over LOCF to justify a constrained correction model.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate constrained oracle ceilings for LOCF-anchored correction policies.")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--manifest_csv", type=str, default=None)
    parser.add_argument("--splits", type=str, default="val,test")
    parser.add_argument("--fit_sessions", type=int, default=3)
    parser.add_argument("--horizons", type=str, default="1,2,3")
    parser.add_argument("--allowed_tiers", type=str, default=None)
    parser.add_argument("--budget_fractions", type=str, default="0,0.005,0.01,0.02,0.05,0.1,0.2")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_samples(args)
    budget_fractions = parse_float_csv(args.budget_fractions)
    oracle, budget = compute_oracles(samples, Path(args.dataset_root), budget_fractions)

    overall = summarize(oracle.assign(overall="all"), ["overall"])
    by_direction = summarize(oracle, ["net_direction"])
    by_horizon = summarize(oracle, ["horizon"])
    by_tier = summarize(oracle, ["tier"])
    budget_overall = summarize_budget(budget.assign(overall="all"), ["overall", "budget_policy"])
    budget_by_direction = summarize_budget(budget, ["net_direction", "budget_policy"])

    summary = {
        "dataset_root": str(Path(args.dataset_root)),
        "manifest_csv": args.manifest_csv,
        "splits": parse_csv(args.splits),
        "n_samples": int(len(oracle)),
        "n_patients": int(oracle["patient_id"].nunique()),
        "budget_fractions": budget_fractions,
        "outputs": {
            "oracle_samples_csv": str(output_dir / "locf_correction_oracle_samples.csv"),
            "budget_samples_csv": str(output_dir / "locf_correction_budget_oracle_samples.csv"),
            "overall_csv": str(output_dir / "locf_correction_oracle_overall.csv"),
            "by_direction_csv": str(output_dir / "locf_correction_oracle_by_direction.csv"),
            "by_horizon_csv": str(output_dir / "locf_correction_oracle_by_horizon.csv"),
            "by_tier_csv": str(output_dir / "locf_correction_oracle_by_tier.csv"),
            "budget_overall_csv": str(output_dir / "locf_correction_budget_oracle_overall.csv"),
            "budget_by_direction_csv": str(output_dir / "locf_correction_budget_oracle_by_direction.csv"),
            "report_md": str(output_dir / "locf_correction_oracle_report.md"),
        },
    }

    oracle.to_csv(output_dir / "locf_correction_oracle_samples.csv", index=False)
    budget.to_csv(output_dir / "locf_correction_budget_oracle_samples.csv", index=False)
    overall.to_csv(output_dir / "locf_correction_oracle_overall.csv", index=False)
    by_direction.to_csv(output_dir / "locf_correction_oracle_by_direction.csv", index=False)
    by_horizon.to_csv(output_dir / "locf_correction_oracle_by_horizon.csv", index=False)
    by_tier.to_csv(output_dir / "locf_correction_oracle_by_tier.csv", index=False)
    budget_overall.to_csv(output_dir / "locf_correction_budget_oracle_overall.csv", index=False)
    budget_by_direction.to_csv(output_dir / "locf_correction_budget_oracle_by_direction.csv", index=False)
    with (output_dir / "locf_correction_oracle_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (output_dir / "locf_correction_oracle_samples.json").open("w", encoding="utf-8") as f:
        json.dump([asdict(s) for s in samples], f, indent=2)
    write_report(output_dir, summary, overall, by_direction, budget_overall)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
