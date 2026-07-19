#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd


def parse_list(text: str | None) -> List[str]:
    if not text:
        return []
    return [x.strip() for x in text.split(",") if x.strip()]


def summarize_once(df: pd.DataFrame, dice_col: str, locf_col: str) -> dict:
    gap = df[dice_col].astype(float) - df[locf_col].astype(float)
    return {
        "n": int(len(df)),
        "n_patients": int(df["patient_id"].nunique()),
        "mean_dice": float(df[dice_col].mean()),
        "locf_mean": float(df[locf_col].mean()),
        "mean_gap_vs_locf": float(gap.mean()),
        "median_gap_vs_locf": float(gap.median()),
        "win_rate_vs_locf": float((gap > 0).mean()),
    }


def patient_bootstrap(
    df: pd.DataFrame,
    dice_col: str,
    locf_col: str,
    n_bootstrap: int,
    seed: int,
) -> tuple[dict, pd.DataFrame]:
    patients = np.array(sorted(df["patient_id"].astype(str).unique()))
    observed = summarize_once(df, dice_col=dice_col, locf_col=locf_col)
    observed["bootstrap_unit"] = "patient"
    observed["n_bootstrap"] = int(n_bootstrap)

    if n_bootstrap <= 0 or len(patients) == 0:
        return observed, pd.DataFrame()

    rng = np.random.default_rng(seed)
    boot_rows = []
    patient_groups = {pid: part.copy() for pid, part in df.groupby("patient_id", observed=True, dropna=False)}
    for b in range(n_bootstrap):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        parts = []
        for draw_id, pid in enumerate(sampled):
            part = patient_groups[str(pid)].copy()
            part["bootstrap_draw_patient_index"] = draw_id
            parts.append(part)
        boot = pd.concat(parts, ignore_index=True)
        stats = summarize_once(boot, dice_col=dice_col, locf_col=locf_col)
        stats["bootstrap_iter"] = int(b)
        boot_rows.append(stats)

    boot_df = pd.DataFrame(boot_rows)
    for metric in ["mean_dice", "locf_mean", "mean_gap_vs_locf", "median_gap_vs_locf", "win_rate_vs_locf"]:
        values = boot_df[metric].to_numpy(dtype=float)
        observed[f"{metric}_ci_low"] = float(np.nanpercentile(values, 2.5))
        observed[f"{metric}_ci_high"] = float(np.nanpercentile(values, 97.5))
    observed["prob_mean_gap_gt_0"] = float((boot_df["mean_gap_vs_locf"] > 0).mean())
    observed["prob_median_gap_gt_0"] = float((boot_df["median_gap_vs_locf"] > 0).mean())
    return observed, boot_df


def run_grouped_bootstrap(
    samples: pd.DataFrame,
    group_cols: Iterable[str],
    dice_col: str,
    locf_col: str,
    n_bootstrap: int,
    seed: int,
    min_patients: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_cols_l = list(group_cols)
    rows = []
    draws = []
    if group_cols_l:
        grouped = samples.groupby(group_cols_l, observed=True, dropna=False)
    else:
        samples = samples.copy()
        samples["_overall"] = "overall"
        group_cols_l = ["_overall"]
        grouped = samples.groupby(group_cols_l, observed=True, dropna=False)

    for key, part in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        group_payload = dict(zip(group_cols_l, key))
        n_patients = int(part["patient_id"].nunique())
        if n_patients < min_patients:
            row = {
                **group_payload,
                "n": int(len(part)),
                "n_patients": n_patients,
                "skipped": True,
                "skip_reason": f"n_patients < {min_patients}",
            }
            rows.append(row)
            continue
        observed, boot_df = patient_bootstrap(
            part,
            dice_col=dice_col,
            locf_col=locf_col,
            n_bootstrap=n_bootstrap,
            seed=seed + len(rows),
        )
        observed.update(group_payload)
        observed["skipped"] = False
        rows.append(observed)
        if not boot_df.empty:
            for col, value in group_payload.items():
                boot_df[col] = value
            draws.append(boot_df)

    summary = pd.DataFrame(rows)
    if "_overall" in summary.columns:
        summary = summary.drop(columns=["_overall"])
    draw_df = pd.concat(draws, ignore_index=True) if draws else pd.DataFrame()
    if "_overall" in draw_df.columns:
        draw_df = draw_df.drop(columns=["_overall"])
    return summary, draw_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Patient-level bootstrap for growth-only residual policy versus LOCF.")
    parser.add_argument("--samples_csv", type=str, required=True)
    parser.add_argument("--dice_col", type=str, default="dice")
    parser.add_argument("--locf_col", type=str, default="locf_dice_recomputed")
    parser.add_argument("--groupings", type=str, default="split;split,net_direction")
    parser.add_argument("--n_bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_patients", type=int, default=2)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    samples = pd.read_csv(args.samples_csv)
    required = {"patient_id", args.dice_col, args.locf_col}
    missing = sorted(required - set(samples.columns))
    if missing:
        raise ValueError(f"Missing required columns in samples CSV: {missing}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    all_draws = []
    for grouping_text in [g.strip() for g in args.groupings.split(";") if g.strip()]:
        group_cols = parse_list(grouping_text)
        summary, draws = run_grouped_bootstrap(
            samples,
            group_cols=group_cols,
            dice_col=args.dice_col,
            locf_col=args.locf_col,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
            min_patients=args.min_patients,
        )
        summary.insert(0, "grouping", grouping_text)
        all_summaries.append(summary)
        if not draws.empty:
            draws.insert(0, "grouping", grouping_text)
            all_draws.append(draws)

    overall_summary, overall_draws = run_grouped_bootstrap(
        samples,
        group_cols=[],
        dice_col=args.dice_col,
        locf_col=args.locf_col,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed + 10000,
        min_patients=args.min_patients,
    )
    overall_summary.insert(0, "grouping", "overall")
    all_summaries.insert(0, overall_summary)
    if not overall_draws.empty:
        overall_draws.insert(0, "grouping", "overall")
        all_draws.insert(0, overall_draws)

    summary_df = pd.concat(all_summaries, ignore_index=True)
    draws_df = pd.concat(all_draws, ignore_index=True) if all_draws else pd.DataFrame()

    summary_path = output_dir / "growth_only_policy_patient_bootstrap_summary.csv"
    draws_path = output_dir / "growth_only_policy_patient_bootstrap_draws.csv"
    report_path = output_dir / "growth_only_policy_patient_bootstrap_report.json"
    summary_df.to_csv(summary_path, index=False)
    draws_df.to_csv(draws_path, index=False)

    report = {
        "samples_csv": str(Path(args.samples_csv).resolve()),
        "dice_col": args.dice_col,
        "locf_col": args.locf_col,
        "groupings": ["overall"] + [g.strip() for g in args.groupings.split(";") if g.strip()],
        "n_bootstrap": int(args.n_bootstrap),
        "seed": int(args.seed),
        "min_patients": int(args.min_patients),
        "outputs": {
            "summary_csv": str(summary_path),
            "draws_csv": str(draws_path),
            "report_json": str(report_path),
        },
    }
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
