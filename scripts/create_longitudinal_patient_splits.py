#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def parse_fracs(payload: str) -> Dict[str, float]:
    parts = [p.strip() for p in payload.split(",") if p.strip()]
    out: Dict[str, float] = {}
    for part in parts:
        if "=" not in part:
            raise ValueError(f"Expected split=fraction entry, got: {part}")
        name, value = part.split("=", 1)
        out[name.strip()] = float(value)
    if set(out) != {"train", "val", "test"}:
        raise ValueError("Fractions must define exactly train, val, and test.")
    total = sum(out.values())
    if total <= 0:
        raise ValueError("Fractions must sum to a positive value.")
    return {k: v / total for k, v in out.items()}


def summarize_windows(windows: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if windows.empty:
        return pd.DataFrame()
    work = windows.copy()
    by = [c for c in group_cols if c in work.columns]
    if not by:
        work["_overall"] = "overall"
        by = ["_overall"]
    agg = (
        work.groupby(by, observed=True, dropna=False)
        .agg(
            n_windows=("patient_id", "size"),
            n_patients=("patient_id", "nunique"),
            mean_locf_dice=("locf_dice", "mean"),
            median_locf_dice=("locf_dice", "median"),
            mean_delta_days=("delta_days", "mean"),
            mean_growth_volume_vox=("growth_volume_vox", "mean"),
            mean_relative_new_growth=("relative_new_growth", "mean"),
            net_growth_rate=("net_direction", lambda x: float((x == "net_growth").mean())),
            treatment_change_rate=("target_treatment_changed", "mean")
            if "target_treatment_changed" in work.columns
            else ("patient_id", lambda x: np.nan),
        )
        .reset_index()
    )
    if "_overall" in agg.columns:
        agg = agg.drop(columns=["_overall"])
    return agg


def build_patient_table(windows: pd.DataFrame) -> pd.DataFrame:
    required = {"patient_id", "locf_dice", "net_direction", "growth_volume_vox", "relative_new_growth", "delta_days"}
    missing = sorted(required - set(windows.columns))
    if missing:
        raise ValueError(f"Window CSV is missing required columns: {missing}")

    treatment_col = "target_treatment_changed" if "target_treatment_changed" in windows.columns else None
    agg_spec = {
        "n_windows": ("patient_id", "size"),
        "mean_locf_dice": ("locf_dice", "mean"),
        "median_locf_dice": ("locf_dice", "median"),
        "mean_delta_days": ("delta_days", "mean"),
        "mean_growth_volume_vox": ("growth_volume_vox", "mean"),
        "mean_relative_new_growth": ("relative_new_growth", "mean"),
        "net_growth_windows": ("net_direction", lambda x: int((x == "net_growth").sum())),
        "net_shrinkage_windows": ("net_direction", lambda x: int((x == "net_shrinkage").sum())),
    }
    if treatment_col:
        agg_spec["target_treatment_change_rate"] = (treatment_col, "mean")
        agg_spec["target_treatment_change_windows"] = (treatment_col, lambda x: int(np.asarray(x, dtype=bool).sum()))
    else:
        agg_spec["target_treatment_change_rate"] = ("patient_id", lambda x: np.nan)
        agg_spec["target_treatment_change_windows"] = ("patient_id", lambda x: 0)

    patient = windows.groupby("patient_id", observed=True).agg(**agg_spec).reset_index()
    patient["net_growth_rate"] = patient["net_growth_windows"] / patient["n_windows"].clip(lower=1)
    return patient


def greedy_assign(patient: pd.DataFrame, fracs: Dict[str, float], seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    split_names = ["train", "val", "test"]
    n_patients = len(patient)
    total_windows = float(patient["n_windows"].sum())
    total_net_growth = float(patient["net_growth_windows"].sum())
    total_treat = float(patient["target_treatment_change_windows"].sum())

    raw_counts = {split: fracs[split] * n_patients for split in split_names}
    patient_quota = {split: int(np.floor(raw_counts[split])) for split in split_names}
    for split in split_names:
        if patient_quota[split] == 0 and n_patients >= len(split_names):
            patient_quota[split] = 1
    while sum(patient_quota.values()) > n_patients:
        split = max(split_names, key=lambda s: (patient_quota[s], s != "train"))
        patient_quota[split] -= 1
    while sum(patient_quota.values()) < n_patients:
        split = max(split_names, key=lambda s: (raw_counts[s] - patient_quota[s], s == "train"))
        patient_quota[split] += 1

    targets = {
        split: {
            "patients": float(patient_quota[split]),
            "windows": max(1.0, fracs[split] * total_windows),
            "net_growth": max(1.0, fracs[split] * total_net_growth) if total_net_growth > 0 else 0.0,
            "treat": max(1.0, fracs[split] * total_treat) if total_treat > 0 else 0.0,
        }
        for split in split_names
    }
    state = {
        split: {"patients": 0.0, "windows": 0.0, "net_growth": 0.0, "treat": 0.0}
        for split in split_names
    }

    work = patient.copy()
    work["_jitter"] = rng.uniform(0.0, 1e-6, size=len(work))
    work = work.sort_values(
        ["n_windows", "net_growth_rate", "mean_growth_volume_vox", "_jitter"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    assignments = []
    for _, row in work.iterrows():
        candidate_scores = []
        for split in split_names:
            if state[split]["patients"] >= patient_quota[split]:
                continue
            projected = state[split].copy()
            projected["patients"] += 1
            projected["windows"] += float(row["n_windows"])
            projected["net_growth"] += float(row["net_growth_windows"])
            projected["treat"] += float(row["target_treatment_change_windows"])

            score = 0.0
            score += abs(projected["windows"] - targets[split]["windows"]) / max(1.0, total_windows)
            score += 0.75 * abs(projected["patients"] - targets[split]["patients"]) / max(1.0, n_patients)
            if total_net_growth > 0:
                score += 0.50 * abs(projected["net_growth"] - targets[split]["net_growth"]) / max(1.0, total_net_growth)
            if total_treat > 0:
                score += 0.35 * abs(projected["treat"] - targets[split]["treat"]) / max(1.0, total_treat)

            candidate_scores.append((score, split))

        if not candidate_scores:
            raise RuntimeError("No split has remaining patient capacity; this should not happen.")
        _, chosen = min(candidate_scores, key=lambda x: (x[0], split_names.index(x[1])))
        state[chosen]["patients"] += 1
        state[chosen]["windows"] += float(row["n_windows"])
        state[chosen]["net_growth"] += float(row["net_growth_windows"])
        state[chosen]["treat"] += float(row["target_treatment_change_windows"])
        assignments.append({"patient_id": row["patient_id"], "split": chosen})

    assigned = patient.merge(pd.DataFrame(assignments), on="patient_id", how="left")
    return assigned.sort_values(["split", "patient_id"]).reset_index(drop=True)


def write_report(path: Path, assigned: pd.DataFrame, summaries: Dict[str, pd.DataFrame], fracs: Dict[str, float]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Longitudinal Patient Split Manifest\n\n")
        f.write(
            "This manifest assigns complete patients to train/validation/test splits. "
            "The goal is to prevent overlapping-window leakage in longitudinal forecasting experiments.\n\n"
        )
        f.write("## Requested Fractions\n\n")
        for split, frac in fracs.items():
            f.write(f"- {split}: {frac:.3f}\n")
        f.write("\n## Assigned Patients\n\n")
        f.write(assigned.to_markdown(index=False))
        for name, table in summaries.items():
            f.write(f"\n\n## {name}\n\n")
            f.write(table.to_markdown(index=False) if not table.empty else "No rows.")
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create patient-level splits from a longitudinal window audit.")
    parser.add_argument("--window_csv", type=str, required=True)
    parser.add_argument("--fractions", type=str, default="train=0.6,val=0.2,test=0.2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    window_csv = Path(args.window_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fracs = parse_fracs(args.fractions)

    windows = pd.read_csv(window_csv)
    if windows.empty:
        raise ValueError(f"Window CSV has no rows: {window_csv}")
    patient = build_patient_table(windows)
    assigned = greedy_assign(patient, fracs=fracs, seed=args.seed)
    split_map = {
        split: assigned.loc[assigned["split"] == split, "patient_id"].tolist()
        for split in ["train", "val", "test"]
    }

    manifest = windows.merge(assigned[["patient_id", "split"]], on="patient_id", how="left")
    summaries = {
        "Split Overall": summarize_windows(manifest, ["split"]),
        "Split by Net Direction": summarize_windows(manifest, ["split", "net_direction"]),
        "Split by Growth Bin": summarize_windows(manifest, ["split", "absolute_growth_bin"]),
        "Split by Patient": summarize_windows(manifest, ["split", "patient_id"]),
    }

    assigned.to_csv(output_dir / "longitudinal_patient_split_assignments.csv", index=False)
    manifest.to_csv(output_dir / "longitudinal_window_manifest.csv", index=False)
    for name, table in summaries.items():
        filename = name.lower().replace(" ", "_")
        table.to_csv(output_dir / f"{filename}.csv", index=False)
    with (output_dir / "longitudinal_patient_splits.json").open("w", encoding="utf-8") as f:
        json.dump({"seed": args.seed, "fractions": fracs, "splits": split_map}, f, indent=2)
    write_report(output_dir / "longitudinal_patient_split_report.md", assigned, summaries, fracs)

    print(
        json.dumps(
            {
                "window_csv": str(window_csv),
                "n_patients": int(len(assigned)),
                "n_windows": int(len(manifest)),
                "splits": {k: {"n_patients": len(v)} for k, v in split_map.items()},
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
