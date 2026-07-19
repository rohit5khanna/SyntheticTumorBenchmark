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


def compute_patient_quota(n_patients: int, fracs: Dict[str, float]) -> Dict[str, int]:
    split_names = ["train", "val", "test"]
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
    return patient_quota


def score_assignment(assigned: pd.DataFrame, fracs: Dict[str, float], patient_quota: Dict[str, int]) -> float:
    split_names = ["train", "val", "test"]
    totals = {
        "n_windows": float(assigned["n_windows"].sum()),
        "net_growth_windows": float(assigned["net_growth_windows"].sum()),
        "target_treatment_change_windows": float(assigned["target_treatment_change_windows"].sum()),
    }
    global_means = {
        "mean_locf_dice": float(np.average(assigned["mean_locf_dice"], weights=assigned["n_windows"])),
        "mean_relative_new_growth": float(np.average(assigned["mean_relative_new_growth"], weights=assigned["n_windows"])),
        "mean_delta_days": float(np.average(assigned["mean_delta_days"], weights=assigned["n_windows"])),
    }

    score = 0.0
    for split in split_names:
        part = assigned[assigned["split"] == split]
        if part.empty:
            return float("inf")
        n_windows = float(part["n_windows"].sum())
        net_growth = float(part["net_growth_windows"].sum())
        treatment_change = float(part["target_treatment_change_windows"].sum())
        locf = float(np.average(part["mean_locf_dice"], weights=part["n_windows"]))
        rel_growth = float(np.average(part["mean_relative_new_growth"], weights=part["n_windows"]))
        delta = float(np.average(part["mean_delta_days"], weights=part["n_windows"]))

        score += 3.00 * abs(n_windows - fracs[split] * totals["n_windows"]) / max(1.0, totals["n_windows"])
        score += 1.00 * abs(len(part) - patient_quota[split]) / max(1.0, len(assigned))
        if totals["net_growth_windows"] > 0:
            score += 1.25 * abs(net_growth - fracs[split] * totals["net_growth_windows"]) / totals["net_growth_windows"]
        if totals["target_treatment_change_windows"] > 0:
            score += (
                0.50
                * abs(treatment_change - fracs[split] * totals["target_treatment_change_windows"])
                / totals["target_treatment_change_windows"]
            )
        score += 1.00 * abs(locf - global_means["mean_locf_dice"])
        score += 0.40 * abs(rel_growth - global_means["mean_relative_new_growth"]) / max(
            1e-6, abs(global_means["mean_relative_new_growth"])
        )
        score += 0.35 * abs(delta - global_means["mean_delta_days"]) / max(1e-6, abs(global_means["mean_delta_days"]))
    return float(score)


def assign_from_order(patient: pd.DataFrame, order: np.ndarray, patient_quota: Dict[str, int]) -> pd.DataFrame:
    split_order = ["train", "val", "test"]
    labels: List[dict] = []
    start = 0
    for split in split_order:
        stop = start + patient_quota[split]
        for idx in order[start:stop]:
            labels.append({"patient_id": patient.iloc[int(idx)]["patient_id"], "split": split})
        start = stop
    return patient.merge(pd.DataFrame(labels), on="patient_id", how="left")


def search_assign(patient: pd.DataFrame, fracs: Dict[str, float], seed: int, n_candidates: int) -> tuple[pd.DataFrame, float]:
    rng = np.random.default_rng(seed)
    n_patients = len(patient)
    patient_quota = compute_patient_quota(n_patients, fracs)
    order_base = np.arange(n_patients)
    best: pd.DataFrame | None = None
    best_score = float("inf")

    # Include one deterministic high-window ordering, then random permutations.
    deterministic = patient.sort_values(["n_windows", "net_growth_rate"], ascending=[False, False]).index.to_numpy()
    candidate_orders = [deterministic]
    for _ in range(max(1, n_candidates)):
        candidate_orders.append(rng.permutation(order_base))

    for order in candidate_orders:
        assigned = assign_from_order(patient, order, patient_quota)
        score = score_assignment(assigned, fracs, patient_quota)
        if score < best_score:
            best_score = score
            best = assigned

    if best is None:
        raise RuntimeError("Could not produce a split assignment.")
    assigned = best
    assigned["assignment_score"] = best_score
    return assigned.sort_values(["split", "patient_id"]).reset_index(drop=True), float(best_score)


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
    parser.add_argument("--n_candidates", type=int, default=20000)
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
    assigned, assignment_score = search_assign(patient, fracs=fracs, seed=args.seed, n_candidates=args.n_candidates)
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
        json.dump(
            {
                "seed": args.seed,
                "n_candidates": args.n_candidates,
                "assignment_score": assignment_score,
                "fractions": fracs,
                "splits": split_map,
            },
            f,
            indent=2,
        )
    write_report(output_dir / "longitudinal_patient_split_report.md", assigned, summaries, fracs)

    print(
        json.dumps(
            {
                "window_csv": str(window_csv),
                "n_patients": int(len(assigned)),
                "n_windows": int(len(manifest)),
                "assignment_score": float(assignment_score),
                "splits": {k: {"n_patients": len(v)} for k, v in split_map.items()},
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
