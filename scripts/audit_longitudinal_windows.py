#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.metrics import dice_np
from baselines.tasks import list_patient_ids, load_splits, patient_paths


def parse_int_list(payload: str) -> List[int]:
    vals = []
    for item in payload.split(","):
        item = item.strip()
        if item:
            vals.append(int(item))
    vals = sorted(set(v for v in vals if v >= 1))
    if not vals:
        raise ValueError("Need at least one positive integer.")
    return vals


def parse_patients(payload: str | None) -> List[str] | None:
    if payload is None:
        return None
    vals = [x.strip() for x in payload.split(",") if x.strip()]
    return vals or None


def standardize_label(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 5 and arr.shape[1] == 1:
        return (arr[:, 0] > 0).astype(np.uint8)
    if arr.ndim == 4:
        return (arr > 0).astype(np.uint8)
    raise ValueError(f"Unsupported label shape: {arr.shape}")


def load_patient_arrays(dataset_root: Path, patient_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    paths = patient_paths(dataset_root, patient_id)
    labels = standardize_label(np.load(paths["label"]))
    days = np.load(paths["days"]).astype(np.float32)
    treatment = np.load(paths["treatment"]).astype(np.float32)
    if labels.shape[0] != days.shape[0] or labels.shape[0] != treatment.shape[0]:
        raise ValueError(
            f"Session mismatch for {patient_id}: labels={labels.shape[0]}, "
            f"days={days.shape[0]}, treatment={treatment.shape[0]}"
        )
    return labels, days, treatment


def select_patient_ids(dataset_root: Path, split: str | None, patient_ids: List[str] | None) -> List[str]:
    all_ids = list_patient_ids(dataset_root)
    if patient_ids is not None:
        keep = set(patient_ids)
        return [pid for pid in all_ids if pid in keep]
    if split is None or split == "all":
        return all_ids
    splits = load_splits(dataset_root)
    split_ids = set(splits.get(split, []))
    if not split_ids:
        raise ValueError(f"No patients found for split='{split}'.")
    return [pid for pid in all_ids if pid in split_ids]


def summarize_patient(patient_id: str, labels: np.ndarray, days: np.ndarray, treatment: np.ndarray) -> dict:
    session_volumes = labels.reshape(labels.shape[0], -1).sum(axis=1).astype(int)
    intervals = np.diff(days) if len(days) >= 2 else np.asarray([], dtype=np.float32)
    return {
        "patient_id": patient_id,
        "n_sessions": int(labels.shape[0]),
        "first_day": float(days[0]) if len(days) else np.nan,
        "last_day": float(days[-1]) if len(days) else np.nan,
        "followup_days": float(days[-1] - days[0]) if len(days) else np.nan,
        "mean_interval_days": float(intervals.mean()) if len(intervals) else np.nan,
        "min_interval_days": float(intervals.min()) if len(intervals) else np.nan,
        "max_interval_days": float(intervals.max()) if len(intervals) else np.nan,
        "first_volume_vox": int(session_volumes[0]) if len(session_volumes) else 0,
        "last_volume_vox": int(session_volumes[-1]) if len(session_volumes) else 0,
        "min_volume_vox": int(session_volumes.min()) if len(session_volumes) else 0,
        "max_volume_vox": int(session_volumes.max()) if len(session_volumes) else 0,
        "mean_volume_vox": float(session_volumes.mean()) if len(session_volumes) else np.nan,
        "n_treatment_states": int(pd.Series(treatment).nunique()),
        "ever_treated": bool(np.any(treatment > 0)),
    }


def window_rows_for_patient(
    patient_id: str,
    labels: np.ndarray,
    days: np.ndarray,
    treatment: np.ndarray,
    input_lengths: Iterable[int],
    horizons: Iterable[int],
) -> List[dict]:
    rows = []
    n_sessions = int(labels.shape[0])
    for input_len in input_lengths:
        for start_idx in range(0, n_sessions):
            input_end_idx = start_idx + input_len - 1
            if input_end_idx >= n_sessions:
                continue
            input_mask = labels[input_end_idx] > 0
            prev_mask = labels[input_end_idx - 1] > 0 if input_end_idx > 0 else None
            input_volume = int(input_mask.sum())
            previous_growth = int((input_mask & ~prev_mask).sum()) if prev_mask is not None else 0
            previous_loss = int((prev_mask & ~input_mask).sum()) if prev_mask is not None else 0
            for horizon in horizons:
                target_idx = input_end_idx + horizon
                if target_idx >= n_sessions:
                    continue
                target_mask = labels[target_idx] > 0
                target_volume = int(target_mask.sum())
                growth = target_mask & ~input_mask
                loss = input_mask & ~target_mask
                growth_volume = int(growth.sum())
                loss_volume = int(loss.sum())
                rows.append(
                    {
                        "patient_id": patient_id,
                        "input_window_len": int(input_len),
                        "window_start_idx": int(start_idx),
                        "input_end_idx": int(input_end_idx),
                        "target_idx": int(target_idx),
                        "horizon": int(horizon),
                        "input_start_day": float(days[start_idx]),
                        "input_end_day": float(days[input_end_idx]),
                        "target_day": float(days[target_idx]),
                        "input_span_days": float(days[input_end_idx] - days[start_idx]),
                        "delta_days": float(days[target_idx] - days[input_end_idx]),
                        "input_start_treatment": float(treatment[start_idx]),
                        "input_end_treatment": float(treatment[input_end_idx]),
                        "target_treatment": float(treatment[target_idx]),
                        "treatment_changed_in_input": bool(np.any(np.diff(treatment[start_idx : input_end_idx + 1]) != 0))
                        if input_len > 1
                        else False,
                        "target_treatment_changed": bool(treatment[target_idx] != treatment[input_end_idx]),
                        "input_volume_vox": input_volume,
                        "target_volume_vox": target_volume,
                        "net_delta_volume_vox": int(target_volume - input_volume),
                        "growth_volume_vox": growth_volume,
                        "loss_volume_vox": loss_volume,
                        "relative_new_growth": float(growth_volume / max(1, input_volume)),
                        "relative_loss": float(loss_volume / max(1, input_volume)),
                        "relative_net_growth": float((target_volume - input_volume) / max(1, input_volume)),
                        "previous_growth_volume_vox": previous_growth,
                        "previous_loss_volume_vox": previous_loss,
                        "previous_growth_ratio": float(previous_growth / max(1, int(prev_mask.sum()) if prev_mask is not None else input_volume)),
                        "locf_dice": dice_np(input_mask.astype(np.float32), target_mask.astype(np.float32)),
                    }
                )
    return rows


def add_bins(windows: pd.DataFrame) -> pd.DataFrame:
    out = windows.copy()
    out["absolute_growth_bin"] = np.select(
        [
            out["growth_volume_vox"] <= 0,
            out["growth_volume_vox"] <= 250,
            out["growth_volume_vox"] <= 1500,
        ],
        ["zero", "small_nonzero", "medium_nonzero"],
        default="large_nonzero",
    )
    out["net_direction"] = np.select(
        [
            out["net_delta_volume_vox"] > 0,
            out["net_delta_volume_vox"] < 0,
        ],
        ["net_growth", "net_shrinkage"],
        default="net_stable",
    )
    return out


def summarize(windows: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if windows.empty:
        return pd.DataFrame()
    cols = [c for c in group_cols if c in windows.columns]
    work = windows if cols else windows.assign(_overall="overall")
    by = cols if cols else ["_overall"]
    out = (
        work.groupby(by, observed=True, dropna=False)
        .agg(
            n_windows=("patient_id", "size"),
            n_patients=("patient_id", "nunique"),
            mean_locf_dice=("locf_dice", "mean"),
            median_locf_dice=("locf_dice", "median"),
            mean_delta_days=("delta_days", "mean"),
            median_delta_days=("delta_days", "median"),
            mean_growth_volume_vox=("growth_volume_vox", "mean"),
            median_growth_volume_vox=("growth_volume_vox", "median"),
            mean_relative_new_growth=("relative_new_growth", "mean"),
            zero_growth_rate=("growth_volume_vox", lambda x: float((x <= 0).mean())),
            target_treatment_change_rate=("target_treatment_changed", "mean"),
        )
        .reset_index()
    )
    if "_overall" in out.columns:
        out = out.drop(columns=["_overall"])
    return out.sort_values(cols).reset_index(drop=True) if cols else out


def write_report(path: Path, patient_summary: pd.DataFrame, windows: pd.DataFrame, summaries: dict[str, pd.DataFrame]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Longitudinal Window Audit\n\n")
        f.write(
            "This audit enumerates patient-level longitudinal forecasting windows. "
            "It is intended to check real-data sample availability before model training.\n\n"
        )
        f.write("## Patient Summary\n\n")
        f.write(patient_summary.describe(include="all").to_markdown())
        f.write("\n\n## Window Summary\n\n")
        f.write(windows.describe(include="all").to_markdown())
        for name, table in summaries.items():
            f.write(f"\n\n## {name}\n\n")
            f.write(table.to_markdown(index=False) if not table.empty else "No rows.")
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit sliding longitudinal forecasting windows.")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--split", type=str, default="all", help="Optional split name from splits/splits.json, or 'all'.")
    parser.add_argument("--patient_ids", type=str, default=None, help="Optional comma-separated patient IDs.")
    parser.add_argument("--input_lengths", type=str, default="3,4,5")
    parser.add_argument("--horizons", type=str, default="1")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_lengths = parse_int_list(args.input_lengths)
    horizons = parse_int_list(args.horizons)
    patient_ids = select_patient_ids(dataset_root, args.split, parse_patients(args.patient_ids))
    if not patient_ids:
        raise ValueError(f"No patient files found under {dataset_root}")

    patient_rows = []
    window_rows = []
    failures = []
    for patient_id in patient_ids:
        try:
            labels, days, treatment = load_patient_arrays(dataset_root, patient_id)
            patient_rows.append(summarize_patient(patient_id, labels, days, treatment))
            window_rows.extend(window_rows_for_patient(patient_id, labels, days, treatment, input_lengths, horizons))
        except Exception as e:
            failures.append({"patient_id": patient_id, "error": str(e)})

    patient_summary = pd.DataFrame(patient_rows)
    windows = add_bins(pd.DataFrame(window_rows)) if window_rows else pd.DataFrame()

    summaries = {
        "Overall": summarize(windows, []),
        "By Input Window Length": summarize(windows, ["input_window_len"]),
        "By Horizon": summarize(windows, ["horizon"]),
        "By Growth Bin": summarize(windows, ["absolute_growth_bin"]),
        "By Input Window and Growth Bin": summarize(windows, ["input_window_len", "absolute_growth_bin"]),
        "By Net Direction": summarize(windows, ["net_direction"]),
        "By Patient": summarize(windows, ["patient_id"]),
    }

    patient_summary.to_csv(output_dir / "longitudinal_patient_summary.csv", index=False)
    windows.to_csv(output_dir / "longitudinal_window_samples.csv", index=False)
    for name, table in summaries.items():
        filename = name.lower().replace(" ", "_").replace("and", "by")
        table.to_csv(output_dir / f"longitudinal_{filename}.csv", index=False)
    pd.DataFrame(failures).to_csv(output_dir / "longitudinal_audit_failures.csv", index=False)
    write_report(output_dir / "longitudinal_window_audit_report.md", patient_summary, windows, summaries)

    print(
        json.dumps(
            {
                "dataset_root": str(dataset_root),
                "split": args.split,
                "input_lengths": input_lengths,
                "horizons": horizons,
                "n_patients_loaded": int(len(patient_summary)),
                "n_failed_patients": int(len(failures)),
                "n_windows": int(len(windows)),
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
