#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd


KEY_COLS = ["patient_id", "input_idx", "target_idx", "horizon", "delta_days"]


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _selected_identity(selection_dir: Path) -> tuple[str, str]:
    selected_path = selection_dir / "selected_policy_validation_test_row.csv"
    selected = _safe_read_csv(selected_path)
    if selected.empty:
        raise ValueError(f"No selected policy row found in {selected_path}")
    return str(selected.iloc[0]["score_source"]), str(selected.iloc[0]["budget_policy"])


def _policy_rows(samples: pd.DataFrame, score_source: str, budget_policy: str) -> pd.DataFrame:
    return samples[
        (samples["score_source"] == score_source)
        & (samples["budget_policy"] == budget_policy)
    ].copy()


def _base_samples(samples: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        "tier",
        "input_volume_vox",
        "target_volume_vox",
        "growth_volume_vox",
        "relative_new_growth",
        "locf_dice",
        "absolute_growth_bin",
        "relative_growth_bin",
    ]
    available = KEY_COLS + [c for c in feature_cols if c in samples.columns]
    return samples[available].drop_duplicates(KEY_COLS).copy()


def _bootstrap_gap(rows: pd.DataFrame, n_bootstrap: int, seed: int) -> dict:
    if rows.empty:
        return {"n_samples": 0, "mean_gap": np.nan, "ci_low": np.nan, "ci_high": np.nan, "win_rate": np.nan}
    gaps = rows["dice_gap_vs_locf"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(gaps), len(gaps))
        boot.append(float(gaps[idx].mean()))
    boot = np.asarray(boot, dtype=float)
    return {
        "n_samples": int(len(gaps)),
        "mean_gap": float(gaps.mean()),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "win_rate": float((gaps > 0).mean()),
    }


def summarize(rows: pd.DataFrame, group_cols: Iterable[str]) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    cols = [c for c in group_cols if c in rows.columns]
    if not cols:
        grouped = rows.assign(_overall="overall").groupby("_overall", observed=True, dropna=False)
    else:
        grouped = rows.groupby(cols, observed=True, dropna=False)
    out = (
        grouped.agg(
            count=("persistence_growth_dice", "size"),
            mean_dice=("persistence_growth_dice", "mean"),
            mean_locf_dice=("locf_dice", "mean"),
            mean_gap_vs_locf=("dice_gap_vs_locf", "mean"),
            median_gap_vs_locf=("dice_gap_vs_locf", "median"),
            win_rate_vs_locf=("dice_gap_vs_locf", lambda x: float((x > 0).mean())),
            mean_growth_volume_vox=("growth_volume_vox", "mean"),
            median_growth_volume_vox=("growth_volume_vox", "median"),
            mean_growth_budget_vox=("growth_budget_vox", "mean"),
            median_growth_budget_vox=("growth_budget_vox", "median"),
            mean_budget_to_true_growth_ratio=("budget_to_true_growth_ratio", "mean"),
        )
        .reset_index()
    )
    if "_overall" in out.columns:
        out = out.drop(columns=["_overall"])
    return out.sort_values(cols).reset_index(drop=True) if cols else out


def compare_split_distributions(val_rows: pd.DataFrame, test_rows: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "growth_volume_vox",
        "relative_new_growth",
        "input_volume_vox",
        "locf_dice",
        "growth_budget_vox",
        "budget_to_true_growth_ratio",
        "dice_gap_vs_locf",
    ]
    rows = []
    for metric in metrics:
        if metric not in val_rows.columns or metric not in test_rows.columns:
            continue
        v = val_rows[metric].dropna().to_numpy(dtype=float)
        t = test_rows[metric].dropna().to_numpy(dtype=float)
        if len(v) == 0 or len(t) == 0:
            continue
        rows.append(
            {
                "metric": metric,
                "validation_mean": float(np.mean(v)),
                "test_mean": float(np.mean(t)),
                "mean_shift_test_minus_validation": float(np.mean(t) - np.mean(v)),
                "validation_median": float(np.median(v)),
                "test_median": float(np.median(t)),
                "median_shift_test_minus_validation": float(np.median(t) - np.median(v)),
            }
        )
    return pd.DataFrame(rows)


def missing_selected_samples(all_samples: pd.DataFrame, selected_rows: pd.DataFrame) -> pd.DataFrame:
    expected = _base_samples(all_samples)
    present = selected_rows[KEY_COLS].drop_duplicates().copy()
    marked = expected.merge(present.assign(_selected_present=1), on=KEY_COLS, how="left")
    missing = marked[marked["_selected_present"].isna()].drop(columns=["_selected_present"]).copy()
    return missing.reset_index(drop=True)


def write_report(
    path: Path,
    score_source: str,
    budget_policy: str,
    val_summary: pd.DataFrame,
    test_summary: pd.DataFrame,
    bootstrap: dict,
    shift: pd.DataFrame,
    missing: pd.DataFrame,
) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Selected Budget Policy Audit\n\n")
        f.write("This audit explains the policy selected on validation and evaluated on held-out test.\n\n")
        f.write("## Selected Policy\n\n")
        f.write(f"- score source: `{score_source}`\n")
        f.write(f"- budget policy: `{budget_policy}`\n\n")
        f.write("## Validation Summary\n\n")
        f.write(val_summary.to_markdown(index=False) if not val_summary.empty else "No validation rows.")
        f.write("\n\n## Test Summary\n\n")
        f.write(test_summary.to_markdown(index=False) if not test_summary.empty else "No test rows.")
        f.write("\n\n## Test Bootstrap\n\n")
        f.write(pd.DataFrame([bootstrap]).to_markdown(index=False))
        f.write("\n\n## Validation-To-Test Distribution Shift\n\n")
        f.write(shift.to_markdown(index=False) if not shift.empty else "No shift summary.")
        f.write("\n\n## Missing Test Samples\n\n")
        if missing.empty:
            f.write("No selected-policy test samples were missing relative to the full sample set.\n")
        else:
            f.write(
                f"{len(missing)} test samples were absent for the selected score source. "
                "Inspect `missing_selected_policy_test_samples.csv` before using this as a final headline number.\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a validation-selected persistence-growth budget policy.")
    parser.add_argument("--selection_dir", type=str, required=True)
    parser.add_argument("--validation_dir", type=str, required=True)
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    selection_dir = Path(args.selection_dir)
    validation_dir = Path(args.validation_dir)
    test_dir = Path(args.test_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    score_source, budget_policy = _selected_identity(selection_dir)
    val_samples = _safe_read_csv(validation_dir / "persistence_growth_budget_samples.csv")
    test_samples = _safe_read_csv(test_dir / "persistence_growth_budget_samples.csv")
    val_rows = _policy_rows(val_samples, score_source, budget_policy)
    test_rows = _policy_rows(test_samples, score_source, budget_policy)
    missing_test = missing_selected_samples(test_samples, test_rows)

    outputs = {
        "selected_policy_validation_samples.csv": val_rows,
        "selected_policy_test_samples.csv": test_rows,
        "missing_selected_policy_test_samples.csv": missing_test,
        "selected_policy_validation_overall.csv": summarize(val_rows, []),
        "selected_policy_test_overall.csv": summarize(test_rows, []),
        "selected_policy_test_by_tier.csv": summarize(test_rows, ["tier"]),
        "selected_policy_test_by_horizon.csv": summarize(test_rows, ["horizon"]),
        "selected_policy_test_by_absolute_growth_bin.csv": summarize(test_rows, ["absolute_growth_bin"]),
        "selected_policy_test_by_relative_growth_bin.csv": summarize(test_rows, ["relative_growth_bin"]),
        "selected_policy_test_by_tier_horizon.csv": summarize(test_rows, ["tier", "horizon"]),
        "selected_policy_test_by_tier_growth_bin.csv": summarize(test_rows, ["tier", "absolute_growth_bin"]),
        "selected_policy_validation_test_shift.csv": compare_split_distributions(val_rows, test_rows),
    }
    for filename, df in outputs.items():
        df.to_csv(output_dir / filename, index=False)

    bootstrap = _bootstrap_gap(test_rows, n_bootstrap=args.n_bootstrap, seed=args.seed)
    bootstrap_df = pd.DataFrame([{**{"score_source": score_source, "budget_policy": budget_policy}, **bootstrap}])
    bootstrap_df.to_csv(output_dir / "selected_policy_test_bootstrap.csv", index=False)

    report_payload = {
        "selection_dir": str(selection_dir),
        "validation_dir": str(validation_dir),
        "test_dir": str(test_dir),
        "score_source": score_source,
        "budget_policy": budget_policy,
        "validation_rows": int(len(val_rows)),
        "test_rows": int(len(test_rows)),
        "full_test_samples": int(len(_base_samples(test_samples))),
        "missing_test_samples": int(len(missing_test)),
        "test_bootstrap": bootstrap,
        "output_dir": str(output_dir),
    }
    with (output_dir / "selected_budget_policy_audit.json").open("w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    write_report(
        output_dir / "selected_budget_policy_audit.md",
        score_source=score_source,
        budget_policy=budget_policy,
        val_summary=outputs["selected_policy_validation_overall.csv"],
        test_summary=outputs["selected_policy_test_overall.csv"],
        bootstrap=bootstrap,
        shift=outputs["selected_policy_validation_test_shift.csv"],
        missing=missing_test,
    )
    print(json.dumps(report_payload, indent=2))


if __name__ == "__main__":
    main()
