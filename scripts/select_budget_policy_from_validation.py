#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def policy_family(policy: str) -> str:
    if policy == "oracle_true_growth_volume":
        return "oracle"
    if policy in {"one_pct_candidates", "five_pct_candidates"}:
        return "fixed_candidate_fraction"
    if "_cap_input_" in policy:
        return "capped_previous_growth"
    if "_zero_if_prev_le_" in policy:
        return "zero_rule_previous_growth"
    if policy.startswith("prev_growth_x") or policy == "previous_growth_volume":
        return "scaled_previous_growth"
    return "other"


def load_overall(run_dir: Path, split_name: str) -> pd.DataFrame:
    path = run_dir / "persistence_growth_budget_overall.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df["split_source"] = split_name
    df["policy_family"] = df["budget_policy"].map(policy_family)
    return df


def load_samples(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "persistence_growth_budget_samples.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df["policy_family"] = df["budget_policy"].map(policy_family)
    return df


def filter_candidates(
    df: pd.DataFrame,
    include_oracle: bool,
    include_fixed_candidate_fraction: bool,
    include_distance_only: bool,
) -> pd.DataFrame:
    out = df.copy()
    if not include_oracle:
        out = out[out["policy_family"] != "oracle"]
    if not include_fixed_candidate_fraction:
        out = out[out["policy_family"] != "fixed_candidate_fraction"]
    if not include_distance_only:
        out = out[out["score_source"] != "distance_to_input_mask"]
    return out


def select_policy(candidates: pd.DataFrame, objective: str) -> pd.Series:
    if candidates.empty:
        raise ValueError("No candidate policies remain after filtering.")
    if objective == "mean_gap":
        sort_cols = ["mean_gap_vs_locf", "win_rate_vs_locf", "mean_dice"]
    elif objective == "win_rate":
        sort_cols = ["win_rate_vs_locf", "mean_gap_vs_locf", "mean_dice"]
    elif objective == "mean_dice":
        sort_cols = ["mean_dice", "mean_gap_vs_locf", "win_rate_vs_locf"]
    else:
        raise ValueError(f"Unsupported objective: {objective}")
    return candidates.sort_values(sort_cols, ascending=False).iloc[0]


def bootstrap_selected(samples: pd.DataFrame, selected: pd.Series, n_bootstrap: int, seed: int) -> Dict:
    rows = samples[
        (samples["score_source"] == selected["score_source"])
        & (samples["budget_policy"] == selected["budget_policy"])
    ].copy()
    if rows.empty:
        return {
            "n_samples": 0,
            "mean_gap": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "win_rate": float("nan"),
        }

    gaps = rows["dice_gap_vs_locf"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(gaps), size=len(gaps))
        boot.append(float(gaps[idx].mean()))
    boot_arr = np.asarray(boot, dtype=float)
    return {
        "n_samples": int(len(gaps)),
        "mean_gap": float(gaps.mean()),
        "ci_low": float(np.quantile(boot_arr, 0.025)),
        "ci_high": float(np.quantile(boot_arr, 0.975)),
        "win_rate": float((gaps > 0).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select a deployable persistence-growth budget policy on validation and evaluate it on held-out test."
    )
    parser.add_argument("--validation_dir", type=str, required=True)
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--objective", type=str, default="mean_gap", choices=["mean_gap", "win_rate", "mean_dice"])
    parser.add_argument("--include_oracle", action="store_true")
    parser.add_argument("--include_fixed_candidate_fraction", action="store_true")
    parser.add_argument("--include_distance_only", action="store_true")
    parser.add_argument("--n_bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    val_dir = Path(args.validation_dir)
    test_dir = Path(args.test_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    val = load_overall(val_dir, "validation")
    test = load_overall(test_dir, "test")
    val_candidates = filter_candidates(
        val,
        include_oracle=args.include_oracle,
        include_fixed_candidate_fraction=args.include_fixed_candidate_fraction,
        include_distance_only=args.include_distance_only,
    )
    selected = select_policy(val_candidates, objective=args.objective)

    merged = val.merge(
        test,
        on=["score_source", "budget_policy", "policy_family"],
        how="inner",
        suffixes=("_validation", "_test"),
    )
    merged_candidates = filter_candidates(
        merged.rename(columns={"split_source_validation": "split_source"}),
        include_oracle=args.include_oracle,
        include_fixed_candidate_fraction=args.include_fixed_candidate_fraction,
        include_distance_only=args.include_distance_only,
    )
    selected_pair = merged[
        (merged["score_source"] == selected["score_source"])
        & (merged["budget_policy"] == selected["budget_policy"])
    ].copy()

    test_samples = load_samples(test_dir)
    boot = bootstrap_selected(test_samples, selected, n_bootstrap=args.n_bootstrap, seed=args.seed)
    boot_df = pd.DataFrame([{**selected[["score_source", "budget_policy", "policy_family"]].to_dict(), **boot}])

    merged.to_csv(output_dir / "validation_test_policy_table.csv", index=False)
    merged_candidates.to_csv(output_dir / "validation_test_deployable_candidates.csv", index=False)
    selected_pair.to_csv(output_dir / "selected_policy_validation_test_row.csv", index=False)
    boot_df.to_csv(output_dir / "selected_policy_test_bootstrap.csv", index=False)

    report = {
        "validation_dir": str(val_dir),
        "test_dir": str(test_dir),
        "objective": args.objective,
        "include_oracle": bool(args.include_oracle),
        "include_fixed_candidate_fraction": bool(args.include_fixed_candidate_fraction),
        "include_distance_only": bool(args.include_distance_only),
        "n_validation_candidates": int(len(val_candidates)),
        "selected_score_source": str(selected["score_source"]),
        "selected_budget_policy": str(selected["budget_policy"]),
        "selected_policy_family": str(selected["policy_family"]),
        "validation_mean_gap_vs_locf": float(selected["mean_gap_vs_locf"]),
        "validation_win_rate_vs_locf": float(selected["win_rate_vs_locf"]),
        "test_bootstrap": boot,
        "output_dir": str(output_dir),
    }
    with (output_dir / "budget_policy_selection_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with (output_dir / "budget_policy_selection_report.md").open("w", encoding="utf-8") as f:
        f.write("# Budget Policy Validation-To-Test Selection\n\n")
        f.write("This report selects a deployable policy on validation and evaluates the selected policy on test.\n\n")
        f.write("## Selected Policy\n\n")
        f.write(pd.DataFrame([report]).to_markdown(index=False))
        f.write("\n\n## Selected Policy Validation/Test Row\n\n")
        f.write(selected_pair.to_markdown(index=False) if not selected_pair.empty else "Selected row not found in test table.")
        f.write("\n\n## Test Bootstrap\n\n")
        f.write(boot_df.to_markdown(index=False))
        f.write("\n")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
