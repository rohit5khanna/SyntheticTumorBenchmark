#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


KEY_COLS = ["patient_id", "input_idx", "target_idx", "horizon", "delta_days"]


def infer_base_model(method: str) -> str | None:
    if method in {"random_prevalence", "distance_to_input_mask"}:
        return None
    if method.startswith("hybrid_distance_"):
        match = re.match(r"hybrid_distance_(.+)_a[0-9.]+$", method)
        if match:
            return match.group(1)
    return method


def load_inputs(growth_eval_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranking_path = growth_eval_dir / "growth_ranking_metrics.csv"
    features_path = growth_eval_dir / "growth_sample_features.csv"
    if not ranking_path.exists():
        raise FileNotFoundError(f"Missing ranking metrics: {ranking_path}")
    if not features_path.exists():
        raise FileNotFoundError(f"Missing growth features: {features_path}")

    ranking = pd.read_csv(ranking_path)
    features = pd.read_csv(features_path)
    feature_cols = [
        c
        for c in features.columns
        if c not in ranking.columns or c in KEY_COLS
    ]
    merged = ranking.merge(features[feature_cols], on=KEY_COLS, how="left")
    return merged, features


def candidate_methods_for_model(ranking: pd.DataFrame, base_model: str) -> List[str]:
    methods = set(ranking["method"].dropna().astype(str))
    candidates = ["distance_to_input_mask", base_model]
    candidates.extend(sorted(m for m in methods if m.startswith(f"hybrid_distance_{base_model}_a")))
    return [m for m in candidates if m in methods]


def build_candidate_matrix(ranking: pd.DataFrame, base_model: str, metric: str) -> pd.DataFrame:
    candidates = candidate_methods_for_model(ranking, base_model)
    if len(candidates) < 2:
        raise ValueError(f"Need at least two candidate methods for {base_model}; found {candidates}")

    keep_cols = KEY_COLS + [
        "tier",
        "new_growth_bin",
        "abs_change_bin",
        "net_growth_bin",
        "relative_new_growth",
        "relative_abs_change",
        "growth_volume_vox",
    ]
    keep_cols = [c for c in keep_cols if c in ranking.columns]

    sub = ranking[ranking["method"].isin(candidates)].copy()
    pivot = sub.pivot_table(index=keep_cols, columns="method", values=metric, aggfunc="mean").reset_index()
    pivot = pivot.dropna(subset=candidates, how="any").copy()
    if pivot.empty:
        raise ValueError(f"No complete candidate rows for {base_model}.")

    values = pivot[candidates]
    pivot["oracle_best_method"] = values.idxmax(axis=1)
    pivot["oracle_best_metric"] = values.max(axis=1)
    pivot["distance_metric"] = pivot["distance_to_input_mask"]
    pivot["model_metric"] = pivot[base_model]
    pivot["best_static_candidate"] = values.mean(axis=0).idxmax()
    pivot["best_static_metric_for_sample"] = pivot[pivot["best_static_candidate"].iloc[0]]
    pivot["oracle_gain_vs_distance"] = pivot["oracle_best_metric"] - pivot["distance_metric"]
    pivot["oracle_gain_vs_model"] = pivot["oracle_best_metric"] - pivot["model_metric"]
    pivot["oracle_gain_vs_best_static"] = pivot["oracle_best_metric"] - pivot["best_static_metric_for_sample"]
    pivot["base_model"] = base_model
    return pivot


def summarize_oracle(matrix: pd.DataFrame, metric: str) -> pd.DataFrame:
    candidate_cols = [
        c
        for c in matrix.columns
        if c in {"distance_to_input_mask", matrix["base_model"].iloc[0]} or c.startswith("hybrid_distance_")
    ]
    means = matrix[candidate_cols].mean(axis=0).sort_values(ascending=False)
    base_model = matrix["base_model"].iloc[0]
    return pd.DataFrame(
        [
            {
                "base_model": base_model,
                "metric": metric,
                "n_samples": int(len(matrix)),
                "distance_mean": float(matrix["distance_metric"].mean()),
                "model_mean": float(matrix["model_metric"].mean()),
                "best_static_method": str(means.index[0]),
                "best_static_mean": float(means.iloc[0]),
                "oracle_mean": float(matrix["oracle_best_metric"].mean()),
                "oracle_gain_vs_distance": float(matrix["oracle_gain_vs_distance"].mean()),
                "oracle_gain_vs_model": float(matrix["oracle_gain_vs_model"].mean()),
                "oracle_gain_vs_best_static": float(matrix["oracle_gain_vs_best_static"].mean()),
            }
        ]
    )


def summarize_choice_counts(matrix: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    available = [c for c in group_cols if c in matrix.columns]
    if not available:
        available = ["base_model"]
    counts = (
        matrix.groupby(available + ["oracle_best_method"], dropna=False, observed=True)
        .size()
        .reset_index(name="count")
    )
    totals = counts.groupby(available)["count"].transform("sum")
    counts["fraction"] = counts["count"] / totals
    return counts.sort_values(available + ["fraction"], ascending=[True] * len(available) + [False])


def group_policy_summary(matrix: pd.DataFrame, group_col: str, metric: str) -> pd.DataFrame:
    if group_col not in matrix.columns:
        return pd.DataFrame()
    candidate_cols = [
        c
        for c in matrix.columns
        if c in {"distance_to_input_mask", matrix["base_model"].iloc[0]} or c.startswith("hybrid_distance_")
    ]
    rows = []
    for group_value, group in matrix.groupby(group_col, dropna=False, observed=True):
        means = group[candidate_cols].mean(axis=0).sort_values(ascending=False)
        best_method = str(means.index[0])
        rows.append(
            {
                "base_model": matrix["base_model"].iloc[0],
                "group_col": group_col,
                "group_value": group_value,
                "metric": metric,
                "count": int(len(group)),
                "best_group_method": best_method,
                "best_group_mean": float(means.iloc[0]),
                "distance_mean": float(group["distance_metric"].mean()),
                "model_mean": float(group["model_metric"].mean()),
                "oracle_mean": float(group["oracle_best_metric"].mean()),
            }
        )
    return pd.DataFrame(rows)


def write_report(path: Path, oracle_summary: pd.DataFrame, group_policy: pd.DataFrame) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Adaptive Ranking Gate Analysis\n\n")
        f.write("This report estimates whether a case- or group-adaptive selector could improve over static distance/model/hybrid ranking.\n\n")
        f.write("## Oracle Summary\n\n")
        f.write(oracle_summary.to_markdown(index=False))
        f.write("\n\n## Group Policy Summary\n\n")
        if group_policy.empty:
            f.write("No group policy summary available.")
        else:
            f.write(group_policy.to_markdown(index=False))
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate oracle and group-policy gains for adaptive distance/model growth-ranking gates."
    )
    parser.add_argument("--growth_eval_dir", type=str, required=True)
    parser.add_argument("--metric", type=str, default="growth_average_precision")
    parser.add_argument("--base_models", type=str, default="unet_image_mask,resunet_image_mask")
    parser.add_argument("--group_cols", type=str, default="new_growth_bin,tier,horizon")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    growth_eval_dir = Path(args.growth_eval_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ranking, _ = load_inputs(growth_eval_dir)
    base_models = [m.strip() for m in args.base_models.split(",") if m.strip()]
    group_cols = [c.strip() for c in args.group_cols.split(",") if c.strip()]

    matrices = []
    oracle_summaries = []
    choice_summaries = []
    policy_summaries = []

    for base_model in base_models:
        matrix = build_candidate_matrix(ranking, base_model, args.metric)
        matrices.append(matrix)
        oracle_summaries.append(summarize_oracle(matrix, args.metric))
        choice_summaries.append(summarize_choice_counts(matrix, ["base_model"]))
        for group_col in group_cols:
            choice_summaries.append(summarize_choice_counts(matrix, ["base_model", group_col]))
            policy_summaries.append(group_policy_summary(matrix, group_col, args.metric))

    oracle_samples = pd.concat(matrices, ignore_index=True)
    oracle_summary = pd.concat(oracle_summaries, ignore_index=True)
    choice_summary = pd.concat(choice_summaries, ignore_index=True)
    group_policy = pd.concat([p for p in policy_summaries if not p.empty], ignore_index=True)

    oracle_samples.to_csv(output_dir / "adaptive_oracle_samples.csv", index=False)
    oracle_summary.to_csv(output_dir / "adaptive_oracle_summary.csv", index=False)
    choice_summary.to_csv(output_dir / "adaptive_choice_counts.csv", index=False)
    group_policy.to_csv(output_dir / "adaptive_group_policy_summary.csv", index=False)
    write_report(output_dir / "adaptive_ranking_gate_report.md", oracle_summary, group_policy)

    print(
        json.dumps(
            {
                "growth_eval_dir": str(growth_eval_dir),
                "metric": args.metric,
                "base_models": base_models,
                "n_oracle_rows": int(len(oracle_samples)),
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
