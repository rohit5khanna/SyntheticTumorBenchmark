#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


KEY_COLS = ["patient_id", "input_idx", "target_idx", "horizon", "delta_days"]


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


def read_samples(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "persistence_growth_budget_samples.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df["policy_family"] = df["budget_policy"].map(policy_family)
    return df


def parse_list(payload: str | None) -> List[str]:
    if payload is None:
        return []
    return [x.strip() for x in payload.split(",") if x.strip()]


def parse_quantiles(payload: str) -> List[float]:
    return [float(x.strip()) for x in payload.split(",") if x.strip()]


def base_samples(samples: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "tier",
        "input_volume_vox",
        "target_volume_vox",
        "growth_volume_vox",
        "relative_new_growth",
        "locf_dice",
        "absolute_growth_bin",
        "relative_growth_bin",
    ]
    available = KEY_COLS + [c for c in cols if c in samples.columns]
    return samples[available].drop_duplicates(KEY_COLS).copy()


def filter_candidates(
    samples: pd.DataFrame,
    score_sources: Iterable[str],
    include_oracle: bool,
    include_fixed_candidate_fraction: bool,
) -> pd.DataFrame:
    out = samples.copy()
    score_sources_l = [s for s in score_sources if s]
    if score_sources_l:
        out = out[out["score_source"].isin(score_sources_l)]
    if not include_oracle:
        out = out[out["policy_family"] != "oracle"]
    if not include_fixed_candidate_fraction:
        out = out[out["policy_family"] != "fixed_candidate_fraction"]
    return out


def summarize_policy_candidates(samples: pd.DataFrame) -> pd.DataFrame:
    if samples.empty:
        return pd.DataFrame()
    return (
        samples.groupby(["score_source", "budget_policy", "policy_family"], observed=True, dropna=False)
        .agg(
            count=("persistence_growth_dice", "size"),
            mean_dice=("persistence_growth_dice", "mean"),
            mean_locf_dice=("locf_dice", "mean"),
            mean_gap_vs_locf=("dice_gap_vs_locf", "mean"),
            median_gap_vs_locf=("dice_gap_vs_locf", "median"),
            win_rate_vs_locf=("dice_gap_vs_locf", lambda x: float((x > 0).mean())),
            mean_growth_budget_vox=("growth_budget_vox", "mean"),
            mean_budget_to_true_growth_ratio=("budget_to_true_growth_ratio", "mean"),
        )
        .reset_index()
    )


def select_budget_per_score(validation_summary: pd.DataFrame, objective: str) -> pd.DataFrame:
    if validation_summary.empty:
        raise ValueError("No validation candidate policies available after filtering.")
    if objective == "mean_gap":
        sort_cols = ["score_source", "mean_gap_vs_locf", "win_rate_vs_locf", "mean_dice"]
    elif objective == "win_rate":
        sort_cols = ["score_source", "win_rate_vs_locf", "mean_gap_vs_locf", "mean_dice"]
    elif objective == "mean_dice":
        sort_cols = ["score_source", "mean_dice", "mean_gap_vs_locf", "win_rate_vs_locf"]
    else:
        raise ValueError(f"Unsupported objective: {objective}")
    ascending = [True] + [False] * (len(sort_cols) - 1)
    return (
        validation_summary.sort_values(sort_cols, ascending=ascending)
        .groupby("score_source", observed=True, dropna=False)
        .head(1)
        .reset_index(drop=True)
    )


def policy_samples(samples: pd.DataFrame, score_source: str, budget_policy: str) -> pd.DataFrame:
    base = base_samples(samples)
    selected = samples[
        (samples["score_source"] == score_source)
        & (samples["budget_policy"] == budget_policy)
    ].copy()
    keep = KEY_COLS + [
        "persistence_growth_dice",
        "dice_gap_vs_locf",
        "growth_budget_vox",
        "budget_to_true_growth_ratio",
    ]
    selected = selected[[c for c in keep if c in selected.columns]].copy()
    merged = base.merge(selected, on=KEY_COLS, how="left", indicator=True)
    merged["selected_policy_available"] = merged["_merge"].eq("both")
    merged = merged.drop(columns=["_merge"])

    # Missing rows usually correspond to score sources that are undefined for
    # empty-input cases. A deployable policy should fall back to persistence.
    merged["persistence_growth_dice"] = merged["persistence_growth_dice"].fillna(merged["locf_dice"])
    merged["dice_gap_vs_locf"] = merged["dice_gap_vs_locf"].fillna(0.0)
    merged["growth_budget_vox"] = merged["growth_budget_vox"].fillna(0.0)
    merged["budget_to_true_growth_ratio"] = merged["budget_to_true_growth_ratio"].fillna(0.0)
    merged["input_has_tumor"] = merged["input_volume_vox"] > 0
    merged["selected_budget_to_input_ratio"] = merged["growth_budget_vox"] / merged["input_volume_vox"].clip(lower=1)
    merged["selected_budget_to_input_pct"] = 100.0 * merged["selected_budget_to_input_ratio"]
    merged["score_source"] = score_source
    merged["budget_policy"] = budget_policy
    return merged


def threshold_values(values: pd.Series, quantiles: Iterable[float]) -> List[float]:
    clean = values.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if clean.empty:
        return []
    out = {float(clean.min()), float(clean.max())}
    out.update(float(clean.quantile(q)) for q in quantiles)
    return sorted(out)


def build_gate_candidates(validation: pd.DataFrame, quantiles: Iterable[float]) -> pd.DataFrame:
    rows = [
        {
            "gate_name": "always_off_locf",
            "feature": "constant",
            "operator": "always_off",
            "threshold": np.nan,
        },
        {
            "gate_name": "always_on_ranked_growth_if_available",
            "feature": "selected_policy_available_and_input_has_tumor",
            "operator": "always_on",
            "threshold": np.nan,
        },
    ]
    feature_specs = [
        ("growth_budget_vox", ">="),
        ("selected_budget_to_input_ratio", ">="),
        ("selected_budget_to_input_pct", ">="),
        ("input_volume_vox", ">="),
        ("delta_days", ">="),
    ]
    for feature, operator in feature_specs:
        if feature not in validation.columns:
            continue
        for threshold in threshold_values(validation[feature], quantiles):
            rows.append(
                {
                    "gate_name": f"{feature}_{operator}_{threshold:.6g}",
                    "feature": feature,
                    "operator": operator,
                    "threshold": threshold,
                }
            )
    return pd.DataFrame(rows)


def apply_gate(samples: pd.DataFrame, gate: pd.Series) -> pd.DataFrame:
    out = samples.copy()
    if gate["operator"] == "always_off":
        active = pd.Series(False, index=out.index)
    elif gate["operator"] == "always_on":
        active = out["selected_policy_available"] & out["input_has_tumor"]
    elif gate["operator"] == ">=":
        active = out[str(gate["feature"])].astype(float) >= float(gate["threshold"])
        active = active & out["selected_policy_available"] & out["input_has_tumor"]
    else:
        raise ValueError(f"Unsupported gate operator: {gate['operator']}")

    out["gate_active"] = active.astype(bool)
    out["hybrid_policy_dice"] = np.where(out["gate_active"], out["persistence_growth_dice"], out["locf_dice"])
    out["hybrid_policy_gap_vs_locf"] = out["hybrid_policy_dice"] - out["locf_dice"]
    out["gate_name"] = gate["gate_name"]
    out["gate_feature"] = gate["feature"]
    out["gate_operator"] = gate["operator"]
    out["gate_threshold"] = gate["threshold"]
    return out


def summarize_gated(samples: pd.DataFrame, group_cols: Iterable[str]) -> pd.DataFrame:
    if samples.empty:
        return pd.DataFrame()
    cols = [c for c in group_cols if c in samples.columns]
    group_df = samples if cols else samples.assign(_overall="overall")
    by = cols if cols else ["_overall"]
    out = (
        group_df.groupby(by, observed=True, dropna=False)
        .agg(
            count=("hybrid_policy_dice", "size"),
            mean_dice=("hybrid_policy_dice", "mean"),
            mean_locf_dice=("locf_dice", "mean"),
            mean_gap_vs_locf=("hybrid_policy_gap_vs_locf", "mean"),
            median_gap_vs_locf=("hybrid_policy_gap_vs_locf", "median"),
            win_rate_vs_locf=("hybrid_policy_gap_vs_locf", lambda x: float((x > 0).mean())),
            gate_active_rate=("gate_active", "mean"),
            selected_policy_available_rate=("selected_policy_available", "mean"),
            mean_growth_volume_vox=("growth_volume_vox", "mean"),
            median_growth_volume_vox=("growth_volume_vox", "median"),
            mean_growth_budget_vox=("growth_budget_vox", "mean"),
            mean_budget_to_true_growth_ratio=("budget_to_true_growth_ratio", "mean"),
        )
        .reset_index()
    )
    if "_overall" in out.columns:
        out = out.drop(columns=["_overall"])
    return out.sort_values(cols).reset_index(drop=True) if cols else out


def select_gate(validation_samples: pd.DataFrame, quantiles: Iterable[float], objective: str) -> tuple[pd.Series, pd.DataFrame]:
    gates = build_gate_candidates(validation_samples, quantiles=quantiles)
    rows = []
    for _, gate in gates.iterrows():
        gated = apply_gate(validation_samples, gate)
        overall = summarize_gated(gated, []).iloc[0].to_dict()
        rows.append({**gate.to_dict(), **overall})
    summary = pd.DataFrame(rows)
    if objective == "mean_gap":
        sort_cols = ["mean_gap_vs_locf", "win_rate_vs_locf", "mean_dice"]
    elif objective == "win_rate":
        sort_cols = ["win_rate_vs_locf", "mean_gap_vs_locf", "mean_dice"]
    elif objective == "mean_dice":
        sort_cols = ["mean_dice", "mean_gap_vs_locf", "win_rate_vs_locf"]
    else:
        raise ValueError(f"Unsupported gate objective: {objective}")
    selected = summary.sort_values(sort_cols, ascending=False).iloc[0]
    return selected, summary


def bootstrap_gap(samples: pd.DataFrame, gap_col: str, n_bootstrap: int, seed: int) -> Dict:
    gaps = samples[gap_col].dropna().to_numpy(dtype=float)
    if len(gaps) == 0:
        return {"n_samples": 0, "mean_gap": np.nan, "ci_low": np.nan, "ci_high": np.nan, "win_rate": np.nan}
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(gaps), len(gaps))
        boot.append(float(gaps[idx].mean()))
    boot_arr = np.asarray(boot, dtype=float)
    return {
        "n_samples": int(len(gaps)),
        "mean_gap": float(gaps.mean()),
        "ci_low": float(np.quantile(boot_arr, 0.025)),
        "ci_high": float(np.quantile(boot_arr, 0.975)),
        "win_rate": float((gaps > 0).mean()),
    }


def write_report(
    output_dir: Path,
    selected_budgets: pd.DataFrame,
    selected_gates: pd.DataFrame,
    test_overall: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> None:
    with (output_dir / "hybrid_forecast_policy_report.md").open("w", encoding="utf-8") as f:
        f.write("# Hybrid Forecast Policy Evaluation\n\n")
        f.write(
            "This report selects a budget rule and a simple activation gate on validation for each score source, "
            "then evaluates the selected policy on held-out test.\n\n"
        )
        f.write("## Selected Budget Rules\n\n")
        f.write(selected_budgets.to_markdown(index=False) if not selected_budgets.empty else "No selected budgets.")
        f.write("\n\n## Selected Gates\n\n")
        f.write(selected_gates.to_markdown(index=False) if not selected_gates.empty else "No selected gates.")
        f.write("\n\n## Test Overall\n\n")
        f.write(test_overall.to_markdown(index=False) if not test_overall.empty else "No test summary.")
        f.write("\n\n## Test Bootstrap\n\n")
        f.write(bootstrap.to_markdown(index=False) if not bootstrap.empty else "No bootstrap summary.")
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate validation-selected hybrid forecast policies built from persistence plus ranked growth."
    )
    parser.add_argument("--validation_dir", type=str, required=True)
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--score_sources",
        type=str,
        default="distance_to_input_mask,resunet_image_mask,hybrid_distance_resunet_image_mask_a0.75",
    )
    parser.add_argument("--budget_objective", type=str, default="mean_gap", choices=["mean_gap", "win_rate", "mean_dice"])
    parser.add_argument("--gate_objective", type=str, default="mean_gap", choices=["mean_gap", "win_rate", "mean_dice"])
    parser.add_argument("--include_oracle", action="store_true")
    parser.add_argument("--include_fixed_candidate_fraction", action="store_true")
    parser.add_argument("--quantiles", type=str, default="0,0.1,0.25,0.5,0.75,0.9,1.0")
    parser.add_argument("--n_bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    validation_dir = Path(args.validation_dir)
    test_dir = Path(args.test_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    score_sources = parse_list(args.score_sources)
    quantiles = parse_quantiles(args.quantiles)
    validation_all = read_samples(validation_dir)
    test_all = read_samples(test_dir)

    validation_candidates = filter_candidates(
        validation_all,
        score_sources=score_sources,
        include_oracle=args.include_oracle,
        include_fixed_candidate_fraction=args.include_fixed_candidate_fraction,
    )
    validation_candidate_summary = summarize_policy_candidates(validation_candidates)
    selected_budgets = select_budget_per_score(validation_candidate_summary, objective=args.budget_objective)

    all_validation_gates = []
    selected_gate_rows = []
    gated_validation_rows = []
    gated_test_rows = []

    for _, budget in selected_budgets.iterrows():
        score_source = str(budget["score_source"])
        budget_policy = str(budget["budget_policy"])
        val_policy = policy_samples(validation_all, score_source, budget_policy)
        test_policy = policy_samples(test_all, score_source, budget_policy)
        selected_gate, gate_summary = select_gate(val_policy, quantiles=quantiles, objective=args.gate_objective)

        gate_summary = gate_summary.copy()
        gate_summary["score_source"] = score_source
        gate_summary["budget_policy"] = budget_policy
        all_validation_gates.append(gate_summary)

        val_gated = apply_gate(val_policy, selected_gate)
        test_gated = apply_gate(test_policy, selected_gate)
        gated_validation_rows.append(val_gated)
        gated_test_rows.append(test_gated)

        selected_gate_payload = selected_gate.to_dict()
        selected_gate_payload["score_source"] = score_source
        selected_gate_payload["budget_policy"] = budget_policy
        selected_gate_rows.append(selected_gate_payload)

    selected_gates = pd.DataFrame(selected_gate_rows)
    validation_gate_candidates = pd.concat(all_validation_gates, ignore_index=True) if all_validation_gates else pd.DataFrame()
    gated_validation = pd.concat(gated_validation_rows, ignore_index=True) if gated_validation_rows else pd.DataFrame()
    gated_test = pd.concat(gated_test_rows, ignore_index=True) if gated_test_rows else pd.DataFrame()

    validation_overall = summarize_gated(gated_validation, ["score_source", "budget_policy", "gate_name"])
    test_overall = summarize_gated(gated_test, ["score_source", "budget_policy", "gate_name"])
    test_by_tier = summarize_gated(gated_test, ["score_source", "tier"])
    test_by_horizon = summarize_gated(gated_test, ["score_source", "horizon"])
    test_by_growth = summarize_gated(gated_test, ["score_source", "absolute_growth_bin"])
    test_by_tier_growth = summarize_gated(gated_test, ["score_source", "tier", "absolute_growth_bin"])

    bootstrap_rows = []
    for score_source, rows in gated_test.groupby("score_source", observed=True, dropna=False):
        boot = bootstrap_gap(rows, "hybrid_policy_gap_vs_locf", n_bootstrap=args.n_bootstrap, seed=args.seed)
        bootstrap_rows.append({"score_source": score_source, **boot})
    bootstrap = pd.DataFrame(bootstrap_rows)

    validation_candidate_summary.to_csv(output_dir / "validation_budget_candidates.csv", index=False)
    selected_budgets.to_csv(output_dir / "selected_budget_per_score_source.csv", index=False)
    validation_gate_candidates.to_csv(output_dir / "validation_gate_candidates_by_score_source.csv", index=False)
    selected_gates.to_csv(output_dir / "selected_gate_per_score_source.csv", index=False)
    gated_validation.to_csv(output_dir / "hybrid_policy_validation_samples.csv", index=False)
    gated_test.to_csv(output_dir / "hybrid_policy_test_samples.csv", index=False)
    validation_overall.to_csv(output_dir / "hybrid_policy_validation_overall.csv", index=False)
    test_overall.to_csv(output_dir / "hybrid_policy_test_overall.csv", index=False)
    test_by_tier.to_csv(output_dir / "hybrid_policy_test_by_tier.csv", index=False)
    test_by_horizon.to_csv(output_dir / "hybrid_policy_test_by_horizon.csv", index=False)
    test_by_growth.to_csv(output_dir / "hybrid_policy_test_by_absolute_growth_bin.csv", index=False)
    test_by_tier_growth.to_csv(output_dir / "hybrid_policy_test_by_tier_growth_bin.csv", index=False)
    bootstrap.to_csv(output_dir / "hybrid_policy_test_bootstrap.csv", index=False)

    report = {
        "validation_dir": str(validation_dir),
        "test_dir": str(test_dir),
        "score_sources": score_sources,
        "budget_objective": args.budget_objective,
        "gate_objective": args.gate_objective,
        "include_oracle": bool(args.include_oracle),
        "include_fixed_candidate_fraction": bool(args.include_fixed_candidate_fraction),
        "n_selected_score_sources": int(len(selected_budgets)),
        "output_dir": str(output_dir),
        "files": [
            "validation_budget_candidates.csv",
            "selected_budget_per_score_source.csv",
            "validation_gate_candidates_by_score_source.csv",
            "selected_gate_per_score_source.csv",
            "hybrid_policy_test_overall.csv",
            "hybrid_policy_test_by_tier.csv",
            "hybrid_policy_test_by_horizon.csv",
            "hybrid_policy_test_by_absolute_growth_bin.csv",
            "hybrid_policy_test_bootstrap.csv",
            "hybrid_forecast_policy_report.md",
        ],
    }
    with (output_dir / "hybrid_forecast_policy_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    write_report(output_dir, selected_budgets, selected_gates, test_overall, bootstrap)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
