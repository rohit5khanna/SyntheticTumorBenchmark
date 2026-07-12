#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List

import numpy as np
import pandas as pd


KEY_COLS = ["patient_id", "input_idx", "target_idx", "horizon", "delta_days"]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _selected_policy(selection_dir: Path) -> tuple[str, str]:
    selected = _read_csv(selection_dir / "selected_policy_validation_test_row.csv")
    if selected.empty:
        raise ValueError("Selected policy table is empty.")
    return str(selected.iloc[0]["score_source"]), str(selected.iloc[0]["budget_policy"])


def _base_samples(samples: pd.DataFrame) -> pd.DataFrame:
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


def _selected_rows(samples: pd.DataFrame, score_source: str, budget_policy: str) -> pd.DataFrame:
    rows = samples[
        (samples["score_source"] == score_source)
        & (samples["budget_policy"] == budget_policy)
    ].copy()
    keep = KEY_COLS + [
        "persistence_growth_dice",
        "dice_gap_vs_locf",
        "growth_budget_vox",
        "budget_to_true_growth_ratio",
    ]
    return rows[[c for c in keep if c in rows.columns]].copy()


def load_policy_samples(run_dir: Path, score_source: str, budget_policy: str) -> pd.DataFrame:
    all_rows = _read_csv(run_dir / "persistence_growth_budget_samples.csv")
    base = _base_samples(all_rows)
    selected = _selected_rows(all_rows, score_source, budget_policy)
    merged = base.merge(selected, on=KEY_COLS, how="left", indicator=True)
    merged["selected_policy_available"] = merged["_merge"].eq("both")
    merged = merged.drop(columns=["_merge"])

    # If the selected ranking source is unavailable, fall back to LOCF. This
    # handles empty-mask samples where distance-based hybrid ranking is undefined.
    merged["persistence_growth_dice"] = merged["persistence_growth_dice"].fillna(merged["locf_dice"])
    merged["dice_gap_vs_locf"] = merged["dice_gap_vs_locf"].fillna(0.0)
    merged["growth_budget_vox"] = merged["growth_budget_vox"].fillna(0.0)
    merged["budget_to_true_growth_ratio"] = merged["budget_to_true_growth_ratio"].fillna(0.0)

    merged["input_has_tumor"] = merged["input_volume_vox"] > 0
    merged["selected_budget_to_input_ratio"] = merged["growth_budget_vox"] / merged["input_volume_vox"].clip(lower=1)
    merged["selected_budget_to_input_pct"] = 100.0 * merged["selected_budget_to_input_ratio"]
    return merged


def _thresholds(values: pd.Series, quantiles: Iterable[float]) -> List[float]:
    clean = values.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if clean.empty:
        return []
    out = {float(clean.quantile(q)) for q in quantiles}
    out.update({float(clean.min()), float(clean.max())})
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
        for threshold in _thresholds(validation[feature], quantiles):
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
        active = out[gate["feature"]].astype(float) >= float(gate["threshold"])
        active = active & out["selected_policy_available"] & out["input_has_tumor"]
    else:
        raise ValueError(f"Unsupported gate operator: {gate['operator']}")

    out["gate_active"] = active.astype(bool)
    out["gated_dice"] = np.where(out["gate_active"], out["persistence_growth_dice"], out["locf_dice"])
    out["gated_gap_vs_locf"] = out["gated_dice"] - out["locf_dice"]
    out["gate_name"] = gate["gate_name"]
    out["gate_feature"] = gate["feature"]
    out["gate_operator"] = gate["operator"]
    out["gate_threshold"] = gate["threshold"]
    return out


def summarize(samples: pd.DataFrame, group_cols: Iterable[str]) -> pd.DataFrame:
    if samples.empty:
        return pd.DataFrame()
    cols = [c for c in group_cols if c in samples.columns]
    group_df = samples if cols else samples.assign(_overall="overall")
    by = cols if cols else ["_overall"]
    out = (
        group_df.groupby(by, observed=True, dropna=False)
        .agg(
            count=("gated_dice", "size"),
            mean_dice=("gated_dice", "mean"),
            mean_locf_dice=("locf_dice", "mean"),
            mean_gap_vs_locf=("gated_gap_vs_locf", "mean"),
            median_gap_vs_locf=("gated_gap_vs_locf", "median"),
            win_rate_vs_locf=("gated_gap_vs_locf", lambda x: float((x > 0).mean())),
            gate_active_rate=("gate_active", "mean"),
            mean_growth_volume_vox=("growth_volume_vox", "mean"),
            median_growth_volume_vox=("growth_volume_vox", "median"),
            mean_growth_budget_vox=("growth_budget_vox", "mean"),
            selected_policy_available_rate=("selected_policy_available", "mean"),
        )
        .reset_index()
    )
    if "_overall" in out.columns:
        out = out.drop(columns=["_overall"])
    return out.sort_values(cols).reset_index(drop=True) if cols else out


def evaluate_gates(samples: pd.DataFrame, gates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, gate in gates.iterrows():
        pred = apply_gate(samples, gate)
        overall = summarize(pred, []).iloc[0].to_dict()
        rows.append(
            {
                **gate.to_dict(),
                **overall,
            }
        )
    return pd.DataFrame(rows)


def select_gate(gate_summary: pd.DataFrame, objective: str) -> pd.Series:
    if objective == "mean_gap":
        sort_cols = ["mean_gap_vs_locf", "win_rate_vs_locf", "mean_dice"]
    elif objective == "win_rate":
        sort_cols = ["win_rate_vs_locf", "mean_gap_vs_locf", "mean_dice"]
    elif objective == "mean_dice":
        sort_cols = ["mean_dice", "mean_gap_vs_locf", "win_rate_vs_locf"]
    else:
        raise ValueError(f"Unsupported objective: {objective}")
    return gate_summary.sort_values(sort_cols, ascending=False).iloc[0]


def bootstrap_gap(samples: pd.DataFrame, n_bootstrap: int, seed: int) -> Dict:
    gaps = samples["gated_gap_vs_locf"].to_numpy(dtype=float)
    if len(gaps) == 0:
        return {"n_samples": 0, "mean_gap": np.nan, "ci_low": np.nan, "ci_high": np.nan, "win_rate": np.nan}
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


def write_report(
    output_dir: Path,
    score_source: str,
    budget_policy: str,
    selected_gate: pd.Series,
    validation_overall: pd.DataFrame,
    test_overall: pd.DataFrame,
    test_by_growth: pd.DataFrame,
    bootstrap: Dict,
) -> None:
    with (output_dir / "gated_budget_policy_report.md").open("w", encoding="utf-8") as f:
        f.write("# Validation-Selected Gated Budget Policy\n\n")
        f.write("This report selects a simple growth-activity gate on validation and evaluates it on held-out test.\n\n")
        f.write("## Base Ranked-Growth Policy\n\n")
        f.write(f"- score source: `{score_source}`\n")
        f.write(f"- budget policy: `{budget_policy}`\n\n")
        f.write("## Selected Gate\n\n")
        f.write(pd.DataFrame([selected_gate]).to_markdown(index=False))
        f.write("\n\n## Validation Overall\n\n")
        f.write(validation_overall.to_markdown(index=False))
        f.write("\n\n## Test Overall\n\n")
        f.write(test_overall.to_markdown(index=False))
        f.write("\n\n## Test Bootstrap\n\n")
        f.write(pd.DataFrame([bootstrap]).to_markdown(index=False))
        f.write("\n\n## Test By Absolute Growth Bin\n\n")
        f.write(test_by_growth.to_markdown(index=False) if not test_by_growth.empty else "No growth-bin summary.")
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select a simple growth-activity gate on validation for a persistence-ranked growth budget policy."
    )
    parser.add_argument("--selection_dir", type=str, required=True)
    parser.add_argument("--validation_dir", type=str, required=True)
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--objective", type=str, default="mean_gap", choices=["mean_gap", "win_rate", "mean_dice"])
    parser.add_argument("--quantiles", type=str, default="0,0.1,0.25,0.5,0.75,0.9,1.0")
    parser.add_argument("--n_bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    selection_dir = Path(args.selection_dir)
    validation_dir = Path(args.validation_dir)
    test_dir = Path(args.test_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    score_source, budget_policy = _selected_policy(selection_dir)
    validation = load_policy_samples(validation_dir, score_source, budget_policy)
    test = load_policy_samples(test_dir, score_source, budget_policy)
    quantiles = [float(q.strip()) for q in args.quantiles.split(",") if q.strip()]

    gates = build_gate_candidates(validation, quantiles=quantiles)
    validation_gate_summary = evaluate_gates(validation, gates)
    selected_gate = select_gate(validation_gate_summary, objective=args.objective)

    gated_validation = apply_gate(validation, selected_gate)
    gated_test = apply_gate(test, selected_gate)
    test_bootstrap = bootstrap_gap(gated_test, n_bootstrap=args.n_bootstrap, seed=args.seed)

    validation_gate_summary.to_csv(output_dir / "validation_gate_candidates.csv", index=False)
    pd.DataFrame([selected_gate]).to_csv(output_dir / "selected_gate.csv", index=False)
    gated_validation.to_csv(output_dir / "gated_validation_samples.csv", index=False)
    gated_test.to_csv(output_dir / "gated_test_samples.csv", index=False)

    validation_overall = summarize(gated_validation, [])
    test_overall = summarize(gated_test, [])
    test_by_tier = summarize(gated_test, ["tier"])
    test_by_horizon = summarize(gated_test, ["horizon"])
    test_by_growth = summarize(gated_test, ["absolute_growth_bin"])
    test_by_tier_growth = summarize(gated_test, ["tier", "absolute_growth_bin"])

    validation_overall.to_csv(output_dir / "gated_validation_overall.csv", index=False)
    test_overall.to_csv(output_dir / "gated_test_overall.csv", index=False)
    test_by_tier.to_csv(output_dir / "gated_test_by_tier.csv", index=False)
    test_by_horizon.to_csv(output_dir / "gated_test_by_horizon.csv", index=False)
    test_by_growth.to_csv(output_dir / "gated_test_by_absolute_growth_bin.csv", index=False)
    test_by_tier_growth.to_csv(output_dir / "gated_test_by_tier_growth_bin.csv", index=False)
    pd.DataFrame([test_bootstrap]).to_csv(output_dir / "gated_test_bootstrap.csv", index=False)

    report = {
        "selection_dir": str(selection_dir),
        "validation_dir": str(validation_dir),
        "test_dir": str(test_dir),
        "score_source": score_source,
        "budget_policy": budget_policy,
        "objective": args.objective,
        "n_gate_candidates": int(len(gates)),
        "selected_gate": selected_gate.to_dict(),
        "test_bootstrap": test_bootstrap,
        "output_dir": str(output_dir),
    }
    with (output_dir / "gated_budget_policy_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    write_report(
        output_dir=output_dir,
        score_source=score_source,
        budget_policy=budget_policy,
        selected_gate=selected_gate,
        validation_overall=validation_overall,
        test_overall=test_overall,
        test_by_growth=test_by_growth,
        bootstrap=test_bootstrap,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
