#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


PREDICTION_TIME_FEATURES = [
    "input_volume_vox",
    "delta_days",
    "growth_budget_vox",
    "selected_budget_to_input_ratio",
    "selected_budget_to_input_pct",
]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _thresholds(values: pd.Series, quantiles: Iterable[float]) -> List[float]:
    clean = values.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if clean.empty:
        return []
    out = {float(clean.quantile(q)) for q in quantiles}
    out.update({float(clean.min()), float(clean.max())})
    return sorted(out)


def add_case_class(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    growth = out["growth_volume_vox"].fillna(0).astype(float)
    active = out["gate_active"].astype(bool)

    conditions = [
        active & growth.eq(0),
        active & growth.gt(0),
        (~active) & growth.eq(0),
        (~active) & growth.gt(0),
    ]
    labels = [
        "false_activation_zero_growth",
        "active_true_growth",
        "protected_zero_growth",
        "inactive_missed_growth",
    ]
    out["gate_case_class"] = np.select(conditions, labels, default="unknown")

    if "absolute_growth_bin" in out.columns:
        out["gate_case_detail"] = out["gate_case_class"] + "__" + out["absolute_growth_bin"].astype(str)
    else:
        out["gate_case_detail"] = out["gate_case_class"]
    return out


def summarize_cases(df: pd.DataFrame, group_cols: Iterable[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    cols = [c for c in group_cols if c in df.columns]
    group_df = df if cols else df.assign(_overall="overall")
    by = cols if cols else ["_overall"]
    out = (
        group_df.groupby(by, observed=True, dropna=False)
        .agg(
            count=("gated_dice", "size"),
            mean_gap_vs_locf=("gated_gap_vs_locf", "mean"),
            mean_dice=("gated_dice", "mean"),
            mean_locf_dice=("locf_dice", "mean"),
            gate_active_rate=("gate_active", "mean"),
            mean_growth_volume_vox=("growth_volume_vox", "mean"),
            median_growth_volume_vox=("growth_volume_vox", "median"),
            mean_growth_budget_vox=("growth_budget_vox", "mean"),
            median_growth_budget_vox=("growth_budget_vox", "median"),
            mean_input_volume_vox=("input_volume_vox", "mean"),
            mean_delta_days=("delta_days", "mean"),
        )
        .reset_index()
    )
    if "_overall" in out.columns:
        out = out.drop(columns=["_overall"])
    return out.sort_values(cols).reset_index(drop=True) if cols else out


def feature_profiles(df: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
    rows = []
    available = [f for f in features if f in df.columns]
    for case_class, group in df.groupby("gate_case_class", observed=True, dropna=False):
        for feature in available:
            values = group[feature].replace([np.inf, -np.inf], np.nan).dropna().astype(float)
            if values.empty:
                continue
            rows.append(
                {
                    "gate_case_class": case_class,
                    "feature": feature,
                    "count": int(len(values)),
                    "mean": float(values.mean()),
                    "median": float(values.median()),
                    "q25": float(values.quantile(0.25)),
                    "q75": float(values.quantile(0.75)),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )
    return pd.DataFrame(rows)


def build_guard_candidates(validation: pd.DataFrame, quantiles: Iterable[float]) -> pd.DataFrame:
    rows = [{"guard_name": "no_guard", "feature": "constant", "operator": "none", "threshold": np.nan}]
    for feature in PREDICTION_TIME_FEATURES:
        if feature not in validation.columns:
            continue
        for threshold in _thresholds(validation[feature], quantiles):
            rows.append(
                {
                    "guard_name": f"suppress_if_{feature}_<=_{threshold:.6g}",
                    "feature": feature,
                    "operator": "<=",
                    "threshold": threshold,
                }
            )
            rows.append(
                {
                    "guard_name": f"suppress_if_{feature}_>=_{threshold:.6g}",
                    "feature": feature,
                    "operator": ">=",
                    "threshold": threshold,
                }
            )
    return pd.DataFrame(rows)


def apply_guard(df: pd.DataFrame, guard: pd.Series) -> pd.DataFrame:
    out = df.copy()
    if guard["operator"] == "none":
        suppress = pd.Series(False, index=out.index)
    elif guard["operator"] == "<=":
        suppress = out[guard["feature"]].astype(float) <= float(guard["threshold"])
    elif guard["operator"] == ">=":
        suppress = out[guard["feature"]].astype(float) >= float(guard["threshold"])
    else:
        raise ValueError(f"Unsupported guard operator: {guard['operator']}")

    suppress = suppress & out["gate_active"].astype(bool)
    out["guard_suppressed"] = suppress.astype(bool)
    out["guarded_gate_active"] = out["gate_active"].astype(bool) & ~out["guard_suppressed"]
    out["guarded_dice"] = np.where(out["guarded_gate_active"], out["persistence_growth_dice"], out["locf_dice"])
    out["guarded_gap_vs_locf"] = out["guarded_dice"] - out["locf_dice"]
    out["guard_name"] = guard["guard_name"]
    out["guard_feature"] = guard["feature"]
    out["guard_operator"] = guard["operator"]
    out["guard_threshold"] = guard["threshold"]
    return out


def summarize_guarded(df: pd.DataFrame, group_cols: Iterable[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    cols = [c for c in group_cols if c in df.columns]
    group_df = df if cols else df.assign(_overall="overall")
    by = cols if cols else ["_overall"]
    out = (
        group_df.groupby(by, observed=True, dropna=False)
        .agg(
            count=("guarded_dice", "size"),
            mean_dice=("guarded_dice", "mean"),
            mean_locf_dice=("locf_dice", "mean"),
            mean_gap_vs_locf=("guarded_gap_vs_locf", "mean"),
            median_gap_vs_locf=("guarded_gap_vs_locf", "median"),
            win_rate_vs_locf=("guarded_gap_vs_locf", lambda x: float((x > 0).mean())),
            guarded_gate_active_rate=("guarded_gate_active", "mean"),
            guard_suppressed_rate=("guard_suppressed", "mean"),
            mean_growth_volume_vox=("growth_volume_vox", "mean"),
            median_growth_volume_vox=("growth_volume_vox", "median"),
        )
        .reset_index()
    )
    if "_overall" in out.columns:
        out = out.drop(columns=["_overall"])
    return out.sort_values(cols).reset_index(drop=True) if cols else out


def evaluate_guards(df: pd.DataFrame, guards: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, guard in guards.iterrows():
        guarded = apply_guard(df, guard)
        overall = summarize_guarded(guarded, []).iloc[0].to_dict()
        rows.append({**guard.to_dict(), **overall})
    return pd.DataFrame(rows)


def select_guard(summary: pd.DataFrame, objective: str) -> pd.Series:
    if objective == "mean_gap":
        sort_cols = ["mean_gap_vs_locf", "win_rate_vs_locf", "mean_dice"]
    elif objective == "win_rate":
        sort_cols = ["win_rate_vs_locf", "mean_gap_vs_locf", "mean_dice"]
    elif objective == "mean_dice":
        sort_cols = ["mean_dice", "mean_gap_vs_locf", "win_rate_vs_locf"]
    else:
        raise ValueError(f"Unsupported objective: {objective}")
    return summary.sort_values(sort_cols, ascending=False).iloc[0]


def bootstrap_gap(df: pd.DataFrame, gap_col: str, n_bootstrap: int, seed: int) -> Dict:
    gaps = df[gap_col].to_numpy(dtype=float)
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
    case_summary: pd.DataFrame,
    selected_guard: pd.Series,
    validation_overall: pd.DataFrame,
    test_overall: pd.DataFrame,
    test_by_case: pd.DataFrame,
    bootstrap: Dict,
) -> None:
    with (output_dir / "gate_false_activation_report.md").open("w", encoding="utf-8") as f:
        f.write("# Gate False-Activation Audit\n\n")
        f.write("This audit studies cases where the growth gate activates despite zero future growth.\n\n")
        f.write("## Test Gate Case Summary\n\n")
        f.write(case_summary.to_markdown(index=False) if not case_summary.empty else "No case summary.")
        f.write("\n\n## Validation-Selected Suppression Guard\n\n")
        f.write(pd.DataFrame([selected_guard]).to_markdown(index=False))
        f.write("\n\n## Guarded Validation Overall\n\n")
        f.write(validation_overall.to_markdown(index=False))
        f.write("\n\n## Guarded Test Overall\n\n")
        f.write(test_overall.to_markdown(index=False))
        f.write("\n\n## Guarded Test Bootstrap\n\n")
        f.write(pd.DataFrame([bootstrap]).to_markdown(index=False))
        f.write("\n\n## Guarded Test By Original Gate Case Class\n\n")
        f.write(test_by_case.to_markdown(index=False) if not test_by_case.empty else "No guarded case summary.")
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit growth-gate false activations and test a validation-selected suppression guard."
    )
    parser.add_argument("--gated_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--objective", type=str, default="mean_gap", choices=["mean_gap", "win_rate", "mean_dice"])
    parser.add_argument("--quantiles", type=str, default="0,0.1,0.25,0.5,0.75,0.9,1.0")
    parser.add_argument("--n_bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    gated_dir = Path(args.gated_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    validation = add_case_class(_read_csv(gated_dir / "gated_validation_samples.csv"))
    test = add_case_class(_read_csv(gated_dir / "gated_test_samples.csv"))
    quantiles = [float(q.strip()) for q in args.quantiles.split(",") if q.strip()]

    test_case_summary = summarize_cases(test, ["gate_case_class"])
    test_case_by_tier = summarize_cases(test, ["gate_case_class", "tier"])
    test_case_features = feature_profiles(test, PREDICTION_TIME_FEATURES)

    guards = build_guard_candidates(validation, quantiles=quantiles)
    validation_guard_summary = evaluate_guards(validation, guards)
    selected_guard = select_guard(validation_guard_summary, objective=args.objective)
    guarded_validation = apply_guard(validation, selected_guard)
    guarded_test = apply_guard(test, selected_guard)
    test_bootstrap = bootstrap_gap(guarded_test, "guarded_gap_vs_locf", args.n_bootstrap, args.seed)

    guarded_validation_overall = summarize_guarded(guarded_validation, [])
    guarded_test_overall = summarize_guarded(guarded_test, [])
    guarded_test_by_case = summarize_guarded(guarded_test, ["gate_case_class"])
    guarded_test_by_growth = summarize_guarded(guarded_test, ["absolute_growth_bin"])
    guarded_test_by_tier = summarize_guarded(guarded_test, ["tier"])
    guarded_test_by_tier_growth = summarize_guarded(guarded_test, ["tier", "absolute_growth_bin"])

    validation.to_csv(output_dir / "gate_validation_samples_with_case_class.csv", index=False)
    test.to_csv(output_dir / "gate_test_samples_with_case_class.csv", index=False)
    test_case_summary.to_csv(output_dir / "gate_test_case_summary.csv", index=False)
    test_case_by_tier.to_csv(output_dir / "gate_test_case_summary_by_tier.csv", index=False)
    test_case_features.to_csv(output_dir / "gate_test_case_feature_profiles.csv", index=False)
    validation_guard_summary.to_csv(output_dir / "validation_suppression_guard_candidates.csv", index=False)
    pd.DataFrame([selected_guard]).to_csv(output_dir / "selected_suppression_guard.csv", index=False)
    guarded_validation.to_csv(output_dir / "guarded_validation_samples.csv", index=False)
    guarded_test.to_csv(output_dir / "guarded_test_samples.csv", index=False)
    guarded_validation_overall.to_csv(output_dir / "guarded_validation_overall.csv", index=False)
    guarded_test_overall.to_csv(output_dir / "guarded_test_overall.csv", index=False)
    guarded_test_by_case.to_csv(output_dir / "guarded_test_by_gate_case_class.csv", index=False)
    guarded_test_by_growth.to_csv(output_dir / "guarded_test_by_absolute_growth_bin.csv", index=False)
    guarded_test_by_tier.to_csv(output_dir / "guarded_test_by_tier.csv", index=False)
    guarded_test_by_tier_growth.to_csv(output_dir / "guarded_test_by_tier_growth_bin.csv", index=False)
    pd.DataFrame([test_bootstrap]).to_csv(output_dir / "guarded_test_bootstrap.csv", index=False)

    report = {
        "gated_dir": str(gated_dir),
        "objective": args.objective,
        "n_guard_candidates": int(len(guards)),
        "selected_guard": selected_guard.to_dict(),
        "test_bootstrap": test_bootstrap,
        "output_dir": str(output_dir),
    }
    with (output_dir / "gate_false_activation_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    write_report(
        output_dir,
        case_summary=test_case_summary,
        selected_guard=selected_guard,
        validation_overall=guarded_validation_overall,
        test_overall=guarded_test_overall,
        test_by_case=guarded_test_by_case,
        bootstrap=test_bootstrap,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
