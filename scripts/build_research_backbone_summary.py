#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def read_csv(path: Optional[Path], missing: List[str]) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    if not path.exists():
        missing.append(str(path))
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        missing.append(f"{path} ({exc})")
        return pd.DataFrame()


def fmt(value: Any, digits: int = 3) -> str:
    try:
        x = float(value)
    except Exception:
        return "NA"
    if not np.isfinite(x):
        return "NA"
    return f"{x:.{digits}f}"


def pct(value: Any, digits: int = 1) -> str:
    try:
        x = float(value) * 100.0
    except Exception:
        return "NA"
    if not np.isfinite(x):
        return "NA"
    return f"{x:.{digits}f}%"


def first_value(df: pd.DataFrame, col: str, default: Any = np.nan) -> Any:
    if df.empty or col not in df.columns:
        return default
    val = df.iloc[0][col]
    return default if pd.isna(val) else val


def value_where(df: pd.DataFrame, filters: Dict[str, Any], col: str, default: Any = np.nan) -> Any:
    if df.empty or col not in df.columns:
        return default
    mask = pd.Series(True, index=df.index)
    for key, value in filters.items():
        if key not in df.columns:
            return default
        mask &= df[key].astype(str).eq(str(value))
    part = df[mask]
    if part.empty:
        return default
    val = part.iloc[0][col]
    return default if pd.isna(val) else val


def add_claim(
    rows: List[dict],
    claim_id: str,
    claim: str,
    role: str,
    status: str,
    quantitative_anchor: str,
    evidence_sources: str,
    use_in_core_story: str,
    missing_or_risk: str,
    next_action: str,
) -> None:
    rows.append(
        {
            "claim_id": claim_id,
            "claim": claim,
            "role": role,
            "status": status,
            "quantitative_anchor": quantitative_anchor,
            "evidence_sources": evidence_sources,
            "use_in_core_story": use_in_core_story,
            "missing_or_risk": missing_or_risk,
            "next_action": next_action,
        }
    )


def transition_fraction(transitions: pd.DataFrame, dataset: str, transition_type: str) -> float:
    return float(
        value_where(
            transitions,
            {"dataset": dataset, "transition_type": transition_type},
            "fraction",
            default=np.nan,
        )
    )


def build_transition_claims(args: argparse.Namespace, missing: List[str], rows: List[dict]) -> None:
    root = Path(args.transition_package_dir) if args.transition_package_dir else None
    overall = read_csv(root / "core_tables" / "dataset_overall_comparison.csv" if root else None, missing)
    transitions = read_csv(root / "core_tables" / "transition_type_distribution.csv" if root else None, missing)
    claim_matrix = read_csv(root / "core_tables" / "claim_support_matrix.csv" if root else None, missing)

    sailor = args.sailor_name
    srd = args.srd_name
    sailor_n = value_where(overall, {"dataset": sailor}, "n_transitions")
    sailor_patients = value_where(overall, {"dataset": sailor}, "n_patients")
    srd_n = value_where(overall, {"dataset": srd}, "n_transitions")
    srd_patients = value_where(overall, {"dataset": srd}, "n_patients")
    sailor_locf = value_where(overall, {"dataset": sailor}, "mean_locf_dice")
    srd_locf = value_where(overall, {"dataset": srd}, "mean_locf_dice")
    mixed_sailor = transition_fraction(transitions, sailor, "mixed_growth_loss")
    mixed_srd = transition_fraction(transitions, srd, "mixed_growth_loss")
    persistence_sailor = transition_fraction(transitions, sailor, "persistence_dominant")
    persistence_srd = transition_fraction(transitions, srd, "persistence_dominant")
    distant_sailor = value_where(overall, {"dataset": sailor}, "distant_growth_rate")
    distant_srd = value_where(overall, {"dataset": srd}, "distant_growth_rate")

    source = "transition_evidence_package/core_tables"
    if overall.empty:
        status = "missing_input"
        anchor = "Transition package not available."
    else:
        status = "central_supported"
        anchor = (
            f"{sailor}: n={fmt(sailor_n,0)} transitions/{fmt(sailor_patients,0)} patients, LOCF={fmt(sailor_locf)}; "
            f"{srd}: n={fmt(srd_n,0)} transitions/{fmt(srd_patients,0)} patients, LOCF={fmt(srd_locf)}."
        )
    add_claim(
        rows,
        "C1",
        "Short-horizon tumor forecasting should be analyzed as a transition problem, not only a model leaderboard.",
        "central",
        status,
        anchor,
        source,
        "yes",
        "Must avoid making this sound like a surprising causal claim; LOCF Dice is mechanically tied to transition burden.",
        "Use as the opening empirical spine.",
    )

    if transitions.empty:
        anchor = "Transition-type distribution not available."
        status = "missing_input"
    else:
        anchor = (
            f"Mixed growth/loss: {sailor}={pct(mixed_sailor)}, {srd}={pct(mixed_srd)}; "
            f"distant growth: {sailor}={pct(distant_sailor)}, {srd}={pct(distant_srd)}; "
            f"persistence-dominant: {sailor}={pct(persistence_sailor)}, {srd}={pct(persistence_srd)}."
        )
        status = "central_supported"
    add_claim(
        rows,
        "C2",
        "SRD and SAILOR play complementary roles: SRD isolates controlled regimes, while SAILOR exposes real transition complexity.",
        "central",
        status,
        anchor,
        source,
        "yes",
        "Do not present SRD as a realistic SAILOR surrogate.",
        "Keep one compact transition table plus one figure; relegate full sensitivity tables to audit/support.",
    )

    if not claim_matrix.empty:
        rejected = claim_matrix[claim_matrix.get("status", pd.Series(dtype=str)).astype(str).str.contains("rejected|weak", case=False, na=False)]
        risk = f"Existing claim matrix has {len(rejected)} weak/rejected rows."
    else:
        risk = "Existing claim matrix not found."
    add_claim(
        rows,
        "C3",
        "The strongest contribution is the transition-aware evaluation framework; the current method probes are secondary.",
        "central",
        "central_supported",
        risk,
        "FINAL_DRAFT_MAP + transition claim matrix",
        "yes",
        "Requires discipline: avoid letting small model gains become the storyline.",
        "Organize results by transition concepts before model prototypes.",
    )


def build_locf_claims(args: argparse.Namespace, missing: List[str], rows: List[dict]) -> None:
    root = Path(args.locf_operating_dir) if args.locf_operating_dir else None
    corr = read_csv(root / "locf_operating_correlations.csv" if root else None, missing)
    growth_q = read_csv(root / "locf_operating_by_new_growth_rate_quantile.csv" if root else None, missing)
    abs_q = read_csv(root / "locf_operating_by_absolute_change_rate_quantile.csv" if root else None, missing)

    anchors = []
    for label, table in [("new-growth-rate", growth_q), ("absolute-change-rate", abs_q)]:
        if table.empty or "mean_locf_dice" not in table.columns:
            continue
        first = table.iloc[0]["mean_locf_dice"]
        last = table.iloc[-1]["mean_locf_dice"]
        anchors.append(f"{label} low-to-high LOCF Dice {fmt(first)} -> {fmt(last)}")
    if not corr.empty:
        useful_cols = [c for c in corr.columns if c not in {"metric"}]
        if useful_cols:
            anchors.append("correlations table available")

    add_claim(
        rows,
        "C4",
        "LOCF has an operating range governed by calendar interval and observed transition/change burden.",
        "central",
        "central_supported" if anchors else "missing_input",
        "; ".join(anchors) if anchors else "LOCF operating-range outputs not available.",
        "locf_operating_range outputs",
        "yes",
        "This must be phrased as operating characterization, not as a causal discovery.",
        "Use heatmap/quantile table to define short-term as session + calendar + biological/change horizon.",
    )


def build_predictability_claims(args: argparse.Namespace, missing: List[str], rows: List[dict]) -> None:
    pred_root = Path(args.forecast_origin_predictability_dir) if args.forecast_origin_predictability_dir else None
    stable_root = Path(args.forecast_origin_stability_dir) if args.forecast_origin_stability_dir else None
    budget_root = Path(args.budget_predictability_dir) if args.budget_predictability_dir else None

    pred_claims = read_csv(pred_root / "forecast_origin_predictability_claim_status.csv" if pred_root else None, missing)
    stability_claims = read_csv(stable_root / "forecast_origin_patient_split_stability_claim_status.csv" if stable_root else None, missing)
    budget_summary = read_csv(budget_root / "forecast_origin_budget_predictability_summary.csv" if budget_root else None, missing)

    if not stability_claims.empty:
        stable_counts = stability_claims["status"].astype(str).value_counts().to_dict() if "status" in stability_claims.columns else {}
        anchor = f"Repeated patient-split claim statuses: {stable_counts}."
        status = "supporting_supported"
        source = "forecast_origin_patient_split_stability_claim_status.csv"
    elif not pred_claims.empty:
        pred_counts = pred_claims["status"].astype(str).value_counts().to_dict() if "status" in pred_claims.columns else {}
        anchor = f"Fixed-split predictability claim statuses: {pred_counts}."
        status = "supporting_fixed_split_only"
        source = "forecast_origin_predictability_claim_status.csv"
    else:
        anchor = "Forecast-origin predictability claim tables not available."
        status = "missing_input"
        source = "forecast-origin predictability outputs"

    add_claim(
        rows,
        "C5",
        "Some difficult transition states are partially predictable from forecast-origin features, but not all of them.",
        "supporting",
        status,
        anchor,
        source,
        "yes_supporting",
        "Avoid broad claims that origin features can anticipate every hard transition.",
        "Use as a bridge from evaluation to possible regime-aware priors.",
    )

    if budget_summary.empty:
        anchor = "Budget predictability summary not available."
        status = "missing_input"
    else:
        candidates = budget_summary.copy()
        if "feature_set" in candidates.columns:
            candidates = candidates[candidates["feature_set"].astype(str).eq("history_only")]
        if "split" in candidates.columns:
            candidates = candidates[candidates["split"].astype(str).isin(["val", "test"])]
        metric_cols = [c for c in candidates.columns if "gap" in c or "dice" in c or "oracle" in c]
        if metric_cols and not candidates.empty:
            anchor = f"Budget predictability rows={len(candidates)}; key columns={metric_cols[:6]}."
        else:
            anchor = f"Budget predictability rows={len(candidates)}."
        status = "diagnostic_supported"
    add_claim(
        rows,
        "C6",
        "Correction-budget estimation is a real bottleneck before spatial growth-field design.",
        "diagnostic",
        status,
        anchor,
        "forecast_origin_budget_predictability_summary.csv",
        "yes_diagnostic",
        "Budget-oracle results still assume spatial localization; do not treat them as final mask performance.",
        "Keep budget predictability as method-design motivation, not the final method.",
    )


def build_method_probe_claims(args: argparse.Namespace, missing: List[str], rows: List[dict]) -> None:
    failure_root = Path(args.growth_field_failure_dir) if args.growth_field_failure_dir else None
    policy_root = Path(args.conservative_policy_robustness_dir) if args.conservative_policy_robustness_dir else None
    failure = read_csv(failure_root / "budgeted_growth_field_failure_evidence_map.csv" if failure_root else None, missing)
    policy = read_csv(policy_root / "conservative_policy_claim_status.csv" if policy_root else None, missing)
    boot = read_csv(policy_root / "conservative_policy_patient_bootstrap_summary.csv" if policy_root else None, missing)

    if failure.empty:
        anchor = "Growth-field failure evidence map not available."
        status = "missing_input"
    else:
        if {"evidence", "status"}.issubset(failure.columns):
            anchor = "; ".join(f"{r.evidence}: {r.status}" for r in failure.head(5).itertuples())
        else:
            anchor = f"Failure evidence rows={len(failure)}."
        status = "diagnostic_supported"
    add_claim(
        rows,
        "C7",
        "A learned growth-probability field can be spatially informative but still fail as a Dice-safe mask update.",
        "diagnostic",
        status,
        anchor,
        "budgeted_growth_field_failure_evidence_map.csv",
        "yes_diagnostic",
        "This is a bottleneck result, not a model win.",
        "Use to motivate separate direction, budget, and spatial-localization components.",
    )

    if policy.empty:
        anchor = "Conservative-policy robustness claim table not available."
        status = "missing_input"
    else:
        anchor = "; ".join(f"{r.claim}: {r.status}" for r in policy.itertuples())
        robust_win = policy[
            policy["claim"].astype(str).str.contains("robustly improves test Dice", case=False, na=False)
        ]
        status_val = str(first_value(robust_win, "status", "unknown"))
        status = "rejected_method_claim" if "not_supported" in status_val else "diagnostic_supported"
    if not boot.empty:
        test_gap = boot[
            (boot.get("split", pd.Series(dtype=str)).astype(str) == "test")
            & (boot.get("metric", pd.Series(dtype=str)).astype(str) == "mean_gap_vs_locf")
            & (boot.get("summary_type", pd.Series(dtype=str)).astype(str) == "split")
        ]
        if not test_gap.empty:
            ci_text = "; ".join(
                f"{r.policy_seed} CI [{fmt(r.ci_low)}, {fmt(r.ci_high)}]"
                for r in test_gap.itertuples()
            )
            anchor = f"{anchor}; patient bootstrap: {ci_text}"
    add_claim(
        rows,
        "C8",
        "The current conservative learned-field prototype robustly beats LOCF.",
        "rejected_or_not_central",
        status,
        anchor,
        "conservative_policy_claim_status.csv + patient bootstrap",
        "no",
        "Overall test gains are tiny and patient-bootstrap intervals cross zero.",
        "Stop tuning this prototype unless the method is conceptually redesigned.",
    )


def priority_table(claims: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "priority": 1,
            "workstream": "Core evidence spine",
            "what_to_do": "Finalize transition evidence package plus LOCF operating-range definition.",
            "why": "These are the strongest central claims and do not rely on tiny model gains.",
        },
        {
            "priority": 2,
            "workstream": "Forecast-origin predictability",
            "what_to_do": "Curate which transition states are predictable, weak, or rejected under patient-split stability.",
            "why": "This gives the regime-aware analysis a data-mining mechanism rather than only post-hoc labels.",
        },
        {
            "priority": 3,
            "workstream": "Method probes",
            "what_to_do": "Use growth-field/conservative-policy results as bottleneck evidence, not as a central method.",
            "why": "The prototype does not robustly beat LOCF, but it explains what a future method must handle.",
        },
        {
            "priority": 4,
            "workstream": "Paper discipline",
            "what_to_do": "Remove or demote claims marked rejected/missing/diagnostic-only from the central narrative.",
            "why": "The work becomes stronger when negative results clarify the boundary of the contribution.",
        },
    ]
    missing = claims[claims["status"].astype(str).str.contains("missing", case=False, na=False)]
    if not missing.empty:
        rows.append(
            {
                "priority": 5,
                "workstream": "Missing outputs",
                "what_to_do": "Rerun or locate missing output folders listed by the summary script.",
                "why": f"{len(missing)} claim rows are missing inputs.",
            }
        )
    return pd.DataFrame(rows)


def write_report(path: Path, claims: pd.DataFrame, priorities: pd.DataFrame, missing: List[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Research Backbone Summary\n\n")
        f.write(
            "This document consolidates the current evidence into a claim-level research spine. "
            "It is intentionally not a model leaderboard. The purpose is to decide what can carry the work, "
            "what should remain supporting, and what should be treated as diagnostic or rejected.\n\n"
        )
        f.write("## Priority Workstreams\n\n")
        f.write(priorities.to_markdown(index=False))
        f.write("\n\n## Claim Map\n\n")
        f.write(claims.to_markdown(index=False))
        if missing:
            f.write("\n\n## Missing Inputs\n\n")
            for item in missing:
                f.write(f"- `{item}`\n")
        f.write(
            "\n\nReading rule: central claims should be supported by descriptive transition evidence, "
            "patient counts, and robustness checks. Diagnostic claims can be valuable, but they should not "
            "be written as method wins.\n"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a compact claim-level summary of the current research backbone.")
    parser.add_argument("--transition_package_dir", type=str, default=None)
    parser.add_argument("--locf_operating_dir", type=str, default=None)
    parser.add_argument("--forecast_origin_predictability_dir", type=str, default=None)
    parser.add_argument("--forecast_origin_stability_dir", type=str, default=None)
    parser.add_argument("--budget_predictability_dir", type=str, default=None)
    parser.add_argument("--growth_field_failure_dir", type=str, default=None)
    parser.add_argument("--conservative_policy_robustness_dir", type=str, default=None)
    parser.add_argument("--sailor_name", type=str, default="SAILOR")
    parser.add_argument("--srd_name", type=str, default="SRD")
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    missing: List[str] = []
    rows: List[dict] = []

    build_transition_claims(args, missing, rows)
    build_locf_claims(args, missing, rows)
    build_predictability_claims(args, missing, rows)
    build_method_probe_claims(args, missing, rows)

    claims = pd.DataFrame(rows)
    priorities = priority_table(claims)
    claims.to_csv(output_dir / "research_backbone_claim_map.csv", index=False)
    priorities.to_csv(output_dir / "research_backbone_priority_table.csv", index=False)
    if missing:
        pd.DataFrame({"missing_input": missing}).to_csv(output_dir / "research_backbone_missing_inputs.csv", index=False)
    write_report(output_dir / "research_backbone_summary.md", claims, priorities, missing)

    payload = {
        "output_dir": str(output_dir),
        "n_claims": int(len(claims)),
        "n_missing_inputs": int(len(missing)),
        "outputs": {
            "claim_map_csv": str(output_dir / "research_backbone_claim_map.csv"),
            "priority_table_csv": str(output_dir / "research_backbone_priority_table.csv"),
            "missing_inputs_csv": str(output_dir / "research_backbone_missing_inputs.csv") if missing else None,
            "report_md": str(output_dir / "research_backbone_summary.md"),
        },
    }
    with (output_dir / "research_backbone_summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
