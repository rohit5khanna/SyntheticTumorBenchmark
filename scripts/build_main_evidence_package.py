#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def read_csv(path: Optional[Path], missing: List[str], required: bool = False) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    if not path.exists():
        missing.append(str(path))
        if required:
            raise FileNotFoundError(path)
        return pd.DataFrame()
    return pd.read_csv(path)


def copy_if_exists(src: Optional[Path], dst: Path, missing: List[str]) -> Optional[str]:
    if src is None or not src.exists():
        if src is not None:
            missing.append(str(src))
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)


def fmt(value: Any, digits: int = 3) -> str:
    try:
        x = float(value)
    except Exception:
        return ""
    if not np.isfinite(x):
        return ""
    return f"{x:.{digits}f}"


def pct(value: Any, digits: int = 1) -> str:
    try:
        x = float(value) * 100.0
    except Exception:
        return ""
    if not np.isfinite(x):
        return ""
    return f"{x:.{digits}f}%"


def first_value(df: pd.DataFrame, col: str, default: Any = "") -> Any:
    if df.empty or col not in df.columns:
        return default
    val = df.iloc[0][col]
    return default if pd.isna(val) else val


def value_where(df: pd.DataFrame, filters: Dict[str, Any], col: str, default: Any = "") -> Any:
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


def write_table(df: pd.DataFrame, csv_path: Path, md_path: Optional[Path] = None) -> Optional[str]:
    if df.empty:
        return None
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    if md_path is not None:
        with md_path.open("w", encoding="utf-8") as f:
            f.write(df.to_markdown(index=False))
            f.write("\n")
    return str(csv_path)


def build_claim_table(backbone_dir: Optional[Path], output_dir: Path, missing: List[str]) -> pd.DataFrame:
    claim_map = read_csv(backbone_dir / "research_backbone_claim_map.csv" if backbone_dir else None, missing)
    if claim_map.empty:
        return pd.DataFrame()
    priority = {
        "central": 1,
        "supporting": 2,
        "diagnostic": 3,
        "rejected_or_not_central": 4,
    }
    keep_cols = [
        "claim_id",
        "claim",
        "role",
        "status",
        "quantitative_anchor",
        "use_in_core_story",
        "missing_or_risk",
        "next_action",
    ]
    out = claim_map[[c for c in keep_cols if c in claim_map.columns]].copy()
    out["_role_order"] = out["role"].map(priority).fillna(9)
    out = out.sort_values(["_role_order", "claim_id"]).drop(columns=["_role_order"])
    write_table(out, output_dir / "tables" / "main_claim_map.csv", output_dir / "tables" / "main_claim_map.md")
    return out


def build_transition_artifacts(
    transition_main_dir: Optional[Path],
    transition_package_dir: Optional[Path],
    output_dir: Path,
    missing: List[str],
) -> pd.DataFrame:
    src_table = transition_main_dir / "main_transition_summary_table.csv" if transition_main_dir else None
    table = read_csv(src_table, missing)
    if table.empty and transition_package_dir is not None:
        overall = read_csv(transition_package_dir / "core_tables" / "dataset_overall_comparison.csv", missing)
        transition_dist = read_csv(transition_package_dir / "core_tables" / "transition_type_distribution.csv", missing)
        rows = []
        if not overall.empty:
            for r in overall.itertuples():
                rows.append(
                    {
                        "metric": f"{r.dataset} transitions / patients",
                        "value": f"{int(getattr(r, 'n_transitions', 0))} / {int(getattr(r, 'n_patients', 0))}",
                        "interpretation": "Dataset scale for transition-level analysis.",
                    }
                )
                if hasattr(r, "mean_locf_dice"):
                    rows.append(
                        {
                            "metric": f"{r.dataset} mean LOCF Dice",
                            "value": fmt(getattr(r, "mean_locf_dice")),
                            "interpretation": "Persistence baseline under this transition distribution.",
                        }
                    )
        if not transition_dist.empty:
            for transition_type in ["mixed_growth_loss", "persistence_dominant", "growth_dominant", "loss_dominant"]:
                vals = []
                for dataset in sorted(transition_dist["dataset"].astype(str).unique()):
                    frac = value_where(transition_dist, {"dataset": dataset, "transition_type": transition_type}, "fraction", 0.0)
                    vals.append(f"{dataset}: {pct(frac)}")
                rows.append(
                    {
                        "metric": transition_type,
                        "value": "; ".join(vals),
                        "interpretation": "Transition-type prevalence.",
                    }
                )
        table = pd.DataFrame(rows)

    write_table(table, output_dir / "tables" / "main_transition_summary_table.csv", output_dir / "tables" / "main_transition_summary_table.md")

    copied = copy_if_exists(
        transition_main_dir / "main_transition_type_and_locf_figure.png" if transition_main_dir else None,
        output_dir / "figures" / "main_transition_type_and_locf_figure.png",
        missing,
    )
    if copied:
        pd.DataFrame(
            [
                {
                    "figure_id": "F1",
                    "file": copied,
                    "recommended_use": "Main transition composition and LOCF-difficulty figure.",
                    "caption_note": "Use with compact transition summary table; robustness details remain in audit outputs.",
                }
            ]
        ).to_csv(output_dir / "figures" / "recommended_transition_figures.csv", index=False)
    return table


def build_locf_table(locf_dir: Optional[Path], output_dir: Path, missing: List[str]) -> pd.DataFrame:
    overall = read_csv(locf_dir / "locf_operating_overall.csv" if locf_dir else None, missing)
    growth_q = read_csv(locf_dir / "locf_operating_by_new_growth_rate_quantile.csv" if locf_dir else None, missing)
    abs_q = read_csv(locf_dir / "locf_operating_by_absolute_change_rate_quantile.csv" if locf_dir else None, missing)
    corr = read_csv(locf_dir / "locf_operating_correlations.csv" if locf_dir else None, missing)

    rows: List[dict] = []
    if not overall.empty:
        rows.append(
            {
                "evidence": "Overall LOCF operating context",
                "value": f"n={first_value(overall, 'n_samples')}, patients={first_value(overall, 'n_patients')}, mean Dice={fmt(first_value(overall, 'mean_locf_dice'))}",
                "interpretation": "Baseline persistence level before stratification.",
            }
        )
    for name, table in [("new-growth-rate quantile", growth_q), ("absolute-change-rate quantile", abs_q)]:
        if not table.empty and "mean_locf_dice" in table.columns:
            first = table.iloc[0]
            last = table.iloc[-1]
            label_col = [c for c in table.columns if c.endswith("_qbin")]
            lo = first[label_col[0]] if label_col else "low"
            hi = last[label_col[0]] if label_col else "high"
            rows.append(
                {
                    "evidence": f"LOCF Dice across {name}",
                    "value": f"{lo}: {fmt(first['mean_locf_dice'])}; {hi}: {fmt(last['mean_locf_dice'])}",
                    "interpretation": "Shows the persistence prior has an operating range, not a single universal behavior.",
                }
            )
    if not corr.empty and "spearman_corr_with_locf_dice" in corr.columns:
        strongest = corr.sort_values("spearman_corr_with_locf_dice").head(3)
        rows.append(
            {
                "evidence": "Strongest negative LOCF associations",
                "value": "; ".join(f"{r.feature}: rho={fmt(r.spearman_corr_with_locf_dice)}" for r in strongest.itertuples()),
                "interpretation": "Higher transition burden/change-rate is associated with lower LOCF Dice.",
            }
        )
    out = pd.DataFrame(rows)
    write_table(out, output_dir / "tables" / "main_locf_operating_table.csv", output_dir / "tables" / "main_locf_operating_table.md")

    figure_rows = []
    for filename, figure_id, use in [
        ("locf_operating_range_heatmap.png", "F2", "Main LOCF operating-range heatmap."),
        ("locf_dice_vs_new_growth_rate.png", "S1", "Supporting LOCF-vs-growth-rate scatter."),
        ("locf_dice_vs_delta_days.png", "S2", "Supporting LOCF-vs-calendar-interval scatter."),
    ]:
        copied = copy_if_exists(
            locf_dir / filename if locf_dir else None,
            output_dir / "figures" / filename,
            missing,
        )
        if copied:
            figure_rows.append({"figure_id": figure_id, "file": copied, "recommended_use": use})
    if figure_rows:
        pd.DataFrame(figure_rows).to_csv(output_dir / "figures" / "recommended_locf_figures.csv", index=False)
    return out


def build_forecast_origin_table(stability_dir: Optional[Path], output_dir: Path, missing: List[str]) -> pd.DataFrame:
    claims = read_csv(stability_dir / "forecast_origin_patient_split_stability_claim_status.csv" if stability_dir else None, missing)
    summary = read_csv(stability_dir / "forecast_origin_patient_split_stability_summary.csv" if stability_dir else None, missing)
    rows = []
    if not claims.empty:
        for r in claims.itertuples():
            rows.append(
                {
                    "target": getattr(r, "target", ""),
                    "status": getattr(r, "status", ""),
                    "best_feature_set": getattr(r, "best_reference_feature_set", ""),
                    "quantitative_anchor": (
                        f"AUC mean={fmt(getattr(r, 'test_roc_auc_mean', np.nan))}, "
                        f"AUC q25={fmt(getattr(r, 'test_roc_auc_q25', np.nan))}, "
                        f"balanced acc mean={fmt(getattr(r, 'test_balanced_accuracy_mean', np.nan))}, "
                        f"FNR mean={fmt(getattr(r, 'test_false_negative_rate_mean', np.nan))}"
                    ),
                    "interpretation": "Forecast-origin signal should be used as risk stratification, not as a deterministic gate.",
                }
            )
    elif not summary.empty:
        rows.append(
            {
                "target": "all",
                "status": "summary_available_no_claim_status",
                "best_feature_set": "",
                "quantitative_anchor": f"{len(summary)} stability rows available.",
                "interpretation": "Inspect stability summary manually.",
            }
        )
    out = pd.DataFrame(rows)
    write_table(out, output_dir / "tables" / "main_forecast_origin_predictability_table.csv", output_dir / "tables" / "main_forecast_origin_predictability_table.md")
    return out


def build_method_diagnostic_table(
    failure_dir: Optional[Path],
    policy_robustness_dir: Optional[Path],
    output_dir: Path,
    missing: List[str],
) -> pd.DataFrame:
    failure = read_csv(failure_dir / "budgeted_growth_field_failure_evidence_map.csv" if failure_dir else None, missing)
    policy = read_csv(policy_robustness_dir / "conservative_policy_claim_status.csv" if policy_robustness_dir else None, missing)
    boot = read_csv(policy_robustness_dir / "conservative_policy_patient_bootstrap_summary.csv" if policy_robustness_dir else None, missing)

    rows = []
    if not failure.empty:
        for r in failure.head(8).itertuples():
            rows.append(
                {
                    "diagnostic": getattr(r, "evidence", getattr(r, "claim", "growth_field_failure")),
                    "status": getattr(r, "status", ""),
                    "quantitative_anchor": getattr(r, "value", getattr(r, "summary", "")),
                    "interpretation": "Learned spatial scores alone are not enough for Dice-safe LOCF correction.",
                }
            )
    if not policy.empty:
        for r in policy.itertuples():
            rows.append(
                {
                    "diagnostic": getattr(r, "claim", "conservative_policy"),
                    "status": getattr(r, "status", ""),
                    "quantitative_anchor": getattr(r, "evidence", ""),
                    "interpretation": getattr(r, "interpretation", ""),
                }
            )
    if not boot.empty:
        test_gap = boot[
            (boot.get("split", pd.Series(dtype=str)).astype(str) == "test")
            & (boot.get("summary_type", pd.Series(dtype=str)).astype(str) == "split")
            & (boot.get("metric", pd.Series(dtype=str)).astype(str) == "mean_gap_vs_locf")
        ]
        for r in test_gap.itertuples():
            rows.append(
                {
                    "diagnostic": f"Patient-bootstrap test gap ({r.policy_seed})",
                    "status": "uncertain" if float(r.ci_low) <= 0 <= float(r.ci_high) else "supported",
                    "quantitative_anchor": f"observed={fmt(r.observed)}, CI=[{fmt(r.ci_low)}, {fmt(r.ci_high)}], p>0={fmt(r.p_gt_zero)}",
                    "interpretation": "Do not claim robust method improvement if CI crosses zero.",
                }
            )
    out = pd.DataFrame(rows)
    write_table(out, output_dir / "tables" / "main_method_diagnostic_table.csv", output_dir / "tables" / "main_method_diagnostic_table.md")
    return out


def write_manifest(output_dir: Path, tables: Dict[str, pd.DataFrame], missing: List[str]) -> None:
    rows = []
    for name, df in tables.items():
        if df.empty:
            status = "empty_or_missing"
            n_rows = 0
        else:
            status = "ready"
            n_rows = len(df)
        rows.append(
            {
                "artifact": name,
                "status": status,
                "n_rows": n_rows,
                "recommended_role": {
                    "claim_map": "internal scaffold / possible supplement",
                    "transition": "main table and main figure",
                    "locf": "main figure/table",
                    "forecast_origin": "supporting main table or compact paragraph",
                    "method_diagnostic": "diagnostic table, likely supplement or discussion",
                }.get(name, ""),
            }
        )
    manifest = pd.DataFrame(rows)
    write_table(manifest, output_dir / "main_evidence_manifest.csv", output_dir / "main_evidence_manifest.md")
    if missing:
        pd.DataFrame({"missing_input": missing}).to_csv(output_dir / "main_evidence_missing_inputs.csv", index=False)

    with (output_dir / "main_evidence_package_report.md").open("w", encoding="utf-8") as f:
        f.write("# Main Evidence Package\n\n")
        f.write("This package curates the current analyses into manuscript-facing artifacts. It does not run new experiments.\n\n")
        f.write("## Artifact Manifest\n\n")
        f.write(manifest.to_markdown(index=False))
        f.write("\n\n## Recommended Main-Text Order\n\n")
        f.write("1. Transition composition and SRD-SAILOR roles.\n")
        f.write("2. LOCF operating range.\n")
        f.write("3. Forecast-origin predictability as supporting risk-stratification evidence.\n")
        f.write("4. Method-probe bottlenecks as diagnostic evidence, not as a method win.\n")
        if missing:
            f.write("\n## Missing Inputs\n\n")
            for item in missing:
                f.write(f"- `{item}`\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Curate manuscript-facing evidence artifacts from completed analyses.")
    parser.add_argument("--research_backbone_dir", type=str, default=None)
    parser.add_argument("--transition_main_dir", type=str, default=None)
    parser.add_argument("--transition_package_dir", type=str, default=None)
    parser.add_argument("--locf_operating_dir", type=str, default=None)
    parser.add_argument("--forecast_origin_stability_dir", type=str, default=None)
    parser.add_argument("--growth_field_failure_dir", type=str, default=None)
    parser.add_argument("--conservative_policy_robustness_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    missing: List[str] = []

    tables = {
        "claim_map": build_claim_table(Path(args.research_backbone_dir) if args.research_backbone_dir else None, output_dir, missing),
        "transition": build_transition_artifacts(
            Path(args.transition_main_dir) if args.transition_main_dir else None,
            Path(args.transition_package_dir) if args.transition_package_dir else None,
            output_dir,
            missing,
        ),
        "locf": build_locf_table(Path(args.locf_operating_dir) if args.locf_operating_dir else None, output_dir, missing),
        "forecast_origin": build_forecast_origin_table(
            Path(args.forecast_origin_stability_dir) if args.forecast_origin_stability_dir else None,
            output_dir,
            missing,
        ),
        "method_diagnostic": build_method_diagnostic_table(
            Path(args.growth_field_failure_dir) if args.growth_field_failure_dir else None,
            Path(args.conservative_policy_robustness_dir) if args.conservative_policy_robustness_dir else None,
            output_dir,
            missing,
        ),
    }
    write_manifest(output_dir, tables, missing)

    payload = {
        "output_dir": str(output_dir),
        "n_missing_inputs": int(len(missing)),
        "outputs": {
            "manifest_csv": str(output_dir / "main_evidence_manifest.csv"),
            "report_md": str(output_dir / "main_evidence_package_report.md"),
            "tables_dir": str(output_dir / "tables"),
            "figures_dir": str(output_dir / "figures"),
            "missing_inputs_csv": str(output_dir / "main_evidence_missing_inputs.csv") if missing else None,
        },
    }
    with (output_dir / "main_evidence_package_summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
