#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


CORE_NUMERIC_COLUMNS = [
    "n_transitions",
    "n_patients",
    "mean_delta_days",
    "median_delta_days",
    "mean_locf_dice",
    "median_locf_dice",
    "mean_input_volume_vox",
    "mean_target_volume_vox",
    "mean_persistent_input_fraction",
    "mean_target_covered_by_input_fraction",
    "mean_relative_new_growth",
    "mean_relative_loss",
    "mean_relative_absolute_change",
    "mean_relative_abs_change_rate_per_day",
    "mean_boundary_growth_fraction",
    "mean_distant_growth_fraction",
    "distant_growth_rate",
    "core_loss_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Curate existing SRD/SAILOR transition-taxonomy, LOCF operating-range, "
            "radius-sensitivity, and domain-gap outputs into one evidence package."
        )
    )
    parser.add_argument("--sailor_taxonomy_dir", type=str, required=True)
    parser.add_argument("--srd_taxonomy_dir", type=str, required=True)
    parser.add_argument("--domain_gap_dir", type=str, default=None)
    parser.add_argument("--sailor_radius_dir", type=str, default=None)
    parser.add_argument("--sailor_locf_dir", type=str, default=None)
    parser.add_argument("--sailor_name", type=str, default="SAILOR")
    parser.add_argument("--srd_name", type=str, default="SRD")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--no_plots", action="store_true")
    return parser.parse_args()


def read_csv(path: Path, missing: List[str], required: bool = False) -> pd.DataFrame:
    if not path.exists():
        msg = f"missing: {path}"
        missing.append(msg)
        if required:
            raise FileNotFoundError(msg)
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        msg = f"could not read {path}: {exc}"
        missing.append(msg)
        if required:
            raise
        return pd.DataFrame()


def maybe_write(df: pd.DataFrame, path: Path) -> Optional[str]:
    if df.empty:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return str(path)


def load_taxonomy_tables(root: Path, name: str, missing: List[str]) -> Dict[str, pd.DataFrame]:
    return {
        "samples": read_csv(root / "transition_taxonomy_samples.csv", missing, required=True),
        "overall": read_csv(root / "transition_taxonomy_overall.csv", missing),
        "by_split": read_csv(root / "transition_taxonomy_by_split.csv", missing),
        "by_tier": read_csv(root / "transition_taxonomy_by_tier.csv", missing),
        "by_horizon": read_csv(root / "transition_taxonomy_by_horizon.csv", missing),
        "by_net_direction": read_csv(root / "transition_taxonomy_by_net_direction.csv", missing),
        "by_transition_type": read_csv(root / "transition_taxonomy_by_transition_type.csv", missing),
        "patient_trajectories": read_csv(root / "transition_taxonomy_patient_trajectories.csv", missing),
        "_name": pd.DataFrame({"dataset": [name]}),
    }


def add_dataset(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out.insert(0, "dataset", dataset)
    return out


def summarize_samples(samples: pd.DataFrame, dataset: str) -> pd.DataFrame:
    if samples.empty:
        return pd.DataFrame()
    rows = {
        "dataset": dataset,
        "n_transitions": int(len(samples)),
        "n_patients": int(samples["patient_id"].nunique()) if "patient_id" in samples else 0,
    }
    for col in CORE_NUMERIC_COLUMNS:
        if col in {"n_transitions", "n_patients"}:
            continue
        raw_col = col.removeprefix("mean_")
        if col.startswith("median_"):
            raw_col = col.removeprefix("median_")
        if raw_col in samples.columns:
            vals = pd.to_numeric(samples[raw_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            rows[col] = float(vals.median() if col.startswith("median_") else vals.mean())
    return pd.DataFrame([rows])


def combine_overall(a: Dict[str, pd.DataFrame], b: Dict[str, pd.DataFrame], a_name: str, b_name: str) -> pd.DataFrame:
    parts = []
    for tables, name in [(a, a_name), (b, b_name)]:
        overall = tables.get("overall", pd.DataFrame())
        if overall.empty:
            overall = summarize_samples(tables.get("samples", pd.DataFrame()), name)
        else:
            overall = add_dataset(overall, name)
        parts.append(overall)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def category_counts(samples: pd.DataFrame, dataset: str, col: str) -> pd.DataFrame:
    if samples.empty or col not in samples.columns:
        return pd.DataFrame()
    counts = samples[col].astype(str).value_counts(dropna=False).rename_axis(col).reset_index(name="count")
    counts.insert(0, "dataset", dataset)
    counts["fraction"] = counts["count"] / max(1, int(counts["count"].sum()))
    if "locf_dice" in samples.columns:
        locf = samples.groupby(samples[col].astype(str), observed=True)["locf_dice"].mean().reset_index()
        locf.columns = [col, "mean_locf_dice"]
        counts = counts.merge(locf, on=col, how="left")
    return counts


def combine_category_counts(
    a_samples: pd.DataFrame,
    b_samples: pd.DataFrame,
    a_name: str,
    b_name: str,
    col: str,
) -> pd.DataFrame:
    parts = [category_counts(a_samples, a_name, col), category_counts(b_samples, b_name, col)]
    parts = [p for p in parts if not p.empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def top_domain_gap(domain_gap_dir: Optional[Path], missing: List[str]) -> Dict[str, pd.DataFrame]:
    if domain_gap_dir is None:
        return {}
    out = {
        "distribution_gap_top": read_csv(domain_gap_dir / "distribution_gap.csv", missing),
        "hard_region_summary": read_csv(domain_gap_dir / "hard_region_summary.csv", missing),
        "category_gap": read_csv(domain_gap_dir / "category_gap.csv", missing),
        "reference_quantile_coverage": read_csv(domain_gap_dir / "reference_quantile_coverage.csv", missing),
    }
    if not out["distribution_gap_top"].empty:
        sort_col = "abs_standardized_mean_diff"
        if sort_col in out["distribution_gap_top"].columns:
            out["distribution_gap_top"] = out["distribution_gap_top"].sort_values(sort_col, ascending=False).head(12)
    if not out["category_gap"].empty and "abs_fraction_gap" in out["category_gap"].columns:
        out["category_gap"] = out["category_gap"].sort_values("abs_fraction_gap", ascending=False).head(12)
    return out


def load_radius_tables(radius_dir: Optional[Path], missing: List[str]) -> Dict[str, pd.DataFrame]:
    if radius_dir is None:
        return {}
    return {
        "radius_stability": read_csv(radius_dir / "radius_sensitivity_stability_summary.csv", missing),
        "radius_by_net_direction": read_csv(radius_dir / "radius_sensitivity_by_net_direction.csv", missing),
        "radius_by_transition_type": read_csv(radius_dir / "radius_sensitivity_by_transition_type.csv", missing),
    }


def load_locf_tables(locf_dir: Optional[Path], missing: List[str]) -> Dict[str, pd.DataFrame]:
    if locf_dir is None:
        return {}
    out = {}
    for csv_path in sorted(locf_dir.glob("locf_operating_*.csv")):
        out[csv_path.stem] = read_csv(csv_path, missing)
    return out


def write_claim_support_matrix(path: Path) -> pd.DataFrame:
    rows = [
        {
            "claim": "Short-horizon forecasting is not uniform.",
            "status": "strong_descriptive",
            "primary_evidence": "SAILOR taxonomy; SRD tier/horizon taxonomy; LOCF operating range",
            "main_risk": "Could sound obvious unless tied to measurable transition components.",
            "next_check": "Report patient counts, split summaries, and sensitivity settings.",
        },
        {
            "claim": "LOCF has an operating range rather than a universal role.",
            "status": "strong_but_careful",
            "primary_evidence": "LOCF Dice versus transition burden, growth/loss rate, and interval.",
            "main_risk": "Partly mathematical because LOCF Dice is coupled to change burden.",
            "next_check": "Phrase as characterization, not causal discovery.",
        },
        {
            "claim": "SRD is controlled mechanism isolation, not a SAILOR surrogate.",
            "status": "strong",
            "primary_evidence": "SRD taxonomy and SRD-SAILOR domain gap.",
            "main_risk": "Reviewers may misread the mismatch as synthetic failure.",
            "next_check": "State the SRD/SAILOR roles early and consistently.",
        },
        {
            "claim": "SAILOR contains real transition complexity absent from SRD.",
            "status": "strong_descriptive",
            "primary_evidence": "Mixed growth/loss, distant growth, core loss, scale, and high-change-tail gaps.",
            "main_risk": "Small dataset and heuristic labels.",
            "next_check": "Show radius sensitivity and patient/split counts.",
        },
        {
            "claim": "Growth-front ranking is a separate evaluation axis from full-mask Dice.",
            "status": "strong_conceptual",
            "primary_evidence": "Ranking-vs-Dice tradeoff and distance/model ranking analyses.",
            "main_risk": "Ranking may not convert into better masks without budget control.",
            "next_check": "Always pair ranking metrics with budgeted-mask analysis.",
        },
        {
            "claim": "Boundary proximity is a useful but incomplete growth prior.",
            "status": "strong_but_incomplete",
            "primary_evidence": "Distance prior and delayed-hit analyses; SAILOR spatial taxonomy.",
            "main_risk": "Non-boundary growth is common enough that distance cannot be sufficient.",
            "next_check": "Keep distant-growth cases visible.",
        },
        {
            "claim": "Image/context models add useful signal beyond distance.",
            "status": "moderate_promising",
            "primary_evidence": "Cropped SAILOR hybrid ranking improvement over distance.",
            "main_risk": "Small sample, crop effects, limited seeds/checkpoints.",
            "next_check": "Patient-level and repeated-split validation.",
        },
        {
            "claim": "Growth-only residual forecasting is a mature method contribution.",
            "status": "weak_not_central_yet",
            "primary_evidence": "Small positive SAILOR seed 42/123 trends.",
            "main_risk": "Tiny test gains and possible threshold/budget tuning.",
            "next_check": "Same-resolution baselines, more splits/seeds, failure analysis.",
        },
        {
            "claim": "Growth and loss should be modeled symmetrically.",
            "status": "rejected_for_now",
            "primary_evidence": "Residual-change experiments struggled with loss; loss has ambiguous causes.",
            "main_risk": "Symmetric modeling may conflate treatment, registration, and biology.",
            "next_check": "Analyze loss as separate uncertainty-sensitive component.",
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def safe_get(df: pd.DataFrame, col: str, default: str = "NA") -> str:
    if df.empty or col not in df.columns or pd.isna(df.iloc[0][col]):
        return default
    val = df.iloc[0][col]
    if isinstance(val, (float, np.floating)):
        return f"{val:.3f}"
    return str(val)


def write_report(
    path: Path,
    outputs: Dict[str, str],
    missing: List[str],
    sailor_overall: pd.DataFrame,
    srd_overall: pd.DataFrame,
    claim_matrix: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Transition Evidence Package\n\n")
        f.write(
            "This package curates existing transition analyses. It does not rerun models or generate a new synthetic dataset. "
            "Its purpose is to make the current evidence auditable and to separate strong descriptive findings from weaker method claims.\n\n"
        )
        f.write("## Inputs\n\n")
        f.write(f"- SAILOR taxonomy: `{args.sailor_taxonomy_dir}`\n")
        f.write(f"- SRD taxonomy: `{args.srd_taxonomy_dir}`\n")
        if args.domain_gap_dir:
            f.write(f"- Domain gap: `{args.domain_gap_dir}`\n")
        if args.sailor_radius_dir:
            f.write(f"- SAILOR radius sensitivity: `{args.sailor_radius_dir}`\n")
        if args.sailor_locf_dir:
            f.write(f"- SAILOR LOCF operating range: `{args.sailor_locf_dir}`\n")
        f.write("\n")

        f.write("## Core Interpretation\n\n")
        f.write("- SRD should be treated as a controlled mechanism-isolation environment.\n")
        f.write("- SAILOR should be treated as the real transition-complexity audit.\n")
        f.write("- The SRD-SAILOR domain gap is evidence, not a failure to hide.\n")
        f.write("- Method claims remain weaker than transition-analysis claims at this stage.\n\n")

        f.write("## Compact Dataset Summary\n\n")
        f.write(
            f"- SAILOR: n={safe_get(sailor_overall, 'n_transitions')}, patients={safe_get(sailor_overall, 'n_patients')}, "
            f"mean LOCF Dice={safe_get(sailor_overall, 'mean_locf_dice')}, "
            f"mean relative absolute change={safe_get(sailor_overall, 'mean_relative_absolute_change')}.\n"
        )
        f.write(
            f"- SRD: n={safe_get(srd_overall, 'n_transitions')}, patients={safe_get(srd_overall, 'n_patients')}, "
            f"mean LOCF Dice={safe_get(srd_overall, 'mean_locf_dice')}, "
            f"mean relative absolute change={safe_get(srd_overall, 'mean_relative_absolute_change')}.\n\n"
        )

        f.write("## Claim Support Matrix\n\n")
        f.write(claim_matrix.to_markdown(index=False))
        f.write("\n\n")

        f.write("## Outputs\n\n")
        for label, out_path in outputs.items():
            f.write(f"- {label}: `{out_path}`\n")
        if missing:
            f.write("\n## Missing Or Skipped Inputs\n\n")
            for item in missing:
                f.write(f"- {item}\n")
        f.write("\n## Recommended Next Use\n\n")
        f.write(
            "Use this package to decide which figures/tables are central before running more models. "
            "The next experiment should be justified by one of the weak or moderate rows in the claim matrix.\n"
        )


def bar_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    path: Path,
    title: str,
    ylabel: str,
    rotation: int = 30,
) -> Optional[str]:
    if df.empty or not {x, y, hue}.issubset(df.columns):
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    work = df[[x, y, hue]].dropna().copy()
    if work.empty:
        return None
    x_vals = list(dict.fromkeys(work[x].astype(str)))
    hue_vals = list(dict.fromkeys(work[hue].astype(str)))
    width = 0.8 / max(1, len(hue_vals))
    fig, ax = plt.subplots(figsize=(max(6, len(x_vals) * 0.8), 4.2))
    idx = np.arange(len(x_vals))
    for i, hv in enumerate(hue_vals):
        vals = []
        for xv in x_vals:
            part = work[(work[x].astype(str) == xv) & (work[hue].astype(str) == hv)]
            vals.append(float(part[y].iloc[0]) if not part.empty else np.nan)
        ax.bar(idx + (i - (len(hue_vals) - 1) / 2) * width, vals, width=width, label=hv)
    ax.set_xticks(idx)
    ax.set_xticklabels(x_vals, rotation=rotation, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def line_plot(df: pd.DataFrame, x: str, ys: Iterable[str], path: Path, title: str, ylabel: str) -> Optional[str]:
    if df.empty or x not in df.columns:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    any_line = False
    for y in ys:
        if y in df.columns:
            work = df[[x, y]].dropna().sort_values(x)
            if not work.empty:
                ax.plot(work[x], work[y], marker="o", label=y)
                any_line = True
    if not any_line:
        plt.close(fig)
        return None
    ax.set_xlabel(x)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def copy_existing_reports(source_dirs: Dict[str, Optional[Path]], output_dir: Path, missing: List[str]) -> Dict[str, str]:
    copied = {}
    report_dir = output_dir / "source_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    for label, root in source_dirs.items():
        if root is None:
            continue
        reports = list(root.glob("*.md")) + list(root.glob("*summary.json"))
        for report in reports:
            if report.exists():
                dest = report_dir / f"{label}_{report.name}"
                shutil.copy2(report, dest)
                copied[f"source_{label}_{report.name}"] = str(dest)
    if not copied:
        missing.append("no source markdown/json reports found to copy")
    return copied


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    table_dir = output_dir / "core_tables"
    fig_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    missing: List[str] = []
    sailor_dir = Path(args.sailor_taxonomy_dir)
    srd_dir = Path(args.srd_taxonomy_dir)
    domain_gap_dir = Path(args.domain_gap_dir) if args.domain_gap_dir else None
    radius_dir = Path(args.sailor_radius_dir) if args.sailor_radius_dir else None
    locf_dir = Path(args.sailor_locf_dir) if args.sailor_locf_dir else None

    sailor = load_taxonomy_tables(sailor_dir, args.sailor_name, missing)
    srd = load_taxonomy_tables(srd_dir, args.srd_name, missing)

    outputs: Dict[str, str] = {}
    overall = combine_overall(sailor, srd, args.sailor_name, args.srd_name)
    maybe = maybe_write(overall, table_dir / "dataset_overall_comparison.csv")
    if maybe:
        outputs["dataset_overall_comparison"] = maybe

    transition_counts = combine_category_counts(
        sailor["samples"], srd["samples"], args.sailor_name, args.srd_name, "transition_type"
    )
    maybe = maybe_write(transition_counts, table_dir / "transition_type_distribution.csv")
    if maybe:
        outputs["transition_type_distribution"] = maybe

    net_counts = combine_category_counts(sailor["samples"], srd["samples"], args.sailor_name, args.srd_name, "net_direction")
    maybe = maybe_write(net_counts, table_dir / "net_direction_distribution.csv")
    if maybe:
        outputs["net_direction_distribution"] = maybe

    for dataset_name, tables in [(args.sailor_name, sailor), (args.srd_name, srd)]:
        for key in ["by_split", "by_tier", "by_horizon", "by_net_direction", "by_transition_type", "patient_trajectories"]:
            table = tables.get(key, pd.DataFrame())
            maybe = maybe_write(add_dataset(table, dataset_name), table_dir / f"{dataset_name.lower()}_{key}.csv")
            if maybe:
                outputs[f"{dataset_name.lower()}_{key}"] = maybe

    for name, table in top_domain_gap(domain_gap_dir, missing).items():
        maybe = maybe_write(table, table_dir / f"domain_gap_{name}.csv")
        if maybe:
            outputs[f"domain_gap_{name}"] = maybe

    for name, table in load_radius_tables(radius_dir, missing).items():
        maybe = maybe_write(table, table_dir / f"sailor_{name}.csv")
        if maybe:
            outputs[f"sailor_{name}"] = maybe

    for name, table in load_locf_tables(locf_dir, missing).items():
        maybe = maybe_write(table, table_dir / f"sailor_{name}.csv")
        if maybe:
            outputs[f"sailor_{name}"] = maybe

    claim_matrix = write_claim_support_matrix(table_dir / "claim_support_matrix.csv")
    outputs["claim_support_matrix"] = str(table_dir / "claim_support_matrix.csv")

    plot_paths: Dict[str, str] = {}
    if not args.no_plots:
        p = bar_plot(
            transition_counts,
            x="transition_type",
            y="fraction",
            hue="dataset",
            path=fig_dir / "transition_type_distribution.png",
            title="Transition-type distribution",
            ylabel="Fraction of transitions",
        )
        if p:
            plot_paths["transition_type_distribution_plot"] = p
        p = bar_plot(
            transition_counts,
            x="transition_type",
            y="mean_locf_dice",
            hue="dataset",
            path=fig_dir / "locf_dice_by_transition_type.png",
            title="LOCF Dice by transition type",
            ylabel="Mean LOCF Dice",
        )
        if p:
            plot_paths["locf_by_transition_type_plot"] = p
        if not overall.empty:
            comp_cols = ["mean_relative_new_growth", "mean_relative_loss", "mean_relative_absolute_change"]
            plot_df = overall[["dataset"] + [c for c in comp_cols if c in overall.columns]].melt(
                id_vars="dataset", var_name="component", value_name="mean_value"
            )
            p = bar_plot(
                plot_df,
                x="component",
                y="mean_value",
                hue="dataset",
                path=fig_dir / "relative_transition_burden_components.png",
                title="Relative transition burden components",
                ylabel="Mean relative value",
            )
            if p:
                plot_paths["relative_transition_burden_plot"] = p
        radius_stability = load_radius_tables(radius_dir, missing).get("radius_stability", pd.DataFrame())
        p = line_plot(
            radius_stability,
            x="boundary_radius_vox",
            ys=["distant_growth_rate", "mean_distant_growth_fraction", "core_loss_rate"],
            path=fig_dir / "sailor_radius_sensitivity.png",
            title="SAILOR radius sensitivity",
            ylabel="Rate / fraction",
        )
        if p:
            plot_paths["sailor_radius_sensitivity_plot"] = p

    outputs.update(plot_paths)
    outputs.update(
        copy_existing_reports(
            {
                "sailor_taxonomy": sailor_dir,
                "srd_taxonomy": srd_dir,
                "domain_gap": domain_gap_dir,
                "sailor_radius": radius_dir,
                "sailor_locf": locf_dir,
            },
            output_dir,
            missing,
        )
    )

    sailor_overall = overall[overall["dataset"].astype(str) == args.sailor_name].head(1) if "dataset" in overall else pd.DataFrame()
    srd_overall = overall[overall["dataset"].astype(str) == args.srd_name].head(1) if "dataset" in overall else pd.DataFrame()
    report_path = output_dir / "transition_evidence_summary.md"
    write_report(report_path, outputs, missing, sailor_overall, srd_overall, claim_matrix, args)
    outputs["transition_evidence_summary_md"] = str(report_path)

    payload = {
        "sailor_taxonomy_dir": args.sailor_taxonomy_dir,
        "srd_taxonomy_dir": args.srd_taxonomy_dir,
        "domain_gap_dir": args.domain_gap_dir,
        "sailor_radius_dir": args.sailor_radius_dir,
        "sailor_locf_dir": args.sailor_locf_dir,
        "output_dir": str(output_dir),
        "missing_or_skipped": missing,
        "outputs": outputs,
    }
    with (output_dir / "transition_evidence_package_summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
