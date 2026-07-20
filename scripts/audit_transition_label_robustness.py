#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


EPS = 1e-12


def parse_csv_floats(payload: str) -> List[float]:
    vals = []
    for item in payload.split(","):
        item = item.strip()
        if item:
            vals.append(float(item))
    if not vals:
        raise ValueError("Expected at least one numeric value.")
    return vals


def load_samples(path_or_dir: str | Path) -> pd.DataFrame:
    p = Path(path_or_dir)
    if p.is_dir():
        p = p / "transition_taxonomy_samples.csv"
    if not p.exists():
        raise FileNotFoundError(f"Could not find transition samples: {p}")
    df = pd.read_csv(p)
    if df.empty:
        raise ValueError(f"No rows in transition samples: {p}")
    return df


def numeric(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def transition_masks(
    df: pd.DataFrame,
    persistence_thr: float,
    growth_loss_thr: float,
    distant_thr: float,
    core_loss_thr: float,
    high_change_thr: float,
) -> Dict[str, pd.Series]:
    rel_growth = numeric(df, "relative_new_growth")
    rel_loss = numeric(df, "relative_loss")
    rel_abs = numeric(df, "relative_absolute_change")
    distant = numeric(df, "distant_growth_fraction")
    core_loss = numeric(df, "core_loss_fraction")
    growth_vol = numeric(df, "new_growth_volume_vox")
    loss_vol = numeric(df, "loss_volume_vox")

    return {
        "persistence_dominant": rel_abs <= persistence_thr,
        "mixed_growth_loss": (rel_growth >= growth_loss_thr) & (rel_loss >= growth_loss_thr),
        "growth_dominant": (rel_growth >= growth_loss_thr) & (rel_loss < growth_loss_thr),
        "loss_dominant": (rel_loss >= growth_loss_thr) & (rel_growth < growth_loss_thr),
        "distant_growth_present": (growth_vol > 0) & (distant >= distant_thr),
        "core_loss_present": (loss_vol > 0) & (core_loss >= core_loss_thr),
        "high_absolute_change": rel_abs >= high_change_thr,
    }


def patient_concentration(part: pd.DataFrame) -> dict:
    if part.empty or "patient_id" not in part.columns:
        return {
            "n": int(len(part)),
            "n_patients": 0,
            "top_patient_id": "",
            "top_patient_count": 0,
            "max_patient_fraction": float("nan"),
            "effective_n_patients_entropy": float("nan"),
        }
    counts = part["patient_id"].astype(str).value_counts()
    probs = counts / max(1, int(counts.sum()))
    entropy = float(-(probs * np.log(probs + EPS)).sum())
    return {
        "n": int(len(part)),
        "n_patients": int(counts.size),
        "top_patient_id": str(counts.index[0]) if counts.size else "",
        "top_patient_count": int(counts.iloc[0]) if counts.size else 0,
        "max_patient_fraction": float(counts.iloc[0] / max(1, len(part))) if counts.size else float("nan"),
        "effective_n_patients_entropy": float(math.exp(entropy)) if counts.size else float("nan"),
    }


def summarize_region(df: pd.DataFrame, dataset: str, region: str, mask: pd.Series, settings: dict) -> dict:
    part = df[mask].copy()
    row = {
        "dataset": dataset,
        "region": region,
        **settings,
        "n_total": int(len(df)),
        "fraction": float(mask.mean()) if len(mask) else float("nan"),
        "mean_locf_dice": float(numeric(part, "locf_dice").mean()) if len(part) and "locf_dice" in part else float("nan"),
        "mean_relative_absolute_change": float(numeric(part, "relative_absolute_change").mean())
        if len(part) and "relative_absolute_change" in part
        else float("nan"),
    }
    row.update(patient_concentration(part))
    return row


def threshold_sensitivity(
    df: pd.DataFrame,
    dataset: str,
    persistence_thresholds: Iterable[float],
    growth_loss_thresholds: Iterable[float],
    distant_thresholds: Iterable[float],
    core_loss_thresholds: Iterable[float],
    high_change_thresholds: Iterable[float],
) -> pd.DataFrame:
    rows = []
    for persistence_thr in persistence_thresholds:
        for growth_loss_thr in growth_loss_thresholds:
            for distant_thr in distant_thresholds:
                for core_loss_thr in core_loss_thresholds:
                    for high_change_thr in high_change_thresholds:
                        settings = {
                            "persistence_thr": persistence_thr,
                            "growth_loss_thr": growth_loss_thr,
                            "distant_thr": distant_thr,
                            "core_loss_thr": core_loss_thr,
                            "high_change_thr": high_change_thr,
                        }
                        masks = transition_masks(
                            df,
                            persistence_thr=persistence_thr,
                            growth_loss_thr=growth_loss_thr,
                            distant_thr=distant_thr,
                            core_loss_thr=core_loss_thr,
                            high_change_thr=high_change_thr,
                        )
                        for region, mask in masks.items():
                            rows.append(summarize_region(df, dataset, region, mask, settings))
    return pd.DataFrame(rows)


def default_patient_spread(
    df: pd.DataFrame,
    dataset: str,
    persistence_thr: float,
    growth_loss_thr: float,
    distant_thr: float,
    core_loss_thr: float,
    high_change_thr: float,
) -> pd.DataFrame:
    settings = {
        "persistence_thr": persistence_thr,
        "growth_loss_thr": growth_loss_thr,
        "distant_thr": distant_thr,
        "core_loss_thr": core_loss_thr,
        "high_change_thr": high_change_thr,
    }
    masks = transition_masks(df, persistence_thr, growth_loss_thr, distant_thr, core_loss_thr, high_change_thr)
    rows = [summarize_region(df, dataset, region, mask, settings) for region, mask in masks.items()]
    return pd.DataFrame(rows).sort_values(["dataset", "region"])


def pairwise_gap(sensitivity: pd.DataFrame, dataset_a: str, dataset_b: str) -> pd.DataFrame:
    keys = ["region", "persistence_thr", "growth_loss_thr", "distant_thr", "core_loss_thr", "high_change_thr"]
    a = sensitivity[sensitivity["dataset"] == dataset_a].copy()
    b = sensitivity[sensitivity["dataset"] == dataset_b].copy()
    if a.empty or b.empty:
        return pd.DataFrame()
    merged = a[keys + ["fraction", "n", "n_patients", "mean_locf_dice"]].rename(
        columns={
            "fraction": "fraction_a",
            "n": "n_a",
            "n_patients": "n_patients_a",
            "mean_locf_dice": "mean_locf_dice_a",
        }
    ).merge(
        b[keys + ["fraction", "n", "n_patients", "mean_locf_dice"]].rename(
            columns={
                "fraction": "fraction_b",
                "n": "n_b",
                "n_patients": "n_patients_b",
                "mean_locf_dice": "mean_locf_dice_b",
            }
        ),
        on=keys,
        how="inner",
    )
    merged.insert(1, "dataset_a", dataset_a)
    merged.insert(2, "dataset_b", dataset_b)
    merged["fraction_a_minus_b"] = merged["fraction_a"] - merged["fraction_b"]
    merged["abs_fraction_gap"] = merged["fraction_a_minus_b"].abs()
    merged["mean_locf_dice_a_minus_b"] = merged["mean_locf_dice_a"] - merged["mean_locf_dice_b"]
    return merged.sort_values(["region", "abs_fraction_gap"], ascending=[True, False])


def robust_region_summary(gap: pd.DataFrame) -> pd.DataFrame:
    if gap.empty:
        return pd.DataFrame()
    rows = []
    for region, part in gap.groupby("region", observed=True):
        rows.append(
            {
                "region": region,
                "n_threshold_settings": int(len(part)),
                "mean_fraction_a": float(part["fraction_a"].mean()),
                "min_fraction_a": float(part["fraction_a"].min()),
                "max_fraction_a": float(part["fraction_a"].max()),
                "mean_fraction_b": float(part["fraction_b"].mean()),
                "min_fraction_b": float(part["fraction_b"].min()),
                "max_fraction_b": float(part["fraction_b"].max()),
                "mean_fraction_gap_a_minus_b": float(part["fraction_a_minus_b"].mean()),
                "min_fraction_gap_a_minus_b": float(part["fraction_a_minus_b"].min()),
                "max_fraction_gap_a_minus_b": float(part["fraction_a_minus_b"].max()),
                "fraction_gap_positive_rate": float((part["fraction_a_minus_b"] > 0).mean()),
                "mean_abs_fraction_gap": float(part["abs_fraction_gap"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_abs_fraction_gap", ascending=False)


def reference_quantile_region_coverage(reference: pd.DataFrame, candidate: pd.DataFrame, ref_name: str, cand_name: str) -> pd.DataFrame:
    features = [
        "relative_new_growth",
        "relative_loss",
        "relative_absolute_change",
        "relative_absolute_change_rate_per_day",
        "distant_growth_fraction",
        "core_loss_fraction",
        "boundary_loss_fraction",
        "input_volume_vox",
        "target_volume_vox",
    ]
    rows = []
    for feature in features:
        if feature not in reference.columns or feature not in candidate.columns:
            continue
        ref = numeric(reference, feature).replace([np.inf, -np.inf], np.nan).dropna()
        cand = numeric(candidate, feature).replace([np.inf, -np.inf], np.nan).dropna()
        if ref.empty or cand.empty:
            continue
        for q in [0.50, 0.75, 0.90]:
            thr = float(ref.quantile(q))
            for dataset, vals in [(ref_name, ref), (cand_name, cand)]:
                rows.append(
                    {
                        "reference_dataset": ref_name,
                        "candidate_dataset": cand_name,
                        "feature": feature,
                        "reference_quantile": q,
                        "reference_threshold": thr,
                        "dataset": dataset,
                        "fraction_at_or_above_reference_threshold": float((vals >= thr).mean()),
                        "n_at_or_above_reference_threshold": int((vals >= thr).sum()),
                        "n_total": int(len(vals)),
                    }
                )
    return pd.DataFrame(rows)


def write_plots(output_dir: Path, sensitivity: pd.DataFrame, robust: pd.DataFrame, patient_spread: pd.DataFrame) -> List[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: List[str] = []
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    def save(fig, name: str) -> None:
        path = fig_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(str(path))

    for region, threshold_col in [
        ("mixed_growth_loss", "growth_loss_thr"),
        ("distant_growth_present", "distant_thr"),
        ("core_loss_present", "core_loss_thr"),
        ("high_absolute_change", "high_change_thr"),
    ]:
        part = sensitivity[sensitivity["region"] == region].copy()
        if part.empty or threshold_col not in part.columns:
            continue
        summary = (
            part.groupby(["dataset", threshold_col], observed=True)
            .agg(mean_fraction=("fraction", "mean"), min_fraction=("fraction", "min"), max_fraction=("fraction", "max"))
            .reset_index()
        )
        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        for dataset, g in summary.groupby("dataset", observed=True):
            g = g.sort_values(threshold_col)
            ax.plot(g[threshold_col], g["mean_fraction"], marker="o", label=str(dataset))
            ax.fill_between(g[threshold_col], g["min_fraction"], g["max_fraction"], alpha=0.15)
        ax.set_xlabel(threshold_col)
        ax.set_ylabel("Fraction of transitions")
        ax.set_title(f"Threshold sensitivity: {region}")
        ax.legend(frameon=False)
        save(fig, f"threshold_sensitivity_{region}.png")

    if not robust.empty:
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        work = robust.sort_values("mean_abs_fraction_gap", ascending=True)
        ax.barh(work["region"], work["mean_abs_fraction_gap"])
        ax.set_xlabel("Mean absolute SAILOR-SRD fraction gap")
        ax.set_title("Robustness of transition-region gaps")
        save(fig, "robust_region_gap_summary.png")

    if not patient_spread.empty:
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        work = patient_spread[patient_spread["dataset"].astype(str).str.upper().eq("SAILOR")].copy()
        if not work.empty:
            work = work.sort_values("max_patient_fraction", ascending=True)
            ax.barh(work["region"], work["max_patient_fraction"])
            ax.set_xlabel("Largest single-patient share")
            ax.set_title("SAILOR hard-region patient concentration")
            save(fig, "sailor_patient_concentration.png")
    return paths


def write_report(
    output_dir: Path,
    sensitivity: pd.DataFrame,
    robust: pd.DataFrame,
    patient_spread: pd.DataFrame,
    quantile_coverage: pd.DataFrame,
    plot_paths: List[str],
    args: argparse.Namespace,
) -> None:
    report = output_dir / "transition_label_robustness_report.md"
    with report.open("w", encoding="utf-8") as f:
        f.write("# Transition Label Robustness Audit\n\n")
        f.write(
            "This audit checks whether the transition-analysis story depends on a single threshold choice or a small number of patients. "
            "It uses existing transition-taxonomy samples only; it does not rerun models or generate new data.\n\n"
        )
        f.write("## Inputs\n\n")
        f.write(f"- dataset_a_name: `{args.dataset_a_name}`\n")
        f.write(f"- dataset_a: `{args.dataset_a}`\n")
        f.write(f"- dataset_b_name: `{args.dataset_b_name}`\n")
        f.write(f"- dataset_b: `{args.dataset_b}`\n\n")
        f.write("## Threshold Grids\n\n")
        f.write(f"- persistence thresholds: `{args.persistence_thresholds}`\n")
        f.write(f"- growth/loss thresholds: `{args.growth_loss_thresholds}`\n")
        f.write(f"- distant-growth thresholds: `{args.distant_thresholds}`\n")
        f.write(f"- core-loss thresholds: `{args.core_loss_thresholds}`\n")
        f.write(f"- high-change thresholds: `{args.high_change_thresholds}`\n\n")

        f.write("## Robust Region Summary\n\n")
        f.write(robust.to_markdown(index=False) if not robust.empty else "No rows.")
        f.write("\n\n")

        f.write("## Default Patient Spread\n\n")
        f.write(patient_spread.to_markdown(index=False) if not patient_spread.empty else "No rows.")
        f.write("\n\n")

        if not quantile_coverage.empty:
            f.write("## Reference Quantile Region Coverage\n\n")
            f.write(quantile_coverage.to_markdown(index=False))
            f.write("\n\n")

        if plot_paths:
            f.write("## Figures\n\n")
            for path in plot_paths:
                f.write(f"- `{path}`\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit robustness of transition-region labels and patient spread.")
    parser.add_argument("--dataset_a", type=str, required=True, help="Reference taxonomy output dir or samples CSV.")
    parser.add_argument("--dataset_a_name", type=str, default="SAILOR")
    parser.add_argument("--dataset_b", type=str, required=True, help="Candidate taxonomy output dir or samples CSV.")
    parser.add_argument("--dataset_b_name", type=str, default="SRD")
    parser.add_argument("--persistence_thresholds", type=str, default="0.1,0.2,0.3")
    parser.add_argument("--growth_loss_thresholds", type=str, default="0.1,0.2,0.3,0.4")
    parser.add_argument("--distant_thresholds", type=str, default="0.1,0.2,0.3,0.4")
    parser.add_argument("--core_loss_thresholds", type=str, default="0.1,0.2,0.3,0.4")
    parser.add_argument("--high_change_thresholds", type=str, default="1.0,1.5,2.0")
    parser.add_argument("--default_persistence_thr", type=float, default=0.2)
    parser.add_argument("--default_growth_loss_thr", type=float, default=0.2)
    parser.add_argument("--default_distant_thr", type=float, default=0.2)
    parser.add_argument("--default_core_loss_thr", type=float, default=0.2)
    parser.add_argument("--default_high_change_thr", type=float, default=2.0)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--no_plots", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    a = load_samples(args.dataset_a)
    b = load_samples(args.dataset_b)

    sensitivity = pd.concat(
        [
            threshold_sensitivity(
                a,
                args.dataset_a_name,
                parse_csv_floats(args.persistence_thresholds),
                parse_csv_floats(args.growth_loss_thresholds),
                parse_csv_floats(args.distant_thresholds),
                parse_csv_floats(args.core_loss_thresholds),
                parse_csv_floats(args.high_change_thresholds),
            ),
            threshold_sensitivity(
                b,
                args.dataset_b_name,
                parse_csv_floats(args.persistence_thresholds),
                parse_csv_floats(args.growth_loss_thresholds),
                parse_csv_floats(args.distant_thresholds),
                parse_csv_floats(args.core_loss_thresholds),
                parse_csv_floats(args.high_change_thresholds),
            ),
        ],
        ignore_index=True,
    )
    gap = pairwise_gap(sensitivity, args.dataset_a_name, args.dataset_b_name)
    robust = robust_region_summary(gap)
    patient_spread = pd.concat(
        [
            default_patient_spread(
                a,
                args.dataset_a_name,
                args.default_persistence_thr,
                args.default_growth_loss_thr,
                args.default_distant_thr,
                args.default_core_loss_thr,
                args.default_high_change_thr,
            ),
            default_patient_spread(
                b,
                args.dataset_b_name,
                args.default_persistence_thr,
                args.default_growth_loss_thr,
                args.default_distant_thr,
                args.default_core_loss_thr,
                args.default_high_change_thr,
            ),
        ],
        ignore_index=True,
    )
    quantile_coverage = reference_quantile_region_coverage(a, b, args.dataset_a_name, args.dataset_b_name)

    outputs = {
        "threshold_sensitivity_csv": output_dir / "transition_label_threshold_sensitivity.csv",
        "threshold_gap_csv": output_dir / "transition_label_threshold_gap.csv",
        "robust_region_summary_csv": output_dir / "transition_label_robust_region_summary.csv",
        "default_patient_spread_csv": output_dir / "transition_label_default_patient_spread.csv",
        "reference_quantile_coverage_csv": output_dir / "transition_label_reference_quantile_coverage.csv",
    }
    sensitivity.to_csv(outputs["threshold_sensitivity_csv"], index=False)
    gap.to_csv(outputs["threshold_gap_csv"], index=False)
    robust.to_csv(outputs["robust_region_summary_csv"], index=False)
    patient_spread.to_csv(outputs["default_patient_spread_csv"], index=False)
    quantile_coverage.to_csv(outputs["reference_quantile_coverage_csv"], index=False)

    plot_paths = [] if args.no_plots else write_plots(output_dir, sensitivity, robust, patient_spread)
    write_report(output_dir, sensitivity, robust, patient_spread, quantile_coverage, plot_paths, args)

    payload = {
        "dataset_a": args.dataset_a,
        "dataset_a_name": args.dataset_a_name,
        "dataset_b": args.dataset_b,
        "dataset_b_name": args.dataset_b_name,
        "n_a": int(len(a)),
        "n_b": int(len(b)),
        "n_patients_a": int(a["patient_id"].nunique()) if "patient_id" in a else None,
        "n_patients_b": int(b["patient_id"].nunique()) if "patient_id" in b else None,
        "output_dir": str(output_dir),
        "outputs": {k: str(v) for k, v in outputs.items()},
        "report_md": str(output_dir / "transition_label_robustness_report.md"),
        "plots": plot_paths,
    }
    with (output_dir / "transition_label_robustness_summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
