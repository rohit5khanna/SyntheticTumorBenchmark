#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

DEFAULT_FEATURES = [
    "locf_dice",
    "delta_days",
    "input_volume_vox",
    "target_volume_vox",
    "persistent_input_fraction",
    "target_covered_by_input_fraction",
    "relative_new_growth",
    "relative_loss",
    "relative_absolute_change",
    "relative_absolute_change_rate_per_day",
    "boundary_growth_fraction",
    "distant_growth_fraction",
    "boundary_loss_fraction",
    "core_loss_fraction",
]


def parse_csv(payload: str | None) -> List[str]:
    if payload is None:
        return []
    return [x.strip() for x in payload.split(",") if x.strip()]


def load_samples(path_or_dir: str | Path) -> pd.DataFrame:
    p = Path(path_or_dir)
    if p.is_dir():
        p = p / "transition_taxonomy_samples.csv"
    if not p.exists():
        raise FileNotFoundError(f"Could not find transition taxonomy samples: {p}")
    df = pd.read_csv(p)
    if df.empty:
        raise ValueError(f"No rows in taxonomy samples: {p}")
    return df


def available_features(a: pd.DataFrame, b: pd.DataFrame, requested: List[str]) -> List[str]:
    out = []
    for feature in requested:
        if feature in a.columns and feature in b.columns:
            out.append(feature)
    if not out:
        raise ValueError("No requested comparison features are present in both datasets.")
    return out


def describe_feature(df: pd.DataFrame, dataset_name: str, features: List[str]) -> pd.DataFrame:
    rows = []
    for feature in features:
        x = pd.to_numeric(df[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if x.empty:
            continue
        rows.append(
            {
                "dataset": dataset_name,
                "feature": feature,
                "n": int(len(x)),
                "mean": float(x.mean()),
                "std": float(x.std()),
                "median": float(x.median()),
                "q10": float(x.quantile(0.10)),
                "q25": float(x.quantile(0.25)),
                "q75": float(x.quantile(0.75)),
                "q90": float(x.quantile(0.90)),
                "min": float(x.min()),
                "max": float(x.max()),
            }
        )
    return pd.DataFrame(rows)


def distribution_gap(a: pd.DataFrame, b: pd.DataFrame, a_name: str, b_name: str, features: List[str]) -> pd.DataFrame:
    try:
        from scipy.stats import ks_2samp, wasserstein_distance
    except Exception:
        ks_2samp = None
        wasserstein_distance = None

    rows = []
    for feature in features:
        x = pd.to_numeric(a[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        y = pd.to_numeric(b[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        if len(x) == 0 or len(y) == 0:
            continue
        pooled = np.concatenate([x, y])
        pooled_std = float(np.std(pooled, ddof=1)) if len(pooled) > 1 else float("nan")
        mean_gap = float(np.mean(a[feature]) - np.mean(b[feature]))
        median_gap = float(np.median(x) - np.median(y))
        row = {
            "feature": feature,
            "dataset_a": a_name,
            "dataset_b": b_name,
            "n_a": int(len(x)),
            "n_b": int(len(y)),
            "mean_a": float(np.mean(x)),
            "mean_b": float(np.mean(y)),
            "mean_a_minus_b": mean_gap,
            "median_a": float(np.median(x)),
            "median_b": float(np.median(y)),
            "median_a_minus_b": median_gap,
            "standardized_mean_diff": float(mean_gap / pooled_std) if pooled_std and np.isfinite(pooled_std) and pooled_std > 0 else float("nan"),
        }
        if ks_2samp is not None:
            ks = ks_2samp(x, y, alternative="two-sided", mode="auto")
            row["ks_statistic"] = float(ks.statistic)
            row["ks_pvalue"] = float(ks.pvalue)
        else:
            row["ks_statistic"] = float("nan")
            row["ks_pvalue"] = float("nan")
        if wasserstein_distance is not None:
            row["wasserstein_distance"] = float(wasserstein_distance(x, y))
        else:
            row["wasserstein_distance"] = float("nan")
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["abs_standardized_mean_diff"] = out["standardized_mean_diff"].abs()
        out = out.sort_values("abs_standardized_mean_diff", ascending=False)
    return out


def category_proportions(df: pd.DataFrame, dataset_name: str, col: str) -> pd.DataFrame:
    if col not in df.columns:
        return pd.DataFrame()
    counts = df[col].astype(str).value_counts(dropna=False).rename_axis(col).reset_index(name="count")
    counts.insert(0, "dataset", dataset_name)
    counts["fraction"] = counts["count"] / max(1, counts["count"].sum())
    return counts


def category_gap(a: pd.DataFrame, b: pd.DataFrame, a_name: str, b_name: str, col: str) -> pd.DataFrame:
    pa = category_proportions(a, a_name, col)
    pb = category_proportions(b, b_name, col)
    if pa.empty or pb.empty:
        return pd.DataFrame()
    merged = pa[[col, "fraction", "count"]].rename(columns={"fraction": "fraction_a", "count": "count_a"}).merge(
        pb[[col, "fraction", "count"]].rename(columns={"fraction": "fraction_b", "count": "count_b"}),
        on=col,
        how="outer",
    )
    merged[["fraction_a", "fraction_b"]] = merged[["fraction_a", "fraction_b"]].fillna(0.0)
    merged[["count_a", "count_b"]] = merged[["count_a", "count_b"]].fillna(0).astype(int)
    merged.insert(0, "category", col)
    merged.insert(1, "dataset_a", a_name)
    merged.insert(2, "dataset_b", b_name)
    merged["fraction_a_minus_b"] = merged["fraction_a"] - merged["fraction_b"]
    merged["abs_fraction_gap"] = merged["fraction_a_minus_b"].abs()
    return merged.sort_values("abs_fraction_gap", ascending=False)


def reference_quantile_coverage(reference: pd.DataFrame, candidate: pd.DataFrame, ref_name: str, cand_name: str, features: List[str]) -> pd.DataFrame:
    rows = []
    for feature in features:
        ref = pd.to_numeric(reference[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        cand = pd.to_numeric(candidate[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if ref.empty or cand.empty:
            continue
        q10, q25, q50, q75, q90 = [float(ref.quantile(q)) for q in [0.10, 0.25, 0.50, 0.75, 0.90]]
        rows.append(
            {
                "reference_dataset": ref_name,
                "candidate_dataset": cand_name,
                "feature": feature,
                "reference_q10": q10,
                "reference_q25": q25,
                "reference_median": q50,
                "reference_q75": q75,
                "reference_q90": q90,
                "candidate_fraction_below_ref_q10": float((cand < q10).mean()),
                "candidate_fraction_between_ref_q10_q90": float(((cand >= q10) & (cand <= q90)).mean()),
                "candidate_fraction_above_ref_q90": float((cand > q90).mean()),
                "candidate_fraction_above_ref_q75": float((cand > q75).mean()),
                "candidate_median_minus_ref_median": float(cand.median() - q50),
            }
        )
    return pd.DataFrame(rows)


def hard_region_summary(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    rows = []
    masks: Dict[str, pd.Series] = {}
    if "transition_type" in df.columns:
        masks["mixed_growth_loss"] = df["transition_type"].astype(str).eq("mixed_growth_loss")
        masks["growth_dominant"] = df["transition_type"].astype(str).eq("growth_dominant")
        masks["loss_dominant"] = df["transition_type"].astype(str).eq("loss_dominant")
    if "distant_growth_fraction" in df.columns:
        masks["distant_growth_fraction_ge_0p20"] = pd.to_numeric(df["distant_growth_fraction"], errors="coerce").fillna(0) >= 0.20
    if "core_loss_fraction" in df.columns:
        masks["core_loss_fraction_ge_0p20"] = pd.to_numeric(df["core_loss_fraction"], errors="coerce").fillna(0) >= 0.20
    if {"relative_new_growth", "relative_loss"}.issubset(df.columns):
        growth = pd.to_numeric(df["relative_new_growth"], errors="coerce").fillna(0)
        loss = pd.to_numeric(df["relative_loss"], errors="coerce").fillna(0)
        masks["relative_growth_and_loss_ge_0p20"] = (growth >= 0.20) & (loss >= 0.20)
    if "relative_absolute_change" in df.columns:
        rel_abs = pd.to_numeric(df["relative_absolute_change"], errors="coerce").fillna(0)
        masks["relative_abs_change_ge_1p0"] = rel_abs >= 1.0
        masks["relative_abs_change_ge_2p0"] = rel_abs >= 2.0

    for region, mask in masks.items():
        part = df[mask].copy()
        rows.append(
            {
                "dataset": dataset_name,
                "region": region,
                "n": int(len(part)),
                "n_patients": int(part["patient_id"].nunique()) if "patient_id" in part else 0,
                "fraction": float(len(part) / max(1, len(df))),
                "mean_locf_dice": float(part["locf_dice"].mean()) if len(part) and "locf_dice" in part else float("nan"),
                "mean_relative_absolute_change": float(part["relative_absolute_change"].mean()) if len(part) and "relative_absolute_change" in part else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def write_report(path: Path, tables: Dict[str, pd.DataFrame], args: argparse.Namespace) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Transition Taxonomy Domain Gap\n\n")
        f.write(
            "This analysis compares two transition-taxonomy outputs. It is intended to test whether a controlled synthetic transition space covers the same operating regions observed in real longitudinal data.\n\n"
        )
        f.write("## Inputs\n\n")
        f.write(f"- dataset_a_name: `{args.dataset_a_name}`\n")
        f.write(f"- dataset_a: `{args.dataset_a}`\n")
        f.write(f"- dataset_b_name: `{args.dataset_b_name}`\n")
        f.write(f"- dataset_b: `{args.dataset_b}`\n\n")
        for name, table in tables.items():
            f.write(f"## {name}\n\n")
            f.write(table.to_markdown(index=False) if not table.empty else "No rows.")
            f.write("\n\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare transition taxonomy distributions between two datasets.")
    parser.add_argument("--dataset_a", type=str, required=True, help="Taxonomy output dir or samples CSV for reference dataset.")
    parser.add_argument("--dataset_a_name", type=str, default="SAILOR")
    parser.add_argument("--dataset_b", type=str, required=True, help="Taxonomy output dir or samples CSV for candidate dataset.")
    parser.add_argument("--dataset_b_name", type=str, default="SRD")
    parser.add_argument("--features", type=str, default=",".join(DEFAULT_FEATURES))
    parser.add_argument("--categories", type=str, default="transition_type,net_direction")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    a = load_samples(args.dataset_a)
    b = load_samples(args.dataset_b)
    features = available_features(a, b, parse_csv(args.features))
    categories = parse_csv(args.categories)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_summary = pd.concat(
        [describe_feature(a, args.dataset_a_name, features), describe_feature(b, args.dataset_b_name, features)],
        ignore_index=True,
    )
    gap = distribution_gap(a, b, args.dataset_a_name, args.dataset_b_name, features)
    coverage = reference_quantile_coverage(a, b, args.dataset_a_name, args.dataset_b_name, features)
    hard_regions = pd.concat(
        [hard_region_summary(a, args.dataset_a_name), hard_region_summary(b, args.dataset_b_name)],
        ignore_index=True,
    )

    category_tables = []
    category_gap_tables = []
    for category in categories:
        if category in a.columns and category in b.columns:
            category_tables.append(category_proportions(a, args.dataset_a_name, category))
            category_tables.append(category_proportions(b, args.dataset_b_name, category))
            category_gap_tables.append(category_gap(a, b, args.dataset_a_name, args.dataset_b_name, category))
    category_props = pd.concat(category_tables, ignore_index=True) if category_tables else pd.DataFrame()
    category_gaps = pd.concat(category_gap_tables, ignore_index=True) if category_gap_tables else pd.DataFrame()

    tables = {
        "Feature Summary": feature_summary,
        "Distribution Gap": gap,
        "Reference Quantile Coverage": coverage,
        "Category Proportions": category_props,
        "Category Gap": category_gaps,
        "Hard Region Summary": hard_regions,
    }

    outputs = {}
    for name, table in tables.items():
        filename = name.lower().replace(" ", "_") + ".csv"
        path = output_dir / filename
        table.to_csv(path, index=False)
        outputs[name] = str(path)

    report_path = output_dir / "transition_taxonomy_domain_gap_report.md"
    write_report(report_path, tables, args)
    outputs["report_md"] = str(report_path)

    payload = {
        "dataset_a": args.dataset_a,
        "dataset_a_name": args.dataset_a_name,
        "dataset_b": args.dataset_b,
        "dataset_b_name": args.dataset_b_name,
        "features": features,
        "categories": categories,
        "n_a": int(len(a)),
        "n_b": int(len(b)),
        "n_patients_a": int(a["patient_id"].nunique()) if "patient_id" in a else None,
        "n_patients_b": int(b["patient_id"].nunique()) if "patient_id" in b else None,
        "output_dir": str(output_dir),
        "outputs": outputs,
    }
    with (output_dir / "transition_taxonomy_domain_gap_summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
