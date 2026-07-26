#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


def parse_policy_dirs(text: str) -> Dict[str, Path]:
    dirs: Dict[str, Path] = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            name, path = item.split("=", 1)
            label = name.strip()
            p = Path(path.strip())
        else:
            p = Path(item)
            label = p.name
        if not label:
            raise ValueError(f"Empty policy label in: {item}")
        dirs[label] = p
    if not dirs:
        raise ValueError("No policy directories were provided.")
    return dirs


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def read_policy_dir(seed_label: str, policy_dir: Path) -> Dict[str, pd.DataFrame | dict]:
    selected_samples = pd.read_csv(require_file(policy_dir / "conservative_growth_policy_selected_samples.csv"))
    selected_by_split = pd.read_csv(require_file(policy_dir / "conservative_growth_policy_selected_by_split.csv"))
    selected_by_direction = pd.read_csv(require_file(policy_dir / "conservative_growth_policy_selected_by_direction.csv"))
    validation_sweep = pd.read_csv(require_file(policy_dir / "conservative_growth_policy_validation_sweep.csv"))
    summary_path = require_file(policy_dir / "conservative_growth_policy_run_summary.json")
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    for df in (selected_samples, selected_by_split, selected_by_direction, validation_sweep):
        df.insert(0, "policy_seed", seed_label)
        df.insert(1, "policy_dir", str(policy_dir))
    return {
        "selected_samples": selected_samples,
        "selected_by_split": selected_by_split,
        "selected_by_direction": selected_by_direction,
        "validation_sweep": validation_sweep,
        "summary": summary,
    }


def mean_or_nan(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.nanmean(arr))


def summarize_rows(rows: pd.DataFrame) -> dict:
    if rows.empty:
        return {
            "n": 0,
            "n_patients": 0,
            "mean_dice": float("nan"),
            "mean_locf_dice": float("nan"),
            "mean_gap_vs_locf": float("nan"),
            "median_gap_vs_locf": float("nan"),
            "win_rate_vs_locf": float("nan"),
            "gate_open_rate": float("nan"),
            "edit_on_growth_rate": float("nan"),
            "edit_on_shrinkage_rate": float("nan"),
            "added_growth_mean": float("nan"),
            "growth_precision_mean": float("nan"),
            "growth_recall_mean": float("nan"),
        }
    out = {
        "n": int(len(rows)),
        "n_patients": int(rows["patient_id"].nunique()) if "patient_id" in rows.columns else 0,
        "mean_dice": mean_or_nan(rows["dice"]),
        "mean_locf_dice": mean_or_nan(rows["locf_dice"]),
        "mean_gap_vs_locf": mean_or_nan(rows["gap_vs_locf"]),
        "median_gap_vs_locf": float(np.nanmedian(rows["gap_vs_locf"].to_numpy(dtype=float))),
        "win_rate_vs_locf": float((rows["gap_vs_locf"].to_numpy(dtype=float) > 0).mean()),
    }
    for col in [
        "gate_open",
        "edit_on_growth",
        "edit_on_shrinkage",
        "added_growth_vox",
        "growth_precision",
        "growth_recall",
    ]:
        if col in rows.columns:
            out_name = {
                "gate_open": "gate_open_rate",
                "edit_on_growth": "edit_on_growth_rate",
                "edit_on_shrinkage": "edit_on_shrinkage_rate",
                "added_growth_vox": "added_growth_mean",
                "growth_precision": "growth_precision_mean",
                "growth_recall": "growth_recall_mean",
            }[col]
            out[out_name] = mean_or_nan(rows[col])
    return out


def bootstrap_group(rows: pd.DataFrame, group_keys: List[str], n_bootstrap: int, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    draw_rows = []
    summary_rows = []
    metric_cols = [
        "mean_dice",
        "mean_locf_dice",
        "mean_gap_vs_locf",
        "median_gap_vs_locf",
        "win_rate_vs_locf",
        "gate_open_rate",
        "edit_on_growth_rate",
        "edit_on_shrinkage_rate",
        "added_growth_mean",
        "growth_precision_mean",
        "growth_recall_mean",
    ]

    grouped = rows.groupby(group_keys, observed=True, dropna=False) if group_keys else [((), rows)]
    for key, part in grouped:
        key_tuple = key if isinstance(key, tuple) else (key,)
        key_dict = dict(zip(group_keys, key_tuple))
        patients = np.asarray(sorted(part["patient_id"].astype(str).unique()))
        if patients.size == 0:
            continue
        patient_to_rows = {pid: part[part["patient_id"].astype(str) == pid] for pid in patients}

        observed = summarize_rows(part)
        for metric in metric_cols:
            if metric in observed:
                summary_rows.append(
                    {
                        **key_dict,
                        "metric": metric,
                        "observed": observed[metric],
                        "bootstrap_mean": float("nan"),
                        "ci_low": float("nan"),
                        "ci_median": float("nan"),
                        "ci_high": float("nan"),
                        "p_gt_zero": float("nan"),
                        "p_lt_zero": float("nan"),
                        "n_bootstrap": int(n_bootstrap),
                        "n_patients": int(patients.size),
                    }
                )

        if n_bootstrap <= 0:
            continue

        draws_by_metric = {metric: [] for metric in metric_cols}
        for draw_idx in range(n_bootstrap):
            sampled = rng.choice(patients, size=patients.size, replace=True)
            draw = pd.concat([patient_to_rows[pid] for pid in sampled], ignore_index=True)
            rec = summarize_rows(draw)
            draw_record = {**key_dict, "draw_idx": int(draw_idx)}
            for metric in metric_cols:
                value = rec.get(metric, float("nan"))
                draw_record[metric] = value
                draws_by_metric[metric].append(value)
            draw_rows.append(draw_record)

        for row in summary_rows:
            if all(row.get(k) == v for k, v in key_dict.items()):
                vals = np.asarray(draws_by_metric.get(row["metric"], []), dtype=float)
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    row["bootstrap_mean"] = float(np.mean(vals))
                    row["ci_low"] = float(np.quantile(vals, 0.025))
                    row["ci_median"] = float(np.quantile(vals, 0.5))
                    row["ci_high"] = float(np.quantile(vals, 0.975))
                    row["p_gt_zero"] = float((vals > 0).mean())
                    row["p_lt_zero"] = float((vals < 0).mean())

    return pd.DataFrame(draw_rows), pd.DataFrame(summary_rows)


def seed_level_summary(by_split: pd.DataFrame, by_direction: pd.DataFrame, validation_sweep: pd.DataFrame) -> pd.DataFrame:
    split_rows = by_split.copy()
    split_rows["summary_type"] = "split"
    if "net_direction" not in split_rows.columns:
        split_rows["net_direction"] = "all"

    direction_rows = by_direction.copy()
    direction_rows["summary_type"] = "direction"

    keep = [
        "policy_seed",
        "summary_type",
        "split",
        "net_direction",
        "n",
        "n_patients",
        "mean_dice",
        "mean_locf_dice",
        "mean_gap_vs_locf",
        "median_gap_vs_locf",
        "win_rate_vs_locf",
        "gate_open_rate",
        "edit_on_growth_rate",
        "edit_on_shrinkage_rate",
        "added_growth_mean",
        "growth_precision_mean",
        "growth_recall_mean",
    ]
    combined = pd.concat([split_rows, direction_rows], ignore_index=True, sort=False)
    keep_l = [c for c in keep if c in combined.columns]
    combined = combined[keep_l]

    selected_rows = []
    for seed_label, part in validation_sweep.groupby("policy_seed", observed=True):
        row = part.sort_values("mean_dice", ascending=False).iloc[0]
        selected_rows.append(
            {
                "policy_seed": seed_label,
                "selected_gate_threshold": row.get("gate_threshold", np.nan),
                "selected_budget_scale": row.get("budget_scale", np.nan),
                "selected_validation_mean_dice": row.get("mean_dice", np.nan),
                "selected_validation_gap_vs_locf": row.get("mean_gap_vs_locf", np.nan),
            }
        )
    selected = pd.DataFrame(selected_rows)
    return combined.merge(selected, on="policy_seed", how="left")


def cross_seed_summary(seed_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["summary_type", "split", "net_direction"]
    for key, part in seed_summary.groupby(group_cols, observed=True, dropna=False):
        key_dict = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        vals = part["mean_gap_vs_locf"].to_numpy(dtype=float)
        rows.append(
            {
                **key_dict,
                "n_seeds": int(part["policy_seed"].nunique()),
                "gap_mean_across_seeds": float(np.nanmean(vals)),
                "gap_min_across_seeds": float(np.nanmin(vals)),
                "gap_max_across_seeds": float(np.nanmax(vals)),
                "all_seed_gaps_positive": bool(np.all(vals > 0)),
                "all_seed_gaps_nonnegative": bool(np.all(vals >= 0)),
                "any_seed_gap_negative": bool(np.any(vals < 0)),
                "seed_gaps": "; ".join(f"{r.policy_seed}:{r.mean_gap_vs_locf:.6f}" for r in part.itertuples()),
            }
        )
    return pd.DataFrame(rows)


def claim_status(seed_summary: pd.DataFrame, bootstrap_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    selected = seed_summary[["policy_seed", "selected_gate_threshold", "selected_budget_scale"]].drop_duplicates()
    same_gate_scale = selected[["selected_gate_threshold", "selected_budget_scale"]].drop_duplicates().shape[0] == 1
    rows.append(
        {
            "claim": "Validation selects the same conservative operating point across seeds",
            "status": "supported" if same_gate_scale else "not_supported",
            "evidence": selected.to_dict(orient="records"),
            "interpretation": "Stable selection suggests the validation sweep is not completely arbitrary, but does not prove held-out improvement.",
        }
    )

    test_split = seed_summary[
        (seed_summary["summary_type"] == "split") & (seed_summary["split"].astype(str) == "test")
    ]
    test_vals = test_split["mean_gap_vs_locf"].to_numpy(dtype=float)
    if test_vals.size and np.all(test_vals > 0):
        status = "supported_weakly"
    elif test_vals.size and np.all(test_vals >= 0):
        status = "borderline"
    else:
        status = "not_supported"
    rows.append(
        {
            "claim": "Conservative learned-field correction robustly improves test Dice over LOCF",
            "status": status,
            "evidence": "; ".join(f"{r.policy_seed}: gap {r.mean_gap_vs_locf:.6f}" for r in test_split.itertuples()),
            "interpretation": "This should not be a central method claim unless patient-bootstrap and additional seeds/splits turn positive.",
        }
    )

    growth_test = seed_summary[
        (seed_summary["summary_type"] == "direction")
        & (seed_summary["split"].astype(str) == "test")
        & (seed_summary["net_direction"].astype(str) == "net_growth")
    ]
    growth_vals = growth_test["mean_gap_vs_locf"].to_numpy(dtype=float)
    rows.append(
        {
            "claim": "Conservative edits help net-growth cases more than shrinkage cases",
            "status": "supported_weakly" if growth_vals.size and np.all(growth_vals > 0) else "mixed",
            "evidence": "; ".join(f"{r.policy_seed}: net-growth gap {r.mean_gap_vs_locf:.6f}" for r in growth_test.itertuples()),
            "interpretation": "This is currently the most coherent method-adjacent signal, but it remains subgroup/small-sample evidence.",
        }
    )

    shrink_test = seed_summary[
        (seed_summary["summary_type"] == "direction")
        & (seed_summary["split"].astype(str) == "test")
        & (seed_summary["net_direction"].astype(str) == "net_shrinkage")
    ]
    shrink_vals = shrink_test["mean_gap_vs_locf"].to_numpy(dtype=float)
    rows.append(
        {
            "claim": "The conservative policy is safe on shrinkage cases",
            "status": "not_supported" if shrink_vals.size and np.any(shrink_vals < 0) else "borderline",
            "evidence": "; ".join(f"{r.policy_seed}: shrinkage gap {r.mean_gap_vs_locf:.6f}" for r in shrink_test.itertuples()),
            "interpretation": "Shrinkage leakage remains a bottleneck; growth-only correction needs a stronger no-edit rule or separate shrinkage/loss analysis.",
        }
    )

    boot_gap = bootstrap_summary[
        (bootstrap_summary.get("summary_type", "") == "split")
        & (bootstrap_summary.get("split", "").astype(str) == "test")
        & (bootstrap_summary.get("metric", "") == "mean_gap_vs_locf")
    ]
    if not boot_gap.empty:
        rows.append(
            {
                "claim": "Patient-level uncertainty supports positive test improvement",
                "status": "supported" if (boot_gap["ci_low"].to_numpy(dtype=float) > 0).all() else "not_supported_or_unresolved",
                "evidence": "; ".join(
                    f"{r.policy_seed}: CI [{r.ci_low:.6f}, {r.ci_high:.6f}], p>0={r.p_gt_zero:.3f}"
                    for r in boot_gap.itertuples()
                ),
                "interpretation": "Patient-level uncertainty is the right bar here because the test split has very few patients.",
            }
        )
    return pd.DataFrame(rows)


def write_report(
    path: Path,
    seed_summary: pd.DataFrame,
    cross_seed: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
    claims: pd.DataFrame,
) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Conservative Policy Seed Robustness Audit\n\n")
        f.write(
            "This report aggregates conservative learned-growth-field policy runs across model seeds. "
            "The goal is not to tune another policy, but to decide whether the current prototype is a credible method result or a diagnostic finding.\n\n"
        )
        f.write("## Claim Status\n\n")
        f.write(claims.to_markdown(index=False))
        f.write("\n\n## Seed-Level Summary\n\n")
        f.write(seed_summary.to_markdown(index=False))
        f.write("\n\n## Cross-Seed Summary\n\n")
        f.write(cross_seed.to_markdown(index=False))
        if not bootstrap_summary.empty:
            f.write("\n\n## Patient Bootstrap Summary\n\n")
            f.write(bootstrap_summary.to_markdown(index=False))
        f.write(
            "\n\nRecommended reading: if validation selection is stable but held-out test gains are not, "
            "the result should be used as bottleneck evidence rather than as a central forecasting improvement claim.\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate conservative growth-policy runs and patient-bootstrap selected-policy gaps.")
    parser.add_argument("--policy_dirs", type=str, required=True, help="Comma-separated paths or label=path entries.")
    parser.add_argument("--n_bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dirs = parse_policy_dirs(args.policy_dirs)

    pieces = {label: read_policy_dir(label, path) for label, path in dirs.items()}
    selected_samples = pd.concat([p["selected_samples"] for p in pieces.values()], ignore_index=True, sort=False)
    selected_by_split = pd.concat([p["selected_by_split"] for p in pieces.values()], ignore_index=True, sort=False)
    selected_by_direction = pd.concat([p["selected_by_direction"] for p in pieces.values()], ignore_index=True, sort=False)
    validation_sweep = pd.concat([p["validation_sweep"] for p in pieces.values()], ignore_index=True, sort=False)

    seed_summary = seed_level_summary(selected_by_split, selected_by_direction, validation_sweep)
    cross_seed = cross_seed_summary(seed_summary)

    split_draws, split_boot = bootstrap_group(
        selected_samples,
        ["policy_seed", "split"],
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    split_boot.insert(1, "summary_type", "split")
    if not split_draws.empty:
        split_draws.insert(1, "summary_type", "split")

    direction_draws, direction_boot = bootstrap_group(
        selected_samples,
        ["policy_seed", "split", "net_direction"],
        n_bootstrap=args.n_bootstrap,
        seed=args.seed + 1009,
    )
    direction_boot.insert(1, "summary_type", "direction")
    if not direction_draws.empty:
        direction_draws.insert(1, "summary_type", "direction")

    bootstrap_draws = pd.concat([split_draws, direction_draws], ignore_index=True, sort=False)
    bootstrap_summary = pd.concat([split_boot, direction_boot], ignore_index=True, sort=False)
    claims = claim_status(seed_summary, bootstrap_summary)

    selected_samples.to_csv(output_dir / "conservative_policy_seed_selected_samples.csv", index=False)
    seed_summary.to_csv(output_dir / "conservative_policy_seed_summary.csv", index=False)
    cross_seed.to_csv(output_dir / "conservative_policy_cross_seed_summary.csv", index=False)
    bootstrap_draws.to_csv(output_dir / "conservative_policy_patient_bootstrap_draws.csv", index=False)
    bootstrap_summary.to_csv(output_dir / "conservative_policy_patient_bootstrap_summary.csv", index=False)
    claims.to_csv(output_dir / "conservative_policy_claim_status.csv", index=False)
    write_report(output_dir / "conservative_policy_seed_robustness_report.md", seed_summary, cross_seed, bootstrap_summary, claims)

    summary = {
        "policy_dirs": {label: str(path) for label, path in dirs.items()},
        "n_policy_seeds": int(len(dirs)),
        "n_selected_rows": int(selected_samples.shape[0]),
        "n_bootstrap": int(args.n_bootstrap),
        "output_dir": str(output_dir),
        "outputs": {
            "selected_samples_csv": str(output_dir / "conservative_policy_seed_selected_samples.csv"),
            "seed_summary_csv": str(output_dir / "conservative_policy_seed_summary.csv"),
            "cross_seed_summary_csv": str(output_dir / "conservative_policy_cross_seed_summary.csv"),
            "patient_bootstrap_draws_csv": str(output_dir / "conservative_policy_patient_bootstrap_draws.csv"),
            "patient_bootstrap_summary_csv": str(output_dir / "conservative_policy_patient_bootstrap_summary.csv"),
            "claim_status_csv": str(output_dir / "conservative_policy_claim_status.csv"),
            "report_md": str(output_dir / "conservative_policy_seed_robustness_report.md"),
        },
    }
    with (output_dir / "conservative_policy_seed_robustness_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
