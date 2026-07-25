#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_forecast_origin_predictability import (  # noqa: E402
    TARGETS,
    add_origin_features,
    add_transition_targets,
    available_features,
    confusion_metrics,
    merge_manifest_features,
    normalize_transition_samples,
    parse_csv,
    resolve_samples_path,
)
from scripts.run_forecast_origin_feature_ablation import FEATURE_SETS  # noqa: E402


METRICS = [
    "accuracy",
    "balanced_accuracy",
    "roc_auc",
    "average_precision",
    "precision",
    "recall",
    "false_positive_rate",
    "false_negative_rate",
    "predicted_positive_rate",
]


def finite_quantile(values: pd.Series, q: float) -> float:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy()
    if len(arr) == 0:
        return float("nan")
    return float(np.quantile(arr, q))


def make_patient_splits(
    patients: List[str],
    rng: np.random.Generator,
    train_fraction: float,
    val_fraction: float,
) -> Dict[str, List[str]]:
    shuffled = np.array(patients, dtype=object)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = max(1, int(round(train_fraction * n)))
    n_val = max(1, int(round(val_fraction * n)))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    train = shuffled[:n_train].tolist()
    val = shuffled[n_train : n_train + n_val].tolist()
    test = shuffled[n_train + n_val :].tolist()
    if not test:
        test = [val.pop()]
    return {"train": train, "val": val, "test": test}


def assign_split(data: pd.DataFrame, split_map: Dict[str, List[str]]) -> pd.DataFrame:
    patient_to_split = {}
    for split, patients in split_map.items():
        for patient_id in patients:
            patient_to_split[str(patient_id)] = split
    out = data.copy()
    out["split"] = out["patient_id"].astype(str).map(patient_to_split)
    out = out[out["split"].notna()].copy()
    return out


def patient_counts(split_map: Dict[str, List[str]]) -> Dict[str, int]:
    return {f"n_{split}_patients": len(patients) for split, patients in split_map.items()}


def fit_logistic_scores(
    train: pd.DataFrame,
    eval_part: pd.DataFrame,
    label_col: str,
    features: List[str],
    seed: int,
) -> np.ndarray:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pre = ColumnTransformer(
        [("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), features)],
        remainder="drop",
    )
    model = Pipeline(
        [
            ("pre", pre),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)),
        ]
    )
    model.fit(train[features], train[label_col].astype(int))
    return model.predict_proba(eval_part[features])[:, 1]


def evaluate_repeat(
    base_data: pd.DataFrame,
    split_map: Dict[str, List[str]],
    repeat_idx: int,
    repeat_seed: int,
    selected_feature_sets: Dict[str, List[str]],
    targets: List[str],
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    split_data = assign_split(base_data, split_map)
    split_data, thresholds = add_transition_targets(
        split_data,
        train_split="train",
        growth_loss_threshold=args.growth_loss_threshold,
        distant_growth_threshold=args.distant_growth_threshold,
        high_burden_quantile=args.high_burden_quantile,
        high_change_rate_quantile=args.high_change_rate_quantile,
        locf_breakdown_threshold=args.locf_breakdown_threshold,
    )
    rows: List[Dict[str, object]] = []
    prevalence_rows: List[Dict[str, object]] = []
    counts = patient_counts(split_map)

    for target in targets:
        label_col = f"label_{target}"
        if label_col not in split_data.columns:
            continue
        for split, part in split_data.groupby("split", observed=True):
            y = part[label_col].dropna().astype(int)
            if y.empty:
                continue
            prevalence_rows.append(
                {
                    "repeat_idx": repeat_idx,
                    "repeat_seed": repeat_seed,
                    "target": target,
                    "split": split,
                    "n_samples": int(len(y)),
                    "n_patients": int(part.loc[y.index, "patient_id"].nunique()),
                    "positive_rate": float(y.mean()),
                    "n_positive": int(y.sum()),
                    **counts,
                    **thresholds,
                }
            )

        train = split_data[(split_data["split"] == "train") & split_data[label_col].notna()].copy()
        if train.empty or train[label_col].astype(int).nunique() < 2:
            for feature_set in selected_feature_sets:
                for split in ["val", "test"]:
                    rows.append(
                        {
                            "repeat_idx": repeat_idx,
                            "repeat_seed": repeat_seed,
                            "target": target,
                            "feature_set": feature_set,
                            "split": split,
                            "status": "skipped_train_one_class",
                            **counts,
                            **thresholds,
                        }
                    )
            continue

        for feature_set, requested_features in selected_feature_sets.items():
            try:
                features = available_features(split_data, requested_features)
            except ValueError as exc:
                for split in ["val", "test"]:
                    rows.append(
                        {
                            "repeat_idx": repeat_idx,
                            "repeat_seed": repeat_seed,
                            "target": target,
                            "feature_set": feature_set,
                            "split": split,
                            "status": f"skipped_features: {exc}",
                            **counts,
                            **thresholds,
                        }
                    )
                continue

            for split in ["val", "test"]:
                eval_part = split_data[(split_data["split"] == split) & split_data[label_col].notna()].copy()
                if eval_part.empty:
                    continue
                y = eval_part[label_col].astype(int).to_numpy()
                row: Dict[str, object] = {
                    "repeat_idx": repeat_idx,
                    "repeat_seed": repeat_seed,
                    "target": target,
                    "feature_set": feature_set,
                    "split": split,
                    "n_train_samples": int(len(train)),
                    "n_eval_samples": int(len(eval_part)),
                    "n_train_patients": int(train["patient_id"].nunique()),
                    "n_eval_patients": int(eval_part["patient_id"].nunique()),
                    "train_positive_rate": float(train[label_col].astype(int).mean()),
                    "eval_positive_rate": float(np.mean(y)),
                    "eval_n_positive": int(y.sum()),
                    "status": "ok",
                    **counts,
                    **thresholds,
                }
                if len(np.unique(y)) < 2:
                    row["status"] = "eval_one_class"
                    for metric in METRICS:
                        row[metric] = float("nan")
                    rows.append(row)
                    continue
                try:
                    scores = fit_logistic_scores(train, eval_part, label_col, features, seed=repeat_seed)
                    row.update(confusion_metrics(y, scores, threshold=0.5))
                except Exception as exc:  # noqa: BLE001 - audit should keep failed repeats visible.
                    row["status"] = f"fit_failed: {type(exc).__name__}: {exc}"
                    for metric in METRICS:
                        row[metric] = float("nan")
                rows.append(row)
    return rows, prevalence_rows


def summarize_runs(runs: pd.DataFrame) -> pd.DataFrame:
    if runs.empty:
        return pd.DataFrame()
    rows = []
    group_cols = ["target", "feature_set", "split"]
    for key, part in runs.groupby(group_cols, observed=True):
        row = dict(zip(group_cols, key))
        row["n_repeats"] = int(part["repeat_idx"].nunique())
        row["n_ok_repeats"] = int((part["status"] == "ok").sum())
        row["ok_fraction"] = float((part["status"] == "ok").mean())
        row["mean_eval_positive_rate"] = float(pd.to_numeric(part["eval_positive_rate"], errors="coerce").mean())
        row["median_eval_n_positive"] = float(pd.to_numeric(part["eval_n_positive"], errors="coerce").median())
        for metric in METRICS:
            vals = pd.to_numeric(part[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(vals.mean()) if len(vals) else float("nan")
            row[f"{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else float("nan")
            row[f"{metric}_q10"] = finite_quantile(vals, 0.10)
            row[f"{metric}_q25"] = finite_quantile(vals, 0.25)
            row[f"{metric}_median"] = finite_quantile(vals, 0.50)
            row[f"{metric}_q75"] = finite_quantile(vals, 0.75)
            row[f"{metric}_q90"] = finite_quantile(vals, 0.90)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["target", "split", "feature_set"]).reset_index(drop=True)


def summarize_prevalence(prevalence: pd.DataFrame) -> pd.DataFrame:
    if prevalence.empty:
        return pd.DataFrame()
    rows = []
    for (target, split), part in prevalence.groupby(["target", "split"], observed=True):
        vals = pd.to_numeric(part["positive_rate"], errors="coerce").dropna()
        pos = pd.to_numeric(part["n_positive"], errors="coerce").dropna()
        rows.append(
            {
                "target": target,
                "split": split,
                "n_repeats": int(part["repeat_idx"].nunique()),
                "positive_rate_mean": float(vals.mean()) if len(vals) else float("nan"),
                "positive_rate_q10": finite_quantile(vals, 0.10),
                "positive_rate_q50": finite_quantile(vals, 0.50),
                "positive_rate_q90": finite_quantile(vals, 0.90),
                "n_positive_median": float(pos.median()) if len(pos) else float("nan"),
                "n_positive_min": float(pos.min()) if len(pos) else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values(["target", "split"]).reset_index(drop=True)


def classify_claims(summary: pd.DataFrame, reference_feature_sets: List[str]) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    test = summary[(summary["split"] == "test") & summary["feature_set"].isin(reference_feature_sets)].copy()
    rows = []
    for target, part in test.groupby("target", observed=True):
        best = part.sort_values(["roc_auc_mean", "balanced_accuracy_mean"], ascending=False).iloc[0]
        mean_auc = float(best.get("roc_auc_mean", np.nan))
        q25_auc = float(best.get("roc_auc_q25", np.nan))
        mean_bacc = float(best.get("balanced_accuracy_mean", np.nan))
        q25_bacc = float(best.get("balanced_accuracy_q25", np.nan))
        ok_fraction = float(best.get("ok_fraction", np.nan))
        median_pos = float(best.get("median_eval_n_positive", np.nan))
        fnr_mean = float(best.get("false_negative_rate_mean", np.nan))

        if ok_fraction < 0.8 or median_pos < 4:
            status = "fragile_sparse"
        elif mean_auc >= 0.70 and q25_auc >= 0.62 and mean_bacc >= 0.62:
            status = "stable_moderate"
        elif mean_auc >= 0.65 and q25_auc >= 0.55 and mean_bacc >= 0.58:
            status = "promising_split_sensitive"
        elif mean_auc < 0.60 or q25_auc < 0.50:
            status = "weak_or_unstable"
        else:
            status = "exploratory"

        rows.append(
            {
                "target": target,
                "best_reference_feature_set": best["feature_set"],
                "status": status,
                "test_roc_auc_mean": mean_auc,
                "test_roc_auc_q25": q25_auc,
                "test_balanced_accuracy_mean": mean_bacc,
                "test_balanced_accuracy_q25": q25_bacc,
                "test_false_negative_rate_mean": fnr_mean,
                "ok_fraction": ok_fraction,
                "median_test_positives": median_pos,
            }
        )
    return pd.DataFrame(rows).sort_values("target").reset_index(drop=True)


def write_report(
    path: Path,
    args: argparse.Namespace,
    selected_feature_sets: Dict[str, List[str]],
    summary: pd.DataFrame,
    prevalence_summary: pd.DataFrame,
    claim_status: pd.DataFrame,
) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Forecast-Origin Patient-Split Stability Audit\n\n")
        f.write(
            "This audit repeats patient-level train/validation/test splits and retrains a simple logistic "
            "forecast-origin classifier. It tests whether transition-state signals survive patient allocation, "
            "rather than relying on one fixed split or one leave-one-patient-out summary.\n\n"
        )
        f.write("## Configuration\n\n")
        f.write(f"- Repeats: `{args.n_repeats}`\n")
        f.write(f"- Train fraction: `{args.train_fraction}`\n")
        f.write(f"- Validation fraction: `{args.val_fraction}`\n")
        f.write(f"- Seed: `{args.seed}`\n")
        f.write(f"- Targets: `{args.targets}`\n")
        f.write("\n## Feature Sets\n\n")
        for name, features in selected_feature_sets.items():
            f.write(f"- `{name}`: " + ", ".join(f"`{x}`" for x in features) + "\n")
        f.write("\n## Claim Status\n\n")
        f.write(claim_status.to_markdown(index=False) if not claim_status.empty else "No claim-status rows.")
        f.write("\n\n## Stability Summary\n\n")
        focus_cols = [
            "target",
            "feature_set",
            "split",
            "n_ok_repeats",
            "ok_fraction",
            "mean_eval_positive_rate",
            "median_eval_n_positive",
            "roc_auc_mean",
            "roc_auc_q25",
            "roc_auc_q75",
            "balanced_accuracy_mean",
            "balanced_accuracy_q25",
            "recall_mean",
            "false_negative_rate_mean",
        ]
        cols = [c for c in focus_cols if c in summary.columns]
        f.write(summary[cols].to_markdown(index=False) if not summary.empty else "No summary rows.")
        f.write("\n\n## Prevalence Stability\n\n")
        f.write(prevalence_summary.to_markdown(index=False) if not prevalence_summary.empty else "No prevalence rows.")
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Repeat patient-level splits for forecast-origin transition-state predictability.")
    parser.add_argument("--samples_csv", type=str, default=None)
    parser.add_argument("--taxonomy_dir", type=str, default=None)
    parser.add_argument("--manifest_csv", type=str, default=None)
    parser.add_argument("--feature_sets", type=str, default="full_origin,no_interval,history_only,time_only,treatment_only")
    parser.add_argument("--targets", type=str, default="mixed_growth_loss,distant_growth_present,high_transition_burden,locf_breakdown,high_change_rate")
    parser.add_argument("--n_repeats", type=int, default=100)
    parser.add_argument("--train_fraction", type=float, default=0.60)
    parser.add_argument("--val_fraction", type=float, default=0.20)
    parser.add_argument("--growth_loss_threshold", type=float, default=0.2)
    parser.add_argument("--distant_growth_threshold", type=float, default=0.2)
    parser.add_argument("--high_burden_quantile", type=float, default=0.75)
    parser.add_argument("--high_change_rate_quantile", type=float, default=0.75)
    parser.add_argument("--locf_breakdown_threshold", type=float, default=0.5)
    parser.add_argument("--reference_feature_sets", type=str, default="full_origin,no_interval,history_only")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    if args.train_fraction <= 0 or args.val_fraction <= 0 or args.train_fraction + args.val_fraction >= 1:
        raise ValueError("--train_fraction and --val_fraction must be positive and sum to less than 1.")
    selected_names = parse_csv(args.feature_sets)
    unknown = [name for name in selected_names if name not in FEATURE_SETS]
    if unknown:
        raise ValueError(f"Unknown feature set(s): {unknown}. Available: {sorted(FEATURE_SETS)}")
    selected_feature_sets = {name: FEATURE_SETS[name] for name in selected_names}
    targets = [target for target in parse_csv(args.targets) if target in TARGETS]
    if not targets:
        raise ValueError("No valid targets requested.")

    samples_path = resolve_samples_path(args)
    base_data = normalize_transition_samples(pd.read_csv(samples_path))
    base_data = merge_manifest_features(base_data, args.manifest_csv)
    base_data = add_origin_features(base_data)
    base_data = base_data[base_data["patient_id"].notna()].copy()
    patients = sorted(base_data["patient_id"].astype(str).unique())
    if len(patients) < 5:
        raise ValueError(f"Need at least 5 patients for repeated patient splits; found {len(patients)}.")

    rng = np.random.default_rng(args.seed)
    run_rows: List[Dict[str, object]] = []
    prevalence_rows: List[Dict[str, object]] = []
    for repeat_idx in range(args.n_repeats):
        repeat_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        repeat_rng = np.random.default_rng(repeat_seed)
        split_map = make_patient_splits(
            patients,
            repeat_rng,
            train_fraction=args.train_fraction,
            val_fraction=args.val_fraction,
        )
        rows, prev_rows = evaluate_repeat(
            base_data=base_data,
            split_map=split_map,
            repeat_idx=repeat_idx,
            repeat_seed=repeat_seed,
            selected_feature_sets=selected_feature_sets,
            targets=targets,
            args=args,
        )
        run_rows.extend(rows)
        prevalence_rows.extend(prev_rows)

    runs = pd.DataFrame(run_rows)
    prevalence = pd.DataFrame(prevalence_rows)
    summary = summarize_runs(runs)
    prevalence_summary = summarize_prevalence(prevalence)
    claim_status = classify_claims(summary, parse_csv(args.reference_feature_sets))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(output_dir / "forecast_origin_patient_split_stability_runs.csv", index=False)
    prevalence.to_csv(output_dir / "forecast_origin_patient_split_stability_prevalence_runs.csv", index=False)
    summary.to_csv(output_dir / "forecast_origin_patient_split_stability_summary.csv", index=False)
    prevalence_summary.to_csv(output_dir / "forecast_origin_patient_split_stability_prevalence_summary.csv", index=False)
    claim_status.to_csv(output_dir / "forecast_origin_patient_split_stability_claim_status.csv", index=False)
    write_report(
        output_dir / "forecast_origin_patient_split_stability_report.md",
        args=args,
        selected_feature_sets=selected_feature_sets,
        summary=summary,
        prevalence_summary=prevalence_summary,
        claim_status=claim_status,
    )
    with (output_dir / "forecast_origin_patient_split_stability_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "samples_path": str(samples_path),
                "manifest_csv": args.manifest_csv,
                "feature_sets": selected_feature_sets,
                "targets": targets,
                "n_repeats": args.n_repeats,
                "train_fraction": args.train_fraction,
                "val_fraction": args.val_fraction,
                "seed": args.seed,
                "n_patients": len(patients),
                "n_rows": int(len(base_data)),
            },
            f,
            indent=2,
        )
    print(
        json.dumps(
            {
                "n_rows": int(len(base_data)),
                "n_patients": len(patients),
                "n_repeats": int(args.n_repeats),
                "output_dir": str(output_dir),
                "outputs": {
                    "runs_csv": str(output_dir / "forecast_origin_patient_split_stability_runs.csv"),
                    "summary_csv": str(output_dir / "forecast_origin_patient_split_stability_summary.csv"),
                    "claim_status_csv": str(output_dir / "forecast_origin_patient_split_stability_claim_status.csv"),
                    "report_md": str(output_dir / "forecast_origin_patient_split_stability_report.md"),
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
