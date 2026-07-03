#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FORECAST_ORIGIN_FEATURES = [
    "input_volume_vox",
    "recent_relative_growth",
    "treated_at_input",
    "delta_days",
    "input_elongation_ratio",
    "input_compactness_proxy",
    "input_connected_component_count",
    "n_sessions",
    "followup_days",
    "mean_interval_days",
]


def prepare_task_frames(case_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}

    win_df = case_df.copy()
    win_df["target"] = (win_df["dice_gap"] > 0).astype(int)
    out["resunet_beats_locf"] = win_df

    tw_be = case_df[case_df["case_type"].isin(["target_wins", "both_easy"])].copy()
    tw_be["target"] = (tw_be["case_type"] == "target_wins").astype(int)
    out["target_wins_vs_both_easy"] = tw_be

    bh_be = case_df[case_df["case_type"].isin(["both_hard", "both_easy"])].copy()
    bh_be["target"] = (bh_be["case_type"] == "both_hard").astype(int)
    out["both_hard_vs_both_easy"] = bh_be

    return out


def _make_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )


def _evaluate_cv(X: pd.DataFrame, y: np.ndarray, seed: int, n_repeats: int) -> tuple[pd.DataFrame, np.ndarray]:
    class_counts = pd.Series(y).value_counts()
    if len(class_counts) < 2 or class_counts.min() < 2:
        raise ValueError("Need at least two classes with >=2 samples each.")

    n_splits = int(min(5, class_counts.min()))
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    metrics: list[dict] = []
    coefs: list[np.ndarray] = []

    for train_idx, test_idx in cv.split(X, y):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        pipe = _make_pipeline()
        pipe.fit(X_train, y_train)
        probs = pipe.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)
        metrics.append(
            {
                "accuracy": accuracy_score(y_test, preds),
                "balanced_accuracy": balanced_accuracy_score(y_test, preds),
                "roc_auc": roc_auc_score(y_test, probs),
            }
        )
        coefs.append(pipe.named_steps["clf"].coef_[0].copy())

    metrics_df = pd.DataFrame(metrics)
    return metrics_df, np.vstack(coefs)


def evaluate_subset(
    df: pd.DataFrame,
    features: list[str],
    seed: int,
    n_repeats: int,
) -> dict[str, pd.DataFrame | dict]:
    available = [f for f in features if f in df.columns]
    task = df[available + ["target"]].copy().dropna(subset=["target"])
    X = task[available].copy()
    y = task["target"].astype(int).to_numpy()

    metrics_df, coef_arr = _evaluate_cv(X, y, seed=seed, n_repeats=n_repeats)
    importance_df = pd.DataFrame(
        {
            "feature": available,
            "mean_coef": coef_arr.mean(axis=0),
            "std_coef": coef_arr.std(axis=0),
            "mean_abs_coef": np.abs(coef_arr).mean(axis=0),
        }
    ).sort_values("mean_abs_coef", ascending=False)

    baseline_auc = float(metrics_df["roc_auc"].mean())
    baseline_bacc = float(metrics_df["balanced_accuracy"].mean())

    ablations: list[dict] = []
    for drop_feature in available:
        keep = [f for f in available if f != drop_feature]
        X_drop = task[keep].copy()
        drop_metrics_df, _ = _evaluate_cv(X_drop, y, seed=seed, n_repeats=n_repeats)
        ablations.append(
            {
                "dropped_feature": drop_feature,
                "roc_auc_mean_without_feature": float(drop_metrics_df["roc_auc"].mean()),
                "roc_auc_drop": baseline_auc - float(drop_metrics_df["roc_auc"].mean()),
                "balanced_accuracy_mean_without_feature": float(drop_metrics_df["balanced_accuracy"].mean()),
                "balanced_accuracy_drop": baseline_bacc - float(drop_metrics_df["balanced_accuracy"].mean()),
            }
        )

    ablation_df = pd.DataFrame(ablations).sort_values("roc_auc_drop", ascending=False)
    summary = {
        "n_samples": int(len(task)),
        "n_positive": int(task["target"].sum()),
        "n_negative": int(len(task) - task["target"].sum()),
        "n_features": int(len(available)),
        "baseline_roc_auc_mean": baseline_auc,
        "baseline_balanced_accuracy_mean": baseline_bacc,
        "baseline_accuracy_mean": float(metrics_df["accuracy"].mean()),
    }
    return {
        "metrics": metrics_df,
        "importance": importance_df,
        "ablation": ablation_df,
        "summary": summary,
    }


def build_strata(task_df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    strata: list[tuple[str, pd.DataFrame]] = [("overall", task_df)]
    if "tier" in task_df.columns:
        for tier, sub in task_df.groupby("tier"):
            strata.append((f"tier_{tier}", sub.copy()))
    if "horizon" in task_df.columns:
        for horizon, sub in task_df.groupby("horizon"):
            strata.append((f"horizon_{horizon}", sub.copy()))
    return strata


def safe_slug(text: str) -> str:
    return str(text).replace(" ", "_").replace("/", "_")


def write_report(path: Path, report_rows: list[dict]) -> None:
    df = pd.DataFrame(report_rows)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Stratified Descriptor Signal Report\n\n")
        if df.empty:
            f.write("No strata were evaluable.\n")
        else:
            f.write(df.to_markdown(index=False))
            f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tier- and horizon-stratified descriptor-signal analysis with leave-one-feature-out ablation.")
    parser.add_argument("--case_type_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_repeats", type=int, default=10)
    parser.add_argument("--min_subset_size", type=int, default=12)
    args = parser.parse_args()

    case_df = pd.read_csv(args.case_type_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = prepare_task_frames(case_df)
    report_rows: list[dict] = []

    for task_name, task_df in tasks.items():
        task_dir = out_dir / task_name
        task_dir.mkdir(parents=True, exist_ok=True)

        for stratum_name, stratum_df in build_strata(task_df):
            class_counts = stratum_df["target"].value_counts()
            if len(stratum_df) < args.min_subset_size or len(class_counts) < 2 or class_counts.min() < 2:
                report_rows.append(
                    {
                        "task": task_name,
                        "stratum": stratum_name,
                        "status": "skipped",
                        "n_samples": int(len(stratum_df)),
                        "min_class_count": int(class_counts.min()) if len(class_counts) else 0,
                    }
                )
                continue

            try:
                result = evaluate_subset(
                    stratum_df,
                    features=FORECAST_ORIGIN_FEATURES,
                    seed=args.seed,
                    n_repeats=args.n_repeats,
                )
            except Exception as e:
                report_rows.append(
                    {
                        "task": task_name,
                        "stratum": stratum_name,
                        "status": f"failed: {e}",
                        "n_samples": int(len(stratum_df)),
                        "min_class_count": int(class_counts.min()) if len(class_counts) else 0,
                    }
                )
                continue

            subdir = task_dir / safe_slug(stratum_name)
            subdir.mkdir(parents=True, exist_ok=True)
            result["metrics"].to_csv(subdir / "metrics_per_fold.csv", index=False)
            result["importance"].to_csv(subdir / "logistic_importance.csv", index=False)
            result["ablation"].to_csv(subdir / "ablation_importance.csv", index=False)
            with (subdir / "summary.json").open("w", encoding="utf-8") as f:
                json.dump(result["summary"], f, indent=2)

            report_rows.append(
                {
                    "task": task_name,
                    "stratum": stratum_name,
                    "status": "ok",
                    **result["summary"],
                    "top_feature": str(result["importance"].iloc[0]["feature"]),
                    "top_ablation_feature": str(result["ablation"].iloc[0]["dropped_feature"]),
                    "top_ablation_roc_auc_drop": float(result["ablation"].iloc[0]["roc_auc_drop"]),
                }
            )

    write_report(out_dir / "stratified_descriptor_report.md", report_rows)
    pd.DataFrame(report_rows).to_csv(out_dir / "stratified_descriptor_summary.csv", index=False)
    manifest = {
        "case_type_csv": str(Path(args.case_type_csv).resolve()),
        "features": FORECAST_ORIGIN_FEATURES,
        "tasks": list(tasks.keys()),
        "n_repeats": args.n_repeats,
        "min_subset_size": args.min_subset_size,
        "output_dir": str(out_dir.resolve()),
    }
    with (out_dir / "stratified_descriptor_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
