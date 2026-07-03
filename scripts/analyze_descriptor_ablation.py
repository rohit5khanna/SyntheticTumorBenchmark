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
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from analyze_anchor_separation import RAW_NUMERIC_FEATURES


CONFIGS = {
    "raw_plus_tier_horizon": {"numeric": RAW_NUMERIC_FEATURES, "categorical": ["tier", "horizon"]},
    "raw_plus_horizon": {"numeric": RAW_NUMERIC_FEATURES, "categorical": ["horizon"]},
    "raw_plus_tier": {"numeric": RAW_NUMERIC_FEATURES, "categorical": ["tier"]},
    "raw_only": {"numeric": RAW_NUMERIC_FEATURES, "categorical": []},
}


def assign_profile_group(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["profile_group"] = "other"
    out.loc[(out["case_type"] == "both_easy") & (out["soft_regime_label"] == "core_aligned"), "profile_group"] = "both_easy_core"
    out.loc[(out["case_type"] == "target_wins") & (out["soft_regime_label"] == "core_aligned"), "profile_group"] = "target_wins_core"
    out.loc[out["soft_regime_label"] == "cross_regime_pull", "profile_group"] = "cross_regime_pull"
    out.loc[out["soft_regime_label"] == "transition", "profile_group"] = "transition"
    return out


def prepare_task_df(df: pd.DataFrame, positive_group: str) -> pd.DataFrame:
    keep = ["both_easy_core", "target_wins_core", positive_group]
    out = df[df["profile_group"].isin(keep)].copy()
    out["target"] = (out["profile_group"] == positive_group).astype(int)
    return out


def build_matrix(df: pd.DataFrame, numeric_features: list[str], categorical_features: list[str]) -> tuple[pd.DataFrame, list[str]]:
    keep_numeric = [c for c in numeric_features if c in df.columns]
    work = df[keep_numeric + [c for c in categorical_features if c in df.columns]].copy()
    if categorical_features:
        work = pd.get_dummies(
            work,
            columns=[c for c in categorical_features if c in work.columns],
            drop_first=False,
            dtype=float,
        )
    return work, work.columns.tolist()


def build_splitter(df: pd.DataFrame, y: np.ndarray, group_col: str | None, seed: int):
    class_counts = pd.Series(y).value_counts()
    if len(class_counts) < 2 or int(class_counts.min()) < 2:
        raise ValueError("Need both classes with at least 2 samples.")

    if group_col and group_col in df.columns:
        group_counts = (
            df[[group_col, "target"]]
            .drop_duplicates()
            .groupby("target")[group_col]
            .nunique()
        )
        n_splits = int(min(5, group_counts.min()))
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        groups = df[group_col].astype(str).to_numpy()
        return splitter, groups

    n_splits = int(min(5, class_counts.min()))
    splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=20, random_state=seed)
    return splitter, None


def fit_task(
    df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    seed: int,
    group_col: str | None,
    permutation_repeats: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X, feature_names = build_matrix(df, numeric_features, categorical_features)
    y = df["target"].astype(int).to_numpy()
    splitter, groups = build_splitter(df, y, group_col, seed)

    metrics = []
    coefs = []
    perm_rows = []

    split_iter = splitter.split(X, y, groups=groups) if groups is not None else splitter.split(X, y)
    for fold_idx, (train_idx, test_idx) in enumerate(split_iter):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        )
        pipe.fit(X_train, y_train)
        prob = pipe.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)
        metrics.append(
            {
                "fold": fold_idx,
                "accuracy": accuracy_score(y_test, pred),
                "balanced_accuracy": balanced_accuracy_score(y_test, pred),
                "roc_auc": roc_auc_score(y_test, prob),
            }
        )
        coefs.append(pipe.named_steps["clf"].coef_[0].copy())

        rng = np.random.default_rng(seed + fold_idx)
        for rep in range(permutation_repeats):
            perm_y_train = rng.permutation(y_train)
            if len(np.unique(perm_y_train)) < 2:
                continue
            perm_pipe = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
                ]
            )
            perm_pipe.fit(X_train, perm_y_train)
            perm_prob = perm_pipe.predict_proba(X_test)[:, 1]
            perm_rows.append(
                {
                    "fold": fold_idx,
                    "repeat": rep,
                    "roc_auc": roc_auc_score(y_test, perm_prob),
                }
            )

    metrics_df = pd.DataFrame(metrics)
    coef_arr = np.vstack(coefs)
    coef_df = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_coef": coef_arr.mean(axis=0),
            "std_coef": coef_arr.std(axis=0),
            "mean_abs_coef": np.abs(coef_arr).mean(axis=0),
        }
    ).sort_values("mean_abs_coef", ascending=False)

    perm_df = pd.DataFrame(perm_rows)
    return metrics_df, coef_df, perm_df


def summarize_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    return metrics_df[["accuracy", "balanced_accuracy", "roc_auc"]].agg(["mean", "std"]).T.reset_index().rename(columns={"index": "metric"})


def summarize_permutation(metrics_df: pd.DataFrame, perm_df: pd.DataFrame) -> pd.DataFrame:
    observed = float(metrics_df["roc_auc"].mean())
    if perm_df.empty:
        return pd.DataFrame(
            [
                {
                    "observed_mean_roc_auc": observed,
                    "perm_mean_roc_auc": np.nan,
                    "perm_std_roc_auc": np.nan,
                    "empirical_pvalue": np.nan,
                    "n_perm_rows": 0,
                }
            ]
        )

    perm_mean = float(perm_df["roc_auc"].mean())
    perm_std = float(perm_df["roc_auc"].std(ddof=0))
    pval = float((np.sum(perm_df["roc_auc"].to_numpy() >= observed) + 1) / (len(perm_df) + 1))
    return pd.DataFrame(
        [
            {
                "observed_mean_roc_auc": observed,
                "perm_mean_roc_auc": perm_mean,
                "perm_std_roc_auc": perm_std,
                "empirical_pvalue": pval,
                "n_perm_rows": int(len(perm_df)),
            }
        ]
    )


def write_report(path: Path, payload: dict[str, dict[str, pd.DataFrame]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Descriptor Ablation Report\n\n")
        f.write("This report tests whether descriptor signal persists when tier and horizon scaffolding are removed.\n\n")
        for task_name, task_payload in payload.items():
            f.write(f"## {task_name}\n\n")
            for config_name, result in task_payload.items():
                f.write(f"### {config_name}\n\n")
                f.write("#### Metrics\n\n")
                f.write(result["metric_summary"].to_markdown(index=False))
                f.write("\n\n#### Permutation Check\n\n")
                f.write(result["perm_summary"].to_markdown(index=False))
                f.write("\n\n#### Leading Coefficients\n\n")
                f.write(result["coef_summary"].head(10).to_markdown(index=False))
                f.write("\n\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run descriptor ablations and permutation checks for soft-regime prediction tasks.")
    parser.add_argument("--soft_profile_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group_col", type=str, default=None)
    parser.add_argument("--permutation_repeats", type=int, default=100)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.soft_profile_csv)
    if "profile_group" not in df.columns:
        df = assign_profile_group(df)

    task_builders = {
        "cross_regime_pull_vs_anchor_cores": prepare_task_df(df, "cross_regime_pull"),
        "transition_vs_anchor_cores": prepare_task_df(df, "transition"),
    }

    payload: dict[str, dict[str, pd.DataFrame]] = {}
    summary_rows = []
    for task_name, task_df in task_builders.items():
        payload[task_name] = {}
        task_dir = out_dir / task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        for config_name, config in CONFIGS.items():
            config_dir = task_dir / config_name
            config_dir.mkdir(parents=True, exist_ok=True)
            metrics_df, coef_df, perm_df = fit_task(
                task_df,
                numeric_features=config["numeric"],
                categorical_features=config["categorical"],
                seed=args.seed,
                group_col=args.group_col,
                permutation_repeats=args.permutation_repeats,
            )
            metric_summary = summarize_metrics(metrics_df)
            perm_summary = summarize_permutation(metrics_df, perm_df)

            metrics_df.to_csv(config_dir / "fold_metrics.csv", index=False)
            metric_summary.to_csv(config_dir / "metric_summary.csv", index=False)
            coef_df.to_csv(config_dir / "coefficient_summary.csv", index=False)
            perm_df.to_csv(config_dir / "permutation_metrics.csv", index=False)
            perm_summary.to_csv(config_dir / "permutation_summary.csv", index=False)

            payload[task_name][config_name] = {
                "metric_summary": metric_summary,
                "perm_summary": perm_summary,
                "coef_summary": coef_df,
            }
            summary_rows.append(
                {
                    "task": task_name,
                    "config": config_name,
                    "group_col": args.group_col,
                    "mean_accuracy": float(metric_summary.loc[metric_summary["metric"] == "accuracy", "mean"].iloc[0]),
                    "mean_balanced_accuracy": float(metric_summary.loc[metric_summary["metric"] == "balanced_accuracy", "mean"].iloc[0]),
                    "mean_roc_auc": float(metric_summary.loc[metric_summary["metric"] == "roc_auc", "mean"].iloc[0]),
                    "perm_mean_roc_auc": float(perm_summary["perm_mean_roc_auc"].iloc[0]) if pd.notna(perm_summary["perm_mean_roc_auc"].iloc[0]) else np.nan,
                    "empirical_pvalue": float(perm_summary["empirical_pvalue"].iloc[0]) if pd.notna(perm_summary["empirical_pvalue"].iloc[0]) else np.nan,
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "descriptor_ablation_summary.csv", index=False)
    write_report(out_dir / "descriptor_ablation_report.md", payload)

    manifest = {
        "soft_profile_csv": str(Path(args.soft_profile_csv).resolve()),
        "configs": CONFIGS,
        "group_col": args.group_col,
        "permutation_repeats": args.permutation_repeats,
        "output_dir": str(out_dir.resolve()),
    }
    with (out_dir / "descriptor_ablation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
