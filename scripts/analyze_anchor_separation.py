#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FULL_NUMERIC_FEATURES = [
    "activity_score",
    "structure_score",
    "input_volume_vox",
    "recent_relative_growth",
    "delta_days",
    "treated_at_input",
    "input_connected_component_count",
    "input_compactness_proxy",
    "input_elongation_ratio",
]

RAW_NUMERIC_FEATURES = [
    "input_volume_vox",
    "recent_relative_growth",
    "delta_days",
    "treated_at_input",
    "input_connected_component_count",
    "input_compactness_proxy",
    "input_elongation_ratio",
]

CATEGORICAL_FEATURES = [
    "tier",
    "horizon",
]


def cohen_d(x1: np.ndarray, x0: np.ndarray) -> float:
    n1 = len(x1)
    n0 = len(x0)
    if n1 < 2 or n0 < 2:
        return 0.0
    v1 = float(np.var(x1, ddof=1))
    v0 = float(np.var(x0, ddof=1))
    pooled = ((n1 - 1) * v1 + (n0 - 1) * v0) / max(n1 + n0 - 2, 1)
    if pooled <= 0:
        return 0.0
    return float((np.mean(x1) - np.mean(x0)) / np.sqrt(pooled))


def build_anchor_separation(df: pd.DataFrame, numeric_features: list[str]) -> pd.DataFrame:
    anchor = df[df["profile_group"].isin(["both_easy_core", "target_wins_core"])].copy()
    pos = anchor[anchor["profile_group"] == "target_wins_core"]
    neg = anchor[anchor["profile_group"] == "both_easy_core"]

    rows = []
    for feature in numeric_features:
        if feature not in anchor.columns:
            continue
        x1 = pos[feature].dropna().to_numpy(dtype=float)
        x0 = neg[feature].dropna().to_numpy(dtype=float)
        rows.append(
            {
                "feature": feature,
                "target_wins_core_mean": float(np.mean(x1)) if len(x1) else np.nan,
                "both_easy_core_mean": float(np.mean(x0)) if len(x0) else np.nan,
                "mean_gap": (float(np.mean(x1)) - float(np.mean(x0))) if len(x1) and len(x0) else np.nan,
                "std_mean_gap": cohen_d(x1, x0),
                "abs_std_mean_gap": abs(cohen_d(x1, x0)),
            }
        )
    out = pd.DataFrame(rows).sort_values("abs_std_mean_gap", ascending=False)
    return out


def prepare_matrix(df: pd.DataFrame, numeric_features: list[str], categorical_features: list[str]) -> tuple[pd.DataFrame, list[str]]:
    keep_numeric = [c for c in numeric_features if c in df.columns]
    work = df[keep_numeric + [c for c in categorical_features if c in df.columns]].copy()
    work = pd.get_dummies(work, columns=[c for c in categorical_features if c in work.columns], drop_first=False, dtype=float)
    feature_names = work.columns.tolist()
    return work, feature_names


def evaluate_binary_task(
    df: pd.DataFrame,
    target_col: str,
    random_state: int,
    numeric_features: list[str],
    group_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X, feature_names = prepare_matrix(df, numeric_features, CATEGORICAL_FEATURES)
    y = df[target_col].astype(int).to_numpy()

    class_counts = pd.Series(y).value_counts()
    if len(class_counts) < 2 or int(class_counts.min()) < 2:
        raise ValueError("Need both classes with at least 2 samples.")

    if group_col and group_col in df.columns:
        groups = df[group_col].astype(str).to_numpy()
        group_counts = (
            df[[group_col, target_col]]
            .drop_duplicates()
            .groupby(target_col)[group_col]
            .nunique()
        )
        n_splits = int(min(5, group_counts.min()))
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_iter = cv.split(X, y, groups=groups)
    else:
        groups = None
        n_splits = int(min(5, class_counts.min()))
        cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=20, random_state=random_state)
        split_iter = cv.split(X, y)

    metrics = []
    coefs = []
    for train_idx, test_idx in split_iter:
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
                "accuracy": accuracy_score(y_test, pred),
                "balanced_accuracy": balanced_accuracy_score(y_test, pred),
                "roc_auc": roc_auc_score(y_test, prob),
            }
        )
        coefs.append(pipe.named_steps["clf"].coef_[0].copy())

    metrics_df = pd.DataFrame(metrics).agg(["mean", "std"]).T.reset_index().rename(columns={"index": "metric"})
    coef_arr = np.vstack(coefs)
    coef_df = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_coef": coef_arr.mean(axis=0),
            "std_coef": coef_arr.std(axis=0),
            "mean_abs_coef": np.abs(coef_arr).mean(axis=0),
        }
    ).sort_values("mean_abs_coef", ascending=False)
    return metrics_df, coef_df


def save_barplot(df: pd.DataFrame, value_col: str, title: str, out_path: Path) -> None:
    top = df.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    ax.barh(top["feature"], top[value_col], color="#3A6EA5")
    ax.set_title(title)
    ax.set_xlabel(value_col.replace("_", " "))
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(
    path: Path,
    anchor_sep: pd.DataFrame,
    pull_metrics: pd.DataFrame,
    pull_coef: pd.DataFrame,
    transition_metrics: pd.DataFrame,
    transition_coef: pd.DataFrame,
) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Anchor Separation And Pull Predictors Report\n\n")
        f.write("This analysis quantifies which descriptors separate the two stable anchor populations and which descriptors predict ambiguous regime behavior.\n\n")
        f.write("## Anchor Separation: Target-Wins Core vs Both-Easy Core\n\n")
        f.write(anchor_sep.to_markdown(index=False))
        f.write("\n\n## Predicting Cross-Regime Pull vs Anchor Cores\n\n")
        f.write("### Metrics\n\n")
        f.write(pull_metrics.to_markdown(index=False))
        f.write("\n\n### Feature Coefficients\n\n")
        f.write(pull_coef.to_markdown(index=False))
        f.write("\n\n## Predicting Transition vs Anchor Cores\n\n")
        f.write("### Metrics\n\n")
        f.write(transition_metrics.to_markdown(index=False))
        f.write("\n\n### Feature Coefficients\n\n")
        f.write(transition_coef.to_markdown(index=False))
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantify anchor separation and predictors of ambiguous soft-regime behavior.")
    parser.add_argument("--soft_profile_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature_mode", type=str, choices=["full", "raw_only"], default="full")
    parser.add_argument("--group_col", type=str, default=None)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.soft_profile_csv)
    numeric_features = FULL_NUMERIC_FEATURES if args.feature_mode == "full" else RAW_NUMERIC_FEATURES
    anchor_sep = build_anchor_separation(df, numeric_features=numeric_features)

    pull_df = df[df["profile_group"].isin(["both_easy_core", "target_wins_core", "cross_regime_pull"])].copy()
    pull_df["target"] = (pull_df["profile_group"] == "cross_regime_pull").astype(int)
    pull_metrics, pull_coef = evaluate_binary_task(
        pull_df,
        "target",
        random_state=args.seed,
        numeric_features=numeric_features,
        group_col=args.group_col,
    )

    transition_df = df[df["profile_group"].isin(["both_easy_core", "target_wins_core", "transition"])].copy()
    transition_df["target"] = (transition_df["profile_group"] == "transition").astype(int)
    transition_metrics, transition_coef = evaluate_binary_task(
        transition_df,
        "target",
        random_state=args.seed,
        numeric_features=numeric_features,
        group_col=args.group_col,
    )

    anchor_sep.to_csv(out_dir / "anchor_feature_separation.csv", index=False)
    pull_metrics.to_csv(out_dir / "cross_regime_pull_metrics.csv", index=False)
    pull_coef.to_csv(out_dir / "cross_regime_pull_coefficients.csv", index=False)
    transition_metrics.to_csv(out_dir / "transition_metrics.csv", index=False)
    transition_coef.to_csv(out_dir / "transition_coefficients.csv", index=False)

    save_barplot(anchor_sep, "abs_std_mean_gap", "Anchor separation by standardized mean gap", out_dir / "anchor_feature_separation.png")
    save_barplot(pull_coef, "mean_abs_coef", "Cross-regime pull predictors", out_dir / "cross_regime_pull_predictors.png")
    save_barplot(transition_coef, "mean_abs_coef", "Transition predictors", out_dir / "transition_predictors.png")

    write_report(
        out_dir / "anchor_separation_report.md",
        anchor_sep,
        pull_metrics,
        pull_coef,
        transition_metrics,
        transition_coef,
    )

    manifest = {
        "soft_profile_csv": str(Path(args.soft_profile_csv).resolve()),
        "numeric_features": numeric_features,
        "categorical_features": CATEGORICAL_FEATURES,
        "feature_mode": args.feature_mode,
        "group_col": args.group_col,
        "output_dir": str(out_dir.resolve()),
    }
    with (out_dir / "anchor_separation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
