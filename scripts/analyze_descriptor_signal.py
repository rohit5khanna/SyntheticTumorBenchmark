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
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


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


def evaluate_binary_task(df: pd.DataFrame, features: list[str], random_state: int) -> dict[str, pd.DataFrame | dict]:
    available = [f for f in features if f in df.columns]
    task = df[available + ["target"]].copy()
    task = task.dropna(subset=["target"])

    class_counts = task["target"].value_counts()
    if len(class_counts) < 2 or class_counts.min() < 2:
        raise ValueError("Need at least two classes with >=2 samples each for evaluation.")

    X = task[available].copy()
    y = task["target"].astype(int).to_numpy()

    n_splits = int(min(5, class_counts.min()))
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=20, random_state=random_state)

    logit_metrics: list[dict] = []
    tree_metrics: list[dict] = []
    logit_coefs: list[np.ndarray] = []
    tree_imps: list[np.ndarray] = []

    for train_idx, test_idx in cv.split(X, y):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        logit = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        )
        logit.fit(X_train, y_train)
        p_logit = logit.predict_proba(X_test)[:, 1]
        yhat_logit = (p_logit >= 0.5).astype(int)
        logit_metrics.append(
            {
                "model": "logistic_regression",
                "accuracy": accuracy_score(y_test, yhat_logit),
                "balanced_accuracy": balanced_accuracy_score(y_test, yhat_logit),
                "roc_auc": roc_auc_score(y_test, p_logit),
            }
        )
        logit_coefs.append(logit.named_steps["clf"].coef_[0].copy())

        tree = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    DecisionTreeClassifier(
                        max_depth=3,
                        min_samples_leaf=max(3, int(round(0.08 * len(train_idx)))),
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        )
        tree.fit(X_train, y_train)
        p_tree = tree.predict_proba(X_test)[:, 1]
        yhat_tree = (p_tree >= 0.5).astype(int)
        tree_metrics.append(
            {
                "model": "decision_tree_depth3",
                "accuracy": accuracy_score(y_test, yhat_tree),
                "balanced_accuracy": balanced_accuracy_score(y_test, yhat_tree),
                "roc_auc": roc_auc_score(y_test, p_tree),
            }
        )
        tree_imps.append(tree.named_steps["clf"].feature_importances_.copy())

    metrics_df = (
        pd.DataFrame(logit_metrics + tree_metrics)
        .groupby("model")
        .agg(["mean", "std"])
    )
    metrics_df.columns = [f"{a}_{b}" for a, b in metrics_df.columns]
    metrics_df = metrics_df.reset_index()

    logit_arr = np.vstack(logit_coefs)
    logit_imp = pd.DataFrame(
        {
            "feature": available,
            "mean_coef": logit_arr.mean(axis=0),
            "std_coef": logit_arr.std(axis=0),
            "mean_abs_coef": np.abs(logit_arr).mean(axis=0),
        }
    ).sort_values("mean_abs_coef", ascending=False)

    tree_arr = np.vstack(tree_imps)
    tree_imp = pd.DataFrame(
        {
            "feature": available,
            "mean_importance": tree_arr.mean(axis=0),
            "std_importance": tree_arr.std(axis=0),
        }
    ).sort_values("mean_importance", ascending=False)

    summary = {
        "n_samples": int(len(task)),
        "n_positive": int(task["target"].sum()),
        "n_negative": int(len(task) - task["target"].sum()),
        "features": available,
        "cv_n_splits": n_splits,
        "cv_n_repeats": 20,
    }
    return {
        "metrics": metrics_df,
        "logit_importance": logit_imp,
        "tree_importance": tree_imp,
        "summary": summary,
    }


def save_barplot(df: pd.DataFrame, value_col: str, title: str, out_path: Path) -> None:
    top = df.head(8).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.barh(top["feature"], top[value_col], color="#3A6EA5")
    ax.set_title(title)
    ax.set_xlabel(value_col.replace("_", " "))
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_report(path: Path, task_payloads: dict[str, dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Descriptor Signal Report\n\n")
        f.write("This report studies whether forecast-origin descriptors carry signal about short-horizon forecasting regime behavior.\n\n")
        for task_name, payload in task_payloads.items():
            f.write(f"## {task_name}\n\n")
            f.write("### Task Summary\n\n")
            f.write(pd.DataFrame([payload["summary"]]).to_markdown(index=False))
            f.write("\n\n### Model Metrics\n\n")
            f.write(payload["metrics"].to_markdown(index=False))
            f.write("\n\n### Logistic-Regression Feature Importance\n\n")
            f.write(payload["logit_importance"].to_markdown(index=False))
            f.write("\n\n### Decision-Tree Feature Importance\n\n")
            f.write(payload["tree_importance"].to_markdown(index=False))
            f.write("\n\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe forecast-origin descriptor signal using simple interpretable models."
    )
    parser.add_argument("--case_type_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    case_df = pd.read_csv(args.case_type_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    task_frames = prepare_task_frames(case_df)
    payloads: dict[str, dict] = {}

    for task_name, task_df in task_frames.items():
        result = evaluate_binary_task(task_df, FORECAST_ORIGIN_FEATURES, random_state=args.seed)
        payloads[task_name] = result
        task_dir = out_dir / task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        result["metrics"].to_csv(task_dir / "metrics.csv", index=False)
        result["logit_importance"].to_csv(task_dir / "logistic_importance.csv", index=False)
        result["tree_importance"].to_csv(task_dir / "tree_importance.csv", index=False)
        with (task_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(result["summary"], f, indent=2)
        save_barplot(
            result["logit_importance"],
            "mean_abs_coef",
            f"{task_name}: logistic feature importance",
            task_dir / "logistic_importance.png",
        )
        save_barplot(
            result["tree_importance"],
            "mean_importance",
            f"{task_name}: tree feature importance",
            task_dir / "tree_importance.png",
        )

    write_report(out_dir / "descriptor_signal_report.md", payloads)
    manifest = {
        "case_type_csv": str(Path(args.case_type_csv).resolve()),
        "tasks": list(task_frames.keys()),
        "forecast_origin_features": FORECAST_ORIGIN_FEATURES,
        "output_dir": str(out_dir.resolve()),
    }
    with (out_dir / "descriptor_signal_summary.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
