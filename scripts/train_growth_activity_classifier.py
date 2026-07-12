#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_growth_continuation import (  # noqa: E402
    PREDICTION_TIME_FEATURES,
    build_continuation_table,
)


def _safe_import_sklearn():
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from sklearn.tree import DecisionTreeClassifier
    except Exception as e:  # pragma: no cover - runtime dependency guard
        raise RuntimeError(
            "This script requires scikit-learn. Install it with `pip install scikit-learn`."
        ) from e

    return {
        "ColumnTransformer": ColumnTransformer,
        "RandomForestClassifier": RandomForestClassifier,
        "SimpleImputer": SimpleImputer,
        "LogisticRegression": LogisticRegression,
        "accuracy_score": accuracy_score,
        "balanced_accuracy_score": balanced_accuracy_score,
        "confusion_matrix": confusion_matrix,
        "f1_score": f1_score,
        "precision_score": precision_score,
        "recall_score": recall_score,
        "roc_auc_score": roc_auc_score,
        "Pipeline": Pipeline,
        "OneHotEncoder": OneHotEncoder,
        "StandardScaler": StandardScaler,
        "DecisionTreeClassifier": DecisionTreeClassifier,
    }


def _feature_columns(include_tier: bool, include_horizon: bool) -> tuple[List[str], List[str]]:
    numeric = list(PREDICTION_TIME_FEATURES)
    categorical = []
    if include_tier:
        categorical.append("tier")
    if include_horizon:
        categorical.append("horizon")
    return numeric, categorical


def _make_preprocessor(sk, numeric_features: List[str], categorical_features: List[str]):
    numeric_pipe = sk["Pipeline"](
        [
            ("imputer", sk["SimpleImputer"](strategy="median")),
            ("scaler", sk["StandardScaler"]()),
        ]
    )
    transformers = [("num", numeric_pipe, numeric_features)]
    if categorical_features:
        cat_pipe = sk["Pipeline"](
            [
                ("imputer", sk["SimpleImputer"](strategy="most_frequent")),
                ("onehot", sk["OneHotEncoder"](handle_unknown="ignore")),
            ]
        )
        transformers.append(("cat", cat_pipe, categorical_features))
    return sk["ColumnTransformer"](transformers=transformers)


def _candidate_models(sk, seed: int, preprocessor) -> Dict[str, object]:
    return {
        "logistic_l2_balanced": sk["Pipeline"](
            [
                ("prep", preprocessor),
                (
                    "clf",
                    sk["LogisticRegression"](
                        max_iter=2000,
                        class_weight="balanced",
                        C=1.0,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "decision_tree_depth2_balanced": sk["Pipeline"](
            [
                ("prep", preprocessor),
                (
                    "clf",
                    sk["DecisionTreeClassifier"](
                        max_depth=2,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "decision_tree_depth3_balanced": sk["Pipeline"](
            [
                ("prep", preprocessor),
                (
                    "clf",
                    sk["DecisionTreeClassifier"](
                        max_depth=3,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "random_forest_small_balanced": sk["Pipeline"](
            [
                ("prep", preprocessor),
                (
                    "clf",
                    sk["RandomForestClassifier"](
                        n_estimators=200,
                        max_depth=3,
                        min_samples_leaf=4,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }


def _threshold_grid(probs: np.ndarray) -> List[float]:
    values = np.asarray(probs, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return [0.5]
    qs = np.linspace(0.05, 0.95, 19)
    thresholds = {0.5}
    thresholds.update(float(np.quantile(finite, q)) for q in qs)
    thresholds.update(float(v) for v in np.unique(np.round(finite, 6)))
    return sorted(thresholds)


def _metrics(sk, y_true: np.ndarray, probs: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (probs >= threshold).astype(int)
    out = {
        "count": int(len(y_true)),
        "positive_rate_true": float(np.mean(y_true)) if len(y_true) else np.nan,
        "positive_rate_pred": float(np.mean(y_pred)) if len(y_pred) else np.nan,
        "threshold": float(threshold),
        "accuracy": float(sk["accuracy_score"](y_true, y_pred)),
        "balanced_accuracy": float(sk["balanced_accuracy_score"](y_true, y_pred)),
        "precision": float(sk["precision_score"](y_true, y_pred, zero_division=0)),
        "recall": float(sk["recall_score"](y_true, y_pred, zero_division=0)),
        "f1": float(sk["f1_score"](y_true, y_pred, zero_division=0)),
    }
    try:
        out["roc_auc"] = float(sk["roc_auc_score"](y_true, probs))
    except Exception:
        out["roc_auc"] = np.nan
    tn, fp, fn, tp = sk["confusion_matrix"](y_true, y_pred, labels=[0, 1]).ravel()
    out.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    out["specificity"] = float(tn / max(1, tn + fp))
    return out


def _select_threshold(sk, y_val: np.ndarray, probs_val: np.ndarray, objective: str) -> tuple[float, pd.DataFrame]:
    rows = []
    for threshold in _threshold_grid(probs_val):
        row = _metrics(sk, y_val, probs_val, threshold)
        rows.append(row)
    df = pd.DataFrame(rows)
    if objective == "balanced_accuracy":
        sort_cols = ["balanced_accuracy", "f1", "accuracy"]
    elif objective == "f1":
        sort_cols = ["f1", "balanced_accuracy", "accuracy"]
    elif objective == "accuracy":
        sort_cols = ["accuracy", "balanced_accuracy", "f1"]
    else:
        raise ValueError(f"Unsupported objective: {objective}")
    best = df.sort_values(sort_cols, ascending=False).iloc[0]
    return float(best["threshold"]), df


def _build_split_tables(args: argparse.Namespace) -> Dict[str, pd.DataFrame]:
    out = {}
    for split in ["train", "val", "test"]:
        out[split] = build_continuation_table(
            dataset_root=Path(args.dataset_root),
            split=split,
            fit_sessions=args.fit_sessions,
            horizons=args.horizons,
            allowed_tiers=args.allowed_tiers,
            min_growth_vox=args.min_growth_vox,
        )
        out[split]["future_growth_active_label"] = out[split]["future_growth_active"].astype(int)
    return out


def _available_features(df: pd.DataFrame, numeric: List[str], categorical: List[str]) -> tuple[List[str], List[str]]:
    num = [c for c in numeric if c in df.columns]
    cat = [c for c in categorical if c in df.columns]
    return num, cat


def _prediction_table(df: pd.DataFrame, probs: np.ndarray, threshold: float, split: str) -> pd.DataFrame:
    cols = [
        "patient_id",
        "tier",
        "split",
        "prev_idx",
        "input_idx",
        "target_idx",
        "horizon",
        "prev_interval_days",
        "delta_days",
        "input_volume_vox",
        "prev_new_growth_vox",
        "prev_relative_new_growth",
        "future_new_growth_vox",
        "future_relative_new_growth",
        "continuation_state",
        "future_growth_active_label",
    ]
    out = df[[c for c in cols if c in df.columns]].copy()
    out["eval_split"] = split
    out["growth_active_probability"] = probs
    out["growth_active_prediction"] = (probs >= threshold).astype(int)
    out["classification_correct"] = out["growth_active_prediction"].eq(out["future_growth_active_label"])
    return out


def _summarize_predictions(pred: pd.DataFrame, group_cols: Iterable[str]) -> pd.DataFrame:
    cols = [c for c in group_cols if c in pred.columns]
    group_df = pred if cols else pred.assign(_overall="overall")
    by = cols if cols else ["_overall"]
    out = (
        group_df.groupby(by, observed=True, dropna=False)
        .agg(
            count=("classification_correct", "size"),
            true_active_rate=("future_growth_active_label", "mean"),
            pred_active_rate=("growth_active_prediction", "mean"),
            accuracy=("classification_correct", "mean"),
            mean_probability=("growth_active_probability", "mean"),
            mean_future_growth_vox=("future_new_growth_vox", "mean"),
            median_future_growth_vox=("future_new_growth_vox", "median"),
        )
        .reset_index()
    )
    if "_overall" in out.columns:
        out = out.drop(columns=["_overall"])
    return out.sort_values(cols).reset_index(drop=True) if cols else out


def write_report(path: Path, selected: Dict, metrics: pd.DataFrame, test_by_state: pd.DataFrame) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Growth-Activity Classifier\n\n")
        f.write("This report trains candidate classifiers on train, selects model and threshold on validation, and evaluates on held-out test.\n\n")
        f.write("## Selected Model\n\n")
        f.write(pd.DataFrame([selected]).to_markdown(index=False))
        f.write("\n\n## Split Metrics\n\n")
        f.write(metrics.to_markdown(index=False))
        f.write("\n\n## Test Predictions By Continuation State\n\n")
        f.write(test_by_state.to_markdown(index=False) if not test_by_state.empty else "No state summary.")
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train simple validation-selected classifiers for future growth activity."
    )
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--fit_sessions", type=int, default=3)
    parser.add_argument("--horizons", type=str, default="1,2,3")
    parser.add_argument("--allowed_tiers", type=str, default=None)
    parser.add_argument("--min_growth_vox", type=int, default=250)
    parser.add_argument("--objective", type=str, default="balanced_accuracy", choices=["balanced_accuracy", "f1", "accuracy"])
    parser.add_argument("--include_tier", action="store_true")
    parser.add_argument("--include_horizon", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    sk = _safe_import_sklearn()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_tables = _build_split_tables(args)
    train = split_tables["train"]
    val = split_tables["val"]
    test = split_tables["test"]

    numeric, categorical = _feature_columns(include_tier=args.include_tier, include_horizon=args.include_horizon)
    numeric, categorical = _available_features(train, numeric, categorical)
    feature_cols = numeric + categorical
    if not feature_cols:
        raise ValueError("No usable features found.")

    X_train = train[feature_cols]
    y_train = train["future_growth_active_label"].to_numpy(dtype=int)
    X_val = val[feature_cols]
    y_val = val["future_growth_active_label"].to_numpy(dtype=int)
    X_test = test[feature_cols]
    y_test = test["future_growth_active_label"].to_numpy(dtype=int)

    preprocessor = _make_preprocessor(sk, numeric, categorical)
    candidates = _candidate_models(sk, seed=args.seed, preprocessor=preprocessor)

    candidate_rows = []
    threshold_tables = []
    fitted_models = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        fitted_models[name] = model
        val_probs = model.predict_proba(X_val)[:, 1]
        threshold, threshold_df = _select_threshold(sk, y_val, val_probs, objective=args.objective)
        threshold_df["model"] = name
        threshold_tables.append(threshold_df)
        candidate_rows.append(
            {
                "model": name,
                "selected_threshold": threshold,
                **{f"val_{k}": v for k, v in _metrics(sk, y_val, val_probs, threshold).items()},
            }
        )

    candidate_summary = pd.DataFrame(candidate_rows)
    if args.objective == "balanced_accuracy":
        sort_cols = ["val_balanced_accuracy", "val_f1", "val_accuracy"]
    elif args.objective == "f1":
        sort_cols = ["val_f1", "val_balanced_accuracy", "val_accuracy"]
    else:
        sort_cols = ["val_accuracy", "val_balanced_accuracy", "val_f1"]
    selected = candidate_summary.sort_values(sort_cols, ascending=False).iloc[0].to_dict()
    selected_model_name = str(selected["model"])
    selected_threshold = float(selected["selected_threshold"])
    selected_model = fitted_models[selected_model_name]

    split_metric_rows = []
    prediction_tables = []
    for split_name, df, X, y in [
        ("train", train, X_train, y_train),
        ("val", val, X_val, y_val),
        ("test", test, X_test, y_test),
    ]:
        probs = selected_model.predict_proba(X)[:, 1]
        split_metric_rows.append(
            {
                "split": split_name,
                "model": selected_model_name,
                **_metrics(sk, y, probs, selected_threshold),
            }
        )
        prediction_tables.append(_prediction_table(df, probs, selected_threshold, split_name))

    metrics = pd.DataFrame(split_metric_rows)
    predictions = pd.concat(prediction_tables, ignore_index=True)
    test_predictions = predictions[predictions["eval_split"] == "test"].copy()
    test_by_state = _summarize_predictions(test_predictions, ["continuation_state"])
    test_by_tier = _summarize_predictions(test_predictions, ["tier"])
    test_by_horizon = _summarize_predictions(test_predictions, ["horizon"])
    test_by_tier_state = _summarize_predictions(test_predictions, ["tier", "continuation_state"])

    for split_name, df in split_tables.items():
        df.to_csv(output_dir / f"growth_activity_{split_name}_samples.csv", index=False)
    candidate_summary.to_csv(output_dir / "growth_activity_candidate_models.csv", index=False)
    pd.concat(threshold_tables, ignore_index=True).to_csv(output_dir / "growth_activity_threshold_sweep.csv", index=False)
    metrics.to_csv(output_dir / "growth_activity_selected_model_metrics.csv", index=False)
    predictions.to_csv(output_dir / "growth_activity_predictions.csv", index=False)
    test_by_state.to_csv(output_dir / "growth_activity_test_by_continuation_state.csv", index=False)
    test_by_tier.to_csv(output_dir / "growth_activity_test_by_tier.csv", index=False)
    test_by_horizon.to_csv(output_dir / "growth_activity_test_by_horizon.csv", index=False)
    test_by_tier_state.to_csv(output_dir / "growth_activity_test_by_tier_state.csv", index=False)

    report = {
        "dataset_root": args.dataset_root,
        "min_growth_vox": args.min_growth_vox,
        "objective": args.objective,
        "include_tier": bool(args.include_tier),
        "include_horizon": bool(args.include_horizon),
        "numeric_features": numeric,
        "categorical_features": categorical,
        "selected_model": selected_model_name,
        "selected_threshold": selected_threshold,
        "output_dir": str(output_dir),
    }
    with (output_dir / "growth_activity_classifier_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    write_report(output_dir / "growth_activity_classifier_report.md", {**report, **selected}, metrics, test_by_state)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
