#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


DEFAULT_FEATURES = [
    "delta_days",
    "input_span_days",
    "input_volume_vox",
    "input_end_treatment",
    "target_treatment",
    "treatment_changed_in_input",
    "target_treatment_changed",
    "previous_growth_volume_vox",
    "previous_loss_volume_vox",
    "previous_growth_ratio",
]

KEY_COLS = ["split", "patient_id", "input_idx", "target_idx", "horizon"]


def parse_csv(payload: str | None) -> List[str]:
    if payload is None:
        return []
    return [x.strip() for x in payload.split(",") if x.strip()]


def normalize_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    out = manifest.copy()
    if "input_idx" not in out.columns and "input_end_idx" in out.columns:
        out["input_idx"] = out["input_end_idx"]
    if "split" not in out.columns:
        out["split"] = "all"
    if "relative_absolute_change" not in out.columns:
        if {"relative_new_growth", "relative_loss"}.issubset(out.columns):
            out["relative_absolute_change"] = out["relative_new_growth"] + out["relative_loss"]
        else:
            raise ValueError("Need relative_absolute_change or both relative_new_growth and relative_loss.")
    if "target_treatment_changed" in out.columns:
        out["target_treatment_changed"] = out["target_treatment_changed"].astype(float)
    if "treatment_changed_in_input" in out.columns:
        out["treatment_changed_in_input"] = out["treatment_changed_in_input"].astype(float)
    return out


def add_train_defined_label(
    df: pd.DataFrame,
    train_split: str,
    change_col: str,
    high_quantile: float,
) -> tuple[pd.DataFrame, float]:
    out = df.copy()
    train = out[out["split"] == train_split]
    if train.empty:
        raise ValueError(f"No rows found for train_split='{train_split}'.")
    threshold = float(train[change_col].quantile(high_quantile))
    out["persistence_breakdown_label"] = (out[change_col] >= threshold).astype(int)
    out["persistence_breakdown_threshold"] = threshold
    return out, threshold


def available_features(df: pd.DataFrame, requested: Iterable[str]) -> List[str]:
    available = []
    blocked = {
        "target_idx",
        "target_day",
        "target_volume_vox",
        "growth_volume_vox",
        "loss_volume_vox",
        "relative_new_growth",
        "relative_loss",
        "relative_net_growth",
        "relative_absolute_change",
        "locf_dice",
        "absolute_growth_bin",
        "net_direction",
    }
    for feature in requested:
        if feature in blocked:
            raise ValueError(
                f"Feature '{feature}' is target-derived and would leak future information. "
                "Use only pre-target/origin-known features."
            )
        if feature in df.columns:
            available.append(feature)
    if not available:
        raise ValueError("No requested features were available.")
    return available


def _safe_metric(fn, y_true, y_score_or_pred) -> float:
    try:
        return float(fn(y_true, y_score_or_pred))
    except Exception:
        return float("nan")


def evaluate_model(
    df: pd.DataFrame,
    features: List[str],
    train_split: str,
    eval_splits: List[str],
    model_name: str,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    train = df[df["split"] == train_split].copy()
    if train.empty:
        raise ValueError(f"No rows found for train_split='{train_split}'.")
    if train["persistence_breakdown_label"].nunique() < 2:
        raise ValueError("Training label has only one class; cannot fit classifier.")

    X_train = train[features]
    y_train = train["persistence_breakdown_label"].astype(int)

    numeric_features = [f for f in features if pd.api.types.is_numeric_dtype(df[f])]
    non_numeric_features = [f for f in features if f not in numeric_features]
    if non_numeric_features:
        raise ValueError(f"Non-numeric features are not currently supported: {non_numeric_features}")

    pre = ColumnTransformer(
        [("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features)],
        remainder="drop",
    )
    if model_name == "logistic":
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)
    elif model_name == "tree":
        clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, class_weight="balanced", random_state=random_state)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    model = Pipeline([("pre", pre), ("clf", clf)])
    model.fit(X_train, y_train)

    pred_rows = []
    summary_rows = []
    for split in eval_splits:
        part = df[df["split"] == split].copy()
        if part.empty:
            continue
        score = model.predict_proba(part[features])[:, 1]
        pred = (score >= 0.5).astype(int)
        y = part["persistence_breakdown_label"].astype(int).to_numpy()
        pred_part = part[KEY_COLS + ["relative_absolute_change", "persistence_breakdown_label"]].copy()
        pred_part["model"] = model_name
        pred_part["breakdown_score"] = score
        pred_part["breakdown_pred_0p5"] = pred
        pred_rows.append(pred_part)
        summary_rows.append(
            {
                "model": model_name,
                "split": split,
                "n_samples": int(len(part)),
                "n_patients": int(part["patient_id"].nunique()),
                "positive_rate": float(np.mean(y)),
                "accuracy": _safe_metric(accuracy_score, y, pred),
                "balanced_accuracy": _safe_metric(balanced_accuracy_score, y, pred),
                "roc_auc": _safe_metric(roc_auc_score, y, score),
                "average_precision": _safe_metric(average_precision_score, y, score),
                "precision_at_0p5": _safe_metric(lambda yt, yp: precision_score(yt, yp, zero_division=0), y, pred),
                "recall_at_0p5": _safe_metric(lambda yt, yp: recall_score(yt, yp, zero_division=0), y, pred),
            }
        )

    if model_name == "logistic":
        coefs = model.named_steps["clf"].coef_[0]
        coef_df = pd.DataFrame({"model": model_name, "feature": numeric_features, "coefficient": coefs})
        coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
        coef_df = coef_df.sort_values("abs_coefficient", ascending=False)
    else:
        importances = model.named_steps["clf"].feature_importances_
        coef_df = pd.DataFrame({"model": model_name, "feature": numeric_features, "importance": importances})
        coef_df = coef_df.sort_values("importance", ascending=False)

    preds = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    return preds, summary, coef_df


def threshold_sweep(preds: pd.DataFrame) -> pd.DataFrame:
    from sklearn.metrics import precision_score, recall_score

    rows = []
    if preds.empty:
        return pd.DataFrame()
    for (model, split), part in preds.groupby(["model", "split"], observed=True):
        y = part["persistence_breakdown_label"].astype(int).to_numpy()
        score = part["breakdown_score"].to_numpy()
        for thr in np.linspace(0.1, 0.9, 9):
            pred = (score >= thr).astype(int)
            rows.append(
                {
                    "model": model,
                    "split": split,
                    "threshold": float(thr),
                    "predicted_positive_rate": float(pred.mean()),
                    "precision": float(precision_score(y, pred, zero_division=0)),
                    "recall": float(recall_score(y, pred, zero_division=0)),
                    "false_negative_rate": float(((y == 1) & (pred == 0)).sum() / max(1, (y == 1).sum())),
                    "false_positive_rate": float(((y == 0) & (pred == 1)).sum() / max(1, (y == 0).sum())),
                }
            )
    return pd.DataFrame(rows)


def _classification_metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> Dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    pred = (score >= threshold).astype(int)
    return {
        "accuracy": _safe_metric(accuracy_score, y, pred),
        "balanced_accuracy": _safe_metric(balanced_accuracy_score, y, pred),
        "roc_auc": _safe_metric(roc_auc_score, y, score),
        "average_precision": _safe_metric(average_precision_score, y, score),
        "precision": _safe_metric(lambda yt, yp: precision_score(yt, yp, zero_division=0), y, pred),
        "recall": _safe_metric(lambda yt, yp: recall_score(yt, yp, zero_division=0), y, pred),
        "false_negative_rate": float(((y == 1) & (pred == 0)).sum() / max(1, (y == 1).sum())),
        "false_positive_rate": float(((y == 0) & (pred == 1)).sum() / max(1, (y == 0).sum())),
        "predicted_positive_rate": float(pred.mean()),
    }


def patient_bootstrap_metrics(
    preds: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if preds.empty or n_bootstrap <= 0:
        return pd.DataFrame(), pd.DataFrame()

    rng = np.random.default_rng(seed)
    draw_rows = []
    summary_rows = []
    metric_cols = [
        "accuracy",
        "balanced_accuracy",
        "roc_auc",
        "average_precision",
        "precision",
        "recall",
        "false_negative_rate",
        "false_positive_rate",
        "predicted_positive_rate",
    ]

    for (model, split), part in preds.groupby(["model", "split"], observed=True):
        patients = np.array(sorted(part["patient_id"].dropna().unique()))
        if len(patients) == 0:
            continue

        patient_parts = {pid: rows for pid, rows in part.groupby("patient_id", observed=True)}
        for draw_idx in range(n_bootstrap):
            sampled_patients = rng.choice(patients, size=len(patients), replace=True)
            sampled = pd.concat([patient_parts[pid] for pid in sampled_patients], ignore_index=True)
            y = sampled["persistence_breakdown_label"].astype(int).to_numpy()
            score = sampled["breakdown_score"].to_numpy()
            metrics = _classification_metrics(y, score, threshold=threshold)
            row = {
                "model": model,
                "split": split,
                "draw_idx": int(draw_idx),
                "n_patients_sampled": int(len(sampled_patients)),
                "n_unique_patients": int(len(set(sampled_patients))),
                "n_samples": int(len(sampled)),
                "positive_rate": float(np.mean(y)),
            }
            row.update(metrics)
            draw_rows.append(row)

        group_draws = pd.DataFrame([r for r in draw_rows if r["model"] == model and r["split"] == split])
        row = {
            "model": model,
            "split": split,
            "n_bootstrap": int(n_bootstrap),
            "n_original_patients": int(len(patients)),
            "n_original_samples": int(len(part)),
            "threshold": float(threshold),
        }
        for metric in metric_cols:
            values = group_draws[metric].dropna().to_numpy()
            row[f"{metric}_mean"] = float(np.mean(values)) if len(values) else float("nan")
            row[f"{metric}_ci_low"] = float(np.quantile(values, 0.025)) if len(values) else float("nan")
            row[f"{metric}_ci_high"] = float(np.quantile(values, 0.975)) if len(values) else float("nan")
            row[f"{metric}_valid_draws"] = int(len(values))
        for metric in ["balanced_accuracy", "roc_auc", "average_precision", "recall"]:
            values = group_draws[metric].dropna().to_numpy()
            row[f"prob_{metric}_gt_0p5"] = float(np.mean(values > 0.5)) if len(values) else float("nan")
        summary_rows.append(row)

    return pd.DataFrame(draw_rows), pd.DataFrame(summary_rows)


def feature_univariate_summary(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    rows = []
    for feature in features:
        for label, part in df.groupby("persistence_breakdown_label", observed=True):
            rows.append(
                {
                    "feature": feature,
                    "label": int(label),
                    "n": int(len(part)),
                    "mean": float(part[feature].mean()),
                    "median": float(part[feature].median()),
                    "std": float(part[feature].std()),
                }
            )
    return pd.DataFrame(rows)


def write_report(
    path: Path,
    threshold: float,
    features: List[str],
    summaries: Dict[str, pd.DataFrame],
) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Persistence-Breakdown Predictability Audit\n\n")
        f.write(
            "This audit asks whether high observed change burden can be anticipated from pre-target information. "
            "It does not tune a deployment policy and should not be interpreted as a finished gating model.\n\n"
        )
        f.write(f"Train-defined high-change threshold: `{threshold:.6g}` for relative absolute change.\n\n")
        f.write("Features used:\n\n")
        for feature in features:
            f.write(f"- `{feature}`\n")
        f.write("\n")
        for name, table in summaries.items():
            f.write(f"## {name}\n\n")
            f.write(table.to_markdown(index=False) if not table.empty else "No rows.")
            f.write("\n\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict high-change/persistence-breakdown cases from pre-target features.")
    parser.add_argument("--manifest_csv", type=str, required=True)
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--eval_splits", type=str, default="val,test")
    parser.add_argument("--features", type=str, default=",".join(DEFAULT_FEATURES))
    parser.add_argument("--change_col", type=str, default="relative_absolute_change")
    parser.add_argument("--high_quantile", type=float, default=2.0 / 3.0)
    parser.add_argument("--n_bootstrap", type=int, default=0)
    parser.add_argument("--bootstrap_threshold", type=float, default=0.5)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest = normalize_manifest(pd.read_csv(args.manifest_csv))
    if args.change_col not in manifest.columns:
        raise ValueError(f"Change column '{args.change_col}' not found.")
    data, threshold = add_train_defined_label(manifest, args.train_split, args.change_col, args.high_quantile)
    features = available_features(data, parse_csv(args.features))
    eval_splits = parse_csv(args.eval_splits)

    pred_parts = []
    summary_parts = []
    coef_parts = []
    for model_name in ["logistic", "tree"]:
        preds, summary, coefs = evaluate_model(
            data,
            features=features,
            train_split=args.train_split,
            eval_splits=eval_splits,
            model_name=model_name,
            random_state=args.seed,
        )
        pred_parts.append(preds)
        summary_parts.append(summary)
        coef_parts.append(coefs)

    predictions = pd.concat(pred_parts, ignore_index=True)
    model_summary = pd.concat(summary_parts, ignore_index=True)
    feature_weights = pd.concat(coef_parts, ignore_index=True)
    sweep = threshold_sweep(predictions)
    univariate = feature_univariate_summary(data[data["split"].isin([args.train_split] + eval_splits)], features)
    bootstrap_draws, bootstrap_summary = patient_bootstrap_metrics(
        predictions,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        threshold=args.bootstrap_threshold,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_dir / "persistence_breakdown_labeled_manifest.csv", index=False)
    predictions.to_csv(output_dir / "persistence_breakdown_predictions.csv", index=False)
    model_summary.to_csv(output_dir / "persistence_breakdown_model_summary.csv", index=False)
    feature_weights.to_csv(output_dir / "persistence_breakdown_feature_weights.csv", index=False)
    sweep.to_csv(output_dir / "persistence_breakdown_threshold_sweep.csv", index=False)
    univariate.to_csv(output_dir / "persistence_breakdown_univariate_features.csv", index=False)
    if not bootstrap_draws.empty:
        bootstrap_draws.to_csv(output_dir / "persistence_breakdown_patient_bootstrap_draws.csv", index=False)
        bootstrap_summary.to_csv(output_dir / "persistence_breakdown_patient_bootstrap_summary.csv", index=False)

    tables = {
        "Model Summary": model_summary,
        "Feature Weights": feature_weights,
        "Threshold Sweep": sweep,
        "Univariate Feature Summary": univariate,
    }
    if not bootstrap_summary.empty:
        tables["Patient Bootstrap Summary"] = bootstrap_summary
    write_report(output_dir / "persistence_breakdown_predictability_report.md", threshold, features, tables)

    payload = {
        "manifest_csv": args.manifest_csv,
        "train_split": args.train_split,
        "eval_splits": eval_splits,
        "change_col": args.change_col,
        "high_quantile": float(args.high_quantile),
        "train_defined_threshold": threshold,
        "features": features,
        "n_bootstrap": int(args.n_bootstrap),
        "bootstrap_threshold": float(args.bootstrap_threshold),
        "n_rows": int(len(data)),
        "output_dir": str(output_dir),
        "outputs": {
            "predictions_csv": str(output_dir / "persistence_breakdown_predictions.csv"),
            "model_summary_csv": str(output_dir / "persistence_breakdown_model_summary.csv"),
            "feature_weights_csv": str(output_dir / "persistence_breakdown_feature_weights.csv"),
            "threshold_sweep_csv": str(output_dir / "persistence_breakdown_threshold_sweep.csv"),
            "patient_bootstrap_summary_csv": str(output_dir / "persistence_breakdown_patient_bootstrap_summary.csv")
            if not bootstrap_summary.empty
            else None,
            "report_md": str(output_dir / "persistence_breakdown_predictability_report.md"),
        },
    }
    with (output_dir / "persistence_breakdown_predictability_summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
