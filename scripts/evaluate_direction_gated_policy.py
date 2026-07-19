#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


KEY_COLS = ["split", "patient_id", "input_idx", "target_idx", "horizon"]

NUMERIC_FEATURES = [
    "input_window_len",
    "input_span_days",
    "delta_days",
    "input_end_treatment",
    "input_volume_vox",
    "previous_growth_volume_vox",
    "previous_loss_volume_vox",
    "previous_growth_ratio",
]

CATEGORICAL_FEATURES = [
    "treatment_changed_in_input",
]


def safe_import_sklearn():
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            confusion_matrix,
            precision_score,
            recall_score,
            roc_auc_score,
        )
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
    except Exception as e:  # pragma: no cover
        raise RuntimeError("This script requires scikit-learn. Install it with `pip install scikit-learn`.") from e
    return {
        "ColumnTransformer": ColumnTransformer,
        "SimpleImputer": SimpleImputer,
        "LogisticRegression": LogisticRegression,
        "accuracy_score": accuracy_score,
        "balanced_accuracy_score": balanced_accuracy_score,
        "confusion_matrix": confusion_matrix,
        "precision_score": precision_score,
        "recall_score": recall_score,
        "roc_auc_score": roc_auc_score,
        "Pipeline": Pipeline,
        "OneHotEncoder": OneHotEncoder,
        "StandardScaler": StandardScaler,
    }


def normalize_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    out = manifest.copy()
    if "input_idx" not in out.columns and "input_end_idx" in out.columns:
        out["input_idx"] = out["input_end_idx"]
    if "input_treatment" not in out.columns and "input_end_treatment" in out.columns:
        out["input_treatment"] = out["input_end_treatment"]
    if "treatment_changed_in_input" in out.columns:
        out["treatment_changed_in_input"] = out["treatment_changed_in_input"].astype(str)
    else:
        out["treatment_changed_in_input"] = "False"
    out["future_is_net_growth"] = (out["net_direction"] == "net_growth").astype(int)
    return out


def read_method_csv(path: Path, method_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = df[KEY_COLS + ["dice"]].copy()
    out = out.rename(columns={"dice": f"dice_{method_name}"})
    return out


def merge_policy_frame(manifest: pd.DataFrame, locf_csv: Path, model_csv: Path, model_name: str) -> pd.DataFrame:
    base_cols = [
        "split",
        "patient_id",
        "input_idx",
        "target_idx",
        "horizon",
        "delta_days",
        "net_direction",
        "future_is_net_growth",
        "absolute_growth_bin",
        "growth_volume_vox",
        "relative_new_growth",
    ]
    feature_cols = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c in manifest.columns]
    base = manifest[base_cols + [c for c in feature_cols if c not in base_cols]].copy()
    locf = read_method_csv(locf_csv, "locf")
    model = read_method_csv(model_csv, model_name)
    paired = base.merge(locf, on=KEY_COLS, how="inner").merge(model, on=KEY_COLS, how="inner")
    if paired.empty:
        raise ValueError("No paired LOCF/model rows after merge. Check manifest and per-sample files.")
    return paired


def make_pipeline(sk, numeric_features: List[str], categorical_features: List[str]):
    numeric_pipe = sk["Pipeline"](
        [
            ("imputer", sk["SimpleImputer"](strategy="median")),
            ("scaler", sk["StandardScaler"]()),
        ]
    )
    categorical_pipe = sk["Pipeline"](
        [
            ("imputer", sk["SimpleImputer"](strategy="most_frequent")),
            ("onehot", sk["OneHotEncoder"](handle_unknown="ignore")),
        ]
    )
    pre = sk["ColumnTransformer"](
        [
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ],
        remainder="drop",
    )
    clf = sk["LogisticRegression"](class_weight="balanced", max_iter=2000, solver="lbfgs")
    return sk["Pipeline"]([("preprocess", pre), ("model", clf)])


def classifier_metrics(sk, y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict:
    pred = (prob >= threshold).astype(int)
    try:
        auc = float(sk["roc_auc_score"](y_true, prob)) if len(np.unique(y_true)) > 1 else np.nan
    except Exception:
        auc = np.nan
    tn, fp, fn, tp = sk["confusion_matrix"](y_true, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "threshold": float(threshold),
        "accuracy": float(sk["accuracy_score"](y_true, pred)),
        "balanced_accuracy": float(sk["balanced_accuracy_score"](y_true, pred)),
        "precision_net_growth": float(sk["precision_score"](y_true, pred, zero_division=0)),
        "recall_net_growth": float(sk["recall_score"](y_true, pred, zero_division=0)),
        "roc_auc": auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def apply_policy(df: pd.DataFrame, model_name: str, threshold: float) -> pd.DataFrame:
    out = df.copy()
    out["predicted_net_growth"] = out["gate_prob_net_growth"] >= threshold
    out["direction_gated_dice"] = np.where(out["predicted_net_growth"], out[f"dice_{model_name}"], out["dice_locf"])
    out["oracle_direction_dice"] = np.where(out["future_is_net_growth"] == 1, out[f"dice_{model_name}"], out["dice_locf"])
    out["best_of_two_dice"] = np.maximum(out["dice_locf"], out[f"dice_{model_name}"])
    out["gap_gated_vs_locf"] = out["direction_gated_dice"] - out["dice_locf"]
    out["gap_gated_vs_model"] = out["direction_gated_dice"] - out[f"dice_{model_name}"]
    out["gap_oracle_vs_locf"] = out["oracle_direction_dice"] - out["dice_locf"]
    out["gap_oracle_vs_model"] = out["oracle_direction_dice"] - out[f"dice_{model_name}"]
    return out


def summarize_policy(df: pd.DataFrame, model_name: str, group_cols: List[str]) -> pd.DataFrame:
    by = group_cols if group_cols else ["_overall"]
    work = df.copy()
    if not group_cols:
        work["_overall"] = "overall"
    out = (
        work.groupby(by, observed=True, dropna=False)
        .agg(
            n=("patient_id", "size"),
            n_patients=("patient_id", "nunique"),
            locf_mean=("dice_locf", "mean"),
            model_mean=(f"dice_{model_name}", "mean"),
            gated_mean=("direction_gated_dice", "mean"),
            oracle_mean=("oracle_direction_dice", "mean"),
            best_of_two_mean=("best_of_two_dice", "mean"),
            gated_gap_vs_locf=("gap_gated_vs_locf", "mean"),
            gated_gap_vs_model=("gap_gated_vs_model", "mean"),
            oracle_gap_vs_locf=("gap_oracle_vs_locf", "mean"),
            predicted_growth_rate=("predicted_net_growth", "mean"),
            true_growth_rate=("future_is_net_growth", "mean"),
        )
        .reset_index()
    )
    if "_overall" in out.columns:
        out = out.drop(columns=["_overall"])
    return out


def bootstrap_patient_ci(df: pd.DataFrame, model_name: str, n_bootstrap: int, seed: int) -> pd.DataFrame:
    if n_bootstrap <= 0:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    rows = []
    for split, part in df.groupby("split", observed=True):
        patients = np.asarray(sorted(part["patient_id"].unique()))
        if len(patients) == 0:
            continue
        boot_gaps = []
        for _ in range(n_bootstrap):
            sampled = rng.choice(patients, size=len(patients), replace=True)
            boot = pd.concat([part[part["patient_id"] == pid] for pid in sampled], ignore_index=True)
            boot_gaps.append(float((boot["direction_gated_dice"] - boot["dice_locf"]).mean()))
        lo, hi = np.percentile(boot_gaps, [2.5, 97.5])
        rows.append(
            {
                "split": split,
                "metric": "gated_gap_vs_locf",
                "mean": float((part["direction_gated_dice"] - part["dice_locf"]).mean()),
                "ci_low": float(lo),
                "ci_high": float(hi),
                "n_patients": int(len(patients)),
                "n_bootstrap": int(n_bootstrap),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a direction-gated LOCF/model policy on longitudinal windows.")
    parser.add_argument("--manifest_csv", type=str, required=True)
    parser.add_argument("--locf_csv", type=str, required=True)
    parser.add_argument("--model_csv", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="resunet")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="val")
    parser.add_argument("--test_split", type=str, default="test")
    parser.add_argument("--threshold_grid", type=str, default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95")
    parser.add_argument("--n_bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    sk = safe_import_sklearn()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = normalize_manifest(pd.read_csv(args.manifest_csv))
    paired = merge_policy_frame(manifest, Path(args.locf_csv), Path(args.model_csv), args.model_name)

    numeric_features = [c for c in NUMERIC_FEATURES if c in paired.columns]
    categorical_features = [c for c in CATEGORICAL_FEATURES if c in paired.columns]
    if not numeric_features and not categorical_features:
        raise ValueError("No gate features are available in the manifest.")

    train_df = manifest[manifest["split"] == args.train_split].copy()
    val_df = paired[paired["split"] == args.val_split].copy()
    test_df = paired[paired["split"] == args.test_split].copy()
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("Train/validation/test splits must all be non-empty.")

    train_features = train_df[numeric_features + categorical_features].copy()
    train_y = train_df["future_is_net_growth"].to_numpy(dtype=int)
    model = make_pipeline(sk, numeric_features, categorical_features)
    model.fit(train_features, train_y)

    for frame in [val_df, test_df]:
        frame["gate_prob_net_growth"] = model.predict_proba(frame[numeric_features + categorical_features])[:, 1]

    thresholds = [float(x.strip()) for x in args.threshold_grid.split(",") if x.strip()]
    threshold_rows = []
    for threshold in thresholds:
        val_policy = apply_policy(val_df, args.model_name, threshold)
        row = summarize_policy(val_policy, args.model_name, ["split"]).iloc[0].to_dict()
        row["threshold"] = threshold
        threshold_rows.append(row)
    threshold_df = pd.DataFrame(threshold_rows).sort_values(
        ["gated_gap_vs_locf", "gated_gap_vs_model", "gated_mean"], ascending=False
    )
    selected_threshold = float(threshold_df.iloc[0]["threshold"])

    all_policy = pd.concat(
        [
            apply_policy(val_df, args.model_name, selected_threshold),
            apply_policy(test_df, args.model_name, selected_threshold),
        ],
        ignore_index=True,
    )
    classifier_rows = []
    for split, part in all_policy.groupby("split", observed=True):
        classifier_rows.append(
            {"split": split, **classifier_metrics(sk, part["future_is_net_growth"].to_numpy(dtype=int), part["gate_prob_net_growth"].to_numpy(), selected_threshold)}
        )
    classifier_df = pd.DataFrame(classifier_rows)

    summary = summarize_policy(all_policy, args.model_name, ["split"])
    by_direction = summarize_policy(all_policy, args.model_name, ["split", "net_direction"])
    by_patient = summarize_policy(all_policy, args.model_name, ["split", "patient_id"])
    boot = bootstrap_patient_ci(all_policy, args.model_name, args.n_bootstrap, args.seed)

    all_policy.to_csv(output_dir / "direction_gated_policy_samples.csv", index=False)
    threshold_df.to_csv(output_dir / "direction_gated_threshold_selection.csv", index=False)
    classifier_df.to_csv(output_dir / "direction_gate_classifier_metrics.csv", index=False)
    summary.to_csv(output_dir / "direction_gated_policy_summary.csv", index=False)
    by_direction.to_csv(output_dir / "direction_gated_policy_by_net_direction.csv", index=False)
    by_patient.to_csv(output_dir / "direction_gated_policy_by_patient.csv", index=False)
    boot.to_csv(output_dir / "direction_gated_policy_patient_bootstrap.csv", index=False)
    with (output_dir / "direction_gated_policy_report.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "selected_threshold": selected_threshold,
                "features": {"numeric": numeric_features, "categorical": categorical_features},
                "train_split": args.train_split,
                "val_split": args.val_split,
                "test_split": args.test_split,
                "n_train_windows": int(len(train_df)),
                "n_val_windows": int(len(val_df)),
                "n_test_windows": int(len(test_df)),
                "output_dir": str(output_dir),
            },
            f,
            indent=2,
        )

    print(
        json.dumps(
            {
                "selected_threshold": selected_threshold,
                "features": {"numeric": numeric_features, "categorical": categorical_features},
                "summary_csv": str(output_dir / "direction_gated_policy_summary.csv"),
                "classifier_csv": str(output_dir / "direction_gate_classifier_metrics.csv"),
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
