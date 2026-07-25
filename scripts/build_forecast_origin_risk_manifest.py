#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

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
    merge_manifest_features,
    normalize_manifest,
    normalize_transition_samples,
    parse_csv,
    resolve_samples_path,
    safe_metric,
)
from scripts.run_forecast_origin_feature_ablation import FEATURE_SETS  # noqa: E402


KEY_COLS = ["patient_id", "input_idx", "target_idx", "horizon"]


def normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "input_idx" not in out.columns and "input_end_idx" in out.columns:
        out["input_idx"] = out["input_end_idx"]
    for col in ["input_idx", "target_idx", "horizon"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    return out


def parse_risk_specs(payload: str) -> List[Tuple[str, str]]:
    specs: List[Tuple[str, str]] = []
    for item in parse_csv(payload):
        if ":" not in item:
            raise ValueError(f"Risk spec must have form target:feature_set, got: {item}")
        target, feature_set = [x.strip() for x in item.split(":", 1)]
        if target not in TARGETS:
            raise ValueError(f"Unknown target in risk spec: {target}. Available: {TARGETS}")
        if feature_set not in FEATURE_SETS:
            raise ValueError(f"Unknown feature set in risk spec: {feature_set}. Available: {sorted(FEATURE_SETS)}")
        specs.append((target, feature_set))
    if not specs:
        raise ValueError("At least one risk spec is required.")
    return specs


def risk_col_name(target: str, feature_set: str) -> str:
    return f"risk_{target}_{feature_set}"


def fit_score_logistic(train: pd.DataFrame, score_part: pd.DataFrame, label_col: str, features: List[str], seed: int) -> np.ndarray:
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
    return model.predict_proba(score_part[features])[:, 1]


def add_crossfit_train_scores(
    data: pd.DataFrame,
    target: str,
    feature_set: str,
    train_split: str,
    n_folds: int,
    seed: int,
) -> Tuple[pd.Series, List[dict]]:
    from sklearn.model_selection import GroupKFold

    label_col = f"label_{target}"
    score_col = risk_col_name(target, feature_set)
    scores = pd.Series(np.nan, index=data.index, dtype=float)
    rows: List[dict] = []
    train = data[(data["split"].astype(str) == train_split) & data[label_col].notna()].copy()
    if train.empty or train[label_col].astype(int).nunique() < 2:
        rows.append({"target": target, "feature_set": feature_set, "stage": "train_crossfit", "status": "skipped_one_class"})
        return scores, rows
    features = available_features(data, FEATURE_SETS[feature_set])
    patients = train["patient_id"].astype(str).to_numpy()
    unique_patients = np.unique(patients)
    folds = min(int(n_folds), len(unique_patients))
    if folds < 2:
        rows.append({"target": target, "feature_set": feature_set, "stage": "train_crossfit", "status": "skipped_too_few_patients"})
        return scores, rows

    splitter = GroupKFold(n_splits=folds)
    for fold_idx, (tr_pos, held_pos) in enumerate(splitter.split(train, train[label_col].astype(int), groups=patients)):
        fit_part = train.iloc[tr_pos].copy()
        held_part = train.iloc[held_pos].copy()
        row = {
            "target": target,
            "feature_set": feature_set,
            "stage": "train_crossfit",
            "fold_idx": int(fold_idx),
            "n_fit_samples": int(len(fit_part)),
            "n_heldout_samples": int(len(held_part)),
            "n_fit_patients": int(fit_part["patient_id"].nunique()),
            "n_heldout_patients": int(held_part["patient_id"].nunique()),
            "fit_positive_rate": float(fit_part[label_col].astype(int).mean()),
            "heldout_positive_rate": float(held_part[label_col].astype(int).mean()),
        }
        if fit_part[label_col].astype(int).nunique() < 2:
            row["status"] = "skipped_fold_one_class"
            rows.append(row)
            continue
        try:
            pred = fit_score_logistic(fit_part, held_part, label_col, features, seed=seed + fold_idx)
            scores.loc[held_part.index] = pred
            row["status"] = "ok"
        except Exception as exc:  # noqa: BLE001 - keep failed folds visible.
            row["status"] = f"fit_failed:{type(exc).__name__}:{exc}"
        rows.append(row)
    return scores.rename(score_col), rows


def add_eval_scores(
    data: pd.DataFrame,
    target: str,
    feature_set: str,
    train_split: str,
    eval_splits: List[str],
    seed: int,
) -> Tuple[pd.Series, List[dict]]:
    label_col = f"label_{target}"
    score_col = risk_col_name(target, feature_set)
    scores = pd.Series(np.nan, index=data.index, dtype=float)
    rows: List[dict] = []
    train = data[(data["split"].astype(str) == train_split) & data[label_col].notna()].copy()
    if train.empty or train[label_col].astype(int).nunique() < 2:
        rows.append({"target": target, "feature_set": feature_set, "stage": "eval_score", "status": "skipped_train_one_class"})
        return scores, rows
    features = available_features(data, FEATURE_SETS[feature_set])
    for split in eval_splits:
        eval_part = data[(data["split"].astype(str) == split) & data[label_col].notna()].copy()
        row = {
            "target": target,
            "feature_set": feature_set,
            "stage": "eval_score",
            "split": split,
            "n_fit_samples": int(len(train)),
            "n_score_samples": int(len(eval_part)),
            "n_fit_patients": int(train["patient_id"].nunique()),
            "n_score_patients": int(eval_part["patient_id"].nunique()) if not eval_part.empty else 0,
            "fit_positive_rate": float(train[label_col].astype(int).mean()),
            "score_positive_rate": float(eval_part[label_col].astype(int).mean()) if not eval_part.empty else np.nan,
        }
        if eval_part.empty:
            row["status"] = "empty_split"
            rows.append(row)
            continue
        try:
            pred = fit_score_logistic(train, eval_part, label_col, features, seed=seed)
            scores.loc[eval_part.index] = pred
            row["status"] = "ok"
        except Exception as exc:  # noqa: BLE001
            row["status"] = f"fit_failed:{type(exc).__name__}:{exc}"
        rows.append(row)
    return scores.rename(score_col), rows


def score_summary(data: pd.DataFrame, specs: List[Tuple[str, str]]) -> pd.DataFrame:
    from sklearn.metrics import average_precision_score, roc_auc_score

    rows = []
    for target, feature_set in specs:
        label_col = f"label_{target}"
        score_col = risk_col_name(target, feature_set)
        if label_col not in data.columns or score_col not in data.columns:
            continue
        for split, part in data.groupby("split", observed=True):
            y = pd.to_numeric(part[label_col], errors="coerce")
            score = pd.to_numeric(part[score_col], errors="coerce")
            mask = y.notna() & score.notna()
            yv = y[mask].astype(int).to_numpy()
            sv = score[mask].to_numpy(dtype=float)
            pred = (sv >= 0.5).astype(int) if len(sv) else np.array([])
            pos = yv == 1
            neg = yv == 0
            rows.append(
                {
                    "target": target,
                    "feature_set": feature_set,
                    "risk_column": score_col,
                    "split": split,
                    "n_samples": int(len(yv)),
                    "n_patients": int(part.loc[mask, "patient_id"].nunique()) if len(yv) else 0,
                    "positive_rate": float(yv.mean()) if len(yv) else np.nan,
                    "mean_risk_score": float(np.mean(sv)) if len(sv) else np.nan,
                    "roc_auc": safe_metric(roc_auc_score, yv, sv) if len(np.unique(yv)) > 1 else np.nan,
                    "average_precision": safe_metric(average_precision_score, yv, sv) if len(np.unique(yv)) > 1 else np.nan,
                    "recall_at_0p5": float(((pred == 1) & pos).sum() / max(1, pos.sum())) if len(yv) else np.nan,
                    "false_positive_rate_at_0p5": float(((pred == 1) & neg).sum() / max(1, neg.sum())) if len(yv) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a leakage-controlled forecast-origin risk-augmented manifest.")
    parser.add_argument("--samples_csv", type=str, default=None)
    parser.add_argument("--taxonomy_dir", type=str, default=None)
    parser.add_argument("--manifest_csv", type=str, required=True)
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--eval_splits", type=str, default="val,test")
    parser.add_argument(
        "--risk_specs",
        type=str,
        default=(
            "mixed_growth_loss:history_only,"
            "high_transition_burden:full_origin,"
            "distant_growth_present:no_interval,"
            "locf_breakdown:history_only"
        ),
    )
    parser.add_argument("--train_cv_folds", type=int, default=5)
    parser.add_argument("--growth_loss_threshold", type=float, default=0.2)
    parser.add_argument("--distant_growth_threshold", type=float, default=0.2)
    parser.add_argument("--high_burden_quantile", type=float, default=0.75)
    parser.add_argument("--high_change_rate_quantile", type=float, default=0.75)
    parser.add_argument("--locf_breakdown_threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest_csv)
    manifest = normalize_keys(normalize_manifest(pd.read_csv(manifest_path)))
    samples_path = resolve_samples_path(args)
    data = normalize_keys(normalize_transition_samples(pd.read_csv(samples_path)))
    data = merge_manifest_features(data, args.manifest_csv)
    data = add_origin_features(data)
    data, thresholds = add_transition_targets(
        data,
        train_split=args.train_split,
        growth_loss_threshold=args.growth_loss_threshold,
        distant_growth_threshold=args.distant_growth_threshold,
        high_burden_quantile=args.high_burden_quantile,
        high_change_rate_quantile=args.high_change_rate_quantile,
        locf_breakdown_threshold=args.locf_breakdown_threshold,
    )

    specs = parse_risk_specs(args.risk_specs)
    eval_splits = parse_csv(args.eval_splits)
    status_rows: List[dict] = []
    for target, feature_set in specs:
        train_scores, rows = add_crossfit_train_scores(
            data=data,
            target=target,
            feature_set=feature_set,
            train_split=args.train_split,
            n_folds=args.train_cv_folds,
            seed=args.seed,
        )
        status_rows.extend(rows)
        eval_scores, rows = add_eval_scores(
            data=data,
            target=target,
            feature_set=feature_set,
            train_split=args.train_split,
            eval_splits=eval_splits,
            seed=args.seed,
        )
        status_rows.extend(rows)
        col = risk_col_name(target, feature_set)
        data[col] = train_scores.combine_first(eval_scores)

    label_cols = [f"label_{target}" for target, _ in specs if f"label_{target}" in data.columns]
    risk_cols = [risk_col_name(target, feature_set) for target, feature_set in specs]
    merge_cols = KEY_COLS + [c for c in ["split"] if c in manifest.columns and c in data.columns]
    keep_cols = KEY_COLS + ["split"] + risk_cols + label_cols
    keep_cols = [c for c in keep_cols if c in data.columns]
    scored = data[keep_cols].drop_duplicates(merge_cols)
    risk_manifest = manifest.merge(scored, on=merge_cols, how="left", suffixes=("", "__risk"))
    for col in risk_cols + label_cols:
        alt = f"{col}__risk"
        if alt in risk_manifest.columns:
            risk_manifest[col] = risk_manifest[col].where(risk_manifest[col].notna(), risk_manifest[alt])
            risk_manifest = risk_manifest.drop(columns=[alt])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    risk_manifest_path = output_dir / "forecast_origin_risk_manifest.csv"
    risk_manifest.to_csv(risk_manifest_path, index=False)
    data.to_csv(output_dir / "forecast_origin_risk_samples.csv", index=False)
    pd.DataFrame(status_rows).to_csv(output_dir / "forecast_origin_risk_fit_status.csv", index=False)
    summary = score_summary(data, specs)
    summary.to_csv(output_dir / "forecast_origin_risk_score_summary.csv", index=False)
    with (output_dir / "forecast_origin_risk_manifest_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "samples_path": str(samples_path),
                "manifest_csv": str(manifest_path),
                "train_split": args.train_split,
                "eval_splits": eval_splits,
                "risk_specs": [{"target": t, "feature_set": fs, "column": risk_col_name(t, fs)} for t, fs in specs],
                "thresholds": thresholds,
                "train_cv_folds": int(args.train_cv_folds),
                "seed": int(args.seed),
                "n_manifest_rows": int(len(manifest)),
                "n_scored_rows": int(len(data)),
            },
            f,
            indent=2,
        )
    with (output_dir / "forecast_origin_risk_manifest_report.md").open("w", encoding="utf-8") as f:
        f.write("# Forecast-Origin Risk Manifest\n\n")
        f.write(
            "This manifest adds leakage-controlled forecast-origin risk scores to each longitudinal window. "
            "Training-window scores are patient-cross-fitted within the training split; validation/test scores "
            "are produced by risk models fit only on the training split.\n\n"
        )
        f.write("## Risk Columns\n\n")
        for target, feature_set in specs:
            f.write(f"- `{risk_col_name(target, feature_set)}` from `{target}` using `{feature_set}` features\n")
        f.write("\n## Score Summary\n\n")
        f.write(summary.to_markdown(index=False) if not summary.empty else "No summary rows.")
        f.write("\n")
    print(
        json.dumps(
            {
                "risk_manifest_csv": str(risk_manifest_path),
                "risk_columns": risk_cols,
                "summary_csv": str(output_dir / "forecast_origin_risk_score_summary.csv"),
                "status_csv": str(output_dir / "forecast_origin_risk_fit_status.csv"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
