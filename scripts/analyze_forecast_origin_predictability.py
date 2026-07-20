#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

KEY_COLS = ["split", "patient_id", "input_idx", "target_idx", "horizon"]
EPS = 1e-8

DEFAULT_FEATURES = [
    "log_delta_days",
    "log_input_span_days",
    "log_input_volume_vox",
    "current_treatment",
    "treatment_changed_in_input",
    "log_previous_growth_volume_vox",
    "log_previous_loss_volume_vox",
    "previous_growth_ratio",
]

TARGET_DERIVED_BLOCKLIST = {
    "target_idx",
    "target_day",
    "target_treatment",
    "target_treatment_changed",
    "target_volume_vox",
    "persistent_volume_vox",
    "union_volume_vox",
    "new_growth_volume_vox",
    "growth_volume_vox",
    "loss_volume_vox",
    "net_delta_volume_vox",
    "absolute_change_volume_vox",
    "relative_new_growth",
    "relative_loss",
    "relative_net_growth",
    "relative_absolute_change",
    "relative_absolute_change_rate_per_day",
    "locf_dice",
    "transition_type",
    "net_direction",
    "absolute_growth_bin",
    "boundary_growth_volume_vox",
    "distant_growth_volume_vox",
    "boundary_loss_volume_vox",
    "core_loss_volume_vox",
    "boundary_growth_fraction",
    "distant_growth_fraction",
    "boundary_loss_fraction",
    "core_loss_fraction",
}

TARGETS = [
    "mixed_growth_loss",
    "distant_growth_present",
    "high_transition_burden",
    "locf_breakdown",
    "high_change_rate",
]


def parse_csv(payload: str | None) -> List[str]:
    if payload is None:
        return []
    return [x.strip() for x in payload.split(",") if x.strip()]


def resolve_samples_path(args: argparse.Namespace) -> Path:
    if args.samples_csv:
        path = Path(args.samples_csv)
    elif args.taxonomy_dir:
        path = Path(args.taxonomy_dir) / "transition_taxonomy_samples.csv"
    else:
        raise ValueError("Provide either --samples_csv or --taxonomy_dir.")
    if not path.exists():
        raise FileNotFoundError(f"Transition taxonomy samples not found: {path}")
    return path


def normalize_transition_samples(samples: pd.DataFrame) -> pd.DataFrame:
    out = samples.copy()
    if "input_idx" not in out.columns and "input_end_idx" in out.columns:
        out["input_idx"] = out["input_end_idx"]
    if "split" not in out.columns:
        out["split"] = "all"
    for col in ["input_idx", "target_idx", "horizon"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    if "relative_absolute_change" not in out.columns and {"relative_new_growth", "relative_loss"}.issubset(out.columns):
        out["relative_absolute_change"] = pd.to_numeric(out["relative_new_growth"], errors="coerce") + pd.to_numeric(
            out["relative_loss"], errors="coerce"
        )
    if "relative_absolute_change_rate_per_day" not in out.columns and {"relative_absolute_change", "delta_days"}.issubset(
        out.columns
    ):
        out["relative_absolute_change_rate_per_day"] = pd.to_numeric(out["relative_absolute_change"], errors="coerce") / np.maximum(
            pd.to_numeric(out["delta_days"], errors="coerce"), EPS
        )
    return out


def normalize_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    out = manifest.copy()
    if "input_idx" not in out.columns and "input_end_idx" in out.columns:
        out["input_idx"] = out["input_end_idx"]
    if "split" not in out.columns:
        out["split"] = "all"
    if "current_treatment" not in out.columns and "input_end_treatment" in out.columns:
        out["current_treatment"] = out["input_end_treatment"]
    return out


def merge_manifest_features(samples: pd.DataFrame, manifest_csv: str | None) -> pd.DataFrame:
    if not manifest_csv:
        return samples
    manifest_path = Path(manifest_csv)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    manifest = normalize_manifest(pd.read_csv(manifest_path))
    keys = [c for c in ["patient_id", "input_idx", "target_idx", "horizon", "split"] if c in samples.columns and c in manifest.columns]
    if len(keys) < 3:
        raise ValueError(f"Could not find enough shared merge keys. Found: {keys}")

    useful = [
        "patient_id",
        "input_idx",
        "target_idx",
        "horizon",
        "split",
        "delta_days",
        "input_span_days",
        "input_volume_vox",
        "current_treatment",
        "input_end_treatment",
        "treatment_changed_in_input",
        "previous_growth_volume_vox",
        "previous_loss_volume_vox",
        "previous_growth_ratio",
    ]
    keep = [c for c in useful if c in manifest.columns]
    merged = samples.merge(manifest[keep].drop_duplicates(keys), on=keys, how="left", suffixes=("", "__manifest"))
    for col in keep:
        alt = f"{col}__manifest"
        if alt in merged.columns:
            if col in merged.columns:
                merged[col] = merged[col].where(merged[col].notna(), merged[alt])
            else:
                merged[col] = merged[alt]
            merged = merged.drop(columns=[alt])
    if "current_treatment" not in merged.columns and "input_end_treatment" in merged.columns:
        merged["current_treatment"] = merged["input_end_treatment"]
    return merged


def add_origin_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = [
        "delta_days",
        "input_span_days",
        "input_volume_vox",
        "current_treatment",
        "input_end_treatment",
        "treatment_changed_in_input",
        "previous_growth_volume_vox",
        "previous_loss_volume_vox",
        "previous_growth_ratio",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "current_treatment" not in out.columns and "input_end_treatment" in out.columns:
        out["current_treatment"] = out["input_end_treatment"]
    if "treatment_changed_in_input" not in out.columns:
        out["treatment_changed_in_input"] = 0.0
    for src, dst in [
        ("delta_days", "log_delta_days"),
        ("input_span_days", "log_input_span_days"),
        ("input_volume_vox", "log_input_volume_vox"),
        ("previous_growth_volume_vox", "log_previous_growth_volume_vox"),
        ("previous_loss_volume_vox", "log_previous_loss_volume_vox"),
    ]:
        if src in out.columns:
            out[dst] = np.log1p(np.clip(pd.to_numeric(out[src], errors="coerce"), a_min=0, a_max=None))
    return out


def add_transition_targets(
    df: pd.DataFrame,
    train_split: str,
    growth_loss_threshold: float,
    distant_growth_threshold: float,
    high_burden_quantile: float,
    high_change_rate_quantile: float,
    locf_breakdown_threshold: float,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    out = df.copy()
    train = out[out["split"].astype(str) == train_split]
    if train.empty:
        raise ValueError(f"No rows found for train_split='{train_split}'.")

    thresholds: Dict[str, float] = {
        "mixed_growth_loss_growth_threshold": float(growth_loss_threshold),
        "mixed_growth_loss_loss_threshold": float(growth_loss_threshold),
        "distant_growth_threshold": float(distant_growth_threshold),
        "locf_breakdown_threshold": float(locf_breakdown_threshold),
    }

    required_for_mixed = {"relative_new_growth", "relative_loss"}
    if required_for_mixed.issubset(out.columns):
        out["label_mixed_growth_loss"] = (
            (pd.to_numeric(out["relative_new_growth"], errors="coerce") >= growth_loss_threshold)
            & (pd.to_numeric(out["relative_loss"], errors="coerce") >= growth_loss_threshold)
        ).astype(int)

    if "distant_growth_fraction" in out.columns:
        out["label_distant_growth_present"] = (
            pd.to_numeric(out["distant_growth_fraction"], errors="coerce") >= distant_growth_threshold
        ).astype(int)

    if "relative_absolute_change" in out.columns:
        burden_thr = float(pd.to_numeric(train["relative_absolute_change"], errors="coerce").quantile(high_burden_quantile))
        thresholds["high_transition_burden_threshold"] = burden_thr
        thresholds["high_transition_burden_train_quantile"] = float(high_burden_quantile)
        out["label_high_transition_burden"] = (
            pd.to_numeric(out["relative_absolute_change"], errors="coerce") >= burden_thr
        ).astype(int)

    if "locf_dice" in out.columns:
        out["label_locf_breakdown"] = (pd.to_numeric(out["locf_dice"], errors="coerce") <= locf_breakdown_threshold).astype(int)

    if "relative_absolute_change_rate_per_day" in out.columns:
        rate_thr = float(
            pd.to_numeric(train["relative_absolute_change_rate_per_day"], errors="coerce").quantile(high_change_rate_quantile)
        )
        thresholds["high_change_rate_threshold"] = rate_thr
        thresholds["high_change_rate_train_quantile"] = float(high_change_rate_quantile)
        out["label_high_change_rate"] = (
            pd.to_numeric(out["relative_absolute_change_rate_per_day"], errors="coerce") >= rate_thr
        ).astype(int)

    return out, thresholds


def available_features(df: pd.DataFrame, requested: Iterable[str]) -> List[str]:
    features = []
    for feature in requested:
        if feature in TARGET_DERIVED_BLOCKLIST:
            raise ValueError(
                f"Feature '{feature}' is target-derived and would leak future information. "
                "Use only forecast-origin features."
            )
        if feature in df.columns:
            features.append(feature)
    if not features:
        raise ValueError("No requested origin-known features are available.")
    non_numeric = [f for f in features if not pd.api.types.is_numeric_dtype(df[f])]
    if non_numeric:
        raise ValueError(f"Only numeric predictors are supported for this audit. Non-numeric: {non_numeric}")
    return features


def safe_metric(fn, y_true, values) -> float:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(fn(y_true, values))
    except Exception:
        return float("nan")


def confusion_metrics(y: np.ndarray, score: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    pred = (score >= threshold).astype(int)
    pos = y == 1
    neg = y == 0
    return {
        "accuracy": safe_metric(accuracy_score, y, pred),
        "balanced_accuracy": safe_metric(balanced_accuracy_score, y, pred),
        "roc_auc": safe_metric(roc_auc_score, y, score),
        "average_precision": safe_metric(average_precision_score, y, score),
        "precision": safe_metric(lambda yt, yp: precision_score(yt, yp, zero_division=0), y, pred),
        "recall": safe_metric(lambda yt, yp: recall_score(yt, yp, zero_division=0), y, pred),
        "f1": safe_metric(lambda yt, yp: f1_score(yt, yp, zero_division=0), y, pred),
        "specificity": float(((pred == 0) & neg).sum() / max(1, neg.sum())),
        "false_positive_rate": float(((pred == 1) & neg).sum() / max(1, neg.sum())),
        "false_negative_rate": float(((pred == 0) & pos).sum() / max(1, pos.sum())),
        "predicted_positive_rate": float(pred.mean()) if len(pred) else float("nan"),
    }


def fit_predict_task(
    data: pd.DataFrame,
    target: str,
    features: List[str],
    train_split: str,
    eval_splits: List[str],
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier, export_text

    label_col = f"label_{target}"
    train = data[data["split"].astype(str) == train_split].copy()
    if label_col not in data.columns:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), f"Skipped {target}: label column missing."
    train = train[train[label_col].notna()].copy()
    if train.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), f"Skipped {target}: empty train split."
    if train[label_col].astype(int).nunique() < 2:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), f"Skipped {target}: train labels contain one class."

    pre = ColumnTransformer(
        [("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), features)],
        remainder="drop",
    )
    models = {
        "logistic": LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed),
        "tree_depth3": DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, class_weight="balanced", random_state=seed),
    }

    pred_rows = []
    summary_rows = []
    importance_rows = []
    tree_rules = []
    for model_name, clf in models.items():
        model = Pipeline([("pre", pre), ("clf", clf)])
        model.fit(train[features], train[label_col].astype(int))
        if model_name == "logistic":
            for feature, coef in zip(features, model.named_steps["clf"].coef_[0]):
                importance_rows.append(
                    {
                        "target": target,
                        "model": model_name,
                        "feature": feature,
                        "weight": float(coef),
                        "abs_weight": float(abs(coef)),
                    }
                )
        else:
            tree = model.named_steps["clf"]
            for feature, imp in zip(features, tree.feature_importances_):
                importance_rows.append(
                    {
                        "target": target,
                        "model": model_name,
                        "feature": feature,
                        "weight": float(imp),
                        "abs_weight": float(abs(imp)),
                    }
                )
            tree_rules.append(f"\n## {target} / {model_name}\n")
            tree_rules.append(export_text(tree, feature_names=features))

        for split in eval_splits:
            part = data[(data["split"].astype(str) == split) & data[label_col].notna()].copy()
            if part.empty:
                continue
            y = part[label_col].astype(int).to_numpy()
            if len(np.unique(y)) < 2:
                score = model.predict_proba(part[features])[:, 1]
            else:
                score = model.predict_proba(part[features])[:, 1]
            metrics = confusion_metrics(y, score, threshold=0.5)
            row = {
                "target": target,
                "model": model_name,
                "split": split,
                "n_samples": int(len(part)),
                "n_patients": int(part["patient_id"].nunique()) if "patient_id" in part.columns else 0,
                "positive_rate": float(np.mean(y)),
                "train_positive_rate": float(train[label_col].astype(int).mean()),
            }
            row.update(metrics)
            summary_rows.append(row)

            pred_part = part[[c for c in KEY_COLS if c in part.columns]].copy()
            pred_part["target"] = target
            pred_part["model"] = model_name
            pred_part["label"] = y
            pred_part["score"] = score
            pred_part["pred_0p5"] = (score >= 0.5).astype(int)
            for col in ["relative_absolute_change", "relative_new_growth", "relative_loss", "distant_growth_fraction", "locf_dice"]:
                if col in part.columns:
                    pred_part[col] = part[col].to_numpy()
            pred_rows.append(pred_part)

    preds = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    importance = pd.DataFrame(importance_rows).sort_values(["target", "model", "abs_weight"], ascending=[True, True, False])
    return preds, summary, importance, "\n".join(tree_rules)


def patient_bootstrap(preds: pd.DataFrame, n_bootstrap: int, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if preds.empty or n_bootstrap <= 0:
        return pd.DataFrame(), pd.DataFrame()
    rng = np.random.default_rng(seed)
    draw_rows = []
    summary_rows = []
    metric_cols = [
        "balanced_accuracy",
        "roc_auc",
        "average_precision",
        "precision",
        "recall",
        "false_positive_rate",
        "false_negative_rate",
    ]
    for (target, model, split), part in preds.groupby(["target", "model", "split"], observed=True):
        patients = np.array(sorted(part["patient_id"].dropna().unique())) if "patient_id" in part.columns else np.array([])
        if len(patients) < 2:
            continue
        patient_parts = {pid: rows for pid, rows in part.groupby("patient_id", observed=True)}
        for draw_idx in range(n_bootstrap):
            sampled_patients = rng.choice(patients, size=len(patients), replace=True)
            sampled = pd.concat([patient_parts[pid] for pid in sampled_patients], ignore_index=True)
            y = sampled["label"].astype(int).to_numpy()
            score = sampled["score"].to_numpy()
            metrics = confusion_metrics(y, score, threshold=0.5)
            row = {
                "target": target,
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
        draws = pd.DataFrame([r for r in draw_rows if r["target"] == target and r["model"] == model and r["split"] == split])
        summary = {
            "target": target,
            "model": model,
            "split": split,
            "n_bootstrap": int(n_bootstrap),
            "n_original_patients": int(len(patients)),
            "n_original_samples": int(len(part)),
        }
        for metric in metric_cols:
            vals = draws[metric].dropna().to_numpy()
            summary[f"{metric}_mean"] = float(np.mean(vals)) if len(vals) else float("nan")
            summary[f"{metric}_ci_low"] = float(np.quantile(vals, 0.025)) if len(vals) else float("nan")
            summary[f"{metric}_ci_high"] = float(np.quantile(vals, 0.975)) if len(vals) else float("nan")
            summary[f"{metric}_valid_draws"] = int(len(vals))
        summary_rows.append(summary)
    return pd.DataFrame(draw_rows), pd.DataFrame(summary_rows)


def target_prevalence(data: pd.DataFrame, targets: List[str]) -> pd.DataFrame:
    rows = []
    for target in targets:
        col = f"label_{target}"
        if col not in data.columns:
            continue
        for split, part in data.groupby("split", observed=True):
            y = part[col].dropna().astype(int)
            if y.empty:
                continue
            rows.append(
                {
                    "target": target,
                    "split": split,
                    "n_samples": int(len(y)),
                    "n_patients": int(part.loc[y.index, "patient_id"].nunique()) if "patient_id" in part.columns else 0,
                    "positive_rate": float(y.mean()),
                    "n_positive": int(y.sum()),
                }
            )
    return pd.DataFrame(rows)


def write_report(
    path: Path,
    features: List[str],
    thresholds: Dict[str, float],
    prevalence: pd.DataFrame,
    summary: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
    importance: pd.DataFrame,
    skipped: List[str],
) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Forecast-Origin Transition Predictability Audit\n\n")
        f.write(
            "This audit asks whether difficult transition states can be anticipated using only information "
            "available at the input scan. It is not a forecasting model and it intentionally excludes target-side "
            "measurements from the predictor set.\n\n"
        )
        f.write("## Origin-known predictors\n\n")
        for feature in features:
            f.write(f"- `{feature}`\n")
        f.write("\n## Target definitions\n\n")
        for key, value in thresholds.items():
            f.write(f"- `{key}`: `{value:.6g}`\n")
        if skipped:
            f.write("\n## Skipped tasks\n\n")
            for item in skipped:
                f.write(f"- {item}\n")
        f.write("\n## Target prevalence\n\n")
        f.write(prevalence.to_markdown(index=False) if not prevalence.empty else "No prevalence rows.")
        f.write("\n\n## Predictability summary\n\n")
        f.write(summary.to_markdown(index=False) if not summary.empty else "No model summary rows.")
        if not bootstrap_summary.empty:
            f.write("\n\n## Patient bootstrap summary\n\n")
            f.write(bootstrap_summary.to_markdown(index=False))
        f.write("\n\n## Feature weights/importances\n\n")
        f.write(importance.head(60).to_markdown(index=False) if not importance.empty else "No feature importance rows.")
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit whether transition states are predictable from forecast-origin information only."
    )
    parser.add_argument("--samples_csv", type=str, default=None)
    parser.add_argument("--taxonomy_dir", type=str, default=None)
    parser.add_argument("--manifest_csv", type=str, default=None)
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--eval_splits", type=str, default="val,test")
    parser.add_argument("--features", type=str, default=",".join(DEFAULT_FEATURES))
    parser.add_argument("--targets", type=str, default=",".join(TARGETS))
    parser.add_argument("--growth_loss_threshold", type=float, default=0.2)
    parser.add_argument("--distant_growth_threshold", type=float, default=0.2)
    parser.add_argument("--high_burden_quantile", type=float, default=0.75)
    parser.add_argument("--high_change_rate_quantile", type=float, default=0.75)
    parser.add_argument("--locf_breakdown_threshold", type=float, default=0.5)
    parser.add_argument("--n_bootstrap", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    samples_path = resolve_samples_path(args)
    data = normalize_transition_samples(pd.read_csv(samples_path))
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
    features = available_features(data, parse_csv(args.features))
    targets = [t for t in parse_csv(args.targets) if t in TARGETS]
    eval_splits = parse_csv(args.eval_splits)

    pred_parts = []
    summary_parts = []
    importance_parts = []
    skipped = []
    rules = []
    for target in targets:
        preds, summary, importance, rule_text = fit_predict_task(
            data=data,
            target=target,
            features=features,
            train_split=args.train_split,
            eval_splits=eval_splits,
            seed=args.seed,
        )
        if preds.empty and summary.empty:
            skipped.append(rule_text)
            continue
        pred_parts.append(preds)
        summary_parts.append(summary)
        importance_parts.append(importance)
        if rule_text.strip():
            rules.append(rule_text)

    predictions = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
    summary = pd.concat(summary_parts, ignore_index=True) if summary_parts else pd.DataFrame()
    importance = pd.concat(importance_parts, ignore_index=True) if importance_parts else pd.DataFrame()
    prevalence = target_prevalence(data, targets)
    bootstrap_draws, bootstrap_summary = patient_bootstrap(predictions, n_bootstrap=args.n_bootstrap, seed=args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_dir / "forecast_origin_predictability_labeled_samples.csv", index=False)
    predictions.to_csv(output_dir / "forecast_origin_predictability_predictions.csv", index=False)
    summary.to_csv(output_dir / "forecast_origin_predictability_summary.csv", index=False)
    prevalence.to_csv(output_dir / "forecast_origin_predictability_prevalence.csv", index=False)
    importance.to_csv(output_dir / "forecast_origin_predictability_feature_weights.csv", index=False)
    if not bootstrap_summary.empty:
        bootstrap_draws.to_csv(output_dir / "forecast_origin_predictability_patient_bootstrap_draws.csv", index=False)
        bootstrap_summary.to_csv(output_dir / "forecast_origin_predictability_patient_bootstrap_summary.csv", index=False)
    with (output_dir / "forecast_origin_predictability_tree_rules.txt").open("w", encoding="utf-8") as f:
        f.write("\n".join(rules) if rules else "No tree rules generated.\n")
    write_report(
        output_dir / "forecast_origin_predictability_report.md",
        features=features,
        thresholds=thresholds,
        prevalence=prevalence,
        summary=summary,
        bootstrap_summary=bootstrap_summary,
        importance=importance,
        skipped=skipped,
    )

    payload = {
        "samples_csv": str(samples_path),
        "manifest_csv": args.manifest_csv,
        "train_split": args.train_split,
        "eval_splits": eval_splits,
        "features": features,
        "targets_requested": targets,
        "targets_skipped": skipped,
        "thresholds": thresholds,
        "n_rows": int(len(data)),
        "n_bootstrap": int(args.n_bootstrap),
        "output_dir": str(output_dir),
        "outputs": {
            "labeled_samples_csv": str(output_dir / "forecast_origin_predictability_labeled_samples.csv"),
            "summary_csv": str(output_dir / "forecast_origin_predictability_summary.csv"),
            "prevalence_csv": str(output_dir / "forecast_origin_predictability_prevalence.csv"),
            "feature_weights_csv": str(output_dir / "forecast_origin_predictability_feature_weights.csv"),
            "patient_bootstrap_summary_csv": str(output_dir / "forecast_origin_predictability_patient_bootstrap_summary.csv")
            if not bootstrap_summary.empty
            else None,
            "tree_rules_txt": str(output_dir / "forecast_origin_predictability_tree_rules.txt"),
            "report_md": str(output_dir / "forecast_origin_predictability_report.md"),
        },
    }
    with (output_dir / "forecast_origin_predictability_summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
