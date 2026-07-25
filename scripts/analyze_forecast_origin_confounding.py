#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

FEATURE_SETS: Dict[str, List[str]] = {
    "full_origin": [
        "log_delta_days",
        "log_input_span_days",
        "log_input_volume_vox",
        "current_treatment",
        "treatment_changed_in_input",
        "log_previous_growth_volume_vox",
        "log_previous_loss_volume_vox",
        "previous_growth_ratio",
    ],
    "no_interval": [
        "log_input_volume_vox",
        "current_treatment",
        "treatment_changed_in_input",
        "log_previous_growth_volume_vox",
        "log_previous_loss_volume_vox",
        "previous_growth_ratio",
    ],
    "time_only": ["log_delta_days", "log_input_span_days"],
    "history_only": [
        "log_input_volume_vox",
        "log_previous_growth_volume_vox",
        "log_previous_loss_volume_vox",
        "previous_growth_ratio",
    ],
    "treatment_only": ["current_treatment", "treatment_changed_in_input"],
}

TARGETS = [
    "mixed_growth_loss",
    "distant_growth_present",
    "high_transition_burden",
    "locf_breakdown",
    "high_change_rate",
]

KEY_COLS = ["split", "patient_id", "input_idx", "target_idx", "horizon"]


def parse_csv(payload: str | None) -> List[str]:
    if payload is None:
        return []
    return [x.strip() for x in payload.split(",") if x.strip()]


def safe_metric(fn, y_true, values) -> float:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(fn(y_true, values))
    except Exception:
        return float("nan")


def classification_metrics(y: np.ndarray, score: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
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
        "false_positive_rate": float(((pred == 1) & neg).sum() / max(1, neg.sum())),
        "false_negative_rate": float(((pred == 0) & pos).sum() / max(1, pos.sum())),
        "predicted_positive_rate": float(pred.mean()) if len(pred) else float("nan"),
    }


def resolve_labeled_samples(args: argparse.Namespace) -> Path:
    if args.labeled_samples_csv:
        path = Path(args.labeled_samples_csv)
    elif args.ablation_dir:
        path = Path(args.ablation_dir) / args.reference_feature_set / "forecast_origin_predictability_labeled_samples.csv"
    else:
        raise ValueError("Provide either --labeled_samples_csv or --ablation_dir.")
    if not path.exists():
        raise FileNotFoundError(f"Labeled samples CSV not found: {path}")
    return path


def label_columns(data: pd.DataFrame, targets: Iterable[str]) -> List[Tuple[str, str]]:
    pairs = []
    for target in targets:
        col = f"label_{target}"
        if col in data.columns:
            pairs.append((target, col))
    if not pairs:
        raise ValueError("No requested label columns found in labeled samples.")
    return pairs


def add_all_split(data: pd.DataFrame) -> pd.DataFrame:
    all_rows = data.copy()
    all_rows["split"] = "all"
    return pd.concat([data, all_rows], ignore_index=True)


def patient_concentration(data: pd.DataFrame, target_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
    rows = []
    data2 = add_all_split(data)
    for target, col in target_pairs:
        for split, part in data2.groupby("split", observed=True):
            y = part[col].dropna().astype(int)
            if y.empty:
                continue
            patient_pos = part.loc[y.index].groupby("patient_id", observed=True)[col].sum().sort_values(ascending=False)
            total_pos = float(patient_pos.sum())
            shares = patient_pos[patient_pos > 0] / total_pos if total_pos > 0 else pd.Series(dtype=float)
            rows.append(
                {
                    "target": target,
                    "split": split,
                    "n_samples": int(len(y)),
                    "n_patients": int(part.loc[y.index, "patient_id"].nunique()),
                    "positive_rate": float(y.mean()),
                    "n_positive": int(y.sum()),
                    "n_positive_patients": int((patient_pos > 0).sum()),
                    "positive_patient_fraction": float((patient_pos > 0).sum() / max(1, patient_pos.shape[0])),
                    "top_positive_patient": str(patient_pos.index[0]) if total_pos > 0 and len(patient_pos) else "",
                    "top_positive_patient_count": int(patient_pos.iloc[0]) if total_pos > 0 and len(patient_pos) else 0,
                    "max_positive_patient_share": float(shares.iloc[0]) if len(shares) else float("nan"),
                    "effective_positive_patients": float(1.0 / np.sum(np.square(shares))) if len(shares) else 0.0,
                }
            )
    return pd.DataFrame(rows)


def patient_target_table(data: pd.DataFrame, target_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
    rows = []
    for target, col in target_pairs:
        for (split, patient_id), part in data.groupby(["split", "patient_id"], observed=True):
            y = part[col].dropna().astype(int)
            if y.empty:
                continue
            rows.append(
                {
                    "target": target,
                    "split": split,
                    "patient_id": patient_id,
                    "n_samples": int(len(y)),
                    "positive_rate": float(y.mean()),
                    "n_positive": int(y.sum()),
                    "current_treatment_mean": float(pd.to_numeric(part.get("current_treatment"), errors="coerce").mean())
                    if "current_treatment" in part
                    else float("nan"),
                    "treatment_changed_in_input_rate": float(
                        pd.to_numeric(part.get("treatment_changed_in_input"), errors="coerce").mean()
                    )
                    if "treatment_changed_in_input" in part
                    else float("nan"),
                    "delta_days_mean": float(pd.to_numeric(part.get("delta_days"), errors="coerce").mean())
                    if "delta_days" in part
                    else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def treatment_distribution(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    data2 = add_all_split(data)
    for split, part in data2.groupby("split", observed=True):
        row = {
            "split": split,
            "n_samples": int(len(part)),
            "n_patients": int(part["patient_id"].nunique()) if "patient_id" in part.columns else 0,
        }
        for col in ["current_treatment", "treatment_changed_in_input"]:
            if col in part.columns:
                vals = pd.to_numeric(part[col], errors="coerce")
                row[f"{col}_mean"] = float(vals.mean())
                row[f"{col}_n_nonmissing"] = int(vals.notna().sum())
        rows.append(row)
    return pd.DataFrame(rows)


def treatment_target_gaps(data: pd.DataFrame, target_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
    rows = []
    data2 = add_all_split(data)
    treatment_cols = [c for c in ["current_treatment", "treatment_changed_in_input"] if c in data2.columns]
    for target, label_col in target_pairs:
        for split, split_part in data2.groupby("split", observed=True):
            for treatment_col in treatment_cols:
                part = split_part[["patient_id", label_col, treatment_col]].dropna().copy()
                if part.empty:
                    continue
                part[label_col] = part[label_col].astype(int)
                part[treatment_col] = pd.to_numeric(part[treatment_col], errors="coerce")
                part = part.dropna()
                if part.empty:
                    continue
                for value, group in part.groupby(treatment_col, observed=True):
                    rows.append(
                        {
                            "target": target,
                            "split": split,
                            "treatment_col": treatment_col,
                            "treatment_value": value,
                            "n_samples": int(len(group)),
                            "n_patients": int(group["patient_id"].nunique()),
                            "positive_rate": float(group[label_col].mean()),
                            "n_positive": int(group[label_col].sum()),
                        }
                    )
                values = sorted(part[treatment_col].dropna().unique())
                if 0 in values and 1 in values:
                    g0 = part[part[treatment_col] == 0]
                    g1 = part[part[treatment_col] == 1]
                    rows.append(
                        {
                            "target": target,
                            "split": split,
                            "treatment_col": treatment_col,
                            "treatment_value": "1_minus_0_gap",
                            "n_samples": int(len(part)),
                            "n_patients": int(part["patient_id"].nunique()),
                            "positive_rate": float(g1[label_col].mean() - g0[label_col].mean()),
                            "n_positive": int(part[label_col].sum()),
                        }
                    )
    return pd.DataFrame(rows)


def fit_lopo_predictions(
    data: pd.DataFrame,
    target_pairs: List[Tuple[str, str]],
    feature_sets: Dict[str, List[str]],
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pred_rows = []
    patient_rows = []
    skipped_rows = []
    patients = sorted(data["patient_id"].dropna().unique())
    for target, label_col in target_pairs:
        for feature_set, requested_features in feature_sets.items():
            features = [f for f in requested_features if f in data.columns]
            if not features:
                skipped_rows.append({"target": target, "feature_set": feature_set, "reason": "no_features"})
                continue
            for heldout_patient in patients:
                test = data[data["patient_id"] == heldout_patient].copy()
                train = data[data["patient_id"] != heldout_patient].copy()
                train = train[train[label_col].notna()].copy()
                test = test[test[label_col].notna()].copy()
                if train.empty or test.empty:
                    continue
                y_train = train[label_col].astype(int)
                if y_train.nunique() < 2:
                    skipped_rows.append(
                        {
                            "target": target,
                            "feature_set": feature_set,
                            "heldout_patient": heldout_patient,
                            "reason": "train_one_class",
                        }
                    )
                    continue
                model = Pipeline(
                    [
                        (
                            "pre",
                            ColumnTransformer(
                                [
                                    (
                                        "num",
                                        Pipeline(
                                            [
                                                ("impute", SimpleImputer(strategy="median")),
                                                ("scale", StandardScaler()),
                                            ]
                                        ),
                                        features,
                                    )
                                ],
                                remainder="drop",
                            ),
                        ),
                        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)),
                    ]
                )
                model.fit(train[features], y_train)
                score = model.predict_proba(test[features])[:, 1]
                y_test = test[label_col].astype(int).to_numpy()
                base_cols = [c for c in KEY_COLS if c in test.columns]
                out = test[base_cols].copy()
                out["target"] = target
                out["feature_set"] = feature_set
                out["heldout_patient"] = heldout_patient
                out["label"] = y_test
                out["score"] = score
                out["pred_0p5"] = (score >= 0.5).astype(int)
                pred_rows.append(out)

                metrics = classification_metrics(y_test, score)
                row = {
                    "target": target,
                    "feature_set": feature_set,
                    "heldout_patient": heldout_patient,
                    "n_samples": int(len(test)),
                    "positive_rate": float(np.mean(y_test)),
                    "n_positive": int(np.sum(y_test)),
                }
                row.update(metrics)
                patient_rows.append(row)

    predictions = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    patient_summary = pd.DataFrame(patient_rows)
    skipped = pd.DataFrame(skipped_rows)
    return predictions, patient_summary, skipped


def lopo_overall_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    rows = []
    for (target, feature_set), part in predictions.groupby(["target", "feature_set"], observed=True):
        y = part["label"].astype(int).to_numpy()
        score = part["score"].to_numpy()
        metrics = classification_metrics(y, score)
        row = {
            "target": target,
            "feature_set": feature_set,
            "n_samples": int(len(part)),
            "n_patients": int(part["heldout_patient"].nunique()),
            "positive_rate": float(np.mean(y)),
            "n_positive": int(np.sum(y)),
        }
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(
    path: Path,
    concentration: pd.DataFrame,
    treatment_gaps: pd.DataFrame,
    lopo_summary: pd.DataFrame,
    skipped: pd.DataFrame,
) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Forecast-Origin Confounding Audit\n\n")
        f.write(
            "This audit checks whether transition-state predictability may be explained by patient concentration, "
            "treatment/treatment-change structure, or failure under patient-held-out prediction. It is a guardrail "
            "against overinterpreting scalar risk predictors.\n\n"
        )
        f.write("## Patient Concentration\n\n")
        f.write(concentration.to_markdown(index=False) if not concentration.empty else "No rows.")
        f.write("\n\n## Treatment Target Gaps\n\n")
        f.write(treatment_gaps.to_markdown(index=False) if not treatment_gaps.empty else "No rows.")
        f.write("\n\n## Leave-One-Patient-Out Summary\n\n")
        f.write(lopo_summary.to_markdown(index=False) if not lopo_summary.empty else "No rows.")
        if not skipped.empty:
            f.write("\n\n## Skipped LOPO Fits\n\n")
            f.write(skipped.to_markdown(index=False))
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit patient/treatment confounding in forecast-origin transition-state predictability.")
    parser.add_argument("--ablation_dir", type=str, default=None)
    parser.add_argument("--labeled_samples_csv", type=str, default=None)
    parser.add_argument("--reference_feature_set", type=str, default="full_origin")
    parser.add_argument("--targets", type=str, default=",".join(TARGETS))
    parser.add_argument("--feature_sets", type=str, default="full_origin,no_interval,time_only,history_only,treatment_only")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    labeled_path = resolve_labeled_samples(args)
    data = pd.read_csv(labeled_path)
    if "patient_id" not in data.columns:
        raise ValueError("labeled samples must contain patient_id")
    if "split" not in data.columns:
        data["split"] = "all"
    targets = parse_csv(args.targets)
    target_pairs = label_columns(data, targets)
    selected_feature_sets = {name: FEATURE_SETS[name] for name in parse_csv(args.feature_sets) if name in FEATURE_SETS}
    if not selected_feature_sets:
        raise ValueError("No valid feature sets selected.")

    concentration = patient_concentration(data, target_pairs)
    patient_table = patient_target_table(data, target_pairs)
    treatment_dist = treatment_distribution(data)
    treatment_gaps = treatment_target_gaps(data, target_pairs)
    lopo_predictions, lopo_patient_summary, lopo_skipped = fit_lopo_predictions(
        data=data,
        target_pairs=target_pairs,
        feature_sets=selected_feature_sets,
        seed=args.seed,
    )
    lopo_summary = lopo_overall_summary(lopo_predictions)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    concentration.to_csv(output_dir / "confounding_patient_concentration.csv", index=False)
    patient_table.to_csv(output_dir / "confounding_patient_target_table.csv", index=False)
    treatment_dist.to_csv(output_dir / "confounding_treatment_distribution.csv", index=False)
    treatment_gaps.to_csv(output_dir / "confounding_treatment_target_gaps.csv", index=False)
    lopo_predictions.to_csv(output_dir / "confounding_lopo_predictions.csv", index=False)
    lopo_patient_summary.to_csv(output_dir / "confounding_lopo_patient_summary.csv", index=False)
    lopo_summary.to_csv(output_dir / "confounding_lopo_overall_summary.csv", index=False)
    lopo_skipped.to_csv(output_dir / "confounding_lopo_skipped.csv", index=False)
    write_report(
        output_dir / "forecast_origin_confounding_report.md",
        concentration=concentration,
        treatment_gaps=treatment_gaps,
        lopo_summary=lopo_summary,
        skipped=lopo_skipped,
    )

    payload = {
        "labeled_samples_csv": str(labeled_path),
        "targets": [t for t, _ in target_pairs],
        "feature_sets": selected_feature_sets,
        "n_rows": int(len(data)),
        "n_patients": int(data["patient_id"].nunique()),
        "output_dir": str(output_dir),
        "outputs": {
            "patient_concentration_csv": str(output_dir / "confounding_patient_concentration.csv"),
            "patient_target_table_csv": str(output_dir / "confounding_patient_target_table.csv"),
            "treatment_distribution_csv": str(output_dir / "confounding_treatment_distribution.csv"),
            "treatment_target_gaps_csv": str(output_dir / "confounding_treatment_target_gaps.csv"),
            "lopo_overall_summary_csv": str(output_dir / "confounding_lopo_overall_summary.csv"),
            "lopo_patient_summary_csv": str(output_dir / "confounding_lopo_patient_summary.csv"),
            "report_md": str(output_dir / "forecast_origin_confounding_report.md"),
        },
    }
    with (output_dir / "forecast_origin_confounding_summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
