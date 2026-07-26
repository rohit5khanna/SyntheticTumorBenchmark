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

from baselines.metrics import dice_np
from baselines.tasks import patient_paths
from scripts.analyze_forecast_origin_predictability import add_origin_features, available_features, parse_csv
from scripts.run_forecast_origin_feature_ablation import FEATURE_SETS


def normalize_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    out = manifest.copy()
    if "input_idx" not in out.columns and "input_end_idx" in out.columns:
        out["input_idx"] = out["input_end_idx"]
    if "current_treatment" not in out.columns and "input_end_treatment" in out.columns:
        out["current_treatment"] = out["input_end_treatment"]
    for col in ["input_idx", "target_idx", "horizon"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    for col in [
        "input_volume_vox",
        "target_volume_vox",
        "growth_volume_vox",
        "loss_volume_vox",
        "previous_growth_volume_vox",
        "previous_loss_volume_vox",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    if "net_direction" not in out.columns and {"input_volume_vox", "target_volume_vox"}.issubset(out.columns):
        out["net_direction"] = np.where(
            out["target_volume_vox"] > out["input_volume_vox"],
            "net_growth",
            np.where(out["target_volume_vox"] < out["input_volume_vox"], "net_shrinkage", "net_stable"),
        )
    if "relative_new_growth" not in out.columns and {"growth_volume_vox", "input_volume_vox"}.issubset(out.columns):
        out["relative_new_growth"] = out["growth_volume_vox"] / out["input_volume_vox"].clip(lower=1)
    if "relative_loss" not in out.columns and {"loss_volume_vox", "input_volume_vox"}.issubset(out.columns):
        out["relative_loss"] = out["loss_volume_vox"] / out["input_volume_vox"].clip(lower=1)
    return add_origin_features(out)


def standardize_label(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 5 and arr.shape[1] == 1:
        return arr > 0
    if arr.ndim == 4:
        return arr[:, None, ...] > 0
    raise ValueError(f"Unsupported label shape: {arr.shape}")


def safe_corr(a: pd.Series, b: pd.Series, method: str) -> float:
    data = pd.DataFrame({"a": pd.to_numeric(a, errors="coerce"), "b": pd.to_numeric(b, errors="coerce")}).dropna()
    if len(data) < 3 or data["a"].nunique() < 2 or data["b"].nunique() < 2:
        return float("nan")
    return float(data["a"].corr(data["b"], method=method))


def fit_predict_regressors(train: pd.DataFrame, score: pd.DataFrame, features: List[str], target_col: str, seed: int) -> Dict[str, np.ndarray]:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    y = np.log1p(np.clip(pd.to_numeric(train[target_col], errors="coerce").to_numpy(dtype=float), 0.0, None))
    pre_linear = ColumnTransformer(
        [("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), features)],
        remainder="drop",
    )
    ridge = Pipeline([("pre", pre_linear), ("model", Ridge(alpha=1.0))])
    ridge.fit(train[features], y)

    pre_tree = ColumnTransformer([("num", SimpleImputer(strategy="median"), features)], remainder="drop")
    forest = Pipeline(
        [
            ("pre", pre_tree),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=250,
                    min_samples_leaf=5,
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    forest.fit(train[features], y)

    input_volume = pd.to_numeric(score["input_volume_vox"], errors="coerce").fillna(0).to_numpy(dtype=float)
    train_ratio = pd.to_numeric(train[target_col], errors="coerce").fillna(0) / pd.to_numeric(
        train["input_volume_vox"], errors="coerce"
    ).clip(lower=1)
    median_ratio = float(train_ratio.median()) if len(train_ratio) else 0.0

    prev_col = "previous_growth_volume_vox" if "growth" in target_col else "previous_loss_volume_vox"
    previous = (
        pd.to_numeric(score[prev_col], errors="coerce").fillna(0).to_numpy(dtype=float)
        if prev_col in score.columns
        else np.zeros(len(score), dtype=float)
    )
    return {
        "zero": np.zeros(len(score), dtype=float),
        "previous_volume": np.clip(previous, 0.0, None),
        "train_median_ratio": np.clip(median_ratio * input_volume, 0.0, None),
        "ridge_log": np.clip(np.expm1(ridge.predict(score[features])), 0.0, None),
        "random_forest_log": np.clip(np.expm1(forest.predict(score[features])), 0.0, None),
    }


def fit_predict_direction(train: pd.DataFrame, score: pd.DataFrame, features: List[str], seed: int) -> np.ndarray:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    y = (train["net_direction"].astype(str) == "net_growth").astype(int)
    if y.nunique() < 2:
        return np.full(len(score), float(y.mean()) if len(y) else 0.5)
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
    model.fit(train[features], y)
    return model.predict_proba(score[features])[:, 1]


def load_labels_for_needed(dataset_root: Path, patient_ids: Iterable[str]) -> Dict[str, np.ndarray]:
    cache: Dict[str, np.ndarray] = {}
    for pid in sorted(set(str(x) for x in patient_ids)):
        cache[pid] = standardize_label(np.load(patient_paths(dataset_root, pid)["label"]))
    return cache


def set_true_subset(base_mask: np.ndarray, candidate: np.ndarray, k: int, value: bool) -> np.ndarray:
    pred = base_mask.copy()
    idx = np.flatnonzero(candidate.reshape(-1))
    if len(idx) == 0 or k <= 0:
        return pred
    chosen = idx[: min(int(k), len(idx))]
    flat = pred.reshape(-1)
    flat[chosen] = value
    return flat.reshape(pred.shape)


def add_budget_oracle_dice(df: pd.DataFrame, dataset_root: Path) -> pd.DataFrame:
    labels_by_pid = load_labels_for_needed(dataset_root, df["patient_id"].astype(str).unique())
    rows = []
    for _, row in df.iterrows():
        labels = labels_by_pid[str(row["patient_id"])]
        input_idx = int(row["input_idx"])
        target_idx = int(row["target_idx"])
        input_mask = labels[input_idx, 0].astype(bool)
        target_mask = labels[target_idx, 0].astype(bool)
        true_growth = target_mask & ~input_mask
        true_loss = input_mask & ~target_mask
        locf = float(dice_np(input_mask, target_mask))
        growth_k = int(round(max(0.0, float(row["pred_growth_budget_vox"]))))
        loss_k = int(round(max(0.0, float(row["pred_loss_budget_vox"]))))
        growth_pred = set_true_subset(input_mask, true_growth, growth_k, True)
        loss_pred = set_true_subset(input_mask, true_loss, loss_k, False)
        actual_growth_pred = growth_pred if str(row["net_direction"]) == "net_growth" else loss_pred
        predicted_growth_pred = growth_pred if float(row["pred_net_growth_prob"]) >= 0.5 else loss_pred
        rows.append(
            {
                **row.to_dict(),
                "locf_dice": locf,
                "growth_budget_oracle_dice": float(dice_np(growth_pred, target_mask)),
                "loss_budget_oracle_dice": float(dice_np(loss_pred, target_mask)),
                "actual_direction_budget_oracle_dice": float(dice_np(actual_growth_pred, target_mask)),
                "predicted_direction_budget_oracle_dice": float(dice_np(predicted_growth_pred, target_mask)),
                "growth_budget_oracle_gap_vs_locf": float(dice_np(growth_pred, target_mask)) - locf,
                "loss_budget_oracle_gap_vs_locf": float(dice_np(loss_pred, target_mask)) - locf,
                "actual_direction_budget_oracle_gap_vs_locf": float(dice_np(actual_growth_pred, target_mask)) - locf,
                "predicted_direction_budget_oracle_gap_vs_locf": float(dice_np(predicted_growth_pred, target_mask)) - locf,
            }
        )
    return pd.DataFrame(rows)


def evaluate_budget_predictions(
    data: pd.DataFrame,
    train_split: str,
    eval_splits: List[str],
    feature_sets: List[str],
    seed: int,
    dataset_root: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = data[data["split"].astype(str) == train_split].copy()
    eval_df = data[data["split"].astype(str).isin(eval_splits)].copy()
    if train.empty or eval_df.empty:
        raise ValueError("Need non-empty train and evaluation splits.")

    prediction_rows = []
    for fs_name in feature_sets:
        if fs_name not in FEATURE_SETS:
            raise ValueError(f"Unknown feature set: {fs_name}. Available: {sorted(FEATURE_SETS)}")
        features = available_features(data, FEATURE_SETS[fs_name])
        direction_prob = fit_predict_direction(train, eval_df, features, seed=seed)
        growth_preds = fit_predict_regressors(train, eval_df, features, "growth_volume_vox", seed=seed)
        loss_preds = fit_predict_regressors(train, eval_df, features, "loss_volume_vox", seed=seed + 11)

        for model_name in sorted(growth_preds):
            cur = eval_df.copy()
            cur["feature_set"] = fs_name
            cur["budget_model"] = model_name
            cur["pred_net_growth_prob"] = direction_prob
            cur["pred_net_growth"] = (direction_prob >= 0.5).astype(int)
            cur["true_net_growth"] = (cur["net_direction"].astype(str) == "net_growth").astype(int)
            cur["pred_growth_budget_vox"] = growth_preds[model_name]
            cur["pred_loss_budget_vox"] = loss_preds[model_name]
            cur["growth_budget_error_vox"] = cur["pred_growth_budget_vox"] - cur["growth_volume_vox"]
            cur["loss_budget_error_vox"] = cur["pred_loss_budget_vox"] - cur["loss_volume_vox"]
            cur["pred_growth_budget_ratio"] = cur["pred_growth_budget_vox"] / cur["input_volume_vox"].clip(lower=1)
            cur["pred_loss_budget_ratio"] = cur["pred_loss_budget_vox"] / cur["input_volume_vox"].clip(lower=1)
            prediction_rows.append(cur)

    pred = pd.concat(prediction_rows, ignore_index=True)
    if dataset_root is not None:
        pred = add_budget_oracle_dice(pred, dataset_root)

    summary = summarize_predictions(pred)
    direction_summary = summarize_direction(pred)
    return pred, summary, direction_summary


def summarize_predictions(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in pred.groupby(["feature_set", "budget_model", "split"], observed=True):
        fs, model, split = keys
        row = {
            "feature_set": fs,
            "budget_model": model,
            "split": split,
            "n": int(len(group)),
            "n_patients": int(group["patient_id"].nunique()),
            "growth_mae_vox": float(np.mean(np.abs(group["growth_budget_error_vox"]))),
            "loss_mae_vox": float(np.mean(np.abs(group["loss_budget_error_vox"]))),
            "growth_pearson": safe_corr(group["pred_growth_budget_vox"], group["growth_volume_vox"], "pearson"),
            "growth_spearman": safe_corr(group["pred_growth_budget_vox"], group["growth_volume_vox"], "spearman"),
            "loss_pearson": safe_corr(group["pred_loss_budget_vox"], group["loss_volume_vox"], "pearson"),
            "loss_spearman": safe_corr(group["pred_loss_budget_vox"], group["loss_volume_vox"], "spearman"),
            "mean_true_growth_vox": float(group["growth_volume_vox"].mean()),
            "mean_pred_growth_vox": float(group["pred_growth_budget_vox"].mean()),
            "mean_true_loss_vox": float(group["loss_volume_vox"].mean()),
            "mean_pred_loss_vox": float(group["pred_loss_budget_vox"].mean()),
        }
        for col in [
            "growth_budget_oracle_dice",
            "loss_budget_oracle_dice",
            "actual_direction_budget_oracle_dice",
            "predicted_direction_budget_oracle_dice",
            "growth_budget_oracle_gap_vs_locf",
            "loss_budget_oracle_gap_vs_locf",
            "actual_direction_budget_oracle_gap_vs_locf",
            "predicted_direction_budget_oracle_gap_vs_locf",
            "locf_dice",
        ]:
            if col in group.columns:
                row[f"mean_{col}"] = float(group[col].mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["split", "feature_set", "budget_model"])


def summarize_direction(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in pred.groupby(["feature_set", "budget_model", "split"], observed=True):
        fs, model, split = keys
        y = group["true_net_growth"].astype(int).to_numpy()
        p = group["pred_net_growth_prob"].to_numpy(dtype=float)
        pred_label = group["pred_net_growth"].astype(int).to_numpy()
        rows.append(
            {
                "feature_set": fs,
                "budget_model": model,
                "split": split,
                "n": int(len(group)),
                "accuracy": float((pred_label == y).mean()) if len(y) else np.nan,
                "net_growth_prevalence": float(y.mean()) if len(y) else np.nan,
                "pred_net_growth_rate": float(pred_label.mean()) if len(y) else np.nan,
                "mean_pred_net_growth_prob": float(p.mean()) if len(p) else np.nan,
                "prob_net_growth_pearson": safe_corr(pd.Series(p), pd.Series(y), "pearson"),
                "prob_net_growth_spearman": safe_corr(pd.Series(p), pd.Series(y), "spearman"),
            }
        )
    return pd.DataFrame(rows).sort_values(["split", "feature_set", "budget_model"])


def write_report(output_dir: Path, summary: pd.DataFrame, direction_summary: pd.DataFrame) -> None:
    with (output_dir / "forecast_origin_budget_predictability_report.md").open("w", encoding="utf-8") as f:
        f.write("# Forecast-Origin Budget Predictability\n\n")
        f.write(
            "This audit asks whether input-side features can estimate the amount of growth/loss correction "
            "needed before training another forecasting network. Spatial localization is treated as oracle-perfect "
            "when Dice ceilings are reported, so the remaining bottleneck is budget calibration.\n\n"
        )
        f.write("## Budget Prediction Summary\n\n")
        f.write(summary.to_markdown(index=False) if not summary.empty else "No rows.")
        f.write("\n\n## Direction Prediction Summary\n\n")
        f.write(direction_summary.to_markdown(index=False) if not direction_summary.empty else "No rows.")
        f.write(
            "\n\nInterpretation rule: if predicted-budget oracle Dice stays near LOCF, then budget prediction is a bottleneck. "
            "If it moves meaningfully toward the constrained oracle ceiling, then a LOCF-correction method has a calibrated path forward.\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit forecast-origin predictability of LOCF-correction growth/loss budgets.")
    parser.add_argument("--manifest_csv", type=str, required=True)
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--eval_splits", type=str, default="val,test")
    parser.add_argument("--feature_sets", type=str, default="full_origin,no_interval,history_only,time_only,treatment_only")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = normalize_manifest(pd.read_csv(args.manifest_csv))
    required = {"split", "patient_id", "input_idx", "target_idx", "horizon", "growth_volume_vox", "loss_volume_vox", "input_volume_vox"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")

    feature_sets = parse_csv(args.feature_sets)
    eval_splits = parse_csv(args.eval_splits)
    dataset_root = Path(args.dataset_root) if args.dataset_root else None
    pred, summary, direction_summary = evaluate_budget_predictions(
        data=data,
        train_split=args.train_split,
        eval_splits=eval_splits,
        feature_sets=feature_sets,
        seed=args.seed,
        dataset_root=dataset_root,
    )

    pred.to_csv(output_dir / "forecast_origin_budget_predictions.csv", index=False)
    summary.to_csv(output_dir / "forecast_origin_budget_predictability_summary.csv", index=False)
    direction_summary.to_csv(output_dir / "forecast_origin_direction_predictability_summary.csv", index=False)
    run_summary = {
        "manifest_csv": args.manifest_csv,
        "dataset_root": args.dataset_root,
        "train_split": args.train_split,
        "eval_splits": eval_splits,
        "feature_sets": feature_sets,
        "n_prediction_rows": int(len(pred)),
        "outputs": {
            "predictions_csv": str(output_dir / "forecast_origin_budget_predictions.csv"),
            "summary_csv": str(output_dir / "forecast_origin_budget_predictability_summary.csv"),
            "direction_summary_csv": str(output_dir / "forecast_origin_direction_predictability_summary.csv"),
            "report_md": str(output_dir / "forecast_origin_budget_predictability_report.md"),
        },
    }
    with (output_dir / "forecast_origin_budget_predictability_run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)
    write_report(output_dir, summary, direction_summary)
    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    main()
