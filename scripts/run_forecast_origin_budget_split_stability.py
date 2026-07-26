#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_forecast_origin_budget_predictability import (  # noqa: E402
    fit_predict_direction,
    normalize_manifest,
)
from scripts.analyze_forecast_origin_predictability import available_features, parse_csv  # noqa: E402
from scripts.run_forecast_origin_feature_ablation import FEATURE_SETS  # noqa: E402


def make_patient_split(patients: List[str], rng: np.random.Generator, train_fraction: float) -> tuple[List[str], List[str]]:
    shuffled = np.array(patients, dtype=object)
    rng.shuffle(shuffled)
    n_train = int(round(train_fraction * len(shuffled)))
    n_train = min(max(1, n_train), len(shuffled) - 1)
    train = [str(x) for x in shuffled[:n_train]]
    test = [str(x) for x in shuffled[n_train:]]
    return train, test


def assign_repeat_split(data: pd.DataFrame, train_patients: List[str], test_patients: List[str]) -> pd.DataFrame:
    train_set = set(train_patients)
    test_set = set(test_patients)
    out = data.copy()
    out["split"] = pd.Series([None] * len(out), index=out.index, dtype=object)
    out.loc[out["patient_id"].astype(str).isin(train_set), "split"] = "train"
    out.loc[out["patient_id"].astype(str).isin(test_set), "split"] = "test"
    return out[out["split"].isin(["train", "test"])].copy()


def fit_predict_regressors_light(
    train: pd.DataFrame,
    score: pd.DataFrame,
    features: List[str],
    target_col: str,
    seed: int,
    budget_models: List[str],
    rf_estimators: int,
) -> Dict[str, np.ndarray]:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    requested = set(budget_models)
    out: Dict[str, np.ndarray] = {}
    input_volume = pd.to_numeric(score["input_volume_vox"], errors="coerce").fillna(0).to_numpy(dtype=float)
    y = np.log1p(np.clip(pd.to_numeric(train[target_col], errors="coerce").to_numpy(dtype=float), 0.0, None))

    if "zero" in requested:
        out["zero"] = np.zeros(len(score), dtype=float)

    if "previous_volume" in requested:
        prev_col = "previous_growth_volume_vox" if "growth" in target_col else "previous_loss_volume_vox"
        previous = (
            pd.to_numeric(score[prev_col], errors="coerce").fillna(0).to_numpy(dtype=float)
            if prev_col in score.columns
            else np.zeros(len(score), dtype=float)
        )
        out["previous_volume"] = np.clip(previous, 0.0, None)

    if "train_median_ratio" in requested:
        train_ratio = pd.to_numeric(train[target_col], errors="coerce").fillna(0) / pd.to_numeric(
            train["input_volume_vox"], errors="coerce"
        ).clip(lower=1)
        median_ratio = float(train_ratio.median()) if len(train_ratio) else 0.0
        out["train_median_ratio"] = np.clip(median_ratio * input_volume, 0.0, None)

    if "ridge_log" in requested:
        pre_linear = ColumnTransformer(
            [("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), features)],
            remainder="drop",
        )
        ridge = Pipeline([("pre", pre_linear), ("model", Ridge(alpha=1.0))])
        ridge.fit(train[features], y)
        out["ridge_log"] = np.clip(np.expm1(ridge.predict(score[features])), 0.0, None)

    if "random_forest_log" in requested:
        pre_tree = ColumnTransformer([("num", SimpleImputer(strategy="median"), features)], remainder="drop")
        forest = Pipeline(
            [
                ("pre", pre_tree),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=int(rf_estimators),
                        min_samples_leaf=5,
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        forest.fit(train[features], y)
        out["random_forest_log"] = np.clip(np.expm1(forest.predict(score[features])), 0.0, None)

    unknown = requested - {"zero", "previous_volume", "train_median_ratio", "ridge_log", "random_forest_log"}
    if unknown:
        raise ValueError(f"Unknown budget models: {sorted(unknown)}")
    return out


def dice_from_counts(intersection: float, pred_volume: float, target_volume: float, eps: float = 1e-6) -> float:
    return float((2.0 * intersection + eps) / (pred_volume + target_volume + eps))


def add_analytic_budget_dice(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    input_vol = pd.to_numeric(out["input_volume_vox"], errors="coerce").fillna(0).clip(lower=0)
    target_vol = pd.to_numeric(out["target_volume_vox"], errors="coerce").fillna(0).clip(lower=0)
    growth_vol = pd.to_numeric(out["growth_volume_vox"], errors="coerce").fillna(0).clip(lower=0)
    loss_vol = pd.to_numeric(out["loss_volume_vox"], errors="coerce").fillna(0).clip(lower=0)
    persistent = (input_vol - loss_vol).clip(lower=0)
    locf = (2.0 * persistent + 1e-6) / (input_vol + target_vol + 1e-6)

    pred_growth_budget = pd.to_numeric(out["pred_growth_budget_vox"], errors="coerce").fillna(0).clip(lower=0)
    pred_loss_budget = pd.to_numeric(out["pred_loss_budget_vox"], errors="coerce").fillna(0).clip(lower=0)
    added = np.minimum(pred_growth_budget, growth_vol)
    removed = np.minimum(pred_loss_budget, loss_vol)

    growth_dice = (2.0 * (persistent + added) + 1e-6) / (input_vol + added + target_vol + 1e-6)
    loss_dice = (2.0 * persistent + 1e-6) / ((input_vol - removed).clip(lower=0) + target_vol + 1e-6)
    true_growth = out["net_direction"].astype(str).eq("net_growth")
    pred_growth = pd.to_numeric(out["pred_net_growth_prob"], errors="coerce").fillna(0.5) >= 0.5

    out["locf_dice"] = locf.astype(float)
    out["growth_budget_oracle_dice"] = growth_dice.astype(float)
    out["loss_budget_oracle_dice"] = loss_dice.astype(float)
    out["actual_direction_budget_oracle_dice"] = np.where(true_growth, growth_dice, loss_dice).astype(float)
    out["predicted_direction_budget_oracle_dice"] = np.where(pred_growth, growth_dice, loss_dice).astype(float)
    for col in [
        "growth_budget_oracle_dice",
        "loss_budget_oracle_dice",
        "actual_direction_budget_oracle_dice",
        "predicted_direction_budget_oracle_dice",
    ]:
        out[f"{col.replace('_dice', '')}_gap_vs_locf"] = out[col] - out["locf_dice"]
    return out


def run_repeat(
    data: pd.DataFrame,
    repeat_idx: int,
    repeat_seed: int,
    train_fraction: float,
    feature_sets: List[str],
    budget_models: List[str],
    rf_estimators: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    rng = np.random.default_rng(repeat_seed)
    patients = sorted(data["patient_id"].astype(str).unique())
    train_patients, test_patients = make_patient_split(patients, rng, train_fraction)
    split_data = assign_repeat_split(data, train_patients, test_patients)
    train = split_data[split_data["split"] == "train"].copy()
    test = split_data[split_data["split"] == "test"].copy()
    meta = {
        "repeat_idx": int(repeat_idx),
        "repeat_seed": int(repeat_seed),
        "n_train_patients": int(len(train_patients)),
        "n_test_patients": int(len(test_patients)),
        "n_train_samples": int(len(train)),
        "n_test_samples": int(len(test)),
        "train_net_growth_rate": float((train["net_direction"].astype(str) == "net_growth").mean()) if len(train) else np.nan,
        "test_net_growth_rate": float((test["net_direction"].astype(str) == "net_growth").mean()) if len(test) else np.nan,
    }
    if train.empty or test.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {**meta, "status": "empty_split"}

    prediction_rows = []
    for fs_name in feature_sets:
        features = available_features(split_data, FEATURE_SETS[fs_name])
        direction_prob = fit_predict_direction(train, test, features, seed=repeat_seed)
        growth_preds = fit_predict_regressors_light(
            train,
            test,
            features,
            "growth_volume_vox",
            seed=repeat_seed,
            budget_models=budget_models,
            rf_estimators=rf_estimators,
        )
        loss_preds = fit_predict_regressors_light(
            train,
            test,
            features,
            "loss_volume_vox",
            seed=repeat_seed + 11,
            budget_models=budget_models,
            rf_estimators=rf_estimators,
        )

        for model_name in sorted(growth_preds):
            cur = test.copy()
            cur["repeat_idx"] = repeat_idx
            cur["repeat_seed"] = repeat_seed
            cur["feature_set"] = fs_name
            cur["budget_model"] = model_name
            cur["pred_net_growth_prob"] = direction_prob
            cur["pred_net_growth"] = (direction_prob >= 0.5).astype(int)
            cur["true_net_growth"] = (cur["net_direction"].astype(str) == "net_growth").astype(int)
            cur["pred_growth_budget_vox"] = growth_preds[model_name]
            cur["pred_loss_budget_vox"] = loss_preds[model_name]
            cur["growth_budget_error_vox"] = cur["pred_growth_budget_vox"] - cur["growth_volume_vox"]
            cur["loss_budget_error_vox"] = cur["pred_loss_budget_vox"] - cur["loss_volume_vox"]
            prediction_rows.append(cur)

    pred = add_analytic_budget_dice(pd.concat(prediction_rows, ignore_index=True))
    summary, direction = summarize_repeat_predictions(pred)
    for frame in [summary, direction]:
        for key, value in meta.items():
            frame[key] = value
    meta["status"] = "ok"
    return pred, summary, direction, meta


def summarize_repeat_predictions(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    direction_rows = []
    for keys, group in pred.groupby(["repeat_idx", "feature_set", "budget_model"], observed=True):
        repeat_idx, feature_set, budget_model = keys
        y = group["true_net_growth"].astype(int)
        p = group["pred_net_growth_prob"].astype(float)
        pred_label = group["pred_net_growth"].astype(int)
        rows.append(
            {
                "repeat_idx": int(repeat_idx),
                "feature_set": feature_set,
                "budget_model": budget_model,
                "n": int(len(group)),
                "n_patients": int(group["patient_id"].nunique()),
                "mean_locf_dice": float(group["locf_dice"].mean()),
                "mean_growth_budget_oracle_dice": float(group["growth_budget_oracle_dice"].mean()),
                "mean_loss_budget_oracle_dice": float(group["loss_budget_oracle_dice"].mean()),
                "mean_actual_direction_budget_oracle_dice": float(group["actual_direction_budget_oracle_dice"].mean()),
                "mean_predicted_direction_budget_oracle_dice": float(group["predicted_direction_budget_oracle_dice"].mean()),
                "mean_growth_budget_oracle_gap_vs_locf": float(group["growth_budget_oracle_gap_vs_locf"].mean()),
                "mean_loss_budget_oracle_gap_vs_locf": float(group["loss_budget_oracle_gap_vs_locf"].mean()),
                "mean_actual_direction_budget_oracle_gap_vs_locf": float(
                    group["actual_direction_budget_oracle_gap_vs_locf"].mean()
                ),
                "mean_predicted_direction_budget_oracle_gap_vs_locf": float(
                    group["predicted_direction_budget_oracle_gap_vs_locf"].mean()
                ),
                "growth_mae_vox": float(np.mean(np.abs(group["growth_budget_error_vox"]))),
                "loss_mae_vox": float(np.mean(np.abs(group["loss_budget_error_vox"]))),
                "growth_spearman": float(group["pred_growth_budget_vox"].corr(group["growth_volume_vox"], method="spearman"))
                if group["pred_growth_budget_vox"].nunique() > 1 and group["growth_volume_vox"].nunique() > 1
                else np.nan,
                "loss_spearman": float(group["pred_loss_budget_vox"].corr(group["loss_volume_vox"], method="spearman"))
                if group["pred_loss_budget_vox"].nunique() > 1 and group["loss_volume_vox"].nunique() > 1
                else np.nan,
            }
        )
        direction_rows.append(
            {
                "repeat_idx": int(repeat_idx),
                "feature_set": feature_set,
                "budget_model": budget_model,
                "n": int(len(group)),
                "direction_accuracy": float((pred_label == y).mean()),
                "net_growth_prevalence": float(y.mean()),
                "pred_net_growth_rate": float(pred_label.mean()),
                "prob_net_growth_spearman": float(p.corr(y, method="spearman")) if p.nunique() > 1 and y.nunique() > 1 else np.nan,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(direction_rows)


def summarize_stability(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric_cols = [
        "mean_locf_dice",
        "mean_growth_budget_oracle_dice",
        "mean_loss_budget_oracle_dice",
        "mean_actual_direction_budget_oracle_dice",
        "mean_predicted_direction_budget_oracle_dice",
        "mean_predicted_direction_budget_oracle_gap_vs_locf",
        "growth_mae_vox",
        "loss_mae_vox",
        "growth_spearman",
        "loss_spearman",
    ]
    for keys, group in runs.groupby(["feature_set", "budget_model"], observed=True):
        feature_set, budget_model = keys
        row = {
            "feature_set": feature_set,
            "budget_model": budget_model,
            "n_repeats": int(group["repeat_idx"].nunique()),
            "mean_n_test_samples": float(group["n"].mean()),
            "mean_n_test_patients": float(group["n_patients"].mean()),
        }
        for metric in metric_cols:
            vals = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{metric}_q10"] = float(vals.quantile(0.10)) if len(vals) else np.nan
            row[f"{metric}_q25"] = float(vals.quantile(0.25)) if len(vals) else np.nan
            row[f"{metric}_median"] = float(vals.quantile(0.50)) if len(vals) else np.nan
            row[f"{metric}_q75"] = float(vals.quantile(0.75)) if len(vals) else np.nan
            row[f"{metric}_q90"] = float(vals.quantile(0.90)) if len(vals) else np.nan
        row["positive_gap_fraction"] = float((group["mean_predicted_direction_budget_oracle_gap_vs_locf"] > 0).mean())
        row["gap_gt_0p05_fraction"] = float((group["mean_predicted_direction_budget_oracle_gap_vs_locf"] > 0.05).mean())
        row["gap_gt_0p10_fraction"] = float((group["mean_predicted_direction_budget_oracle_gap_vs_locf"] > 0.10).mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["mean_predicted_direction_budget_oracle_gap_vs_locf_mean", "gap_gt_0p10_fraction"],
        ascending=False,
    )


def summarize_direction_stability(direction_runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in direction_runs.groupby(["feature_set", "budget_model"], observed=True):
        feature_set, budget_model = keys
        vals = pd.to_numeric(group["direction_accuracy"], errors="coerce").dropna()
        rows.append(
            {
                "feature_set": feature_set,
                "budget_model": budget_model,
                "n_repeats": int(group["repeat_idx"].nunique()),
                "direction_accuracy_mean": float(vals.mean()) if len(vals) else np.nan,
                "direction_accuracy_q25": float(vals.quantile(0.25)) if len(vals) else np.nan,
                "direction_accuracy_median": float(vals.quantile(0.50)) if len(vals) else np.nan,
                "direction_accuracy_q75": float(vals.quantile(0.75)) if len(vals) else np.nan,
                "prob_net_growth_spearman_mean": float(
                    pd.to_numeric(group["prob_net_growth_spearman"], errors="coerce").mean()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("direction_accuracy_mean", ascending=False)


def write_report(path: Path, stability: pd.DataFrame, direction: pd.DataFrame, run_summary: Dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Forecast-Origin Budget Split Stability\n\n")
        f.write(
            "This audit repeats patient-level train/test splits to test whether correction-budget calibration "
            "is stable or a fixed-split artifact. Dice values use analytic oracle localization from manifest volumes, "
            "so the reported gains isolate budget/direction calibration rather than spatial localization.\n\n"
        )
        f.write("## Run Summary\n\n")
        for key, value in run_summary.items():
            if key != "outputs":
                f.write(f"- {key}: `{value}`\n")
        f.write("\n## Budget Stability Summary\n\n")
        f.write(stability.to_markdown(index=False) if not stability.empty else "No rows.")
        f.write("\n\n## Direction Stability Summary\n\n")
        f.write(direction.to_markdown(index=False) if not direction.empty else "No rows.")
        f.write(
            "\n\nInterpretation rule: prioritize feature/model pairs whose predicted-direction budget-oracle gap "
            "is positive across most patient splits and whose lower quartile remains meaningfully above zero. "
            "If time/treatment-only remains strongest, frame the signal as operating-context calibration rather than tumor-state calibration.\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Repeated patient-split stability audit for forecast-origin correction-budget prediction.")
    parser.add_argument("--manifest_csv", type=str, required=True)
    parser.add_argument("--feature_sets", type=str, default="full_origin,no_interval,history_only,time_only,treatment_only")
    parser.add_argument("--budget_models", type=str, default="previous_volume,train_median_ratio,ridge_log,random_forest_log,zero")
    parser.add_argument("--rf_estimators", type=int, default=100)
    parser.add_argument("--n_repeats", type=int, default=100)
    parser.add_argument("--train_fraction", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = normalize_manifest(pd.read_csv(args.manifest_csv))
    feature_sets = parse_csv(args.feature_sets)
    budget_models = parse_csv(args.budget_models)

    required = {
        "patient_id",
        "input_idx",
        "target_idx",
        "horizon",
        "input_volume_vox",
        "target_volume_vox",
        "growth_volume_vox",
        "loss_volume_vox",
        "net_direction",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")

    rng = np.random.default_rng(args.seed)
    repeat_summaries = []
    direction_summaries = []
    repeat_meta = []
    for repeat_idx in range(args.n_repeats):
        repeat_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        _, summary, direction, meta = run_repeat(
            data,
            repeat_idx,
            repeat_seed,
            args.train_fraction,
            feature_sets,
            budget_models,
            args.rf_estimators,
        )
        if not summary.empty:
            repeat_summaries.append(summary)
        if not direction.empty:
            direction_summaries.append(direction)
        repeat_meta.append(meta)
        if (repeat_idx + 1) % 10 == 0:
            print(f"[INFO] Completed repeat {repeat_idx + 1}/{args.n_repeats}", flush=True)

    runs = pd.concat(repeat_summaries, ignore_index=True) if repeat_summaries else pd.DataFrame()
    direction_runs = pd.concat(direction_summaries, ignore_index=True) if direction_summaries else pd.DataFrame()
    stability = summarize_stability(runs) if not runs.empty else pd.DataFrame()
    meta_df = pd.DataFrame(repeat_meta)
    direction_stability = summarize_direction_stability(direction_runs) if not direction_runs.empty else pd.DataFrame()

    runs.to_csv(output_dir / "forecast_origin_budget_split_stability_runs.csv", index=False)
    direction_runs.to_csv(output_dir / "forecast_origin_budget_split_stability_direction_runs.csv", index=False)
    stability.to_csv(output_dir / "forecast_origin_budget_split_stability_summary.csv", index=False)
    meta_df.to_csv(output_dir / "forecast_origin_budget_split_stability_repeat_meta.csv", index=False)
    direction_stability.to_csv(output_dir / "forecast_origin_budget_split_stability_direction_summary.csv", index=False)

    run_summary = {
        "manifest_csv": args.manifest_csv,
        "feature_sets": feature_sets,
        "budget_models": budget_models,
        "rf_estimators": int(args.rf_estimators),
        "n_repeats": int(args.n_repeats),
        "train_fraction": float(args.train_fraction),
        "seed": int(args.seed),
        "n_rows": int(len(data)),
        "n_patients": int(data["patient_id"].nunique()),
        "output_dir": str(output_dir),
        "outputs": {
            "runs_csv": str(output_dir / "forecast_origin_budget_split_stability_runs.csv"),
            "direction_runs_csv": str(output_dir / "forecast_origin_budget_split_stability_direction_runs.csv"),
            "summary_csv": str(output_dir / "forecast_origin_budget_split_stability_summary.csv"),
            "direction_summary_csv": str(output_dir / "forecast_origin_budget_split_stability_direction_summary.csv"),
            "repeat_meta_csv": str(output_dir / "forecast_origin_budget_split_stability_repeat_meta.csv"),
            "report_md": str(output_dir / "forecast_origin_budget_split_stability_report.md"),
        },
    }
    with (output_dir / "forecast_origin_budget_split_stability_run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)
    write_report(output_dir / "forecast_origin_budget_split_stability_report.md", stability, direction_stability, run_summary)
    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    main()
