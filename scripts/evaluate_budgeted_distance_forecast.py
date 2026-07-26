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
from scripts.analyze_forecast_origin_budget_predictability import (
    fit_predict_direction,
    fit_predict_regressors,
    normalize_manifest,
    standardize_label,
)
from scripts.analyze_forecast_origin_predictability import available_features, parse_csv
from scripts.run_forecast_origin_feature_ablation import FEATURE_SETS


KEY_COLS = ["split", "patient_id", "input_idx", "target_idx", "horizon"]


def topk_edit(mask: np.ndarray, candidate: np.ndarray, score: np.ndarray, k: int, value: bool) -> np.ndarray:
    pred = mask.copy()
    idx = np.flatnonzero(candidate.reshape(-1))
    if len(idx) == 0 or k <= 0:
        return pred
    k = min(int(k), len(idx))
    scores = score.reshape(-1)[idx]
    chosen_local = np.argsort(-scores, kind="mergesort")[:k]
    chosen = idx[chosen_local]
    flat = pred.reshape(-1)
    flat[chosen] = value
    return flat.reshape(mask.shape)


def distance_scores(input_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.ndimage import distance_transform_edt
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("scipy is required for distance-based spatial scores.") from exc
    growth_score = -distance_transform_edt(~input_mask).astype(np.float32)
    loss_score = -distance_transform_edt(input_mask).astype(np.float32)
    return growth_score, loss_score


def load_label_cache(dataset_root: Path, patient_ids: Iterable[str]) -> Dict[str, np.ndarray]:
    cache = {}
    for pid in sorted(set(str(x) for x in patient_ids)):
        cache[pid] = standardize_label(np.load(patient_paths(dataset_root, pid)["label"]))
    return cache


def add_budget_predictions(
    manifest: pd.DataFrame,
    train_split: str,
    eval_splits: List[str],
    feature_set: str,
    budget_model: str,
    seed: int,
) -> pd.DataFrame:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature_set={feature_set}. Available: {sorted(FEATURE_SETS)}")
    data = normalize_manifest(manifest)
    train = data[data["split"].astype(str) == train_split].copy()
    eval_df = data[data["split"].astype(str).isin(eval_splits)].copy()
    if train.empty or eval_df.empty:
        raise ValueError("Need non-empty train and evaluation splits.")

    features = available_features(data, FEATURE_SETS[feature_set])
    direction_prob = fit_predict_direction(train, eval_df, features, seed=seed)
    growth_preds = fit_predict_regressors(train, eval_df, features, "growth_volume_vox", seed=seed)
    loss_preds = fit_predict_regressors(train, eval_df, features, "loss_volume_vox", seed=seed + 11)
    if budget_model not in growth_preds:
        raise ValueError(f"Unknown budget_model={budget_model}. Available: {sorted(growth_preds)}")

    out = eval_df.copy()
    out["feature_set"] = feature_set
    out["budget_model"] = budget_model
    out["pred_net_growth_prob"] = direction_prob
    out["pred_net_growth"] = (direction_prob >= 0.5).astype(int)
    out["pred_growth_budget_vox"] = np.clip(growth_preds[budget_model], 0.0, None)
    out["pred_loss_budget_vox"] = np.clip(loss_preds[budget_model], 0.0, None)
    return out


def evaluate_one(input_mask: np.ndarray, target_mask: np.ndarray, row: pd.Series) -> List[dict]:
    growth_score, loss_score = distance_scores(input_mask)
    true_growth = target_mask & ~input_mask
    true_loss = input_mask & ~target_mask
    locf_dice = float(dice_np(input_mask, target_mask))
    pred_growth_k = int(round(max(0.0, float(row["pred_growth_budget_vox"]))))
    pred_loss_k = int(round(max(0.0, float(row["pred_loss_budget_vox"]))))
    true_growth_k = int(true_growth.sum())
    true_loss_k = int(true_loss.sum())
    pred_is_growth = bool(float(row["pred_net_growth_prob"]) >= 0.5)
    true_is_growth = str(row["net_direction"]) == "net_growth"

    policies = {
        "locf": input_mask,
        "pred_growth_budget_distance": topk_edit(input_mask, ~input_mask, growth_score, pred_growth_k, True),
        "pred_direction_growth_only_distance": (
            topk_edit(input_mask, ~input_mask, growth_score, pred_growth_k, True) if pred_is_growth else input_mask
        ),
        "pred_direction_growth_or_boundary_loss": (
            topk_edit(input_mask, ~input_mask, growth_score, pred_growth_k, True)
            if pred_is_growth
            else topk_edit(input_mask, input_mask, loss_score, pred_loss_k, False)
        ),
        "true_growth_budget_distance": topk_edit(input_mask, ~input_mask, growth_score, true_growth_k, True),
        "true_direction_budget_distance": (
            topk_edit(input_mask, ~input_mask, growth_score, true_growth_k, True)
            if true_is_growth
            else topk_edit(input_mask, input_mask, loss_score, true_loss_k, False)
        ),
    }

    rows = []
    for policy, pred in policies.items():
        pred_growth = pred & ~input_mask
        pred_loss = input_mask & ~pred
        growth_tp = int((pred_growth & true_growth).sum())
        growth_fp = int((pred_growth & ~true_growth).sum())
        loss_tp = int((pred_loss & true_loss).sum())
        loss_fp = int((pred_loss & ~true_loss).sum())
        d = float(dice_np(pred, target_mask))
        rows.append(
            {
                "policy": policy,
                "dice": d,
                "locf_dice": locf_dice,
                "gap_vs_locf": d - locf_dice,
                "predicted_net_growth": int(pred_is_growth),
                "true_net_growth": int(true_is_growth),
                "pred_growth_budget_vox": pred_growth_k,
                "pred_loss_budget_vox": pred_loss_k,
                "true_growth_volume_vox": true_growth_k,
                "true_loss_volume_vox": true_loss_k,
                "added_growth_vox": int(pred_growth.sum()),
                "removed_loss_vox": int(pred_loss.sum()),
                "growth_tp_vox": growth_tp,
                "growth_fp_vox": growth_fp,
                "loss_tp_vox": loss_tp,
                "loss_fp_vox": loss_fp,
                "growth_precision": growth_tp / max(1, int(pred_growth.sum())),
                "growth_recall": growth_tp / max(1, true_growth_k),
                "loss_precision": loss_tp / max(1, int(pred_loss.sum())),
                "loss_recall": loss_tp / max(1, true_loss_k),
            }
        )
    return rows


def evaluate_distance_forecasts(predictions: pd.DataFrame, dataset_root: Path) -> pd.DataFrame:
    label_cache = load_label_cache(dataset_root, predictions["patient_id"].astype(str).unique())
    rows = []
    for _, row in predictions.iterrows():
        labels = label_cache[str(row["patient_id"])]
        input_mask = labels[int(row["input_idx"]), 0].astype(bool)
        target_mask = labels[int(row["target_idx"]), 0].astype(bool)
        base = row.to_dict()
        for result in evaluate_one(input_mask, target_mask, row):
            rows.append({**base, **result})
    return pd.DataFrame(rows)


def summarize(samples: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    group_cols_l = [c for c in group_cols if c in samples.columns]
    return (
        samples.groupby(group_cols_l, observed=True, dropna=False)
        .agg(
            n=("dice", "size"),
            n_patients=("patient_id", "nunique"),
            mean_dice=("dice", "mean"),
            std_dice=("dice", "std"),
            mean_locf_dice=("locf_dice", "mean"),
            mean_gap_vs_locf=("gap_vs_locf", "mean"),
            median_gap_vs_locf=("gap_vs_locf", "median"),
            win_rate_vs_locf=("gap_vs_locf", lambda x: float((x > 0).mean())),
            added_growth_mean=("added_growth_vox", "mean"),
            removed_loss_mean=("removed_loss_vox", "mean"),
            growth_precision_mean=("growth_precision", "mean"),
            growth_recall_mean=("growth_recall", "mean"),
            loss_precision_mean=("loss_precision", "mean"),
            loss_recall_mean=("loss_recall", "mean"),
        )
        .reset_index()
        .sort_values(group_cols_l)
    )


def write_report(path: Path, overall: pd.DataFrame, by_direction: pd.DataFrame) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Budgeted Distance Forecast Evaluation\n\n")
        f.write(
            "This evaluation removes the oracle spatial-localization assumption from the correction-budget audit. "
            "Forecast-origin features predict growth/loss budgets; distance-to-current-mask provides the spatial score field; "
            "actual masks are produced by top-k edits to LOCF.\n\n"
        )
        f.write("## Overall\n\n")
        f.write(overall.to_markdown(index=False))
        f.write("\n\n## By Net Direction\n\n")
        f.write(by_direction.to_markdown(index=False))
        f.write(
            "\n\nInterpretation rule: if predicted-budget distance policies remain near or below LOCF, "
            "the remaining bottleneck is spatial localization rather than budget calibration. "
            "Compare predicted-budget policies to true-budget distance policies to separate budget error from spatial-prior error.\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LOCF-anchored forecast masks using predicted budgets and distance spatial scores.")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--manifest_csv", type=str, required=True)
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--eval_splits", type=str, default="val,test")
    parser.add_argument("--feature_set", type=str, default="history_only")
    parser.add_argument("--budget_model", type=str, default="ridge_log")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest_csv)
    predictions = add_budget_predictions(
        manifest,
        train_split=args.train_split,
        eval_splits=parse_csv(args.eval_splits),
        feature_set=args.feature_set,
        budget_model=args.budget_model,
        seed=args.seed,
    )
    samples = evaluate_distance_forecasts(predictions, Path(args.dataset_root))
    overall = summarize(samples, ["split", "policy"])
    by_direction = summarize(samples, ["split", "net_direction", "policy"])

    samples.to_csv(output_dir / "budgeted_distance_forecast_samples.csv", index=False)
    overall.to_csv(output_dir / "budgeted_distance_forecast_summary_by_split.csv", index=False)
    by_direction.to_csv(output_dir / "budgeted_distance_forecast_summary_by_direction.csv", index=False)
    run_summary = {
        "dataset_root": args.dataset_root,
        "manifest_csv": args.manifest_csv,
        "train_split": args.train_split,
        "eval_splits": parse_csv(args.eval_splits),
        "feature_set": args.feature_set,
        "budget_model": args.budget_model,
        "seed": int(args.seed),
        "n_eval_windows": int(predictions.shape[0]),
        "n_output_rows": int(samples.shape[0]),
        "output_dir": str(output_dir),
        "outputs": {
            "samples_csv": str(output_dir / "budgeted_distance_forecast_samples.csv"),
            "summary_by_split_csv": str(output_dir / "budgeted_distance_forecast_summary_by_split.csv"),
            "summary_by_direction_csv": str(output_dir / "budgeted_distance_forecast_summary_by_direction.csv"),
            "report_md": str(output_dir / "budgeted_distance_forecast_report.md"),
        },
    }
    with (output_dir / "budgeted_distance_forecast_run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)
    write_report(output_dir / "budgeted_distance_forecast_report.md", overall, by_direction)
    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    main()
