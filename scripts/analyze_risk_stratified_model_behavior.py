#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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
    normalize_transition_samples,
    parse_csv,
    resolve_samples_path,
)
from scripts.run_forecast_origin_feature_ablation import FEATURE_SETS  # noqa: E402


KEY_COLS = ["patient_id", "input_idx", "target_idx", "horizon", "delta_days"]
RISK_BIN_LABELS = ["low", "medium", "high"]


def make_patient_splits(
    patients: List[str],
    rng: np.random.Generator,
    train_fraction: float,
    val_fraction: float,
) -> Dict[str, List[str]]:
    shuffled = np.array(patients, dtype=object)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = max(1, int(round(train_fraction * n)))
    n_val = max(1, int(round(val_fraction * n)))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    train = shuffled[:n_train].tolist()
    val = shuffled[n_train : n_train + n_val].tolist()
    test = shuffled[n_train + n_val :].tolist()
    if not test:
        test = [val.pop()]
    return {"train": train, "val": val, "test": test}


def assign_split(data: pd.DataFrame, split_map: Dict[str, List[str]]) -> pd.DataFrame:
    patient_to_split = {}
    for split, patients in split_map.items():
        for patient_id in patients:
            patient_to_split[str(patient_id)] = split
    out = data.copy()
    out["split"] = out["patient_id"].astype(str).map(patient_to_split)
    return out[out["split"].notna()].copy()


def fit_logistic_scores(
    train: pd.DataFrame,
    heldout: pd.DataFrame,
    label_col: str,
    features: List[str],
    seed: int,
) -> np.ndarray:
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
    return model.predict_proba(heldout[features])[:, 1]


def stable_key_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["input_idx", "target_idx", "horizon"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    if "delta_days" in out.columns:
        out["delta_days"] = pd.to_numeric(out["delta_days"], errors="coerce")
    return out


def generate_oof_scores(
    base_data: pd.DataFrame,
    selected_feature_sets: Dict[str, List[str]],
    targets: List[str],
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    patients = sorted(base_data["patient_id"].astype(str).unique())
    rng = np.random.default_rng(args.seed)
    score_rows = []
    repeat_rows = []

    for repeat_idx in range(args.n_repeats):
        repeat_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        split_map = make_patient_splits(
            patients,
            np.random.default_rng(repeat_seed),
            train_fraction=args.train_fraction,
            val_fraction=args.val_fraction,
        )
        split_data = assign_split(base_data, split_map)
        split_data, thresholds = add_transition_targets(
            split_data,
            train_split="train",
            growth_loss_threshold=args.growth_loss_threshold,
            distant_growth_threshold=args.distant_growth_threshold,
            high_burden_quantile=args.high_burden_quantile,
            high_change_rate_quantile=args.high_change_rate_quantile,
            locf_breakdown_threshold=args.locf_breakdown_threshold,
        )
        heldout = split_data[split_data["split"].isin(["val", "test"])].copy()
        repeat_rows.append(
            {
                "repeat_idx": repeat_idx,
                "repeat_seed": repeat_seed,
                "n_train_patients": len(split_map["train"]),
                "n_val_patients": len(split_map["val"]),
                "n_test_patients": len(split_map["test"]),
                "n_train_samples": int((split_data["split"] == "train").sum()),
                "n_val_samples": int((split_data["split"] == "val").sum()),
                "n_test_samples": int((split_data["split"] == "test").sum()),
                **thresholds,
            }
        )

        for target in targets:
            label_col = f"label_{target}"
            if label_col not in split_data.columns:
                continue
            train = split_data[(split_data["split"] == "train") & split_data[label_col].notna()].copy()
            if train.empty or train[label_col].astype(int).nunique() < 2:
                continue

            for feature_set, requested_features in selected_feature_sets.items():
                try:
                    features = available_features(split_data, requested_features)
                except ValueError:
                    continue
                try:
                    scores = fit_logistic_scores(train, heldout, label_col, features, seed=repeat_seed)
                except Exception:
                    continue
                score_part = heldout[[c for c in KEY_COLS if c in heldout.columns]].copy()
                score_part["repeat_idx"] = repeat_idx
                score_part["repeat_seed"] = repeat_seed
                score_part["heldout_split"] = heldout["split"].astype(str).to_numpy()
                score_part["target"] = target
                score_part["feature_set"] = feature_set
                score_part["label"] = heldout[label_col].astype(int).to_numpy()
                score_part["risk_score"] = scores
                score_rows.append(score_part)

    scores = pd.concat(score_rows, ignore_index=True) if score_rows else pd.DataFrame()
    return scores, pd.DataFrame(repeat_rows)


def aggregate_scores(scores: pd.DataFrame) -> pd.DataFrame:
    if scores.empty:
        return pd.DataFrame()
    group_cols = [c for c in KEY_COLS if c in scores.columns] + ["target", "feature_set"]
    agg = (
        scores.groupby(group_cols, dropna=False, observed=True)
        .agg(
            n_oof_scores=("risk_score", "size"),
            mean_risk_score=("risk_score", "mean"),
            std_risk_score=("risk_score", "std"),
            min_risk_score=("risk_score", "min"),
            max_risk_score=("risk_score", "max"),
            mean_label=("label", "mean"),
        )
        .reset_index()
    )
    agg["label"] = (agg["mean_label"] >= 0.5).astype(int)
    return agg


def add_risk_bins(agg: pd.DataFrame) -> pd.DataFrame:
    if agg.empty:
        return agg
    out = agg.copy()
    out["risk_bin"] = "all"
    for (_target, _feature_set), idx in out.groupby(["target", "feature_set"], observed=True).groups.items():
        values = out.loc[idx, "mean_risk_score"]
        if values.dropna().nunique() < len(RISK_BIN_LABELS):
            out.loc[idx, "risk_bin"] = "all"
        else:
            out.loc[idx, "risk_bin"] = pd.qcut(
                values,
                q=len(RISK_BIN_LABELS),
                labels=RISK_BIN_LABELS,
                duplicates="drop",
            ).astype(str)
    return out


def load_per_sample_tables(output_dir: Path, methods: Iterable[str]) -> pd.DataFrame:
    rows = []
    for method in methods:
        json_path = output_dir / f"{method}_per_sample.json"
        csv_path = output_dir / f"{method}_per_sample.csv"
        if json_path.exists():
            with json_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            cur = pd.DataFrame(payload)
        elif csv_path.exists():
            cur = pd.read_csv(csv_path)
        else:
            continue
        if cur.empty:
            continue
        cur["method"] = method
        rows.append(cur)
    if not rows:
        raise FileNotFoundError(f"No per-sample files found for requested methods in {output_dir}")
    out = pd.concat(rows, ignore_index=True)
    out = stable_key_frame(out)
    out["dice"] = pd.to_numeric(out["dice"], errors="coerce")
    return out


def merge_scores_and_performance(risk_scores: pd.DataFrame, per_sample: pd.DataFrame) -> pd.DataFrame:
    key_cols = [c for c in KEY_COLS if c in risk_scores.columns and c in per_sample.columns]
    if len(key_cols) < 4:
        raise ValueError(f"Not enough shared key columns to merge risk and performance tables. Found: {key_cols}")
    return risk_scores.merge(per_sample, on=key_cols, how="inner")


def summarize_method_by_risk(merged: pd.DataFrame) -> pd.DataFrame:
    if merged.empty:
        return pd.DataFrame()
    rows = []
    group_cols = ["target", "feature_set", "risk_bin", "method"]
    for key, part in merged.groupby(group_cols, dropna=False, observed=True):
        row = dict(zip(group_cols, key))
        row.update(
            {
                "count": int(len(part)),
                "n_patients": int(part["patient_id"].nunique()),
                "mean_risk_score": float(part["mean_risk_score"].mean()),
                "positive_rate": float(part["label"].mean()),
                "mean_dice": float(part["dice"].mean()),
                "std_dice": float(part["dice"].std(ddof=1)) if len(part) > 1 else float("nan"),
                "median_dice": float(part["dice"].median()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def summarize_pairwise_by_risk(merged: pd.DataFrame, locf_method: str) -> pd.DataFrame:
    if merged.empty:
        return pd.DataFrame()
    key_cols = [c for c in KEY_COLS if c in merged.columns]
    locf = (
        merged[merged["method"] == locf_method][key_cols + ["target", "feature_set", "dice"]]
        .drop_duplicates(key_cols + ["target", "feature_set"])
        .rename(columns={"dice": "locf_dice"})
    )
    pair = merged[merged["method"] != locf_method].merge(locf, on=key_cols + ["target", "feature_set"], how="inner")
    if pair.empty:
        return pd.DataFrame()
    pair["dice_gap_vs_locf"] = pair["dice"] - pair["locf_dice"]
    pair["beats_locf"] = pair["dice_gap_vs_locf"] > 0

    rows = []
    for key, part in pair.groupby(["target", "feature_set", "risk_bin", "method"], dropna=False, observed=True):
        row = dict(zip(["target", "feature_set", "risk_bin", "method"], key))
        row.update(
            {
                "count": int(len(part)),
                "n_patients": int(part["patient_id"].nunique()),
                "mean_risk_score": float(part["mean_risk_score"].mean()),
                "positive_rate": float(part["label"].mean()),
                "locf_mean_dice": float(part["locf_dice"].mean()),
                "model_mean_dice": float(part["dice"].mean()),
                "mean_gap_vs_locf": float(part["dice_gap_vs_locf"].mean()),
                "median_gap_vs_locf": float(part["dice_gap_vs_locf"].median()),
                "win_rate_vs_locf": float(part["beats_locf"].mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["target", "feature_set", "method", "risk_bin"]).reset_index(drop=True)


def summarize_risk_trends(merged: pd.DataFrame, locf_method: str) -> pd.DataFrame:
    if merged.empty:
        return pd.DataFrame()
    key_cols = [c for c in KEY_COLS if c in merged.columns]
    locf_corr = (
        merged[merged["method"] == locf_method][key_cols + ["target", "feature_set", "mean_risk_score", "dice"]]
        .drop_duplicates(key_cols + ["target", "feature_set"])
        .rename(columns={"dice": "locf_dice"})
    )
    locf_for_pair = locf_corr.drop(columns=["mean_risk_score"])
    rows = []
    for (target, feature_set), part in locf_corr.groupby(["target", "feature_set"], observed=True):
        rows.append(
            {
                "target": target,
                "feature_set": feature_set,
                "method": locf_method,
                "metric": "dice",
                "count": int(len(part)),
                "pearson_risk_corr": float(part["mean_risk_score"].corr(part["locf_dice"], method="pearson")),
                "spearman_risk_corr": float(part["mean_risk_score"].corr(part["locf_dice"], method="spearman")),
            }
        )

    pair = merged[merged["method"] != locf_method].merge(locf_for_pair, on=key_cols + ["target", "feature_set"], how="inner")
    if not pair.empty:
        pair["dice_gap_vs_locf"] = pair["dice"] - pair["locf_dice"]
        for (target, feature_set, method), part in pair.groupby(["target", "feature_set", "method"], observed=True):
            rows.append(
                {
                    "target": target,
                    "feature_set": feature_set,
                    "method": method,
                    "metric": "gap_vs_locf",
                    "count": int(len(part)),
                    "pearson_risk_corr": float(part["mean_risk_score"].corr(part["dice_gap_vs_locf"], method="pearson")),
                    "spearman_risk_corr": float(part["mean_risk_score"].corr(part["dice_gap_vs_locf"], method="spearman")),
                }
            )
    return pd.DataFrame(rows).sort_values(["target", "feature_set", "method"]).reset_index(drop=True)


def compact_risk_contrast(pairwise: pd.DataFrame, method_focus: List[str]) -> pd.DataFrame:
    if pairwise.empty:
        return pd.DataFrame()
    rows = []
    bins = set(pairwise["risk_bin"].dropna().astype(str))
    if not {"low", "high"}.issubset(bins):
        return pd.DataFrame()
    for (target, feature_set, method), part in pairwise.groupby(["target", "feature_set", "method"], observed=True):
        if method_focus and method not in method_focus:
            continue
        low = part[part["risk_bin"].astype(str) == "low"]
        high = part[part["risk_bin"].astype(str) == "high"]
        if low.empty or high.empty:
            continue
        rows.append(
            {
                "target": target,
                "feature_set": feature_set,
                "method": method,
                "low_count": int(low["count"].sum()),
                "high_count": int(high["count"].sum()),
                "low_positive_rate": float(np.average(low["positive_rate"], weights=low["count"])),
                "high_positive_rate": float(np.average(high["positive_rate"], weights=high["count"])),
                "low_locf_mean_dice": float(np.average(low["locf_mean_dice"], weights=low["count"])),
                "high_locf_mean_dice": float(np.average(high["locf_mean_dice"], weights=high["count"])),
                "locf_high_minus_low": float(
                    np.average(high["locf_mean_dice"], weights=high["count"])
                    - np.average(low["locf_mean_dice"], weights=low["count"])
                ),
                "low_model_gap_vs_locf": float(np.average(low["mean_gap_vs_locf"], weights=low["count"])),
                "high_model_gap_vs_locf": float(np.average(high["mean_gap_vs_locf"], weights=high["count"])),
                "gap_high_minus_low": float(
                    np.average(high["mean_gap_vs_locf"], weights=high["count"])
                    - np.average(low["mean_gap_vs_locf"], weights=low["count"])
                ),
                "low_win_rate_vs_locf": float(np.average(low["win_rate_vs_locf"], weights=low["count"])),
                "high_win_rate_vs_locf": float(np.average(high["win_rate_vs_locf"], weights=high["count"])),
            }
        )
    return pd.DataFrame(rows).sort_values(["target", "feature_set", "method"]).reset_index(drop=True)


def write_report(
    path: Path,
    args: argparse.Namespace,
    risk_scores: pd.DataFrame,
    method_summary: pd.DataFrame,
    pairwise_summary: pd.DataFrame,
    risk_contrast: pd.DataFrame,
    trend_summary: pd.DataFrame,
) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Risk-Stratified Model Behavior Report\n\n")
        f.write(
            "This analysis generates out-of-fold forecast-origin risk scores using repeated patient splits, "
            "then asks whether those scores stratify existing forecasting performance. It is an analysis layer, "
            "not a new neural forecasting model.\n\n"
        )
        f.write("## Configuration\n\n")
        f.write(f"- Repeats: `{args.n_repeats}`\n")
        f.write(f"- Feature sets: `{args.feature_sets}`\n")
        f.write(f"- Targets: `{args.targets}`\n")
        f.write(f"- Methods: `{args.methods}`\n")
        f.write(f"- Baseline output directory: `{args.baseline_output_dir}`\n\n")
        f.write("## Risk Score Coverage\n\n")
        if risk_scores.empty:
            f.write("No risk scores generated.\n")
        else:
            coverage = (
                risk_scores.groupby(["target", "feature_set"], observed=True)
                .agg(count=("mean_risk_score", "size"), mean_n_oof_scores=("n_oof_scores", "mean"))
                .reset_index()
            )
            f.write(coverage.to_markdown(index=False))
        f.write("\n\n## High-vs-Low Risk Contrast\n\n")
        f.write(risk_contrast.to_markdown(index=False) if not risk_contrast.empty else "No risk contrast rows.")
        f.write("\n\n## Pairwise Model-vs-LOCF Summary By Risk Bin\n\n")
        focus_cols = [
            "target",
            "feature_set",
            "risk_bin",
            "method",
            "count",
            "positive_rate",
            "locf_mean_dice",
            "model_mean_dice",
            "mean_gap_vs_locf",
            "win_rate_vs_locf",
        ]
        cols = [c for c in focus_cols if c in pairwise_summary.columns]
        f.write(pairwise_summary[cols].to_markdown(index=False) if not pairwise_summary.empty else "No pairwise rows.")
        f.write("\n\n## Risk/Performance Correlations\n\n")
        f.write(trend_summary.to_markdown(index=False) if not trend_summary.empty else "No trend rows.")
        f.write("\n\n## Method Dice By Risk Bin\n\n")
        f.write(method_summary.to_markdown(index=False) if not method_summary.empty else "No method summary rows.")
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate forecast-origin risk scores and stratify model behavior by risk.")
    parser.add_argument("--samples_csv", type=str, default=None)
    parser.add_argument("--taxonomy_dir", type=str, default=None)
    parser.add_argument("--manifest_csv", type=str, default=None)
    parser.add_argument("--baseline_output_dir", type=str, required=True)
    parser.add_argument("--methods", type=str, default="locf,resunet_image_mask")
    parser.add_argument("--locf_method", type=str, default="locf")
    parser.add_argument("--feature_sets", type=str, default="full_origin,no_interval,history_only")
    parser.add_argument("--targets", type=str, default="mixed_growth_loss,distant_growth_present,high_transition_burden,locf_breakdown")
    parser.add_argument("--n_repeats", type=int, default=100)
    parser.add_argument("--train_fraction", type=float, default=0.60)
    parser.add_argument("--val_fraction", type=float, default=0.20)
    parser.add_argument("--growth_loss_threshold", type=float, default=0.2)
    parser.add_argument("--distant_growth_threshold", type=float, default=0.2)
    parser.add_argument("--high_burden_quantile", type=float, default=0.75)
    parser.add_argument("--high_change_rate_quantile", type=float, default=0.75)
    parser.add_argument("--locf_breakdown_threshold", type=float, default=0.5)
    parser.add_argument("--method_focus", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    if args.train_fraction <= 0 or args.val_fraction <= 0 or args.train_fraction + args.val_fraction >= 1:
        raise ValueError("--train_fraction and --val_fraction must be positive and sum to less than 1.")
    selected_names = parse_csv(args.feature_sets)
    unknown = [name for name in selected_names if name not in FEATURE_SETS]
    if unknown:
        raise ValueError(f"Unknown feature set(s): {unknown}. Available: {sorted(FEATURE_SETS)}")
    selected_feature_sets = {name: FEATURE_SETS[name] for name in selected_names}
    targets = [target for target in parse_csv(args.targets) if target in TARGETS]
    if not targets:
        raise ValueError("No valid targets requested.")

    samples_path = resolve_samples_path(args)
    base_data = normalize_transition_samples(pd.read_csv(samples_path))
    base_data = merge_manifest_features(base_data, args.manifest_csv)
    base_data = add_origin_features(base_data)
    base_data = base_data[base_data["patient_id"].notna()].copy()
    base_data = stable_key_frame(base_data)

    raw_scores, repeat_summary = generate_oof_scores(
        base_data=base_data,
        selected_feature_sets=selected_feature_sets,
        targets=targets,
        args=args,
    )
    risk_scores = add_risk_bins(aggregate_scores(stable_key_frame(raw_scores)))

    per_sample = load_per_sample_tables(Path(args.baseline_output_dir), parse_csv(args.methods))
    merged = merge_scores_and_performance(risk_scores, per_sample)
    method_summary = summarize_method_by_risk(merged)
    pairwise_summary = summarize_pairwise_by_risk(merged, locf_method=args.locf_method)
    trend_summary = summarize_risk_trends(merged, locf_method=args.locf_method)
    risk_contrast = compact_risk_contrast(pairwise_summary, method_focus=parse_csv(args.method_focus))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_scores.to_csv(output_dir / "risk_stratified_oof_score_draws.csv", index=False)
    repeat_summary.to_csv(output_dir / "risk_stratified_oof_repeat_summary.csv", index=False)
    risk_scores.to_csv(output_dir / "risk_stratified_scores.csv", index=False)
    merged.to_csv(output_dir / "risk_stratified_model_samples.csv", index=False)
    method_summary.to_csv(output_dir / "risk_stratified_method_summary.csv", index=False)
    pairwise_summary.to_csv(output_dir / "risk_stratified_pairwise_summary.csv", index=False)
    trend_summary.to_csv(output_dir / "risk_stratified_trend_summary.csv", index=False)
    risk_contrast.to_csv(output_dir / "risk_stratified_high_low_contrast.csv", index=False)
    write_report(
        output_dir / "risk_stratified_model_behavior_report.md",
        args=args,
        risk_scores=risk_scores,
        method_summary=method_summary,
        pairwise_summary=pairwise_summary,
        risk_contrast=risk_contrast,
        trend_summary=trend_summary,
    )
    with (output_dir / "risk_stratified_model_behavior_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "samples_path": str(samples_path),
                "manifest_csv": args.manifest_csv,
                "baseline_output_dir": args.baseline_output_dir,
                "methods": parse_csv(args.methods),
                "feature_sets": selected_feature_sets,
                "targets": targets,
                "n_repeats": args.n_repeats,
                "train_fraction": args.train_fraction,
                "val_fraction": args.val_fraction,
                "seed": args.seed,
                "n_rows": int(len(base_data)),
                "n_patients": int(base_data["patient_id"].nunique()),
            },
            f,
            indent=2,
        )
    print(
        json.dumps(
            {
                "n_rows": int(len(base_data)),
                "n_patients": int(base_data["patient_id"].nunique()),
                "n_risk_score_rows": int(len(risk_scores)),
                "n_merged_model_rows": int(len(merged)),
                "output_dir": str(output_dir),
                "outputs": {
                    "risk_scores_csv": str(output_dir / "risk_stratified_scores.csv"),
                    "method_summary_csv": str(output_dir / "risk_stratified_method_summary.csv"),
                    "pairwise_summary_csv": str(output_dir / "risk_stratified_pairwise_summary.csv"),
                    "high_low_contrast_csv": str(output_dir / "risk_stratified_high_low_contrast.csv"),
                    "report_md": str(output_dir / "risk_stratified_model_behavior_report.md"),
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
