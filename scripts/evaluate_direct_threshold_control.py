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
from baselines.tasks import build_samples_for_split, infer_tier_from_patient_id
from scripts.evaluate_calibrated_growth_field import (
    KEY_COLS,
    load_labels,
    load_model,
    model_probability,
    parse_threshold_values,
    sample_context_features,
)


def qbin(series: pd.Series, labels: List[str]) -> pd.Series:
    vals = series.dropna()
    if vals.nunique() < 2:
        return pd.Series(["all"] * len(series), index=series.index)
    try:
        return pd.qcut(series, q=len(labels), labels=labels, duplicates="drop")
    except ValueError:
        codes = pd.qcut(series, q=len(labels), labels=False, duplicates="drop")
        n_bins = int(pd.Series(codes).dropna().nunique())
        use_labels = labels[:n_bins]
        return pd.Series(codes, index=series.index).map({i: use_labels[i] for i in range(n_bins)})


def add_bins(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["new_growth_bin"] = qbin(out["relative_new_growth"], ["low", "medium", "high"])
    out["absolute_growth_bin"] = np.select(
        [
            out["growth_volume_vox"] <= 0,
            out["growth_volume_vox"] <= 250,
            out["growth_volume_vox"] <= 1500,
        ],
        ["zero", "small_nonzero", "medium_nonzero"],
        default="large_nonzero",
    )
    return out


def evaluate_direct_threshold(
    dataset_root: Path,
    sample,
    sample_index: int,
    model,
    ds,
    dev,
    threshold: float,
    label_cache: Dict[str, np.ndarray],
) -> dict:
    labels = load_labels(dataset_root, sample.patient_id, label_cache)
    target_mask = labels[sample.target_idx, 0] > 0
    input_mask = labels[sample.input_idx, 0] > 0
    prob = model_probability(model, ds, sample_index, dev)
    pred = prob >= float(threshold)
    context = sample_context_features(sample, labels)
    context.update({"delta_days": float(sample.delta_days)})
    dice = dice_np(pred.astype(np.float32), target_mask.astype(np.float32))
    return {
        "patient_id": sample.patient_id,
        "input_idx": int(sample.input_idx),
        "target_idx": int(sample.target_idx),
        "horizon": int(sample.horizon),
        "delta_days": float(sample.delta_days),
        "tier": infer_tier_from_patient_id(sample.patient_id),
        "threshold": float(threshold),
        "selected_voxels": int(pred.sum()),
        "dice": float(dice),
        "locf_dice": float(context["locf_dice"]),
        "dice_gap_vs_locf": float(dice - context["locf_dice"]),
        "input_volume_vox": int(context["input_volume_vox"]),
        "target_volume_vox": int(context["target_volume_vox"]),
        "growth_volume_vox": int(context["growth_volume_vox"]),
        "loss_volume_vox": int(context["loss_volume_vox"]),
        "relative_new_growth": float(context["relative_new_growth"]),
        "relative_loss": float(context["relative_loss"]),
        "relative_net_growth": float(context["relative_net_growth"]),
    }


def evaluate_thresholds(
    dataset_root: Path,
    samples,
    start_index: int,
    model,
    ds,
    dev,
    thresholds: Iterable[float],
    label_cache: Dict[str, np.ndarray],
    verbose: bool = False,
) -> pd.DataFrame:
    rows = []
    for idx, sample in enumerate(samples):
        for threshold in thresholds:
            rows.append(
                evaluate_direct_threshold(
                    dataset_root=dataset_root,
                    sample=sample,
                    sample_index=start_index + idx,
                    model=model,
                    ds=ds,
                    dev=dev,
                    threshold=float(threshold),
                    label_cache=label_cache,
                )
            )
        if verbose and (idx + 1) % 25 == 0:
            print(f"[INFO] Evaluated {idx + 1}/{len(samples)} samples")
    return pd.DataFrame(rows)


def summarize_candidates(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("threshold", observed=True, dropna=False)
        .agg(
            count=("dice", "size"),
            mean_dice=("dice", "mean"),
            mean_locf_dice=("locf_dice", "mean"),
            mean_gap_vs_locf=("dice_gap_vs_locf", "mean"),
            median_gap_vs_locf=("dice_gap_vs_locf", "median"),
            win_rate_vs_locf=("dice_gap_vs_locf", lambda x: float((x > 0).mean())),
            mean_selected_voxels=("selected_voxels", "mean"),
        )
        .reset_index()
        .sort_values(["mean_dice", "mean_gap_vs_locf", "win_rate_vs_locf"], ascending=False)
    )


def summarize_selected(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    work = add_bins(df)
    cols = [c for c in group_cols if c in work.columns]
    group = work if cols else work.assign(_overall="overall")
    by = cols if cols else ["_overall"]
    out = (
        group.groupby(by, observed=True, dropna=False)
        .agg(
            count=("dice", "size"),
            mean_dice=("dice", "mean"),
            mean_locf_dice=("locf_dice", "mean"),
            mean_gap_vs_locf=("dice_gap_vs_locf", "mean"),
            median_gap_vs_locf=("dice_gap_vs_locf", "median"),
            win_rate_vs_locf=("dice_gap_vs_locf", lambda x: float((x > 0).mean())),
            mean_gap_vs_default_direct=("gap_vs_default_direct", "mean"),
            median_gap_vs_default_direct=("gap_vs_default_direct", "median"),
            win_rate_vs_default_direct=("beats_default_direct", "mean"),
            mean_selected_voxels=("selected_voxels", "mean"),
            mean_growth_volume_vox=("growth_volume_vox", "mean"),
            mean_relative_new_growth=("relative_new_growth", "mean"),
        )
        .reset_index()
    )
    if "_overall" in out.columns:
        out = out.drop(columns=["_overall"])
    return out.sort_values(cols).reset_index(drop=True) if cols else out


def load_default_direct(baseline_output_dir: Path, method: str) -> pd.DataFrame:
    path = baseline_output_dir / f"{method}_per_sample.json"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        df = pd.DataFrame(json.load(f))
    return df.rename(columns={"dice": "default_direct_dice"})[KEY_COLS + ["default_direct_dice"]]


def attach_default_direct(test_selected: pd.DataFrame, baseline_output_dir: Path, method: str) -> pd.DataFrame:
    out = test_selected.merge(load_default_direct(baseline_output_dir, method), on=KEY_COLS, how="inner")
    out["gap_vs_default_direct"] = out["dice"] - out["default_direct_dice"]
    out["beats_default_direct"] = out["gap_vs_default_direct"] > 0
    return out


def bootstrap_summary(df: pd.DataFrame, n_bootstrap: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for metric in ["dice_gap_vs_locf", "gap_vs_default_direct"]:
        vals = df[metric].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        boot = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, len(vals), len(vals))
            boot.append(float(vals[idx].mean()))
        rows.append(
            {
                "metric": metric,
                "n": int(len(vals)),
                "mean": float(vals.mean()),
                "ci_low": float(np.quantile(boot, 0.025)),
                "ci_high": float(np.quantile(boot, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def write_report(
    path: Path,
    selected_threshold: float,
    validation_summary: pd.DataFrame,
    overall: pd.DataFrame,
    by_growth: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Direct ResUNet Threshold-Control Evaluation\n\n")
        f.write(
            "This control tunes the global probability threshold of the direct ResUNet output on validation, "
            "then evaluates the selected threshold on the held-out test split. It checks whether the additive "
            "growth-field result is merely a threshold-tuning artifact.\n\n"
        )
        f.write(f"Selected threshold: `{selected_threshold}`\n\n")
        f.write("## Validation Candidates\n\n")
        f.write(validation_summary.to_markdown(index=False))
        f.write("\n\n## Test Overall\n\n")
        f.write(overall.to_markdown(index=False))
        f.write("\n\n## Test By Absolute Growth Bin\n\n")
        f.write(by_growth.to_markdown(index=False))
        f.write("\n\n## Bootstrap\n\n")
        f.write(bootstrap.to_markdown(index=False))
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate validation-tuned direct ResUNet threshold control.")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--baseline_output_dir", type=str, required=True)
    parser.add_argument("--model_method", type=str, default="resunet_image_mask")
    parser.add_argument("--validation_split", type=str, default="val")
    parser.add_argument("--test_split", type=str, default="test")
    parser.add_argument("--fit_sessions", type=int, default=3)
    parser.add_argument("--horizons", type=str, default="1,2,3")
    parser.add_argument("--allowed_tiers", type=str, default=None)
    parser.add_argument("--thresholds", type=str, default="0.01,0.02,0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.95,0.975,0.99")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--n_bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    baseline_output_dir = Path(args.baseline_output_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    thresholds = parse_threshold_values(args.thresholds)
    if 0.5 not in thresholds:
        thresholds.append(0.5)
    thresholds = sorted(set(thresholds))

    val_samples = build_samples_for_split(dataset_root, args.validation_split, args.fit_sessions, args.horizons, args.allowed_tiers)
    test_samples = build_samples_for_split(dataset_root, args.test_split, args.fit_sessions, args.horizons, args.allowed_tiers)
    all_samples = val_samples + test_samples
    model, ds_all, dev, _ = load_model(dataset_root, baseline_output_dir, args.model_method, all_samples, args.device)
    test_offset = len(val_samples)
    label_cache: Dict[str, np.ndarray] = {}

    validation_candidates = evaluate_thresholds(
        dataset_root,
        val_samples,
        0,
        model,
        ds_all,
        dev,
        thresholds,
        label_cache,
        verbose=args.verbose,
    )
    validation_summary = summarize_candidates(validation_candidates)
    selected_threshold = float(validation_summary.iloc[0]["threshold"])

    test_candidates = evaluate_thresholds(
        dataset_root,
        test_samples,
        test_offset,
        model,
        ds_all,
        dev,
        [selected_threshold],
        label_cache,
        verbose=args.verbose,
    )
    test_selected = attach_default_direct(test_candidates, baseline_output_dir, args.model_method)
    overall = summarize_selected(test_selected, [])
    by_tier = summarize_selected(test_selected, ["tier"])
    by_horizon = summarize_selected(test_selected, ["horizon"])
    by_growth = summarize_selected(test_selected, ["absolute_growth_bin"])
    by_horizon_growth = summarize_selected(test_selected, ["horizon", "absolute_growth_bin"])
    bootstrap = bootstrap_summary(test_selected, args.n_bootstrap, args.seed)

    validation_candidates.to_csv(output_dir / "direct_threshold_validation_candidates.csv", index=False)
    validation_summary.to_csv(output_dir / "direct_threshold_validation_summary.csv", index=False)
    test_selected.to_csv(output_dir / "direct_threshold_test_samples.csv", index=False)
    overall.to_csv(output_dir / "direct_threshold_test_overall.csv", index=False)
    by_tier.to_csv(output_dir / "direct_threshold_test_by_tier.csv", index=False)
    by_horizon.to_csv(output_dir / "direct_threshold_test_by_horizon.csv", index=False)
    by_growth.to_csv(output_dir / "direct_threshold_test_by_growth_bin.csv", index=False)
    by_horizon_growth.to_csv(output_dir / "direct_threshold_test_by_horizon_growth_bin.csv", index=False)
    bootstrap.to_csv(output_dir / "direct_threshold_bootstrap.csv", index=False)
    with (output_dir / "direct_threshold_selected.json").open("w", encoding="utf-8") as f:
        json.dump({"selected_threshold": selected_threshold}, f, indent=2)
    write_report(
        output_dir / "direct_threshold_control_report.md",
        selected_threshold,
        validation_summary,
        overall,
        by_growth,
        bootstrap,
    )

    print(
        json.dumps(
            {
                "dataset_root": str(dataset_root),
                "baseline_output_dir": str(baseline_output_dir),
                "model_method": args.model_method,
                "selected_threshold": selected_threshold,
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
