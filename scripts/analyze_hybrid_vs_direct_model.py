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
from baselines.tasks import infer_tier_from_patient_id, patient_paths


KEY_COLS = ["patient_id", "input_idx", "target_idx", "horizon", "delta_days"]


def _standardize_label(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 5 and arr.shape[1] == 1:
        return (arr > 0).astype(np.float32)
    if arr.ndim == 4:
        return (arr[:, None, ...] > 0).astype(np.float32)
    raise ValueError(f"Unsupported label shape: {arr.shape}")


def _qbin(series: pd.Series, labels: List[str]) -> pd.Series:
    if series.dropna().nunique() < len(labels):
        return pd.Series(["all"] * len(series), index=series.index)
    return pd.qcut(series, q=len(labels), labels=labels, duplicates="drop")


def _absolute_growth_bins(df: pd.DataFrame) -> pd.Series:
    nonzero = df.loc[df["growth_volume_vox"] > 0, "growth_volume_vox"].dropna()
    if nonzero.empty:
        small_max = 0.0
        large_min = 0.0
    else:
        small_max = float(nonzero.quantile(0.33))
        large_min = float(nonzero.quantile(0.67))

    def label(v: float) -> str:
        if pd.isna(v):
            return "unknown"
        if v <= 0:
            return "zero"
        if v <= small_max:
            return "small_nonzero"
        if v <= large_min:
            return "medium_nonzero"
        return "large_nonzero"

    return df["growth_volume_vox"].apply(label)


def _load_direct_per_sample(baseline_output_dir: Path, direct_method: str) -> pd.DataFrame:
    path = baseline_output_dir / f"{direct_method}_per_sample.json"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No rows found in {path}")
    return df.rename(columns={"dice": "direct_model_dice"})


def _load_hybrid_samples(hybrid_policy_dir: Path, score_source: str | None) -> pd.DataFrame:
    path = hybrid_policy_dir / "hybrid_policy_test_samples.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if score_source:
        df = df[df["score_source"] == score_source].copy()
    if df.empty:
        raise ValueError(f"No hybrid policy rows available for score_source={score_source!r}")
    keep = KEY_COLS + [
        "score_source",
        "budget_policy",
        "gate_name",
        "gate_active",
        "selected_policy_available",
        "growth_budget_vox",
        "budget_to_true_growth_ratio",
        "hybrid_policy_dice",
        "hybrid_policy_gap_vs_locf",
        "locf_dice",
    ]
    return df[[c for c in keep if c in df.columns]].copy()


def compute_mask_features(dataset_root: Path, samples: pd.DataFrame) -> pd.DataFrame:
    label_cache: Dict[str, np.ndarray] = {}
    rows = []

    for row in samples[KEY_COLS].drop_duplicates().itertuples(index=False):
        patient_id = str(row.patient_id)
        if patient_id not in label_cache:
            p = patient_paths(dataset_root, patient_id)
            label_cache[patient_id] = _standardize_label(np.load(p["label"], mmap_mode="r"))
        labels = label_cache[patient_id]
        input_mask = (labels[int(row.input_idx), 0] > 0)
        target_mask = (labels[int(row.target_idx), 0] > 0)
        growth = target_mask & ~input_mask
        loss = input_mask & ~target_mask
        stable_core = input_mask & target_mask
        union = input_mask | target_mask

        input_volume = int(input_mask.sum())
        target_volume = int(target_mask.sum())
        growth_volume = int(growth.sum())
        loss_volume = int(loss.sum())
        stable_core_volume = int(stable_core.sum())
        union_volume = int(union.sum())
        net_delta = target_volume - input_volume

        rows.append(
            {
                "patient_id": patient_id,
                "input_idx": int(row.input_idx),
                "target_idx": int(row.target_idx),
                "horizon": int(row.horizon),
                "delta_days": float(row.delta_days),
                "tier": infer_tier_from_patient_id(patient_id),
                "input_volume_vox": input_volume,
                "target_volume_vox": target_volume,
                "growth_volume_vox": growth_volume,
                "loss_volume_vox": loss_volume,
                "stable_core_volume_vox": stable_core_volume,
                "union_volume_vox": union_volume,
                "net_delta_volume_vox": net_delta,
                "abs_delta_volume_vox": abs(net_delta),
                "relative_new_growth": growth_volume / max(1, input_volume),
                "relative_loss": loss_volume / max(1, input_volume),
                "relative_net_growth": net_delta / max(1, input_volume),
                "growth_to_loss_ratio": growth_volume / max(1, loss_volume),
                "loss_fraction_of_union": loss_volume / max(1, union_volume),
                "growth_fraction_of_union": growth_volume / max(1, union_volume),
                "locf_dice_from_masks": dice_np(input_mask.astype(np.float32), target_mask.astype(np.float32)),
            }
        )

    df = pd.DataFrame(rows)
    df["absolute_growth_bin"] = _absolute_growth_bins(df)
    df["relative_growth_bin"] = _qbin(df["relative_new_growth"], ["low", "medium", "high"])
    df["relative_loss_bin"] = _qbin(df["relative_loss"], ["low", "medium", "high"])
    df["net_growth_direction"] = np.where(
        df["net_delta_volume_vox"] > 0,
        "net_growth",
        np.where(df["net_delta_volume_vox"] < 0, "net_shrinkage", "net_stable"),
    )
    return df


def summarize(df: pd.DataFrame, group_cols: Iterable[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    cols = [c for c in group_cols if c in df.columns]
    group_df = df if cols else df.assign(_overall="overall")
    by = cols if cols else ["_overall"]
    out = (
        group_df.groupby(by, observed=True, dropna=False)
        .agg(
            count=("direct_minus_hybrid_dice", "size"),
            mean_direct_dice=("direct_model_dice", "mean"),
            mean_hybrid_dice=("hybrid_policy_dice", "mean"),
            mean_locf_dice=("locf_dice", "mean"),
            mean_direct_minus_hybrid=("direct_minus_hybrid_dice", "mean"),
            median_direct_minus_hybrid=("direct_minus_hybrid_dice", "median"),
            direct_win_rate_vs_hybrid=("direct_beats_hybrid", "mean"),
            hybrid_win_rate_vs_direct=("hybrid_beats_direct", "mean"),
            mean_direct_gap_vs_locf=("direct_gap_vs_locf", "mean"),
            mean_hybrid_gap_vs_locf=("hybrid_policy_gap_vs_locf", "mean"),
            gate_active_rate=("gate_active", "mean"),
            mean_growth_volume_vox=("growth_volume_vox", "mean"),
            mean_loss_volume_vox=("loss_volume_vox", "mean"),
            mean_relative_new_growth=("relative_new_growth", "mean"),
            mean_relative_loss=("relative_loss", "mean"),
        )
        .reset_index()
    )
    if "_overall" in out.columns:
        out = out.drop(columns=["_overall"])
    return out.sort_values(cols).reset_index(drop=True) if cols else out


def correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    features = [
        "growth_volume_vox",
        "loss_volume_vox",
        "relative_new_growth",
        "relative_loss",
        "net_delta_volume_vox",
        "growth_to_loss_ratio",
        "growth_budget_vox",
        "budget_to_true_growth_ratio",
        "delta_days",
        "input_volume_vox",
    ]
    rows = []
    for feature in features:
        if feature not in df.columns:
            continue
        x = df[feature].replace([np.inf, -np.inf], np.nan)
        y = df["direct_minus_hybrid_dice"].replace([np.inf, -np.inf], np.nan)
        mask = x.notna() & y.notna()
        if int(mask.sum()) < 3:
            continue
        rows.append(
            {
                "feature": feature,
                "n": int(mask.sum()),
                "pearson_corr_with_direct_advantage": float(x[mask].corr(y[mask], method="pearson")),
                "spearman_corr_with_direct_advantage": float(x[mask].corr(y[mask], method="spearman")),
            }
        )
    return pd.DataFrame(rows).sort_values("spearman_corr_with_direct_advantage", ascending=False)


def bootstrap_mean(values: np.ndarray, n_bootstrap: int, seed: int) -> Dict:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return {"n_samples": 0, "mean": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(vals), len(vals))
        boot.append(float(vals[idx].mean()))
    boot = np.asarray(boot)
    return {
        "n_samples": int(len(vals)),
        "mean": float(vals.mean()),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
    }


def write_report(
    path: Path,
    overall: pd.DataFrame,
    by_growth: pd.DataFrame,
    by_loss: pd.DataFrame,
    by_net: pd.DataFrame,
    corr: pd.DataFrame,
    bootstrap: Dict,
) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Hybrid Policy vs Direct Model Bottleneck Analysis\n\n")
        f.write(
            "This analysis compares a validation-selected hybrid growth-front policy against the direct model forecast. "
            "It asks where the direct model's Dice advantage comes from.\n\n"
        )
        f.write("## Overall\n\n")
        f.write(overall.to_markdown(index=False) if not overall.empty else "No overall summary.")
        f.write("\n\n## Bootstrap: Direct Minus Hybrid Dice\n\n")
        f.write(pd.DataFrame([bootstrap]).to_markdown(index=False))
        f.write("\n\n## By Absolute Growth Bin\n\n")
        f.write(by_growth.to_markdown(index=False) if not by_growth.empty else "No growth-bin summary.")
        f.write("\n\n## By Relative Loss Bin\n\n")
        f.write(by_loss.to_markdown(index=False) if not by_loss.empty else "No loss-bin summary.")
        f.write("\n\n## By Net Growth Direction\n\n")
        f.write(by_net.to_markdown(index=False) if not by_net.empty else "No net-direction summary.")
        f.write("\n\n## Correlations With Direct-Model Advantage\n\n")
        f.write(corr.to_markdown(index=False) if not corr.empty else "No correlation summary.")
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare a selected hybrid forecast policy against direct model forecasts and diagnose bottlenecks."
    )
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--baseline_output_dir", type=str, required=True)
    parser.add_argument("--hybrid_policy_dir", type=str, required=True)
    parser.add_argument("--direct_method", type=str, default="resunet_image_mask")
    parser.add_argument("--hybrid_score_source", type=str, default="hybrid_distance_resunet_image_mask_a0.75")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    baseline_output_dir = Path(args.baseline_output_dir)
    hybrid_policy_dir = Path(args.hybrid_policy_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    direct = _load_direct_per_sample(baseline_output_dir, args.direct_method)
    hybrid = _load_hybrid_samples(hybrid_policy_dir, args.hybrid_score_source)
    features = compute_mask_features(dataset_root, hybrid)
    merged = hybrid.merge(direct[KEY_COLS + ["direct_model_dice"]], on=KEY_COLS, how="inner")
    merged = merged.merge(features, on=KEY_COLS, how="left", suffixes=("", "_feature"))
    if "tier_feature" in merged.columns:
        merged["tier"] = merged.get("tier", merged["tier_feature"]).fillna(merged["tier_feature"])
        merged = merged.drop(columns=["tier_feature"])

    merged["direct_gap_vs_locf"] = merged["direct_model_dice"] - merged["locf_dice"]
    merged["direct_minus_hybrid_dice"] = merged["direct_model_dice"] - merged["hybrid_policy_dice"]
    merged["direct_beats_hybrid"] = merged["direct_minus_hybrid_dice"] > 0
    merged["hybrid_beats_direct"] = merged["direct_minus_hybrid_dice"] < 0
    merged["both_beat_locf"] = (merged["direct_gap_vs_locf"] > 0) & (merged["hybrid_policy_gap_vs_locf"] > 0)
    merged["direct_only_beats_locf"] = (merged["direct_gap_vs_locf"] > 0) & (merged["hybrid_policy_gap_vs_locf"] <= 0)
    merged["hybrid_only_beats_locf"] = (merged["direct_gap_vs_locf"] <= 0) & (merged["hybrid_policy_gap_vs_locf"] > 0)

    overall = summarize(merged, [])
    by_tier = summarize(merged, ["tier"])
    by_horizon = summarize(merged, ["horizon"])
    by_growth = summarize(merged, ["absolute_growth_bin"])
    by_loss = summarize(merged, ["relative_loss_bin"])
    by_net = summarize(merged, ["net_growth_direction"])
    by_gate = summarize(merged, ["gate_active"])
    corr = correlation_table(merged)
    boot = bootstrap_mean(merged["direct_minus_hybrid_dice"].to_numpy(), args.n_bootstrap, args.seed)

    merged.to_csv(output_dir / "hybrid_vs_direct_samples.csv", index=False)
    overall.to_csv(output_dir / "hybrid_vs_direct_overall.csv", index=False)
    by_tier.to_csv(output_dir / "hybrid_vs_direct_by_tier.csv", index=False)
    by_horizon.to_csv(output_dir / "hybrid_vs_direct_by_horizon.csv", index=False)
    by_growth.to_csv(output_dir / "hybrid_vs_direct_by_absolute_growth_bin.csv", index=False)
    by_loss.to_csv(output_dir / "hybrid_vs_direct_by_relative_loss_bin.csv", index=False)
    by_net.to_csv(output_dir / "hybrid_vs_direct_by_net_growth_direction.csv", index=False)
    by_gate.to_csv(output_dir / "hybrid_vs_direct_by_gate_active.csv", index=False)
    corr.to_csv(output_dir / "direct_advantage_correlations.csv", index=False)
    pd.DataFrame([boot]).to_csv(output_dir / "direct_minus_hybrid_bootstrap.csv", index=False)
    write_report(output_dir / "hybrid_vs_direct_bottleneck_report.md", overall, by_growth, by_loss, by_net, corr, boot)

    report = {
        "dataset_root": str(dataset_root),
        "baseline_output_dir": str(baseline_output_dir),
        "hybrid_policy_dir": str(hybrid_policy_dir),
        "direct_method": args.direct_method,
        "hybrid_score_source": args.hybrid_score_source,
        "n_samples": int(len(merged)),
        "bootstrap_direct_minus_hybrid": boot,
        "output_dir": str(output_dir),
        "files": [
            "hybrid_vs_direct_samples.csv",
            "hybrid_vs_direct_overall.csv",
            "hybrid_vs_direct_by_tier.csv",
            "hybrid_vs_direct_by_horizon.csv",
            "hybrid_vs_direct_by_absolute_growth_bin.csv",
            "hybrid_vs_direct_by_relative_loss_bin.csv",
            "hybrid_vs_direct_by_net_growth_direction.csv",
            "hybrid_vs_direct_by_gate_active.csv",
            "direct_advantage_correlations.csv",
            "direct_minus_hybrid_bootstrap.csv",
            "hybrid_vs_direct_bottleneck_report.md",
        ],
    }
    with (output_dir / "hybrid_vs_direct_bottleneck_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
