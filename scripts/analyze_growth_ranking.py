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
from baselines.tasks import build_samples_for_split, infer_tier_from_patient_id, patient_paths


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


def _average_precision_binary(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.uint8).reshape(-1)
    s = np.asarray(score, dtype=np.float32).reshape(-1)
    positives = int(y.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    rank = np.arange(1, len(y_sorted) + 1)
    precision = tp / rank
    return float((precision * y_sorted).sum() / positives)


def _recall_at_k(y_true: np.ndarray, score: np.ndarray, k: int) -> float:
    y = np.asarray(y_true, dtype=np.uint8).reshape(-1)
    s = np.asarray(score, dtype=np.float32).reshape(-1)
    positives = int(y.sum())
    if positives == 0:
        return float("nan")
    k = int(max(1, min(k, len(y))))
    order = np.argsort(-s, kind="mergesort")[:k]
    return float(y[order].sum() / positives)


def _ranking_row(
    method: str,
    patient_id: str,
    input_idx: int,
    target_idx: int,
    horizon: int,
    delta_days: float,
    tier: str,
    y_true: np.ndarray,
    score: np.ndarray | None,
    source: str,
) -> Dict:
    y = np.asarray(y_true, dtype=np.uint8).reshape(-1)
    candidate_count = int(len(y))
    growth_count = int(y.sum())
    k_growth = max(1, growth_count)
    k_1pct = max(1, int(0.01 * candidate_count))
    k_5pct = max(1, int(0.05 * candidate_count))

    if growth_count == 0:
        ap = float("nan")
        recall_growth = float("nan")
        recall_1pct = float("nan")
        recall_5pct = float("nan")
    elif score is None:
        prevalence = growth_count / max(1, candidate_count)
        ap = float(prevalence)
        recall_growth = float(min(1.0, k_growth / max(1, candidate_count)))
        recall_1pct = float(min(1.0, k_1pct / max(1, candidate_count)))
        recall_5pct = float(min(1.0, k_5pct / max(1, candidate_count)))
    else:
        ap = _average_precision_binary(y, score)
        recall_growth = _recall_at_k(y, score, k_growth)
        recall_1pct = _recall_at_k(y, score, k_1pct)
        recall_5pct = _recall_at_k(y, score, k_5pct)

    return {
        "method": method,
        "ranking_source": source,
        "patient_id": patient_id,
        "input_idx": input_idx,
        "target_idx": target_idx,
        "horizon": horizon,
        "delta_days": delta_days,
        "tier": tier,
        "growth_volume_vox": growth_count,
        "candidate_vox": candidate_count,
        "growth_average_precision": ap,
        "growth_recall_at_growth_volume": recall_growth,
        "growth_recall_at_1pct_candidates": recall_1pct,
        "growth_recall_at_5pct_candidates": recall_5pct,
    }


def _distance_to_input_score(input_mask: np.ndarray) -> np.ndarray | None:
    if int(input_mask.sum()) == 0:
        return None
    try:
        from scipy.ndimage import distance_transform_edt
    except Exception as e:
        print(f"[WARN] Skipping distance-to-input ranking baseline because scipy is unavailable: {e}")
        return None
    dist = distance_transform_edt(~input_mask)
    return -dist.astype(np.float32)


def _load_per_sample(output_dir: Path, methods: Iterable[str]) -> pd.DataFrame:
    rows = []
    for method in methods:
        path = output_dir / f"{method}_per_sample.json"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        for row in payload:
            out = dict(row)
            out["method"] = method
            rows.append(out)
    if not rows:
        raise FileNotFoundError(f"No '*_per_sample.json' files found in {output_dir}")
    return pd.DataFrame(rows)


def _load_summary(output_dir: Path, method: str) -> Dict | None:
    path = output_dir / f"{method}_summary.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_growth_features(
    dataset_root: Path,
    split: str,
    fit_sessions: int,
    horizons: str,
    allowed_tiers: str | None,
) -> pd.DataFrame:
    samples = build_samples_for_split(
        dataset_root=dataset_root,
        split=split,
        fit_sessions=fit_sessions,
        horizons=horizons,
        allowed_tiers=allowed_tiers,
    )

    label_cache: Dict[str, np.ndarray] = {}
    rows = []
    for s in samples:
        if s.patient_id not in label_cache:
            p = patient_paths(dataset_root, s.patient_id)
            label_cache[s.patient_id] = _standardize_label(np.load(p["label"]))
        labels = label_cache[s.patient_id]
        input_mask = labels[s.input_idx] > 0
        target_mask = labels[s.target_idx] > 0
        growth = target_mask & ~input_mask
        loss = input_mask & ~target_mask
        union = input_mask | target_mask

        input_volume = int(input_mask.sum())
        target_volume = int(target_mask.sum())
        growth_volume = int(growth.sum())
        loss_volume = int(loss.sum())
        union_volume = int(union.sum())
        abs_delta = int(abs(target_volume - input_volume))

        rows.append(
            {
                "patient_id": s.patient_id,
                "input_idx": s.input_idx,
                "target_idx": s.target_idx,
                "horizon": s.horizon,
                "delta_days": s.delta_days,
                "tier": infer_tier_from_patient_id(s.patient_id),
                "current_treatment": s.current_treatment,
                "target_treatment": s.target_treatment,
                "input_volume_vox": input_volume,
                "target_volume_vox": target_volume,
                "net_delta_volume_vox": target_volume - input_volume,
                "abs_delta_volume_vox": abs_delta,
                "growth_volume_vox": growth_volume,
                "loss_volume_vox": loss_volume,
                "union_volume_vox": union_volume,
                "relative_net_growth": (target_volume - input_volume) / max(1, input_volume),
                "relative_abs_change": abs_delta / max(1, input_volume),
                "relative_new_growth": growth_volume / max(1, input_volume),
                "growth_fraction_of_target": growth_volume / max(1, target_volume),
                "loss_fraction_of_input": loss_volume / max(1, input_volume),
                "locf_dice_from_masks": dice_np(input_mask.astype(np.float32), target_mask.astype(np.float32)),
            }
        )

    df = pd.DataFrame(rows)
    df["abs_change_bin"] = _qbin(df["relative_abs_change"], ["low", "medium", "high"])
    df["new_growth_bin"] = _qbin(df["relative_new_growth"], ["low", "medium", "high"])
    df["net_growth_bin"] = _qbin(df["relative_net_growth"], ["low", "medium", "high"])
    return df


def summarize_dice_by_growth(per_sample: pd.DataFrame, features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = per_sample.merge(features, on=KEY_COLS, how="inner")
    group_cols = ["method", "new_growth_bin"]
    dice_by_growth = (
        merged.groupby(group_cols, dropna=False)
        .agg(
            count=("dice", "size"),
            mean_dice=("dice", "mean"),
            std_dice=("dice", "std"),
            mean_relative_new_growth=("relative_new_growth", "mean"),
            mean_growth_volume_vox=("growth_volume_vox", "mean"),
        )
        .reset_index()
        .sort_values(group_cols)
    )

    locf = merged[merged["method"] == "locf"][KEY_COLS + ["dice"]].rename(columns={"dice": "locf_dice"})
    pair_rows = []
    for method in sorted(set(merged["method"]) - {"locf"}):
        cur = merged[merged["method"] == method].merge(locf, on=KEY_COLS, how="inner")
        cur["dice_gap_vs_locf"] = cur["dice"] - cur["locf_dice"]
        cur["beats_locf"] = cur["dice_gap_vs_locf"] > 0
        pair_rows.append(cur)
    if pair_rows:
        pair = pd.concat(pair_rows, ignore_index=True)
        pairwise = (
            pair.groupby(["method", "new_growth_bin"], dropna=False)
            .agg(
                count=("dice_gap_vs_locf", "size"),
                win_rate=("beats_locf", "mean"),
                mean_gap_vs_locf=("dice_gap_vs_locf", "mean"),
                median_gap_vs_locf=("dice_gap_vs_locf", "median"),
                mean_relative_new_growth=("relative_new_growth", "mean"),
            )
            .reset_index()
            .sort_values(["method", "new_growth_bin"])
        )
    else:
        pairwise = pd.DataFrame()
    return dice_by_growth, pairwise


def _infer_checkpoint(output_dir: Path, method: str) -> Path | None:
    summary = _load_summary(output_dir, method)
    if summary and summary.get("checkpoint"):
        ckpt = Path(summary["checkpoint"])
        if ckpt.exists():
            return ckpt
        local = output_dir / ckpt.name
        if local.exists():
            return local
    matches = sorted(output_dir.glob(f"model_best_{method}.pt"))
    if matches:
        return matches[0]
    return None


def _checkpoint_specs(output_dir: Path, methods: Iterable[str]) -> Dict[str, Dict]:
    specs = {}
    for method in methods:
        if method == "locf":
            continue
        summary = _load_summary(output_dir, method)
        ckpt = _infer_checkpoint(output_dir, method)
        if summary is None or ckpt is None:
            continue
        specs[method] = {
            "checkpoint": ckpt,
            "model_variant": summary.get("model_variant", method.split("_")[0]),
            "input_mode": summary.get("input_mode", "image_mask" if "image_mask" in method else "mask"),
        }
    return specs


def compute_ranking_metrics(
    dataset_root: Path,
    split: str,
    fit_sessions: int,
    horizons: str,
    allowed_tiers: str | None,
    output_dir: Path,
    methods: Iterable[str],
    device: str,
) -> pd.DataFrame:
    samples = build_samples_for_split(dataset_root, split, fit_sessions, horizons, allowed_tiers=allowed_tiers)
    label_cache: Dict[str, np.ndarray] = {}
    rows = []

    for s in samples:
        if s.patient_id not in label_cache:
            p = patient_paths(dataset_root, s.patient_id)
            label_cache[s.patient_id] = _standardize_label(np.load(p["label"]))
        labels = label_cache[s.patient_id]
        input_mask = (labels[s.input_idx] > 0)[0]
        target_mask = (labels[s.target_idx] > 0)[0]
        growth = target_mask & ~input_mask
        outside_input = ~input_mask
        y = growth[outside_input]
        tier = infer_tier_from_patient_id(s.patient_id)

        rows.append(
            _ranking_row(
                method="random_prevalence",
                patient_id=s.patient_id,
                input_idx=s.input_idx,
                target_idx=s.target_idx,
                horizon=s.horizon,
                delta_days=s.delta_days,
                tier=tier,
                y_true=y,
                score=None,
                source="reference",
            )
        )

        distance_score_full = _distance_to_input_score(input_mask)
        if distance_score_full is not None:
            rows.append(
                _ranking_row(
                    method="distance_to_input_mask",
                    patient_id=s.patient_id,
                    input_idx=s.input_idx,
                    target_idx=s.target_idx,
                    horizon=s.horizon,
                    delta_days=s.delta_days,
                    tier=tier,
                    y_true=y,
                    score=distance_score_full[outside_input],
                    source="reference",
                )
            )

    specs = _checkpoint_specs(output_dir, methods)
    if not specs:
        return pd.DataFrame(rows)

    try:
        import torch

        from baselines.unet import _TorchForecastDataset, _build_torch_model
    except Exception as e:
        print(f"[WARN] Skipping ranking metrics because PyTorch/model code is unavailable: {e}")
        return pd.DataFrame(rows)

    dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))

    for method, spec in specs.items():
        ds = _TorchForecastDataset(dataset_root, samples, input_mode=spec["input_mode"], cache_arrays=True)
        sample_x, _, _ = ds[0]
        ckpt = torch.load(spec["checkpoint"], map_location=dev, weights_only=False)
        base_channels = int(ckpt.get("base_channels", 12))
        in_channels = int(ckpt.get("in_channels", sample_x.shape[0]))
        model = _build_torch_model(in_channels, base_channels, model_variant=spec["model_variant"])
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(dev)
        model.eval()

        for i, s in enumerate(samples):
            x, _, _ = ds[i]
            if s.patient_id not in label_cache:
                p = patient_paths(dataset_root, s.patient_id)
                label_cache[s.patient_id] = _standardize_label(np.load(p["label"]))
            labels = label_cache[s.patient_id]
            input_mask = (labels[s.input_idx] > 0)[0]
            target_mask = (labels[s.target_idx] > 0)[0]
            growth = target_mask & ~input_mask
            outside_input = ~input_mask

            with torch.no_grad():
                logits = model(x[None].to(dev))
                prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()

            y = growth[outside_input]
            score = prob[outside_input]
            rows.append(
                _ranking_row(
                    method=method,
                    patient_id=s.patient_id,
                    input_idx=s.input_idx,
                    target_idx=s.target_idx,
                    horizon=s.horizon,
                    delta_days=s.delta_days,
                    tier=infer_tier_from_patient_id(s.patient_id),
                    y_true=y,
                    score=score,
                    source="model",
                )
            )

    return pd.DataFrame(rows)


def write_report(
    path: Path,
    features: pd.DataFrame,
    dice_by_growth: pd.DataFrame,
    pairwise: pd.DataFrame,
    ranking: pd.DataFrame,
) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Growth-Aware Forecast Evaluation\n\n")
        f.write("This report tests whether forecast performance changes with the amount of actual future growth.\n\n")
        f.write("## Growth Feature Summary\n\n")
        summary_cols = [
            "input_volume_vox",
            "target_volume_vox",
            "growth_volume_vox",
            "loss_volume_vox",
            "relative_new_growth",
            "relative_abs_change",
            "locf_dice_from_masks",
        ]
        f.write(features[summary_cols].describe().to_markdown())
        f.write("\n\n## Dice By New-Growth Bin\n\n")
        f.write(dice_by_growth.to_markdown(index=False) if not dice_by_growth.empty else "No dice summary available.")
        f.write("\n\n## Model Gain Over LOCF By New-Growth Bin\n\n")
        f.write(pairwise.to_markdown(index=False) if not pairwise.empty else "No pairwise model-vs-LOCF summary available.")
        f.write("\n\n## Forward-Growth Ranking Metrics\n\n")
        if ranking.empty:
            f.write("Ranking metrics were not computed because no accessible model checkpoints were found.")
        else:
            rank_summary = (
                ranking.groupby(["method", "horizon"])
                .agg(
                    count=("growth_average_precision", "size"),
                    mean_ap=("growth_average_precision", "mean"),
                    mean_recall_at_growth_volume=("growth_recall_at_growth_volume", "mean"),
                    mean_recall_at_1pct=("growth_recall_at_1pct_candidates", "mean"),
                    mean_recall_at_5pct=("growth_recall_at_5pct_candidates", "mean"),
                )
                .reset_index()
            )
            f.write(rank_summary.to_markdown(index=False))
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze short-horizon forecasts by actual future-growth size and optional growth-region ranking metrics."
    )
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--fit_sessions", type=int, default=3)
    parser.add_argument("--horizons", type=str, default="1,2,3")
    parser.add_argument("--allowed_tiers", type=str, default=None)
    parser.add_argument("--baseline_output_dir", type=str, required=True)
    parser.add_argument(
        "--methods",
        type=str,
        default="locf,unet_image_mask,resunet_image_mask",
        help="Comma-separated method prefixes matching '*_per_sample.json' and '*_summary.json'.",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--skip_ranking", action="store_true")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    baseline_output_dir = Path(args.baseline_output_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    features = build_growth_features(
        dataset_root=dataset_root,
        split=args.split,
        fit_sessions=args.fit_sessions,
        horizons=args.horizons,
        allowed_tiers=args.allowed_tiers,
    )
    per_sample = _load_per_sample(baseline_output_dir, methods)
    dice_by_growth, pairwise = summarize_dice_by_growth(per_sample, features)
    ranking = (
        pd.DataFrame()
        if args.skip_ranking
        else compute_ranking_metrics(
            dataset_root=dataset_root,
            split=args.split,
            fit_sessions=args.fit_sessions,
            horizons=args.horizons,
            allowed_tiers=args.allowed_tiers,
            output_dir=baseline_output_dir,
            methods=methods,
            device=args.device,
        )
    )

    features.to_csv(output_dir / "growth_sample_features.csv", index=False)
    dice_by_growth.to_csv(output_dir / "dice_by_new_growth_bin.csv", index=False)
    pairwise.to_csv(output_dir / "model_gain_vs_locf_by_new_growth_bin.csv", index=False)
    if not ranking.empty:
        ranking.to_csv(output_dir / "growth_ranking_metrics.csv", index=False)
    write_report(output_dir / "growth_aware_evaluation_report.md", features, dice_by_growth, pairwise, ranking)

    print(
        json.dumps(
            {
                "dataset_root": str(dataset_root),
                "baseline_output_dir": str(baseline_output_dir),
                "methods": methods,
                "n_samples": int(len(features)),
                "ranking_metrics_computed": bool(not ranking.empty),
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
