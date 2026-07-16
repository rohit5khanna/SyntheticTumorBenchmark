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
from scripts.analyze_growth_ranking import _checkpoint_specs, _rank_normalize_score


KEY_COLS = ["patient_id", "input_idx", "target_idx", "horizon", "delta_days"]


def _standardize_label(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 5 and arr.shape[1] == 1:
        return (arr > 0).astype(np.float32)
    if arr.ndim == 4:
        return (arr[:, None, ...] > 0).astype(np.float32)
    raise ValueError(f"Unsupported label shape: {arr.shape}")


def parse_threshold_values(payload: str) -> List[float]:
    vals = []
    for item in payload.split(","):
        item = item.strip()
        if not item:
            continue
        val = float(item)
        if val < 0.0 or val > 1.0:
            raise ValueError("Threshold values must be in [0, 1].")
        vals.append(val)
    if not vals:
        raise ValueError("Need at least one threshold quantile.")
    return sorted(set(vals))


def load_labels(dataset_root: Path, patient_id: str, cache: Dict[str, np.ndarray]) -> np.ndarray:
    if patient_id not in cache:
        cache[patient_id] = _standardize_label(np.load(patient_paths(dataset_root, patient_id)["label"]))
    return cache[patient_id]


def load_model(dataset_root: Path, baseline_output_dir: Path, method: str, samples, device: str):
    try:
        import torch
        from baselines.unet import _TorchForecastDataset, _build_torch_model
    except Exception as e:
        raise RuntimeError("PyTorch and the baseline model code are required for this analysis.") from e

    specs = _checkpoint_specs(baseline_output_dir, [method])
    if method not in specs:
        raise FileNotFoundError(
            f"Could not infer checkpoint/spec for method='{method}' under {baseline_output_dir}."
        )
    spec = specs[method]
    dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))
    ds = _TorchForecastDataset(dataset_root, samples, input_mode=spec["input_mode"], cache_arrays=False)
    sample_x, _, _ = ds[0]
    ckpt = torch.load(spec["checkpoint"], map_location=dev, weights_only=False)
    model = _build_torch_model(
        in_channels=int(ckpt.get("in_channels", sample_x.shape[0])),
        base_channels=int(ckpt.get("base_channels", 12)),
        model_variant=spec["model_variant"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(dev)
    model.eval()
    return model, ds, dev, spec


def model_probability(model, ds, idx: int, dev) -> np.ndarray:
    import torch

    x, _, _ = ds[idx]
    with torch.no_grad():
        logits = model(x[None].to(dev))
        prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
    return prob.astype(np.float32)


def distance_rank_score(input_mask: np.ndarray) -> np.ndarray:
    try:
        from scipy.ndimage import distance_transform_edt
    except Exception as e:
        raise RuntimeError("scipy is required for distance-to-input growth-field scoring.") from e
    if int(input_mask.sum()) == 0:
        return np.zeros(input_mask.shape, dtype=np.float32)
    distance_score = -distance_transform_edt(~input_mask).astype(np.float32)
    out = np.zeros(input_mask.shape, dtype=np.float32)
    outside = ~input_mask
    out[outside] = _rank_normalize_score(distance_score[outside])
    return out


def previous_growth_features(labels: np.ndarray, input_idx: int) -> dict:
    input_mask = labels[input_idx, 0] > 0
    input_volume = int(input_mask.sum())
    if input_idx <= 0:
        return {
            "previous_growth_volume_vox": 0,
            "previous_growth_ratio": 0.0,
            "previous_loss_volume_vox": 0,
            "previous_loss_ratio": 0.0,
        }
    prev_mask = labels[input_idx - 1, 0] > 0
    prev_growth = input_mask & ~prev_mask
    prev_loss = prev_mask & ~input_mask
    return {
        "previous_growth_volume_vox": int(prev_growth.sum()),
        "previous_growth_ratio": float(prev_growth.sum() / max(1, prev_mask.sum())),
        "previous_loss_volume_vox": int(prev_loss.sum()),
        "previous_loss_ratio": float(prev_loss.sum() / max(1, prev_mask.sum())),
    }


def sample_context_features(sample, labels: np.ndarray) -> dict:
    input_mask = labels[sample.input_idx, 0] > 0
    target_mask = labels[sample.target_idx, 0] > 0
    growth = target_mask & ~input_mask
    loss = input_mask & ~target_mask
    input_volume = int(input_mask.sum())
    target_volume = int(target_mask.sum())
    prev = previous_growth_features(labels, sample.input_idx)
    return {
        "tier": infer_tier_from_patient_id(sample.patient_id),
        "input_volume_vox": input_volume,
        "target_volume_vox": target_volume,
        "growth_volume_vox": int(growth.sum()),
        "loss_volume_vox": int(loss.sum()),
        "relative_new_growth": float(growth.sum() / max(1, input_volume)),
        "relative_loss": float(loss.sum() / max(1, input_volume)),
        "relative_net_growth": float((target_volume - input_volume) / max(1, input_volume)),
        "locf_dice": dice_np(input_mask.astype(np.float32), target_mask.astype(np.float32)),
        "current_treatment": float(sample.current_treatment),
        "target_treatment": float(sample.target_treatment),
        **prev,
    }


def voxel_feature_matrix(
    model_score: np.ndarray,
    distance_score: np.ndarray,
    outside_input: np.ndarray,
    context: dict,
) -> np.ndarray:
    model_v = model_score[outside_input].astype(np.float32)
    distance_v = distance_score[outside_input].astype(np.float32)
    n = int(model_v.shape[0])
    delta = np.full(n, float(context["delta_days"]) / 180.0, dtype=np.float32)
    log_input = np.full(n, np.log1p(float(context["input_volume_vox"])) / 10.0, dtype=np.float32)
    prev_growth = np.full(n, float(context["previous_growth_ratio"]), dtype=np.float32)
    treatment = np.full(n, float(context["current_treatment"]), dtype=np.float32)
    interaction = (model_v * distance_v).astype(np.float32)
    return np.column_stack([model_v, distance_v, interaction, delta, log_input, prev_growth, treatment]).astype(np.float32)


def sample_training_voxels(
    x_all: np.ndarray,
    y_all: np.ndarray,
    max_pos: int,
    max_neg: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    pos = np.flatnonzero(y_all > 0)
    neg = np.flatnonzero(y_all <= 0)
    if len(pos) > max_pos:
        pos = rng.choice(pos, size=max_pos, replace=False)
    if len(neg) > max_neg:
        neg = rng.choice(neg, size=max_neg, replace=False)
    idx = np.concatenate([pos, neg])
    rng.shuffle(idx)
    return x_all[idx], y_all[idx]


def train_calibrator(train_payloads: List[dict], max_pos: int, max_neg: int, seed: int):
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as e:
        raise RuntimeError("scikit-learn is required for the calibrated growth-field decoder.") from e

    rng = np.random.default_rng(seed)
    x_parts = []
    y_parts = []
    for payload in train_payloads:
        x_s, y_s = sample_training_voxels(
            payload["features"],
            payload["growth_label"],
            max_pos=max_pos,
            max_neg=max_neg,
            rng=rng,
        )
        if len(y_s):
            x_parts.append(x_s)
            y_parts.append(y_s)
    if not x_parts:
        raise ValueError("No training voxels available for calibrator.")
    x = np.vstack(x_parts)
    y = np.concatenate(y_parts).astype(np.uint8)
    if int(y.sum()) == 0:
        raise ValueError("No positive future-growth voxels found for calibrator training.")

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=500, class_weight="balanced", solver="lbfgs", random_state=seed),
    )
    clf.fit(x, y)
    return clf, {"n_voxels": int(len(y)), "n_positive_voxels": int(y.sum()), "positive_fraction": float(y.mean())}


def build_payloads(
    dataset_root: Path,
    samples,
    model,
    ds,
    dev,
    split_name: str,
    label_cache: Dict[str, np.ndarray],
    verbose: bool = False,
) -> List[dict]:
    payloads = []
    for idx, sample in enumerate(samples):
        labels = load_labels(dataset_root, sample.patient_id, label_cache)
        input_mask = labels[sample.input_idx, 0] > 0
        target_mask = labels[sample.target_idx, 0] > 0
        outside = ~input_mask
        growth = (target_mask & ~input_mask)[outside].astype(np.uint8)
        model_score = model_probability(model, ds, idx, dev)
        distance_score = distance_rank_score(input_mask)
        context = sample_context_features(sample, labels)
        context.update({"delta_days": float(sample.delta_days)})
        features = voxel_feature_matrix(model_score, distance_score, outside, context)
        model_rank = _rank_normalize_score(model_score[outside]).astype(np.float32)
        hybrid_score = 0.25 * distance_score[outside].astype(np.float32) + 0.75 * model_rank
        payloads.append(
            {
                "sample": sample,
                "split": split_name,
                "input_mask": input_mask,
                "target_mask": target_mask,
                "outside_input": outside,
                "growth_label": growth,
                "features": features,
                "context": context,
                "scores": {
                    "model_probability": model_score[outside].astype(np.float32),
                    "model_rank": model_rank,
                    "distance_rank": distance_score[outside].astype(np.float32),
                    "hybrid_distance_model_a0.75": hybrid_score.astype(np.float32),
                },
            }
        )
        if verbose and (idx + 1) % 25 == 0:
            print(f"[INFO] Built {idx + 1}/{len(samples)} {split_name} payloads")
    return payloads


def add_calibrated_scores(payloads: List[dict], calibrator) -> None:
    for payload in payloads:
        payload["scores"]["calibrated_growth_probability"] = calibrator.predict_proba(payload["features"])[:, 1].astype(np.float32)


def threshold_grid(payloads: List[dict], source: str, quantiles: Iterable[float]) -> List[float]:
    vals = []
    for payload in payloads:
        score = payload["scores"][source]
        if len(score):
            vals.append(score[np.isfinite(score)])
    if not vals:
        return [float("inf")]
    all_scores = np.concatenate(vals)
    thresholds = [float(np.quantile(all_scores, q)) for q in quantiles]
    thresholds.append(float("inf"))  # LOCF-equivalent, no added growth.
    return sorted(set(thresholds))


def evaluate_payload(payload: dict, source: str, threshold: float) -> dict:
    sample = payload["sample"]
    input_mask = payload["input_mask"]
    target_mask = payload["target_mask"]
    outside = payload["outside_input"]
    pred = input_mask.copy()
    if np.isfinite(threshold):
        add = payload["scores"][source] >= threshold
        pred[outside] = pred[outside] | add
    context = payload["context"]
    dice = dice_np(pred.astype(np.float32), target_mask.astype(np.float32))
    return {
        "patient_id": sample.patient_id,
        "input_idx": int(sample.input_idx),
        "target_idx": int(sample.target_idx),
        "horizon": int(sample.horizon),
        "delta_days": float(sample.delta_days),
        "tier": context["tier"],
        "score_source": source,
        "threshold": threshold,
        "selected_voxels": int(0 if not np.isfinite(threshold) else (payload["scores"][source] >= threshold).sum()),
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
        "previous_growth_volume_vox": int(context["previous_growth_volume_vox"]),
        "previous_growth_ratio": float(context["previous_growth_ratio"]),
    }


def evaluate_thresholds(payloads: List[dict], score_sources: Iterable[str], quantiles: Iterable[float]) -> pd.DataFrame:
    rows = []
    for source in score_sources:
        for threshold in threshold_grid(payloads, source, quantiles):
            for payload in payloads:
                rows.append(evaluate_payload(payload, source, threshold))
    return pd.DataFrame(rows)


def summarize_candidates(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    return (
        df.groupby(["score_source", "threshold"], observed=True, dropna=False)
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
        .sort_values(["score_source", "mean_gap_vs_locf"], ascending=[True, False])
    )


def select_thresholds(validation_summary: pd.DataFrame) -> pd.DataFrame:
    if validation_summary.empty:
        raise ValueError("No validation threshold candidates available.")
    return (
        validation_summary.sort_values(
            ["score_source", "mean_gap_vs_locf", "win_rate_vs_locf", "mean_dice"],
            ascending=[True, False, False, False],
        )
        .groupby("score_source", observed=True, dropna=False)
        .head(1)
        .reset_index(drop=True)
    )


def apply_selected(payloads: List[dict], selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in selected.iterrows():
        source = str(row["score_source"])
        threshold = float(row["threshold"])
        for payload in payloads:
            rows.append(evaluate_payload(payload, source, threshold))
    return pd.DataFrame(rows)


def build_single_payload(
    dataset_root: Path,
    sample,
    model,
    ds,
    sample_index: int,
    dev,
    split_name: str,
    label_cache: Dict[str, np.ndarray],
    calibrator=None,
) -> dict:
    labels = load_labels(dataset_root, sample.patient_id, label_cache)
    input_mask = labels[sample.input_idx, 0] > 0
    target_mask = labels[sample.target_idx, 0] > 0
    outside = ~input_mask
    growth = (target_mask & ~input_mask)[outside].astype(np.uint8)
    model_score = model_probability(model, ds, sample_index, dev)
    distance_score = distance_rank_score(input_mask)
    context = sample_context_features(sample, labels)
    context.update({"delta_days": float(sample.delta_days)})
    features = voxel_feature_matrix(model_score, distance_score, outside, context)
    model_rank = _rank_normalize_score(model_score[outside]).astype(np.float32)
    hybrid_score = 0.25 * distance_score[outside].astype(np.float32) + 0.75 * model_rank
    scores = {
        "model_probability": model_score[outside].astype(np.float32),
        "model_rank": model_rank,
        "distance_rank": distance_score[outside].astype(np.float32),
        "hybrid_distance_model_a0.75": hybrid_score.astype(np.float32),
    }
    if calibrator is not None:
        scores["calibrated_growth_probability"] = calibrator.predict_proba(features)[:, 1].astype(np.float32)
    return {
        "sample": sample,
        "split": split_name,
        "input_mask": input_mask,
        "target_mask": target_mask,
        "outside_input": outside,
        "growth_label": growth,
        "features": features,
        "context": context,
        "scores": scores,
    }


def fit_calibrator_from_arrays(x_parts: List[np.ndarray], y_parts: List[np.ndarray], seed: int):
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as e:
        raise RuntimeError("scikit-learn is required for the calibrated growth-field decoder.") from e

    if not x_parts:
        raise ValueError("No training voxels available for calibrator.")
    x = np.vstack(x_parts)
    y = np.concatenate(y_parts).astype(np.uint8)
    if int(y.sum()) == 0:
        raise ValueError("No positive future-growth voxels found for calibrator training.")

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=500, class_weight="balanced", solver="lbfgs", random_state=seed),
    )
    clf.fit(x, y)
    return clf, {"n_voxels": int(len(y)), "n_positive_voxels": int(y.sum()), "positive_fraction": float(y.mean())}


def train_calibrator_streaming(
    dataset_root: Path,
    samples,
    model,
    ds,
    start_index: int,
    dev,
    label_cache: Dict[str, np.ndarray],
    max_pos: int,
    max_neg: int,
    seed: int,
    verbose: bool = False,
):
    rng = np.random.default_rng(seed)
    x_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    for idx, sample in enumerate(samples):
        payload = build_single_payload(
            dataset_root,
            sample,
            model,
            ds,
            start_index + idx,
            dev,
            "train",
            label_cache,
        )
        x_s, y_s = sample_training_voxels(
            payload["features"],
            payload["growth_label"],
            max_pos=max_pos,
            max_neg=max_neg,
            rng=rng,
        )
        if len(y_s):
            x_parts.append(x_s)
            y_parts.append(y_s)
        if verbose and (idx + 1) % 25 == 0:
            print(f"[INFO] Sampled calibrator voxels from {idx + 1}/{len(samples)} train samples")
        del payload
    return fit_calibrator_from_arrays(x_parts, y_parts, seed=seed)


def evaluate_thresholds_streaming(
    dataset_root: Path,
    samples,
    model,
    ds,
    start_index: int,
    dev,
    split_name: str,
    label_cache: Dict[str, np.ndarray],
    score_sources: Iterable[str],
    thresholds: Iterable[float],
    calibrator=None,
    verbose: bool = False,
) -> pd.DataFrame:
    rows = []
    thresholds_l = sorted(set([float(t) for t in thresholds] + [float("inf")]))
    for idx, sample in enumerate(samples):
        payload = build_single_payload(
            dataset_root,
            sample,
            model,
            ds,
            start_index + idx,
            dev,
            split_name,
            label_cache,
            calibrator=calibrator,
        )
        for source in score_sources:
            for threshold in thresholds_l:
                rows.append(evaluate_payload(payload, source, threshold))
        if verbose and (idx + 1) % 25 == 0:
            print(f"[INFO] Evaluated {idx + 1}/{len(samples)} {split_name} samples")
        del payload
    return pd.DataFrame(rows)


def apply_selected_streaming(
    dataset_root: Path,
    samples,
    model,
    ds,
    start_index: int,
    dev,
    split_name: str,
    label_cache: Dict[str, np.ndarray],
    selected: pd.DataFrame,
    calibrator=None,
    verbose: bool = False,
) -> pd.DataFrame:
    rows = []
    selected_l = [(str(row["score_source"]), float(row["threshold"])) for _, row in selected.iterrows()]
    for idx, sample in enumerate(samples):
        payload = build_single_payload(
            dataset_root,
            sample,
            model,
            ds,
            start_index + idx,
            dev,
            split_name,
            label_cache,
            calibrator=calibrator,
        )
        for source, threshold in selected_l:
            rows.append(evaluate_payload(payload, source, threshold))
        if verbose and (idx + 1) % 25 == 0:
            print(f"[INFO] Applied selected thresholds to {idx + 1}/{len(samples)} {split_name} samples")
        del payload
    return pd.DataFrame(rows)


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


def summarize_selected(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    work = add_bins(df)
    cols = [c for c in group_cols if c in work.columns]
    group = work if cols else work.assign(_overall="overall")
    by = cols if cols else ["_overall"]
    out = (
        group.groupby(["score_source"] + by, observed=True, dropna=False)
        .agg(
            count=("dice", "size"),
            mean_dice=("dice", "mean"),
            mean_locf_dice=("locf_dice", "mean"),
            mean_gap_vs_locf=("dice_gap_vs_locf", "mean"),
            median_gap_vs_locf=("dice_gap_vs_locf", "median"),
            win_rate_vs_locf=("dice_gap_vs_locf", lambda x: float((x > 0).mean())),
            mean_selected_voxels=("selected_voxels", "mean"),
            mean_growth_volume_vox=("growth_volume_vox", "mean"),
            mean_relative_new_growth=("relative_new_growth", "mean"),
        )
        .reset_index()
    )
    if "_overall" in out.columns:
        out = out.drop(columns=["_overall"])
    return out.sort_values(["score_source"] + cols).reset_index(drop=True) if cols else out


def bootstrap_summary(df: pd.DataFrame, n_bootstrap: int, seed: int) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    for source, sub in df.groupby("score_source", observed=True, dropna=False):
        vals = sub["dice_gap_vs_locf"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            rows.append({"score_source": source, "n": 0, "mean_gap": np.nan, "ci_low": np.nan, "ci_high": np.nan})
            continue
        boot = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, len(vals), len(vals))
            boot.append(float(vals[idx].mean()))
        rows.append(
            {
                "score_source": source,
                "n": int(len(vals)),
                "mean_gap": float(vals.mean()),
                "ci_low": float(np.quantile(boot, 0.025)),
                "ci_high": float(np.quantile(boot, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def write_report(
    path: Path,
    train_info: dict,
    selected: pd.DataFrame,
    overall: pd.DataFrame,
    by_growth: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Calibrated Growth-Field Evaluation\n\n")
        f.write(
            "This analysis tests whether a lightweight voxel-level decoder can convert growth-ranking signals "
            "into a deployable persistence-plus-growth forecast. The forecast preserves the input tumor mask and "
            "adds outside-input voxels whose calibrated growth score exceeds a validation-selected threshold.\n\n"
        )
        f.write("## Calibrator Training\n\n")
        f.write(pd.DataFrame([train_info]).to_markdown(index=False))
        f.write("\n\n## Validation-Selected Thresholds\n\n")
        f.write(selected.to_markdown(index=False))
        f.write("\n\n## Test Overall\n\n")
        f.write(overall.to_markdown(index=False))
        f.write("\n\n## Test By Absolute Growth Bin\n\n")
        f.write(by_growth.to_markdown(index=False))
        f.write("\n\n## Bootstrap Gap vs LOCF\n\n")
        f.write(bootstrap.to_markdown(index=False))
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate calibrated persistence-plus-growth-field forecasts.")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--baseline_output_dir", type=str, required=True)
    parser.add_argument("--model_method", type=str, default="resunet_image_mask")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--validation_split", type=str, default="val")
    parser.add_argument("--test_split", type=str, default="test")
    parser.add_argument("--fit_sessions", type=int, default=3)
    parser.add_argument("--horizons", type=str, default="1,2,3")
    parser.add_argument("--allowed_tiers", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--threshold_quantiles",
        type=str,
        default="0.01,0.02,0.05,0.10,0.20,0.30,0.40,0.50,0.70,0.80,0.90,0.95,0.975,0.99,0.995,0.999",
        help="Comma-separated score thresholds in [0,1]. Name kept for backward compatibility.",
    )
    parser.add_argument("--max_pos_per_sample", type=int, default=1000)
    parser.add_argument("--max_neg_per_sample", type=int, default=3000)
    parser.add_argument("--n_bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    baseline_output_dir = Path(args.baseline_output_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    thresholds = parse_threshold_values(args.threshold_quantiles)
    train_samples = build_samples_for_split(dataset_root, args.train_split, args.fit_sessions, args.horizons, args.allowed_tiers)
    val_samples = build_samples_for_split(dataset_root, args.validation_split, args.fit_sessions, args.horizons, args.allowed_tiers)
    test_samples = build_samples_for_split(dataset_root, args.test_split, args.fit_sessions, args.horizons, args.allowed_tiers)

    all_samples = train_samples + val_samples + test_samples
    model, ds_all, dev, spec = load_model(dataset_root, baseline_output_dir, args.model_method, all_samples, args.device)
    val_offset = len(train_samples)
    test_offset = len(train_samples) + len(val_samples)

    label_cache: Dict[str, np.ndarray] = {}
    calibrator, train_info = train_calibrator_streaming(
        dataset_root,
        train_samples,
        model,
        ds_all,
        0,
        dev,
        label_cache,
        max_pos=args.max_pos_per_sample,
        max_neg=args.max_neg_per_sample,
        seed=args.seed,
        verbose=args.verbose,
    )
    train_info.update(
        {
            "model_method": args.model_method,
            "model_variant": spec["model_variant"],
            "input_mode": spec["input_mode"],
            "n_train_samples": int(len(train_samples)),
            "n_validation_samples": int(len(val_samples)),
            "n_test_samples": int(len(test_samples)),
        }
    )

    score_sources = [
        "model_probability",
        "model_rank",
        "distance_rank",
        "hybrid_distance_model_a0.75",
        "calibrated_growth_probability",
    ]
    validation_candidates = evaluate_thresholds_streaming(
        dataset_root,
        val_samples,
        model,
        ds_all,
        val_offset,
        dev,
        args.validation_split,
        label_cache,
        score_sources,
        thresholds,
        calibrator=calibrator,
        verbose=args.verbose,
    )
    validation_summary = summarize_candidates(validation_candidates)
    selected = select_thresholds(validation_summary)
    test_selected = apply_selected_streaming(
        dataset_root,
        test_samples,
        model,
        ds_all,
        test_offset,
        dev,
        args.test_split,
        label_cache,
        selected,
        calibrator=calibrator,
        verbose=args.verbose,
    )

    test_overall = summarize_selected(test_selected, [])
    test_by_tier = summarize_selected(test_selected, ["tier"])
    test_by_horizon = summarize_selected(test_selected, ["horizon"])
    test_by_growth = summarize_selected(test_selected, ["absolute_growth_bin"])
    test_by_horizon_growth = summarize_selected(test_selected, ["horizon", "absolute_growth_bin"])
    bootstrap = bootstrap_summary(test_selected, args.n_bootstrap, args.seed)

    validation_candidates.to_csv(output_dir / "calibrated_growth_field_validation_candidates.csv", index=False)
    validation_summary.to_csv(output_dir / "calibrated_growth_field_validation_summary.csv", index=False)
    selected.to_csv(output_dir / "calibrated_growth_field_selected_thresholds.csv", index=False)
    test_selected.to_csv(output_dir / "calibrated_growth_field_test_samples.csv", index=False)
    test_overall.to_csv(output_dir / "calibrated_growth_field_test_overall.csv", index=False)
    test_by_tier.to_csv(output_dir / "calibrated_growth_field_test_by_tier.csv", index=False)
    test_by_horizon.to_csv(output_dir / "calibrated_growth_field_test_by_horizon.csv", index=False)
    test_by_growth.to_csv(output_dir / "calibrated_growth_field_test_by_growth_bin.csv", index=False)
    test_by_horizon_growth.to_csv(output_dir / "calibrated_growth_field_test_by_horizon_growth_bin.csv", index=False)
    bootstrap.to_csv(output_dir / "calibrated_growth_field_bootstrap.csv", index=False)
    with (output_dir / "calibrated_growth_field_train_info.json").open("w", encoding="utf-8") as f:
        json.dump(train_info, f, indent=2)
    write_report(
        output_dir / "calibrated_growth_field_report.md",
        train_info,
        selected,
        test_overall,
        test_by_growth,
        bootstrap,
    )

    print(
        json.dumps(
            {
                "dataset_root": str(dataset_root),
                "baseline_output_dir": str(baseline_output_dir),
                "model_method": args.model_method,
                "train_info": train_info,
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
