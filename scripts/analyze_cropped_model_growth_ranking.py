#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.tasks import build_samples_for_split, infer_tier_from_patient_id, patient_paths
from scripts.analyze_growth_ranking import (
    KEY_COLS,
    _checkpoint_specs,
    _distance_to_input_score,
    _parse_float_list,
    _qbin,
    _ranking_row,
    _rank_normalize_score,
    summarize_ranking,
)


def _standardize_label_sessions(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 5 and arr.shape[1] == 1:
        return (arr > 0).astype(np.float32)
    if arr.ndim == 4:
        return (arr[:, None, ...] > 0).astype(np.float32)
    raise ValueError(f"Unsupported label array shape: {arr.shape}")


def _standardize_image_sessions(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 5:
        return arr
    if arr.ndim == 4:
        return arr[:, None, ...]
    raise ValueError(f"Unsupported image array shape: {arr.shape}")


def _load_patient_arrays(dataset_root: Path, patient_id: str, need_image: bool) -> Dict[str, np.ndarray]:
    p = patient_paths(dataset_root, patient_id)
    out = {
        "label": _standardize_label_sessions(np.load(p["label"], mmap_mode="r")),
        "days": np.load(p["days"], mmap_mode="r").astype(np.float32),
        "treatment": np.load(p["treatment"], mmap_mode="r").astype(np.float32),
    }
    if need_image:
        out["image"] = _standardize_image_sessions(np.load(p["image"], mmap_mode="r"))
    return out


def _bbox_from_mask(mask: np.ndarray, margin: int, min_size: int) -> Tuple[slice, slice, slice]:
    mask = np.asarray(mask).astype(bool)
    shape = mask.shape
    coords = np.argwhere(mask)

    if coords.size == 0:
        starts = [max(0, (shape[i] - min_size) // 2) for i in range(3)]
        stops = [min(shape[i], starts[i] + min_size) for i in range(3)]
        starts = [max(0, stops[i] - min_size) for i in range(3)]
    else:
        starts = [int(coords[:, i].min()) - margin for i in range(3)]
        stops = [int(coords[:, i].max()) + margin + 1 for i in range(3)]

        for i in range(3):
            starts[i] = max(0, starts[i])
            stops[i] = min(shape[i], stops[i])
            cur = stops[i] - starts[i]
            if cur < min_size:
                extra = min_size - cur
                left = extra // 2
                right = extra - left
                starts[i] = max(0, starts[i] - left)
                stops[i] = min(shape[i], stops[i] + right)
                if stops[i] - starts[i] < min_size:
                    starts[i] = max(0, stops[i] - min_size)
                    stops[i] = min(shape[i], starts[i] + min_size)

    return tuple(slice(starts[i], stops[i]) for i in range(3))  # type: ignore[return-value]


def _crop_vox(crop: Tuple[slice, slice, slice]) -> int:
    out = 1
    for sl in crop:
        out *= int(sl.stop - sl.start)
    return int(out)


def _build_input_tensor(
    arrs: Dict[str, np.ndarray],
    input_idx: int,
    delta_days: float,
    current_treatment: float,
    target_treatment: float,
    crop: Tuple[slice, slice, slice],
    input_mode: str,
    delta_days_norm: float,
) -> np.ndarray:
    mask_in = np.asarray(arrs["label"][input_idx, :, crop[0], crop[1], crop[2]], dtype=np.float32)
    _, h, w, d = mask_in.shape
    delta_chan = np.full((1, h, w, d), float(delta_days) / float(delta_days_norm), dtype=np.float32)
    cur_treat_chan = np.full((1, h, w, d), float(current_treatment), dtype=np.float32)
    tgt_treat_chan = np.full((1, h, w, d), float(target_treatment), dtype=np.float32)

    feats = []
    if input_mode == "image_mask":
        feats.append(np.asarray(arrs["image"][input_idx, :, crop[0], crop[1], crop[2]], dtype=np.float32))
    feats.extend([mask_in, delta_chan, cur_treat_chan, tgt_treat_chan])
    return np.concatenate(feats, axis=0).astype(np.float32)


def _growth_features_from_rows(rows: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["abs_change_bin"] = _qbin(df["relative_abs_change"], ["low", "medium", "high"])
    df["new_growth_bin"] = _qbin(df["relative_new_growth"], ["low", "medium", "high"])
    df["net_growth_bin"] = _qbin(df["relative_net_growth"], ["low", "medium", "high"])
    return df


def compute_cropped_ranking(
    dataset_root: Path,
    split: str,
    fit_sessions: int,
    horizons: str,
    allowed_tiers: str | None,
    baseline_output_dir: Path,
    methods: Iterable[str],
    device: str,
    hybrid_alphas: Iterable[float],
    crop_margin: int,
    min_crop_size: int,
    delta_days_norm: float,
    max_samples: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        import torch

        from baselines.unet import _build_torch_model
    except Exception as e:
        raise RuntimeError(f"PyTorch/model utilities are required for cropped model ranking: {e}") from e

    samples = build_samples_for_split(dataset_root, split, fit_sessions, horizons, allowed_tiers=allowed_tiers)
    if max_samples is not None:
        samples = samples[: int(max_samples)]

    specs = _checkpoint_specs(baseline_output_dir, methods)
    if not specs:
        raise FileNotFoundError(
            f"No usable model checkpoints found in {baseline_output_dir} for methods: {list(methods)}"
        )

    need_image_by_pid = any(spec["input_mode"] == "image_mask" for spec in specs.values())
    dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))

    models = {}
    for method, spec in specs.items():
        ckpt = torch.load(spec["checkpoint"], map_location=dev, weights_only=False)
        in_channels = int(ckpt["in_channels"])
        base_channels = int(ckpt.get("base_channels", 12))
        model = _build_torch_model(in_channels, base_channels, model_variant=spec["model_variant"])
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(dev)
        model.eval()
        models[method] = {"model": model, "input_mode": spec["input_mode"]}

    array_cache: Dict[str, Dict[str, np.ndarray]] = {}
    ranking_rows: List[Dict] = []
    feature_rows: List[Dict] = []
    hybrid_alpha_l = [float(a) for a in hybrid_alphas]

    for s in samples:
        if s.patient_id not in array_cache:
            array_cache[s.patient_id] = _load_patient_arrays(dataset_root, s.patient_id, need_image=need_image_by_pid)
        arrs = array_cache[s.patient_id]

        labels = arrs["label"]
        input_full = (labels[s.input_idx, 0] > 0)
        target_full = (labels[s.target_idx, 0] > 0)
        growth_full = target_full & ~input_full
        loss_full = input_full & ~target_full
        union_full = input_full | target_full

        crop = _bbox_from_mask(input_full, margin=crop_margin, min_size=min_crop_size)
        crop_shape = tuple(int(sl.stop - sl.start) for sl in crop)
        input_mask = input_full[crop]
        target_mask = target_full[crop]
        growth = target_mask & ~input_mask
        outside_input = ~input_mask
        y = growth[outside_input]
        tier = infer_tier_from_patient_id(s.patient_id)

        input_volume = int(input_full.sum())
        target_volume = int(target_full.sum())
        growth_volume = int(growth_full.sum())
        loss_volume = int(loss_full.sum())
        union_volume = int(union_full.sum())
        crop_growth_volume = int(growth.sum())

        feature_rows.append(
            {
                "patient_id": s.patient_id,
                "input_idx": s.input_idx,
                "target_idx": s.target_idx,
                "horizon": s.horizon,
                "delta_days": s.delta_days,
                "tier": tier,
                "crop_margin": crop_margin,
                "crop_shape": "x".join(str(v) for v in crop_shape),
                "crop_vox": _crop_vox(crop),
                "input_volume_vox": input_volume,
                "target_volume_vox": target_volume,
                "net_delta_volume_vox": target_volume - input_volume,
                "abs_delta_volume_vox": int(abs(target_volume - input_volume)),
                "growth_volume_vox": growth_volume,
                "loss_volume_vox": loss_volume,
                "union_volume_vox": union_volume,
                "crop_growth_volume_vox": crop_growth_volume,
                "crop_growth_capture_fraction": crop_growth_volume / max(1, growth_volume),
                "candidate_vox_in_crop": int(outside_input.sum()),
                "relative_net_growth": (target_volume - input_volume) / max(1, input_volume),
                "relative_abs_change": int(abs(target_volume - input_volume)) / max(1, input_volume),
                "relative_new_growth": growth_volume / max(1, input_volume),
                "growth_fraction_of_target": growth_volume / max(1, target_volume),
                "loss_fraction_of_input": loss_volume / max(1, input_volume),
            }
        )

        ranking_rows.append(
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
                source="reference_cropped",
            )
        )

        distance_score_full = _distance_to_input_score(input_mask)
        distance_score = None
        if distance_score_full is not None:
            distance_score = distance_score_full[outside_input]
            ranking_rows.append(
                _ranking_row(
                    method="distance_to_input_mask",
                    patient_id=s.patient_id,
                    input_idx=s.input_idx,
                    target_idx=s.target_idx,
                    horizon=s.horizon,
                    delta_days=s.delta_days,
                    tier=tier,
                    y_true=y,
                    score=distance_score,
                    source="reference_cropped",
                )
            )

        for method, bundle in models.items():
            x = _build_input_tensor(
                arrs=arrs,
                input_idx=s.input_idx,
                delta_days=s.delta_days,
                current_treatment=s.current_treatment,
                target_treatment=s.target_treatment,
                crop=crop,
                input_mode=bundle["input_mode"],
                delta_days_norm=delta_days_norm,
            )
            with torch.no_grad():
                xt = torch.from_numpy(x[None]).to(dev)
                logits = bundle["model"](xt)
                prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
            score = prob[outside_input]
            ranking_rows.append(
                _ranking_row(
                    method=method,
                    patient_id=s.patient_id,
                    input_idx=s.input_idx,
                    target_idx=s.target_idx,
                    horizon=s.horizon,
                    delta_days=s.delta_days,
                    tier=tier,
                    y_true=y,
                    score=score,
                    source="model_cropped",
                )
            )

            if distance_score is not None:
                distance_rank = _rank_normalize_score(distance_score)
                model_rank = _rank_normalize_score(score)
                for alpha in hybrid_alpha_l:
                    hybrid_score = (1.0 - alpha) * distance_rank + alpha * model_rank
                    ranking_rows.append(
                        _ranking_row(
                            method=f"hybrid_distance_{method}_a{alpha:.2f}",
                            patient_id=s.patient_id,
                            input_idx=s.input_idx,
                            target_idx=s.target_idx,
                            horizon=s.horizon,
                            delta_days=s.delta_days,
                            tier=tier,
                            y_true=y,
                            score=hybrid_score,
                            source="hybrid_cropped",
                        )
                    )

    return pd.DataFrame(ranking_rows), _growth_features_from_rows(feature_rows)


def write_report(
    path: Path,
    ranking: pd.DataFrame,
    features: pd.DataFrame,
    by_horizon: pd.DataFrame,
    by_growth: pd.DataFrame,
    by_tier: pd.DataFrame,
) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Cropped Model Growth-Ranking Evaluation\n\n")
        f.write(
            "This memory-safe evaluation runs trained models only inside an input-mask-centered crop. "
            "It is intended for real-data stress tests where full-volume 3D inference can exceed Colab memory.\n\n"
        )
        f.write("## Crop Coverage\n\n")
        coverage_cols = ["crop_vox", "crop_growth_capture_fraction", "candidate_vox_in_crop"]
        f.write(features[coverage_cols].describe().to_markdown())
        f.write("\n\n## Ranking By Horizon\n\n")
        f.write(by_horizon.to_markdown(index=False) if not by_horizon.empty else "No ranking rows available.")
        f.write("\n\n## Ranking By New-Growth Bin\n\n")
        f.write(by_growth.to_markdown(index=False) if not by_growth.empty else "No growth-bin ranking rows available.")
        f.write("\n\n## Ranking By Tier\n\n")
        f.write(by_tier.to_markdown(index=False) if not by_tier.empty else "No tier ranking rows available.")
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Memory-safe cropped model ranking of future tumor growth regions."
    )
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--fit_sessions", type=int, default=3)
    parser.add_argument("--horizons", type=str, default="1,2,3")
    parser.add_argument("--allowed_tiers", type=str, default=None)
    parser.add_argument("--baseline_output_dir", type=str, required=True)
    parser.add_argument("--methods", type=str, default="resunet_mask")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--hybrid_alphas", type=str, default="0.75")
    parser.add_argument("--crop_margin", type=int, default=48)
    parser.add_argument("--min_crop_size", type=int, default=64)
    parser.add_argument("--delta_days_norm", type=float, default=180.0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    baseline_output_dir = Path(args.baseline_output_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    ranking, features = compute_cropped_ranking(
        dataset_root=dataset_root,
        split=args.split,
        fit_sessions=args.fit_sessions,
        horizons=args.horizons,
        allowed_tiers=args.allowed_tiers,
        baseline_output_dir=baseline_output_dir,
        methods=methods,
        device=args.device,
        hybrid_alphas=_parse_float_list(args.hybrid_alphas),
        crop_margin=args.crop_margin,
        min_crop_size=args.min_crop_size,
        delta_days_norm=args.delta_days_norm,
        max_samples=args.max_samples,
    )

    if not ranking.empty:
        merge_cols = KEY_COLS + [
            "new_growth_bin",
            "abs_change_bin",
            "net_growth_bin",
            "relative_new_growth",
            "relative_abs_change",
            "crop_growth_capture_fraction",
            "crop_vox",
        ]
        ranking = ranking.merge(features[merge_cols], on=KEY_COLS, how="left")
        by_horizon = summarize_ranking(ranking, ["ranking_source", "method", "horizon"])
        by_growth = summarize_ranking(ranking, ["ranking_source", "method", "new_growth_bin"])
        by_tier = summarize_ranking(ranking, ["ranking_source", "method", "tier"])
    else:
        by_horizon = pd.DataFrame()
        by_growth = pd.DataFrame()
        by_tier = pd.DataFrame()

    features.to_csv(output_dir / "cropped_growth_sample_features.csv", index=False)
    ranking.to_csv(output_dir / "cropped_growth_ranking_metrics.csv", index=False)
    by_horizon.to_csv(output_dir / "cropped_ranking_summary_by_horizon.csv", index=False)
    by_growth.to_csv(output_dir / "cropped_ranking_summary_by_new_growth_bin.csv", index=False)
    by_tier.to_csv(output_dir / "cropped_ranking_summary_by_tier.csv", index=False)
    write_report(output_dir / "cropped_growth_ranking_report.md", ranking, features, by_horizon, by_growth, by_tier)

    print(
        json.dumps(
            {
                "dataset_root": str(dataset_root),
                "baseline_output_dir": str(baseline_output_dir),
                "methods": methods,
                "crop_margin": args.crop_margin,
                "min_crop_size": args.min_crop_size,
                "n_samples": int(len(features)),
                "device": args.device,
                "output_dir": str(output_dir),
                "files": [
                    "cropped_growth_sample_features.csv",
                    "cropped_growth_ranking_metrics.csv",
                    "cropped_ranking_summary_by_horizon.csv",
                    "cropped_ranking_summary_by_new_growth_bin.csv",
                    "cropped_ranking_summary_by_tier.csv",
                    "cropped_growth_ranking_report.md",
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
