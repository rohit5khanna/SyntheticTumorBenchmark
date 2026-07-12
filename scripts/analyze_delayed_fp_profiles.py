#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.tasks import build_samples_for_split, infer_tier_from_patient_id, patient_paths
from scripts.analyze_growth_ranking import (  # noqa: E402
    _checkpoint_specs,
    _distance_to_input_score,
    _rank_normalize_score,
    _standardize_label,
)


def _parse_float_list(payload: str) -> List[float]:
    out = []
    for item in str(payload).split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if value <= 0.0 or value > 1.0:
            raise ValueError("top_pct values must be in (0, 1].")
        out.append(value)
    return sorted(set(out))


def _stable_seed(*parts: object) -> int:
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def _topk_mask(score: np.ndarray, k: int) -> np.ndarray:
    s = np.asarray(score, dtype=np.float32).reshape(-1)
    k = int(max(1, min(k, len(s))))
    idx = np.argpartition(-s, kth=k - 1)[:k]
    out = np.zeros(len(s), dtype=bool)
    out[idx] = True
    return out


def _safe_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return float("nan")
    return float(np.nanmean(values))


def _safe_median(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return float("nan")
    return float(np.nanmedian(values))


def _profile_row(
    *,
    method: str,
    source: str,
    patient_id: str,
    tier: str,
    input_idx: int,
    target_idx: int,
    horizon: int,
    delta_days: float,
    budget_name: str,
    score: np.ndarray,
    distance_to_input: np.ndarray,
    immediate_growth: np.ndarray,
    delayed_growth_after_target: np.ndarray,
    eventual_growth_from_input: np.ndarray,
    selected: np.ndarray,
) -> Dict:
    score = np.asarray(score, dtype=np.float32).reshape(-1)
    distance_to_input = np.asarray(distance_to_input, dtype=np.float32).reshape(-1)
    immediate_growth = np.asarray(immediate_growth, dtype=bool).reshape(-1)
    delayed_growth_after_target = np.asarray(delayed_growth_after_target, dtype=bool).reshape(-1)
    eventual_growth_from_input = np.asarray(eventual_growth_from_input, dtype=bool).reshape(-1)
    selected = np.asarray(selected, dtype=bool).reshape(-1)

    immediate_tp = selected & immediate_growth
    immediate_fp = selected & ~immediate_growth
    delayed_fp = immediate_fp & delayed_growth_after_target
    never_fp = immediate_fp & ~delayed_growth_after_target
    eventual_hit = selected & eventual_growth_from_input

    selected_count = int(selected.sum())
    fp_count = int(immediate_fp.sum())
    delayed_count = int(delayed_fp.sum())
    never_count = int(never_fp.sum())

    return {
        "method": method,
        "ranking_source": source,
        "patient_id": patient_id,
        "tier": tier,
        "input_idx": int(input_idx),
        "target_idx": int(target_idx),
        "horizon": int(horizon),
        "delta_days": float(delta_days),
        "budget_name": budget_name,
        "selected_count": selected_count,
        "immediate_tp_count": int(immediate_tp.sum()),
        "immediate_fp_count": fp_count,
        "delayed_fp_count": delayed_count,
        "never_fp_count": never_count,
        "eventual_hit_count": int(eventual_hit.sum()),
        "delayed_fp_fraction_among_fp": float(delayed_count / max(1, fp_count)),
        "never_fp_fraction_among_fp": float(never_count / max(1, fp_count)),
        "immediate_precision": float(immediate_tp.sum() / max(1, selected_count)),
        "eventual_precision": float(eventual_hit.sum() / max(1, selected_count)),
        "eventual_precision_gain": float((eventual_hit.sum() - immediate_tp.sum()) / max(1, selected_count)),
        "mean_score_immediate_tp": _safe_mean(score[immediate_tp]),
        "mean_score_delayed_fp": _safe_mean(score[delayed_fp]),
        "mean_score_never_fp": _safe_mean(score[never_fp]),
        "median_score_delayed_fp": _safe_median(score[delayed_fp]),
        "median_score_never_fp": _safe_median(score[never_fp]),
        "mean_distance_immediate_tp": _safe_mean(distance_to_input[immediate_tp]),
        "mean_distance_delayed_fp": _safe_mean(distance_to_input[delayed_fp]),
        "mean_distance_never_fp": _safe_mean(distance_to_input[never_fp]),
        "median_distance_delayed_fp": _safe_median(distance_to_input[delayed_fp]),
        "median_distance_never_fp": _safe_median(distance_to_input[never_fp]),
        "score_gap_delayed_minus_never_fp": _safe_mean(score[delayed_fp]) - _safe_mean(score[never_fp]),
        "distance_gap_delayed_minus_never_fp": _safe_mean(distance_to_input[delayed_fp]) - _safe_mean(distance_to_input[never_fp]),
    }


def _rows_for_score(
    *,
    method: str,
    source: str,
    sample,
    tier: str,
    score: np.ndarray,
    distance_to_input: np.ndarray,
    immediate_growth: np.ndarray,
    delayed_growth_after_target: np.ndarray,
    eventual_growth_from_input: np.ndarray,
    top_pcts: Iterable[float],
    include_growth_budget: bool,
) -> List[Dict]:
    rows = []
    n = int(np.asarray(score).size)
    budgets: List[tuple[str, int]] = []
    for pct in top_pcts:
        budgets.append((f"top_{pct:g}_candidate_fraction", max(1, int(round(pct * n)))))
    if include_growth_budget:
        budgets.append(("top_immediate_growth_volume", max(1, int(np.asarray(immediate_growth).sum()))))
        budgets.append(("top_eventual_growth_volume", max(1, int(np.asarray(eventual_growth_from_input).sum()))))

    for budget_name, k in budgets:
        selected = _topk_mask(score, k)
        rows.append(
            _profile_row(
                method=method,
                source=source,
                patient_id=sample.patient_id,
                tier=tier,
                input_idx=sample.input_idx,
                target_idx=sample.target_idx,
                horizon=sample.horizon,
                delta_days=sample.delta_days,
                budget_name=budget_name,
                score=score,
                distance_to_input=distance_to_input,
                immediate_growth=immediate_growth,
                delayed_growth_after_target=delayed_growth_after_target,
                eventual_growth_from_input=eventual_growth_from_input,
                selected=selected,
            )
        )
    return rows


def _weighted_summary(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    rows = []
    for keys, g in df.groupby(group_cols, dropna=False, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        total_fp = float(g["immediate_fp_count"].sum())
        total_selected = float(g["selected_count"].sum())
        total_delayed = float(g["delayed_fp_count"].sum())
        total_never = float(g["never_fp_count"].sum())
        total_tp = float(g["immediate_tp_count"].sum())
        total_eventual = float(g["eventual_hit_count"].sum())
        row = {col: val for col, val in zip(group_cols, keys)}
        row.update(
            {
                "count": int(len(g)),
                "total_selected_count": int(total_selected),
                "total_immediate_fp_count": int(total_fp),
                "total_delayed_fp_count": int(total_delayed),
                "total_never_fp_count": int(total_never),
                "weighted_immediate_precision": float(total_tp / max(1.0, total_selected)),
                "weighted_eventual_precision": float(total_eventual / max(1.0, total_selected)),
                "weighted_eventual_precision_gain": float((total_eventual - total_tp) / max(1.0, total_selected)),
                "weighted_delayed_fp_fraction_among_fp": float(total_delayed / max(1.0, total_fp)),
                "mean_sample_delayed_fp_fraction_among_fp": float(g["delayed_fp_fraction_among_fp"].mean()),
                "mean_score_gap_delayed_minus_never_fp": float(g["score_gap_delayed_minus_never_fp"].mean()),
                "mean_distance_gap_delayed_minus_never_fp": float(g["distance_gap_delayed_minus_never_fp"].mean()),
                "mean_distance_delayed_fp": float(g["mean_distance_delayed_fp"].mean()),
                "mean_distance_never_fp": float(g["mean_distance_never_fp"].mean()),
                "mean_score_delayed_fp": float(g["mean_score_delayed_fp"].mean()),
                "mean_score_never_fp": float(g["mean_score_never_fp"].mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def compute_profiles(
    *,
    dataset_root: Path,
    split: str,
    fit_sessions: int,
    horizons: str,
    allowed_tiers: str | None,
    baseline_output_dir: Path,
    methods: List[str],
    device: str,
    top_pcts: List[float],
    hybrid_alphas: List[float],
    include_growth_budget: bool,
) -> pd.DataFrame:
    samples = build_samples_for_split(dataset_root, split, fit_sessions, horizons, allowed_tiers=allowed_tiers)
    label_cache: Dict[str, np.ndarray] = {}
    rows: List[Dict] = []
    contexts = []

    for s in samples:
        if s.patient_id not in label_cache:
            p = patient_paths(dataset_root, s.patient_id)
            label_cache[s.patient_id] = _standardize_label(np.load(p["label"]))
        labels = label_cache[s.patient_id]
        input_mask = (labels[s.input_idx] > 0)[0]
        target_mask = (labels[s.target_idx] > 0)[0]
        outside_input = ~input_mask
        immediate_growth = (target_mask & ~input_mask)[outside_input]

        if s.target_idx + 1 < labels.shape[0]:
            future_after_target = (labels[s.target_idx + 1 :] > 0).any(axis=(0, 1))
        else:
            future_after_target = np.zeros_like(input_mask, dtype=bool)
        delayed_growth_after_target = (future_after_target & ~target_mask & ~input_mask)[outside_input]

        future_from_input = (labels[s.target_idx :] > 0).any(axis=(0, 1))
        eventual_growth_from_input = (future_from_input & ~input_mask)[outside_input]

        distance_score_full = _distance_to_input_score(input_mask)
        if distance_score_full is None:
            continue
        distance_score = distance_score_full[outside_input]
        distance_to_input = -distance_score.astype(np.float32)
        tier = infer_tier_from_patient_id(s.patient_id)
        contexts.append(
            {
                "sample": s,
                "tier": tier,
                "input_mask": input_mask,
                "outside_input": outside_input,
                "distance_score": distance_score,
                "distance_to_input": distance_to_input,
                "immediate_growth": immediate_growth,
                "delayed_growth_after_target": delayed_growth_after_target,
                "eventual_growth_from_input": eventual_growth_from_input,
            }
        )

        rows.extend(
            _rows_for_score(
                method="distance_to_input_mask",
                source="reference",
                sample=s,
                tier=tier,
                score=distance_score,
                distance_to_input=distance_to_input,
                immediate_growth=immediate_growth,
                delayed_growth_after_target=delayed_growth_after_target,
                eventual_growth_from_input=eventual_growth_from_input,
                top_pcts=top_pcts,
                include_growth_budget=include_growth_budget,
            )
        )

        rng = np.random.default_rng(_stable_seed(s.patient_id, s.input_idx, s.target_idx, "random"))
        rows.extend(
            _rows_for_score(
                method="random_score",
                source="reference",
                sample=s,
                tier=tier,
                score=rng.random(int(outside_input.sum()), dtype=np.float32),
                distance_to_input=distance_to_input,
                immediate_growth=immediate_growth,
                delayed_growth_after_target=delayed_growth_after_target,
                eventual_growth_from_input=eventual_growth_from_input,
                top_pcts=top_pcts,
                include_growth_budget=include_growth_budget,
            )
        )

    specs = _checkpoint_specs(baseline_output_dir, methods)
    if not specs:
        return pd.DataFrame(rows)

    try:
        import torch
        from baselines.unet import _TorchForecastDataset, _build_torch_model
    except Exception as e:
        print(f"[WARN] Skipping model profile analysis because PyTorch/model code is unavailable: {e}")
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

        for i, ctx in enumerate(contexts):
            s = ctx["sample"]
            x, _, _ = ds[i]
            with torch.no_grad():
                logits = model(x[None].to(dev))
                prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
            score = prob[ctx["outside_input"]]
            rows.extend(
                _rows_for_score(
                    method=method,
                    source="model",
                    sample=s,
                    tier=ctx["tier"],
                    score=score,
                    distance_to_input=ctx["distance_to_input"],
                    immediate_growth=ctx["immediate_growth"],
                    delayed_growth_after_target=ctx["delayed_growth_after_target"],
                    eventual_growth_from_input=ctx["eventual_growth_from_input"],
                    top_pcts=top_pcts,
                    include_growth_budget=include_growth_budget,
                )
            )

            distance_rank = _rank_normalize_score(ctx["distance_score"])
            model_rank = _rank_normalize_score(score)
            for alpha in hybrid_alphas:
                hybrid_score = (1.0 - alpha) * distance_rank + alpha * model_rank
                rows.extend(
                    _rows_for_score(
                        method=f"hybrid_distance_{method}_a{alpha:.2f}",
                        source="hybrid",
                        sample=s,
                        tier=ctx["tier"],
                        score=hybrid_score,
                        distance_to_input=ctx["distance_to_input"],
                        immediate_growth=ctx["immediate_growth"],
                        delayed_growth_after_target=ctx["delayed_growth_after_target"],
                        eventual_growth_from_input=ctx["eventual_growth_from_input"],
                        top_pcts=top_pcts,
                        include_growth_budget=include_growth_budget,
                    )
                )

    return pd.DataFrame(rows)


def write_report(path: Path, overall: pd.DataFrame, by_tier: pd.DataFrame, by_horizon: pd.DataFrame) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Delayed False-Positive Profile Analysis\n\n")
        f.write(
            "Selected voxels that are false positives at the immediate target are split into delayed-hit and never-hit groups. "
            "The goal is to test whether delayed hits look different from ordinary over-expansion artifacts.\n\n"
        )
        f.write("## Overall\n\n")
        f.write(overall.to_markdown(index=False) if not overall.empty else "No overall summary.")
        f.write("\n\n## By Tier\n\n")
        f.write(by_tier.to_markdown(index=False) if not by_tier.empty else "No tier summary.")
        f.write("\n\n## By Horizon\n\n")
        f.write(by_horizon.to_markdown(index=False) if not by_horizon.empty else "No horizon summary.")
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile delayed-hit versus never-hit immediate false positives.")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--fit_sessions", type=int, default=3)
    parser.add_argument("--horizons", type=str, default="1,2,3")
    parser.add_argument("--allowed_tiers", type=str, default=None)
    parser.add_argument("--baseline_output_dir", type=str, required=True)
    parser.add_argument("--methods", type=str, default="unet_image_mask,resunet_image_mask")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--top_pcts", type=str, default="0.01,0.05")
    parser.add_argument("--hybrid_alphas", type=str, default="0.75")
    parser.add_argument("--no_growth_budgets", action="store_true")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    top_pcts = _parse_float_list(args.top_pcts)
    hybrid_alphas = _parse_float_list(args.hybrid_alphas) if args.hybrid_alphas.strip() else []

    profiles = compute_profiles(
        dataset_root=Path(args.dataset_root),
        split=args.split,
        fit_sessions=args.fit_sessions,
        horizons=args.horizons,
        allowed_tiers=args.allowed_tiers,
        baseline_output_dir=Path(args.baseline_output_dir),
        methods=methods,
        device=args.device,
        top_pcts=top_pcts,
        hybrid_alphas=hybrid_alphas,
        include_growth_budget=not args.no_growth_budgets,
    )

    overall = _weighted_summary(profiles, ["ranking_source", "method", "budget_name"])
    by_tier = _weighted_summary(profiles, ["ranking_source", "method", "budget_name", "tier"])
    by_horizon = _weighted_summary(profiles, ["ranking_source", "method", "budget_name", "horizon"])

    profiles.to_csv(output_dir / "delayed_fp_profile_samples.csv", index=False)
    overall.to_csv(output_dir / "delayed_fp_profile_summary_overall.csv", index=False)
    by_tier.to_csv(output_dir / "delayed_fp_profile_summary_by_tier.csv", index=False)
    by_horizon.to_csv(output_dir / "delayed_fp_profile_summary_by_horizon.csv", index=False)
    write_report(output_dir / "delayed_fp_profile_report.md", overall, by_tier, by_horizon)

    report = {
        "dataset_root": args.dataset_root,
        "baseline_output_dir": args.baseline_output_dir,
        "split": args.split,
        "fit_sessions": args.fit_sessions,
        "horizons": args.horizons,
        "methods": methods,
        "top_pcts": top_pcts,
        "hybrid_alphas": hybrid_alphas,
        "output_dir": str(output_dir),
    }
    with (output_dir / "delayed_fp_profile_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
