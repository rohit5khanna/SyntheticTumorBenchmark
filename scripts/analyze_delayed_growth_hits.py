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


def _row_for_selection(
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
    selected: np.ndarray,
    immediate_growth: np.ndarray,
    delayed_growth_after_target: np.ndarray,
    eventual_growth_from_input: np.ndarray,
) -> Dict:
    selected = np.asarray(selected, dtype=bool).reshape(-1)
    immediate_growth = np.asarray(immediate_growth, dtype=bool).reshape(-1)
    delayed_growth_after_target = np.asarray(delayed_growth_after_target, dtype=bool).reshape(-1)
    eventual_growth_from_input = np.asarray(eventual_growth_from_input, dtype=bool).reshape(-1)

    selected_count = int(selected.sum())
    immediate_tp = selected & immediate_growth
    immediate_fp = selected & ~immediate_growth
    delayed_hits = immediate_fp & delayed_growth_after_target
    eventual_hits = selected & eventual_growth_from_input

    immediate_fp_count = int(immediate_fp.sum())
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
        "immediate_growth_vox": int(immediate_growth.sum()),
        "delayed_growth_after_target_vox": int(delayed_growth_after_target.sum()),
        "eventual_growth_from_input_vox": int(eventual_growth_from_input.sum()),
        "immediate_true_positive_count": int(immediate_tp.sum()),
        "immediate_false_positive_count": immediate_fp_count,
        "delayed_hit_count_from_immediate_fp": int(delayed_hits.sum()),
        "eventual_hit_count": int(eventual_hits.sum()),
        "immediate_precision": float(immediate_tp.sum() / max(1, selected_count)),
        "eventual_precision": float(eventual_hits.sum() / max(1, selected_count)),
        "delayed_hit_rate_among_immediate_fp": float(delayed_hits.sum() / max(1, immediate_fp_count)),
        "eventual_precision_gain": float((eventual_hits.sum() - immediate_tp.sum()) / max(1, selected_count)),
        "has_later_session": bool(delayed_growth_after_target.size > 0),
    }


def _score_rows_for_sample(
    *,
    method: str,
    source: str,
    patient_id: str,
    tier: str,
    input_idx: int,
    target_idx: int,
    horizon: int,
    delta_days: float,
    score: np.ndarray,
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
            _row_for_selection(
                method=method,
                source=source,
                patient_id=patient_id,
                tier=tier,
                input_idx=input_idx,
                target_idx=target_idx,
                horizon=horizon,
                delta_days=delta_days,
                budget_name=budget_name,
                selected=selected,
                immediate_growth=immediate_growth,
                delayed_growth_after_target=delayed_growth_after_target,
                eventual_growth_from_input=eventual_growth_from_input,
            )
        )
    return rows


def _summarize(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    return (
        df.groupby(group_cols, dropna=False, observed=True)
        .agg(
            count=("immediate_precision", "size"),
            mean_selected_count=("selected_count", "mean"),
            mean_immediate_growth_vox=("immediate_growth_vox", "mean"),
            mean_delayed_growth_after_target_vox=("delayed_growth_after_target_vox", "mean"),
            mean_eventual_growth_from_input_vox=("eventual_growth_from_input_vox", "mean"),
            mean_immediate_precision=("immediate_precision", "mean"),
            mean_eventual_precision=("eventual_precision", "mean"),
            mean_eventual_precision_gain=("eventual_precision_gain", "mean"),
            mean_delayed_hit_rate_among_immediate_fp=("delayed_hit_rate_among_immediate_fp", "mean"),
        )
        .reset_index()
        .sort_values(group_cols)
    )


def compute_delayed_hits(
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

    sample_context = []
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

        tier = infer_tier_from_patient_id(s.patient_id)
        sample_context.append(
            {
                "sample": s,
                "tier": tier,
                "input_mask": input_mask,
                "outside_input": outside_input,
                "immediate_growth": immediate_growth,
                "delayed_growth_after_target": delayed_growth_after_target,
                "eventual_growth_from_input": eventual_growth_from_input,
            }
        )

        distance_score_full = _distance_to_input_score(input_mask)
        if distance_score_full is not None:
            rows.extend(
                _score_rows_for_sample(
                    method="distance_to_input_mask",
                    source="reference",
                    patient_id=s.patient_id,
                    tier=tier,
                    input_idx=s.input_idx,
                    target_idx=s.target_idx,
                    horizon=s.horizon,
                    delta_days=s.delta_days,
                    score=distance_score_full[outside_input],
                    immediate_growth=immediate_growth,
                    delayed_growth_after_target=delayed_growth_after_target,
                    eventual_growth_from_input=eventual_growth_from_input,
                    top_pcts=top_pcts,
                    include_growth_budget=include_growth_budget,
                )
            )

        rng = np.random.default_rng(_stable_seed(s.patient_id, s.input_idx, s.target_idx, "random"))
        rows.extend(
            _score_rows_for_sample(
                method="random_score",
                source="reference",
                patient_id=s.patient_id,
                tier=tier,
                input_idx=s.input_idx,
                target_idx=s.target_idx,
                horizon=s.horizon,
                delta_days=s.delta_days,
                score=rng.random(int(outside_input.sum()), dtype=np.float32),
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
        print(f"[WARN] Skipping model delayed-hit analysis because PyTorch/model code is unavailable: {e}")
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

        for i, ctx in enumerate(sample_context):
            s = ctx["sample"]
            x, _, _ = ds[i]
            with torch.no_grad():
                logits = model(x[None].to(dev))
                prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
            score = prob[ctx["outside_input"]]
            rows.extend(
                _score_rows_for_sample(
                    method=method,
                    source="model",
                    patient_id=s.patient_id,
                    tier=ctx["tier"],
                    input_idx=s.input_idx,
                    target_idx=s.target_idx,
                    horizon=s.horizon,
                    delta_days=s.delta_days,
                    score=score,
                    immediate_growth=ctx["immediate_growth"],
                    delayed_growth_after_target=ctx["delayed_growth_after_target"],
                    eventual_growth_from_input=ctx["eventual_growth_from_input"],
                    top_pcts=top_pcts,
                    include_growth_budget=include_growth_budget,
                )
            )

            distance_score_full = _distance_to_input_score(ctx["input_mask"])
            if distance_score_full is not None:
                distance_rank = _rank_normalize_score(distance_score_full[ctx["outside_input"]])
                model_rank = _rank_normalize_score(score)
                for alpha in hybrid_alphas:
                    hybrid_score = (1.0 - alpha) * distance_rank + alpha * model_rank
                    rows.extend(
                        _score_rows_for_sample(
                            method=f"hybrid_distance_{method}_a{alpha:.2f}",
                            source="hybrid",
                            patient_id=s.patient_id,
                            tier=ctx["tier"],
                            input_idx=s.input_idx,
                            target_idx=s.target_idx,
                            horizon=s.horizon,
                            delta_days=s.delta_days,
                            score=hybrid_score,
                            immediate_growth=ctx["immediate_growth"],
                            delayed_growth_after_target=ctx["delayed_growth_after_target"],
                            eventual_growth_from_input=ctx["eventual_growth_from_input"],
                            top_pcts=top_pcts,
                            include_growth_budget=include_growth_budget,
                        )
                    )

    return pd.DataFrame(rows)


def write_report(path: Path, summary_overall: pd.DataFrame, summary_horizon: pd.DataFrame, summary_tier: pd.DataFrame) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Delayed Growth Hit Analysis\n\n")
        f.write(
            "This analysis asks whether top-ranked candidate voxels that miss the immediate target "
            "later become tumor in subsequent sessions. It separates immediate precision from eventual precision.\n\n"
        )
        f.write("## Overall\n\n")
        f.write(summary_overall.to_markdown(index=False) if not summary_overall.empty else "No summary available.")
        f.write("\n\n## By Horizon\n\n")
        f.write(summary_horizon.to_markdown(index=False) if not summary_horizon.empty else "No horizon summary available.")
        f.write("\n\n## By Tier\n\n")
        f.write(summary_tier.to_markdown(index=False) if not summary_tier.empty else "No tier summary available.")
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate whether apparent false-positive growth predictions become delayed true growth.")
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

    rows = compute_delayed_hits(
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

    summary_overall = _summarize(rows, ["ranking_source", "method", "budget_name"])
    summary_horizon = _summarize(rows, ["ranking_source", "method", "budget_name", "horizon"])
    summary_tier = _summarize(rows, ["ranking_source", "method", "budget_name", "tier"])

    rows.to_csv(output_dir / "delayed_growth_hit_samples.csv", index=False)
    summary_overall.to_csv(output_dir / "delayed_growth_hit_summary_overall.csv", index=False)
    summary_horizon.to_csv(output_dir / "delayed_growth_hit_summary_by_horizon.csv", index=False)
    summary_tier.to_csv(output_dir / "delayed_growth_hit_summary_by_tier.csv", index=False)
    write_report(output_dir / "delayed_growth_hit_report.md", summary_overall, summary_horizon, summary_tier)

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
    with (output_dir / "delayed_growth_hit_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
