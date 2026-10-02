#!/usr/bin/env python3
"""Find hidden PDE settings with similar observed scans but different futures.

This is a controlled identifiability probe, not a tumor-forecasting model.
All candidate parameters are fixed before the first simulated session.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.config import load_config
from benchmark.images import make_session_modalities
from benchmark.simulator import (
    _compute_diffusion_maps,
    _make_brain_and_tissues,
    _make_initial_concentration,
    _pde_integrate_session,
)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = int(a.sum()), int(b.sum())
    return 1.0 if na + nb == 0 else 2.0 * float(np.logical_and(a, b).sum()) / (na + nb)


def rollout(
    rng: np.random.Generator,
    u0: np.ndarray,
    brain: np.ndarray,
    tissues: dict,
    days: np.ndarray,
    sim_cfg: dict,
    tier: str,
    rho: float,
    dw: float,
) -> np.ndarray:
    map_cfg = dict(sim_cfg)
    map_cfg["dw_range"] = [dw, dw]
    dx, dy, dz, _ = _compute_diffusion_maps(rng, brain, tissues, map_cfg, tier)
    states = [u0.copy()]
    cur = u0.copy()
    for s in range(1, len(days)):
        steps = max(1, int(round((days[s] - days[s - 1]) * sim_cfg["steps_per_day"])))
        cur = _pde_integrate_session(
            cur, brain, dx, dy, dz, rho, 0.0, 0.0, steps, float(sim_cfg["dt"])
        )
        states.append(cur)
    return np.stack(states)


def observed_images(
    states: np.ndarray,
    brain: np.ndarray,
    tissues: dict,
    image_cfg: dict,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack(
        [
            make_session_modalities(state, brain, tissues, image_cfg, ["t1ce", "flair"], rng)
            for state in states[:3]
        ]
    )


def image_rmse(a: np.ndarray, b: np.ndarray, brain: np.ndarray) -> float:
    delta = a[..., brain > 0] - b[..., brain > 0]
    return float(np.sqrt(np.mean(delta * delta)))


def pairwise_dice(masks: list[np.ndarray]) -> list[float]:
    return [dice(masks[i], masks[j]) for i in range(len(masks)) for j in range(i + 1, len(masks))]


def render_example(path: Path, anchor: np.ndarray, candidate: np.ndarray, threshold: float) -> None:
    import matplotlib.pyplot as plt

    masks = [(anchor >= threshold), (candidate >= threshold)]
    z = int(np.argmax((masks[0] | masks[1]).sum(axis=(0, 1))))
    fig, axes = plt.subplots(2, 4, figsize=(12, 6), constrained_layout=True)
    for row, (states, name) in enumerate([(anchor, "Anchor"), (candidate, "Matched candidate")]):
        for col in range(4):
            axes[row, col].imshow(states[col, :, :, z].T, cmap="magma", vmin=0, vmax=1, origin="lower")
            axes[row, col].contour(masks[row][col, :, :, z].T, levels=[0.5], colors="cyan", linewidths=0.8)
            axes[row, col].set_title(f"{name}, session {col}")
            axes[row, col].axis("off")
    fig.suptitle(f"Same initial state; near-matched observed sessions 0-2; future at session 3 (z={z})")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/benchmark_sprint.yaml")
    parser.add_argument("--output_dir", default="outputs/ambiguity_probe")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--anchors_per_tier", type=int, default=2)
    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument("--tiers", default="B,C")
    parser.add_argument("--shape", default="32,32,24")
    parser.add_argument("--days", default="0,20,40,80")
    parser.add_argument("--steps_per_day", type=int, default=1)
    args = parser.parse_args()

    cfg = load_config(args.config)
    shape = tuple(int(x) for x in args.shape.split(","))
    days = np.array([float(x) for x in args.days.split(",")], dtype=np.float32)
    tiers = [x.strip() for x in args.tiers.split(",") if x.strip()]
    if len(shape) != 3 or min(shape) < 16 or len(days) != 4 or not np.all(np.diff(days) > 0):
        parser.error("Use a 3D shape of at least 16^3 and four increasing session days")
    if args.anchors_per_tier < 1 or args.candidates < 1 or args.steps_per_day < 1:
        parser.error("anchors_per_tier, candidates, and steps_per_day must be positive")
    if not tiers or any(t not in {"B", "C"} for t in tiers):
        parser.error("tiers must be B, C, or B,C")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rows = []
    summaries = []
    figure_written = False
    thresholds = (0.90, 0.95, 0.98)

    for tier in tiers:
        tier_cfg = cfg["tiers"][tier]
        sim_cfg = {**cfg["simulation"], **tier_cfg.get("simulation_overrides", {})}
        sim_cfg["steps_per_day"] = args.steps_per_day
        image_cfg = {**cfg["image_synthesis"], **tier_cfg.get("image_synthesis_overrides", {})}
        mask_threshold = float(cfg["labeling"]["mask_threshold"])

        for anchor_id in range(args.anchors_per_tier):
            brain, tissues = _make_brain_and_tissues(rng, shape)
            u0, _ = _make_initial_concentration(rng, brain, sim_cfg, tier)
            rho0 = float(rng.uniform(*sim_cfg["rho_range"]))
            dw0 = float(rng.uniform(*sim_cfg["dw_range"]))
            anchor = rollout(rng, u0, brain, tissues, days, sim_cfg, tier, rho0, dw0)
            anchor_masks = anchor >= mask_threshold
            anchor_img = observed_images(anchor, brain, tissues, image_cfg, int(rng.integers(2**31)))

            # Independent synthetic scans of the same state define the image-noise tolerance.
            repeat_rmse = [
                image_rmse(
                    anchor_img,
                    observed_images(anchor, brain, tissues, image_cfg, int(rng.integers(2**31))),
                    brain,
                )
                for _ in range(12)
            ]
            image_limit = float(np.quantile(repeat_rmse, 0.95))
            candidates = []
            for candidate_id in range(args.candidates):
                rho = float(rng.uniform(*sim_cfg["rho_range"]))
                dw = float(rng.uniform(*sim_cfg["dw_range"]))
                states = rollout(rng, u0, brain, tissues, days, sim_cfg, tier, rho, dw)
                masks = states >= mask_threshold
                early_dice = min(dice(anchor_masks[s], masks[s]) for s in (1, 2))
                img = observed_images(states, brain, tissues, image_cfg, int(rng.integers(2**31)))
                rmse = image_rmse(anchor_img, img, brain)
                future_dice = dice(anchor_masks[3], masks[3])
                row = {
                    "tier": tier,
                    "anchor": anchor_id,
                    "candidate": candidate_id,
                    "rho_anchor": rho0,
                    "dw_anchor": dw0,
                    "rho_candidate": rho,
                    "dw_candidate": dw,
                    "early_min_dice": early_dice,
                    "early_image_rmse": rmse,
                    "image_noise_95pct": image_limit,
                    "future_dice_to_anchor": future_dice,
                }
                for cutoff in thresholds:
                    row[f"accepted_mask_{cutoff:.2f}"] = int(early_dice >= cutoff and rmse <= image_limit)
                rows.append(row)
                candidates.append((states, masks[3], row))

            for cutoff in thresholds:
                accepted = [c for c in candidates if c[2][f"accepted_mask_{cutoff:.2f}"]]
                # Include the anchor in the accepted-future distribution.
                future_masks = [anchor_masks[3], *[c[1] for c in accepted]]
                agreements = pairwise_dice(future_masks)
                summary = {
                    "tier": tier,
                    "anchor": anchor_id,
                    "early_dice_cutoff": cutoff,
                    "n_candidates": args.candidates,
                    "n_accepted": len(accepted),
                    "image_noise_95pct": image_limit,
                    "future_pairwise_dice_mean": float(np.mean(agreements)) if agreements else None,
                    "future_pairwise_dice_min": float(np.min(agreements)) if agreements else None,
                }
                summaries.append(summary)
                print(json.dumps(summary), flush=True)

                if cutoff == 0.95 and accepted and not figure_written:
                    example = min(accepted, key=lambda item: item[2]["future_dice_to_anchor"])
                    render_example(out / "matched_histories_example.png", anchor, example[0], mask_threshold)
                    figure_written = True

    with (out / "candidates.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (out / "summary.json").open("w") as f:
        json.dump(
            {
                "protocol": {
                    "config": args.config,
                    "seed": args.seed,
                    "tiers": tiers,
                    "shape": shape,
                    "days": days.tolist(),
                    "steps_per_day": args.steps_per_day,
                    "treatment": "none; held fixed",
                    "initial_concentration": "identical within each anchor set",
                    "image_modalities": ["t1ce", "flair"],
                    "image_rule": "candidate RMSE no larger than 95th percentile of same-state repeat scans",
                    "mask_cutoffs": thresholds,
                },
                "results": summaries,
            },
            f,
            indent=2,
        )
    print(f"Saved {len(rows)} candidates and {len(summaries)} summaries to {out}")


if __name__ == "__main__":
    main()
