from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .utils import draw_ellipsoid, normalize01, sample_point_from_mask, smooth3d


def _make_brain_and_tissues(
    rng: np.random.Generator,
    shape: Tuple[int, int, int],
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    h, w, d = shape
    xx, yy, zz = np.indices(shape)
    xn = (xx - 0.5 * h) / (0.42 * h)
    yn = (yy - 0.5 * w) / (0.40 * w)
    zn = (zz - 0.5 * d) / (0.46 * d)
    base_brain = ((xn ** 2 + yn ** 2 + zn ** 2) <= 1.0).astype(np.float32)

    irr = rng.normal(0.0, 1.0, size=shape).astype(np.float32)
    irr = normalize01(smooth3d(irr, 6))
    brain = (base_brain * (irr > 0.18)).astype(np.float32)
    if brain.sum() < 0.2 * base_brain.sum():
        brain = base_brain

    tissue_field = normalize01(smooth3d(rng.normal(0.0, 1.0, size=shape).astype(np.float32), 10))
    q1 = np.quantile(tissue_field[brain > 0], 0.33)
    q2 = np.quantile(tissue_field[brain > 0], 0.66)
    csf = ((tissue_field <= q1) * brain).astype(np.float32)
    gm = ((tissue_field > q1) * (tissue_field <= q2) * brain).astype(np.float32)
    wm = ((tissue_field > q2) * brain).astype(np.float32)
    tissue_maps = {"wm": wm, "gm": gm, "csf": csf}
    return brain.astype(np.float32), tissue_maps


def _sample_schedule(cfg: Dict, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    n_min = int(cfg["n_sessions_min"])
    n_max = int(cfg["n_sessions_max"])
    n_sessions = int(rng.integers(n_min, n_max + 1))

    int_min = int(cfg["days_interval_min"])
    int_max = int(cfg["days_interval_max"])
    intervals = rng.integers(int_min, int_max + 1, size=n_sessions - 1)
    days = np.concatenate([[0], np.cumsum(intervals)]).astype(np.float32)

    treatment = np.zeros(n_sessions, dtype=np.float32)
    if rng.random() < float(cfg["treatment_patient_prob"]):
        start_min = int(cfg["treatment_start_session_min"])
        start_min = min(start_min, n_sessions - 1)
        start_idx = int(rng.integers(start_min, n_sessions))
        treatment[start_idx:] = 1.0

    return days, treatment


def _make_initial_concentration(
    rng: np.random.Generator,
    brain: np.ndarray,
    sim_cfg: Dict,
    tier: str,
) -> Tuple[np.ndarray, int]:
    c = np.zeros_like(brain, dtype=np.float32)
    foci_min = int(sim_cfg["init_foci_min"])
    foci_max = int(sim_cfg["init_foci_max"])
    if tier == "C":
        foci_max += 1
    n_foci = int(rng.integers(foci_min, foci_max + 1))

    xx, yy, zz = np.indices(brain.shape)
    for _ in range(n_foci):
        cx, cy, cz = sample_point_from_mask(rng, brain)
        sig_min, sig_max = sim_cfg["init_sigma_vox_range"]
        sigma = float(rng.uniform(sig_min, sig_max))
        amp_min, amp_max = sim_cfg["init_amp_range"]
        amp = float(rng.uniform(amp_min, amp_max))
        dist2 = (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2
        blob = amp * np.exp(-0.5 * dist2 / (sigma * sigma))
        c = np.maximum(c, blob.astype(np.float32))

    c = np.clip(c, 0.0, 1.0) * brain
    return c.astype(np.float32), n_foci


def _rollout_procedural(
    rng: np.random.Generator,
    brain: np.ndarray,
    days: np.ndarray,
    sim_cfg: Dict,
    label_thr: float,
) -> np.ndarray:
    n_sessions = len(days)
    shape = brain.shape
    states = []

    init, _ = _make_initial_concentration(rng, brain, sim_cfg, tier="A")
    cur = (init >= label_thr).astype(np.float32)
    states.append(cur.copy())

    steps_per_day = int(sim_cfg["steps_per_day"])
    for s in range(1, n_sessions):
        delta_days = float(days[s] - days[s - 1])
        n_steps = max(1, int(round(delta_days * steps_per_day / 8.0)))
        for _ in range(n_steps):
            grown = cur.copy()
            coords = np.argwhere(cur > 0.5)
            if len(coords) == 0:
                coords = np.argwhere(brain > 0.5)
            n_seeds = int(rng.integers(1, 5))
            picks = rng.integers(0, len(coords), size=n_seeds)
            for pi in picks:
                cx, cy, cz = coords[int(pi)]
                rx = int(rng.integers(1, 4))
                ry = int(rng.integers(1, 4))
                rz = int(rng.integers(1, 4))
                draw_ellipsoid(grown, (int(cx), int(cy), int(cz)), (rx, ry, rz), value=1.0)

            if rng.random() < 0.30:
                sx = int(rng.integers(-2, 3))
                sy = int(rng.integers(-2, 3))
                sz = int(rng.integers(-1, 2))
                shifted = np.roll(grown, shift=(sx, sy, sz), axis=(0, 1, 2))
                grown = np.maximum(grown, shifted)

            if rng.random() < 0.18:
                ax, ay, az = sample_point_from_mask(rng, grown)
                ox = int(rng.integers(-10, 11))
                oy = int(rng.integers(-10, 11))
                oz = int(rng.integers(-6, 7))
                cx = int(np.clip(ax + ox, 2, shape[0] - 3))
                cy = int(np.clip(ay + oy, 2, shape[1] - 3))
                cz = int(np.clip(az + oz, 2, shape[2] - 3))
                draw_ellipsoid(grown, (cx, cy, cz), (1, 1, 1), value=1.0)

            cur = (grown * brain > 0.5).astype(np.float32)

        states.append(cur.copy())

    return np.stack(states, axis=0).astype(np.float32)


def _compute_diffusion_maps(
    rng: np.random.Generator,
    brain: np.ndarray,
    tissues: Dict[str, np.ndarray],
    sim_cfg: Dict,
    tier: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    dw_min, dw_max = sim_cfg["dw_range"]
    Dw = float(rng.uniform(dw_min, dw_max))

    wm = tissues["wm"]
    gm = tissues["gm"]
    base = Dw * (0.60 * wm + 0.35 * gm + 0.15) * brain

    if tier == "B":
        Dx = base.copy()
        Dy = base.copy()
        Dz = base.copy()
        extra = {"Dw": Dw, "anisotropy": [1.0, 1.0, 1.0]}
        return Dx, Dy, Dz, extra

    # Tier C: anisotropy + heterogeneity.
    axis = np.array([1.8, 1.1, 0.7], dtype=np.float32)
    rng.shuffle(axis)
    het = normalize01(smooth3d(rng.normal(0, 1, size=brain.shape).astype(np.float32), 6))
    het = 0.7 + 0.6 * het
    Dx = base * axis[0] * het
    Dy = base * axis[1] * het
    Dz = base * axis[2] * het
    extra = {"Dw": Dw, "anisotropy": axis.tolist()}
    return Dx.astype(np.float32), Dy.astype(np.float32), Dz.astype(np.float32), extra


def _pde_integrate_session(
    u0: np.ndarray,
    brain: np.ndarray,
    Dx: np.ndarray,
    Dy: np.ndarray,
    Dz: np.ndarray,
    rho: float,
    treat_flag: float,
    treat_effect: float,
    n_steps: int,
    dt: float,
) -> np.ndarray:
    u = u0.astype(np.float32, copy=True)
    for _ in range(max(1, int(n_steps))):
        dxx = np.roll(u, -1, axis=0) - 2.0 * u + np.roll(u, 1, axis=0)
        dyy = np.roll(u, -1, axis=1) - 2.0 * u + np.roll(u, 1, axis=1)
        dzz = np.roll(u, -1, axis=2) - 2.0 * u + np.roll(u, 1, axis=2)
        lap = Dx * dxx + Dy * dyy + Dz * dzz
        growth = rho * u * (1.0 - u)
        sink = treat_effect * treat_flag * u
        u = u + dt * (lap + growth - sink)
        u = np.clip(u, 0.0, 1.0) * brain
    return u.astype(np.float32)


def _rollout_pde(
    rng: np.random.Generator,
    brain: np.ndarray,
    tissues: Dict[str, np.ndarray],
    days: np.ndarray,
    treatment: np.ndarray,
    sim_cfg: Dict,
    tier: str,
) -> Tuple[np.ndarray, Dict]:
    n_sessions = len(days)
    steps_per_day = int(sim_cfg["steps_per_day"])
    dt = float(sim_cfg["dt"])
    rho_min, rho_max = sim_cfg["rho_range"]
    rho = float(rng.uniform(rho_min, rho_max))
    te_min, te_max = sim_cfg["treatment_effect_range"]
    treatment_effect = float(rng.uniform(te_min, te_max))

    u0, n_foci = _make_initial_concentration(rng, brain, sim_cfg, tier=tier)
    Dx, Dy, Dz, dmeta = _compute_diffusion_maps(rng, brain, tissues, sim_cfg, tier=tier)

    states = [u0.copy()]
    cur = u0.copy()
    for s in range(1, n_sessions):
        delta_days = float(days[s] - days[s - 1])
        n_steps = max(1, int(round(delta_days * steps_per_day)))
        cur = _pde_integrate_session(
            u0=cur,
            brain=brain,
            Dx=Dx,
            Dy=Dy,
            Dz=Dz,
            rho=rho,
            treat_flag=float(treatment[s]),
            treat_effect=treatment_effect,
            n_steps=n_steps,
            dt=dt,
        )
        states.append(cur.copy())

    meta = {
        "rho": rho,
        "treatment_effect": treatment_effect,
        "n_init_foci": n_foci,
        **dmeta,
    }
    return np.stack(states, axis=0).astype(np.float32), meta


def simulate_patient(
    rng: np.random.Generator,
    patient_id: str,
    tier: str,
    cfg: Dict,
) -> Dict:
    shape = tuple(int(v) for v in cfg["dataset"]["volume_shape"])
    schedule_cfg = cfg["schedule"]
    sim_cfg = cfg["simulation"]
    label_cfg = cfg["labeling"]

    brain, tissues = _make_brain_and_tissues(rng, shape)
    days, treatment = _sample_schedule(schedule_cfg, rng)

    mask_thr = float(label_cfg["mask_threshold"])
    if tier == "A":
        concentration = _rollout_procedural(
            rng=rng,
            brain=brain,
            days=days,
            sim_cfg=sim_cfg,
            label_thr=mask_thr,
        )
        sim_meta = {
            "mode": "procedural",
            "rho": None,
            "Dw": None,
            "treatment_effect": None,
            "anisotropy": None,
        }
    else:
        concentration, pde_meta = _rollout_pde(
            rng=rng,
            brain=brain,
            tissues=tissues,
            days=days,
            treatment=treatment,
            sim_cfg=sim_cfg,
            tier=tier,
        )
        sim_meta = {"mode": "reaction_diffusion", **pde_meta}

    labels = (concentration >= mask_thr).astype(np.uint8)[:, None, ...]  # [S,1,H,W,D]

    return {
        "patient_id": patient_id,
        "tier": tier,
        "brain_mask": brain.astype(np.float32),
        "tissues": tissues,
        "concentration": concentration.astype(np.float32),
        "labels": labels.astype(np.uint8),
        "days": days.astype(np.float32),
        "treatment": treatment.astype(np.float32),
        "meta": sim_meta,
    }

