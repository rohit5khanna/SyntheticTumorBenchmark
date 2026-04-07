from __future__ import annotations

from typing import Dict, List

import numpy as np

from .utils import normalize01, smooth3d


def make_session_modalities(
    concentration: np.ndarray,
    brain_mask: np.ndarray,
    tissue_maps: Dict[str, np.ndarray],
    image_cfg: Dict,
    modalities: List[str],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Returns tensor [C, H, W, D] with synthetic MRI-like channels.
    """
    conc = np.clip(concentration.astype(np.float32), 0.0, 1.0)
    brain = brain_mask.astype(np.float32)
    wm = tissue_maps["wm"].astype(np.float32)
    gm = tissue_maps["gm"].astype(np.float32)
    csf = tissue_maps["csf"].astype(np.float32)

    core_thr = float(image_cfg["core_threshold"])
    edema_thr = float(image_cfg["edema_threshold"])
    core = (conc >= core_thr).astype(np.float32)
    edema = (conc >= edema_thr).astype(np.float32) * (1.0 - core)
    edema = smooth3d(edema, 2) * brain

    # Global smooth multiplicative bias field.
    bias_raw = rng.normal(0.0, 1.0, size=conc.shape).astype(np.float32)
    bias = normalize01(smooth3d(bias_raw, int(image_cfg["bias_smooth_steps"])))
    bias = 1.0 + float(image_cfg["bias_amp"]) * (bias - 0.5)

    # Tissue baseline.
    base_t1 = 0.65 * wm + 0.45 * gm + 0.22 * csf
    base_t2 = 0.35 * wm + 0.55 * gm + 0.68 * csf
    base_flair = 0.28 * wm + 0.50 * gm + 0.58 * csf

    out = []
    for mod in modalities:
        mod_l = mod.lower()
        if mod_l == "t1":
            img = base_t1 + float(image_cfg["t1_core_boost"]) * core + 0.10 * edema
        elif mod_l == "t1ce":
            img = base_t1 + float(image_cfg["t1ce_core_boost"]) * core + 0.05 * edema
        elif mod_l == "t2":
            img = base_t2 + 0.10 * core + float(image_cfg["t2_edema_boost"]) * edema
        elif mod_l == "flair":
            img = base_flair + 0.12 * core + float(image_cfg["flair_edema_boost"]) * edema
        else:
            # Generic fallback channel.
            img = base_t1 + 0.20 * core + 0.20 * edema

        noise = rng.normal(0.0, float(image_cfg["noise_std"]), size=img.shape).astype(np.float32)
        img = (img * bias + noise) * brain
        out.append(normalize01(img))

    return np.stack(out, axis=0).astype(np.float32)

