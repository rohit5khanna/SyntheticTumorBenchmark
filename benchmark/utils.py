from __future__ import annotations

from typing import Tuple

import numpy as np


def smooth3d(field: np.ndarray, n_steps: int) -> np.ndarray:
    out = field.astype(np.float32, copy=True)
    for _ in range(max(0, int(n_steps))):
        out = (
            out
            + np.roll(out, 1, axis=0)
            + np.roll(out, -1, axis=0)
            + np.roll(out, 1, axis=1)
            + np.roll(out, -1, axis=1)
            + np.roll(out, 1, axis=2)
            + np.roll(out, -1, axis=2)
        ) / 7.0
    return out


def normalize01(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    xmin = float(x.min())
    xmax = float(x.max())
    if xmax - xmin < eps:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - xmin) / (xmax - xmin)).astype(np.float32)


def draw_ellipsoid(
    target: np.ndarray,
    center: Tuple[int, int, int],
    radii: Tuple[int, int, int],
    value: float = 1.0,
) -> None:
    h, w, d = target.shape
    cx, cy, cz = center
    rx, ry, rz = max(1, radii[0]), max(1, radii[1]), max(1, radii[2])
    xx, yy, zz = np.indices((h, w, d))
    eq = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 + ((zz - cz) / rz) ** 2 <= 1.0
    target[eq] = value


def sample_point_from_mask(rng: np.random.Generator, mask: np.ndarray) -> Tuple[int, int, int]:
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        h, w, d = mask.shape
        return h // 2, w // 2, d // 2
    idx = int(rng.integers(0, len(coords)))
    c = coords[idx]
    return int(c[0]), int(c[1]), int(c[2])

