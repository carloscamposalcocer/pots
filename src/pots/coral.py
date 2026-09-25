"""
Gray-Scott reaction-diffusion texture for the `coral` pattern.

Generating it takes a while, so each size is cached on disk, in
$POTS_CACHE_DIR or ~/.cache/pots, and in memory for the rest of the run.
"""
import logging
import os
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import ndimage

log = logging.getLogger(__name__)

PX = 0.4              # simulation grid spacing (mm)


def cache_dir():
    return Path(os.environ.get("POTS_CACHE_DIR") or Path.home() / ".cache" / "pots")


def texture(pot):
    """Signed distance (mm) to the pattern edge on the unrolled wall, > 0 inside
    a hole, plus its pixel size. Periodic in s, so seamless around the pot."""
    nx = int(round(pot.circ / PX))
    nz = int(round((pot.height + 8) / PX))
    return _texture(nx, nz)


@lru_cache(maxsize=4)
def _texture(nx, nz):
    path = cache_dir() / f"coral_{nx}x{nz}.npy"
    if path.exists():
        log.info("loading coral texture cache %s", path)
        tex = np.load(path)
    else:
        tex = _generate(nx, nz)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, tex)
        log.info("saved coral texture cache %s", path)
    return tex[0], float(tex[1][0, 0])


def _generate(nx, nz):
    log.info("generating coral texture %d x %d (this is slow; cached after first run)", nz, nx)
    t0 = time.perf_counter()
    U = np.ones((nz, nx)); V = np.zeros((nz, nx))
    g = np.random.default_rng(3)
    for _ in range(90):
        i, j = g.integers(0, nz), g.integers(0, nx)
        U[max(i-3,0):i+3, j:j+6] = 0.5; V[max(i-3,0):i+3, j:j+6] = 0.25
    Du, Dv, f, k = 0.16, 0.08, 0.0545, 0.062

    def lap(A):
        return np.roll(A, 1, 0) + np.roll(A, -1, 0) + np.roll(A, 1, 1) + np.roll(A, -1, 1) - 4 * A

    n_steps = 9000
    for step in range(n_steps):
        uvv = U * V * V
        U += Du * lap(U) - uvv + f * (1 - U)
        V += Dv * lap(V) + uvv - (f + k) * V
        if (step + 1) % 1000 == 0:
            log.info("coral reaction-diffusion %d/%d", step + 1, n_steps)
    # upsample the smooth V field 4x before thresholding -> clean edges
    Vt = np.concatenate([V[:, -30:], V, V[:, :30]], 1)
    Vu = ndimage.zoom(ndimage.gaussian_filter(Vt, 0.7), 4, order=3)
    solid = Vu < 0.2
    d = (ndimage.distance_transform_edt(~solid) - ndimage.distance_transform_edt(solid)) * PX / 4
    d = d[:, 120:-120]
    log.info("coral texture generated in %.0f s", time.perf_counter() - t0)
    return np.stack([d, np.full_like(d, PX / 4)])
