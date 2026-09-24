"""
Alternative wall patterns for the single-piece pot (pot_v3 layout).

Each pattern returns the wall MATERIAL field (negative = solid) for points
inside the wall band. Two families:
  - 3D lattices (gyroid, diamond): tortuous channels, no straight line of sight
  - 2D perforations cut radially through the wall (voronoi, hex, coral, slots):
    every hole roof is a bridge no longer than the wall thickness (6 mm)

All patterns wrap seamlessly around the pot (integer repeat counts in theta).
Usage:  PATTERN=voronoi python3 patterns.py [voxel] [out.stl]
"""
import logging
import os
import sys
import time
import numpy as np
from scipy import ndimage

import pot_v2 as P2
from pot_v2 import CELL, T_IN, T_OUT, BLEND, r_out, r_in, smin, gyroid, build
import pot_v3 as P3

log = logging.getLogger(__name__)

R0_REF = 64.0                   # reference radius for unrolled (s, z) coordinates, at P2.H_REF
STRUT = 1.8                     # strut width for 2D perforation patterns

# Cell counts around the pot, at P2.H_REF. set_height() scales them with the
# circumference (rounded to integers -> still seamless), so the cells and
# holes keep their size in mm while the pot grows or shrinks.
DIAMOND_NC_REF = 44
VOR_NC_REF = 76
HEX_NC_REF = 84
SLOT_NC_REF = 60
VOR_H = 5.2                     # voronoi row height (mm), fixed
R0 = CIRC = None                # all set by set_height()
DIAMOND_NC = VOR_NC = VOR_W = VOR_NR = VOR_J = HEX_NC = SLOT_NC = None


def _count(ref):
    return max(3, round(ref * P2.K))


def set_height(h):
    """Scale the whole pot (wall, base, cup, patterns) to height h (mm)."""
    global R0, CIRC, DIAMOND_NC, VOR_NC, VOR_W, VOR_NR, VOR_J, HEX_NC, SLOT_NC
    P3.set_height(h)
    R0 = R0_REF * P2.K
    CIRC = 2 * np.pi * R0
    DIAMOND_NC = _count(DIAMOND_NC_REF)
    VOR_NC = _count(VOR_NC_REF)
    VOR_W = CIRC / VOR_NC
    VOR_NR = int(P2.H / VOR_H) + 3
    # periodic jittered seeds for the voronoi pattern (fixed seed -> repeatable)
    VOR_J = np.random.default_rng(7).uniform(-0.30, 0.30, (VOR_NR, VOR_NC, 2))
    HEX_NC = _count(HEX_NC_REF)
    SLOT_NC = _count(SLOT_NC_REF)
    log.debug("height %.1f mm  k=%.3f  R %.1f-%.1f mm  voronoi %d x %d",
              P2.H, P2.K, P2.R_BOT, P2.R_TOP, VOR_NC, VOR_NR)


set_height(P2.H_REF)


# ---------------------------------------------------------------- 3D lattices
def pat_gyroid(th, Z, r, s):
    return gyroid(th, Z, r, T_IN + (T_OUT - T_IN) * s)


def pat_diamond(th, Z, r, s):
    """Schwarz diamond TPMS: straighter 45-degree channels than the gyroid."""
    n = DIAMOND_NC
    x = th * n + 0.9 * np.sin(4 * th + 2 * np.pi * Z / 60)
    y = 2 * np.pi * Z / (CELL * 1.1) + 0.7 * np.sin(3 * th - 2 * np.pi * Z / 45)
    z = 2 * np.pi * r / (CELL * 1.1)
    d = (np.sin(x) * np.sin(y) * np.sin(z) + np.sin(x) * np.cos(y) * np.cos(z)
         + np.cos(x) * np.sin(y) * np.cos(z) + np.cos(x) * np.cos(y) * np.sin(z))
    level = 0.62 + (0.40 - 0.62) * s
    return (np.abs(d) - level) * CELL * 1.1 / (2 * np.pi * 1.2)


# ------------------------------------------------------ 2D perforations
def unroll(th, Z):
    return th * R0, Z


def _voronoi_edge(S, ZZ):
    """Approximate distance (mm) to the nearest Voronoi cell edge, (F2 - F1) / 2."""
    ci = np.floor(S / VOR_W).astype(int)
    cj = np.floor(ZZ / VOR_H).astype(int)
    f1 = np.full(S.shape, 1e9, np.float32)
    f2 = np.full(S.shape, 1e9, np.float32)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            i = ci + di
            j = np.clip(cj + dj + 1, 0, VOR_NR - 1)
            jit = VOR_J[j, i % VOR_NC]
            px = (i + 0.5 + jit[..., 0]) * VOR_W
            pz = (j - 1 + 0.5 + jit[..., 1]) * VOR_H
            d = np.hypot(S - px, ZZ - pz)
            f2 = np.where(d < f1, f1, np.minimum(f2, d))
            f1 = np.minimum(f1, d)
    return (f2 - f1) / 2


def pat_voronoi(th, Z, r, s):
    """Organic cells: periodic jittered Voronoi, holes = shrunken cells."""
    S, ZZ = unroll(th, Z)
    S = S + 1.2 * np.sin(2 * np.pi * Z / 47)
    return STRUT / 2 - _voronoi_edge(S, ZZ)          # <0 = open (hole)


STRUT_IN, STRUT_OUT = 2.2, 1.1     # strut width on the soil side / outside


def pat_voronoi_taper(th, Z, r, s):
    """Voronoi cells shaped like funnels: small on the soil side, wide outside.
    Struts thin from STRUT_IN to STRUT_OUT through the wall. The pattern is
    shifted down by the same amount each edge recedes, so every hole grows
    sideways and downward while its roof stays level (a plain short bridge,
    never a sagging sloped ceiling)."""
    S, ZZ = unroll(th, Z)
    S = S + 1.2 * np.sin(2 * np.pi * Z / 47)
    strut = STRUT_IN + (STRUT_OUT - STRUT_IN) * s
    shift = (STRUT_IN - strut) / 2
    return strut / 2 - _voronoi_edge(S, ZZ + shift)


def pat_hex(th, Z, r, s):
    """Warped honeycomb, vertex pointing up (self-supporting roofs)."""
    a = CIRC / HEX_NC                    # column pitch (flat-to-flat)
    rc = a / np.sqrt(3)                  # circumradius
    b = 3 * rc
    S, ZZ = unroll(th, Z)
    S = S + 1.6 * np.sin(2 * np.pi * Z / 38 + 3 * th)
    ZZ = ZZ + 1.0 * np.sin(5 * th)
    best = np.full(S.shape, 1e9, np.float32)
    hexd = np.zeros_like(best)
    for ox, oz in ((0, 0), (a / 2, b / 2)):
        px = np.mod(S - ox, a) - a / 2
        pz = np.mod(ZZ - oz, b) - b / 2
        d2 = px * px + pz * pz
        qx, qz = np.abs(px), np.abs(pz)
        h = np.maximum(qx, qx * 0.5 + qz * np.sqrt(3) / 2)      # pointy-top hex metric
        hexd = np.where(d2 < best, h, hexd)
        best = np.minimum(best, d2)
    apothem = a / 2
    return hexd - (apothem - STRUT / 2)


def _coral_texture():
    """Gray-Scott reaction-diffusion on a periodic grid (seamless around the pot)."""
    px = 0.4
    nx = int(round(CIRC / px)); nz = int(round((P2.H + 8) / px))
    # one cache per size; the reference size keeps the original name
    cache = "coral_tex.npy" if P2.K == 1.0 else f"coral_tex_{nx}x{nz}.npy"
    if os.path.exists(cache):
        log.info("loading coral texture cache %s", cache)
        return np.load(cache)
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
    d = (ndimage.distance_transform_edt(~solid) - ndimage.distance_transform_edt(solid)) * px / 4
    d = d[:, 120:-120]
    px = px / 4
    np.save(cache, np.stack([d, np.full_like(d, px)]))
    log.info("saved coral texture cache %s in %.0f s", cache, time.perf_counter() - t0)
    return np.load(cache)


def pat_coral(th, Z, r, s):
    """Reaction-diffusion (Turing) pattern: coral / fingerprint labyrinth."""
    tex = _coral_texture()
    d, px = tex[0], tex[1][0, 0]
    S, ZZ = unroll(th, Z)
    ix = np.mod(S, CIRC) / px
    iz = (ZZ + 4) / px
    val = ndimage.map_coordinates(d, [iz.ravel(), ix.ravel()], order=1, mode="grid-wrap").reshape(S.shape)
    # val > 0 : open (inside a hole) ; shrink holes slightly so struts >= ~1.6 mm
    return -(val - 0.4)


def pat_slots(th, Z, r, s):
    """Air-pruning style: narrow wavy vertical slots with pointed ends."""
    pitch = CIRC / SLOT_NC
    rowh = 22.0
    S, ZZ = unroll(th, Z)
    S = S + 2.2 * np.sin(2 * np.pi * Z / 30 + 2 * th)
    row = np.floor(ZZ / rowh)
    Ss = S + (np.mod(row, 2)) * pitch / 2
    px = np.mod(Ss, pitch) - pitch / 2
    pz = np.mod(ZZ, rowh) - rowh / 2
    halfw, halfl = 1.1, 8.5
    # pointed-end slot: rectangle with 45-degree tips (diamond caps)
    tip = (np.abs(px) + np.abs(pz) - halfl) / np.sqrt(2)
    return np.maximum(np.abs(px) - halfw, tip)


PATTERNS = {
    "gyroid": ("3d", pat_gyroid),
    "diamond": ("3d", pat_diamond),
    "voronoi": ("2d", pat_voronoi),
    "voronoi_taper": ("2d", pat_voronoi_taper),
    "hex": ("2d", pat_hex),
    "coral": ("2d", pat_coral),
    "slots": ("2d", pat_slots),
}


def make_field(name):
    kind, fn = PATTERNS[name]

    def field(X, Y, Z):
        r = np.sqrt(X * X + Y * Y)
        th = np.arctan2(Y, X)
        ro, ri = r_out(Z), r_in(Z)
        shell = np.maximum.reduce([r - ro, ri - r, -Z, Z - P2.H])
        s = np.clip((r - ri) / P2.WALL, 0, 1)
        band_lo = Z - (P3.BASE + 1.0)            # pattern starts just above the base
        band_hi = (P2.H - P2.RIM) - Z
        band = np.minimum(band_lo, band_hi)       # >0 inside the pattern band
        if kind == "3d":
            mat = smin(fn(th, Z, r, s), band, BLEND)      # solid outside the band
        else:
            hole = np.maximum(fn(th, Z, r, s), -band)   # holes only inside the band
            mat = -hole
        wall = np.maximum(shell, mat)
        ci = P3.cup_ri(Z)
        co = ci + P3.CUP_WALL
        cup_wall = np.maximum.reduce([r - co, ci - r, -Z, Z - P3.CUP_H])
        base = np.maximum.reduce([r - co, -Z, Z - P3.BASE])
        body = smin(smin(wall, base, 1.0), cup_wall, 1.0)
        return np.maximum(body, ((r - co) + (0.6 - Z)) / np.sqrt(2))
    return field


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-5s  %(message)s", datefmt="%H:%M:%S")
    name = os.environ.get("PATTERN", "gyroid")
    vox = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
    out = sys.argv[2] if len(sys.argv) > 2 else f"pot_{name}_raw.stl"
    log.info("pattern=%s  voxel=%.3f mm  out=%s", name, vox, out)
    m = build(make_field(name), P2.R_TOP + P3.LIP + P3.CUP_WALL + 1.0, P2.H, vox, out)
    log.info("%s  faces=%s  volume=%.1f cm3", name, f"{len(m.faces):,}", m.volume / 1000)
