"""
Wall patterns.

Each pattern is `fn(pot, th, Z, r, s)` and returns a field for points in the
wall, where s goes from 0 on the soil side to 1 outside. Two kinds:
  - "3d" lattices (gyroid, diamond): the field is the wall MATERIAL
    (negative = solid). Tortuous channels, no straight line of sight.
  - "2d" perforations (voronoi, hex, coral, slots): the field is the HOLE
    (negative = open), cut radially through the wall. Every hole roof is a
    bridge no longer than the hole width.

All patterns wrap seamlessly around the pot: integer repeat counts in theta,
and warp terms with integer frequency in theta. Cell counts are given at the
reference height and scaled with `pot.count`, so holes keep their size in mm.
"""
from functools import lru_cache
from typing import Callable, NamedTuple

import numpy as np

from . import coral
from .sdf import CELL, T_IN, T_OUT, gyroid

GYROID_NC_REF = 48
DIAMOND_NC_REF = 44
VOR_NC_REF = 76
HEX_NC_REF = 84
SLOT_NC_REF = 60
VOR_H = 5.2           # voronoi row height (mm), fixed
VOR_JITTER = 0.30     # seed jitter, fraction of a cell
STRUT = 1.8           # strut width of the fixed-strut 2D patterns


class Pattern(NamedTuple):
    kind: str                     # "3d" or "2d"
    fn: Callable
    description: str


# ---------------------------------------------------------------- 3D lattices
def pat_gyroid(pot, th, Z, r, s):
    return gyroid(th, Z, r, T_IN + (T_OUT - T_IN) * s, pot.count(GYROID_NC_REF))


def pat_diamond(pot, th, Z, r, s):
    """Schwarz diamond TPMS: straighter 45-degree channels than the gyroid."""
    n = pot.count(DIAMOND_NC_REF)
    x = th * n + 0.9 * np.sin(4 * th + 2 * np.pi * Z / 60)
    y = 2 * np.pi * Z / (CELL * 1.1) + 0.7 * np.sin(3 * th - 2 * np.pi * Z / 45)
    z = 2 * np.pi * r / (CELL * 1.1)
    d = (np.sin(x) * np.sin(y) * np.sin(z) + np.sin(x) * np.cos(y) * np.cos(z)
         + np.cos(x) * np.sin(y) * np.cos(z) + np.cos(x) * np.cos(y) * np.sin(z))
    level = 0.62 + (0.40 - 0.62) * s
    return (np.abs(d) - level) * CELL * 1.1 / (2 * np.pi * 1.2)


# ------------------------------------------------------ 2D perforations
def unroll(pot, th, Z):
    """Cylinder -> flat (s, z) in mm, s measured at radius pot.r0."""
    return th * pot.r0, Z


@lru_cache(maxsize=8)
def _vor_jitter(nr, nc):
    # periodic jittered seeds (fixed seed -> repeatable)
    return np.random.default_rng(7).uniform(-VOR_JITTER, VOR_JITTER, (nr, nc, 2))


def voronoi_edge(pot, S, ZZ):
    """Approximate distance (mm) to the nearest Voronoi cell edge, (F2 - F1) / 2."""
    nc = pot.count(VOR_NC_REF)
    nr = int(pot.height / VOR_H) + 3
    w = pot.circ / nc
    jitter = _vor_jitter(nr, nc)
    ci = np.floor(S / w).astype(int)
    cj = np.floor(ZZ / VOR_H).astype(int)
    f1 = np.full(S.shape, 1e9, np.float32)
    f2 = np.full(S.shape, 1e9, np.float32)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            i = ci + di
            j = np.clip(cj + dj + 1, 0, nr - 1)
            jit = jitter[j, i % nc]
            px = (i + 0.5 + jit[..., 0]) * w
            pz = (j - 1 + 0.5 + jit[..., 1]) * VOR_H
            d = np.hypot(S - px, ZZ - pz)
            f2 = np.where(d < f1, f1, np.minimum(f2, d))
            f1 = np.minimum(f1, d)
    return (f2 - f1) / 2


def pat_voronoi(pot, th, Z, r, s):
    """Organic cells: periodic jittered Voronoi, holes = shrunken cells."""
    S, ZZ = unroll(pot, th, Z)
    S = S + 1.2 * np.sin(2 * np.pi * Z / 47)
    return STRUT / 2 - voronoi_edge(pot, S, ZZ)


def pat_voronoi_taper(pot, th, Z, r, s):
    """Voronoi cells shaped like funnels: small on the soil side, wide outside.
    Struts thin from pot.strut_in to pot.strut_out through the wall. The
    pattern is shifted down by the same amount each edge recedes, so every
    hole grows sideways and downward while its roof stays level (a plain
    short bridge, never a sagging sloped ceiling)."""
    S, ZZ = unroll(pot, th, Z)
    S = S + 1.2 * np.sin(2 * np.pi * Z / 47)
    strut = pot.strut_in + (pot.strut_out - pot.strut_in) * s
    shift = (pot.strut_in - strut) / 2
    return strut / 2 - voronoi_edge(pot, S, ZZ + shift)


def pat_hex(pot, th, Z, r, s):
    """Warped honeycomb, vertex pointing up (self-supporting roofs)."""
    a = pot.circ / pot.count(HEX_NC_REF)   # column pitch (flat-to-flat)
    rc = a / np.sqrt(3)                  # circumradius
    b = 3 * rc
    S, ZZ = unroll(pot, th, Z)
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


def pat_coral(pot, th, Z, r, s):
    """Reaction-diffusion (Turing) pattern: coral / fingerprint labyrinth."""
    from scipy import ndimage
    d, px = coral.texture(pot)
    S, ZZ = unroll(pot, th, Z)
    ix = np.mod(S, pot.circ) / px
    iz = (ZZ + 4) / px
    val = ndimage.map_coordinates(d, [iz.ravel(), ix.ravel()], order=1, mode="grid-wrap").reshape(S.shape)
    # val > 0 : open (inside a hole) ; shrink holes slightly so struts >= ~1.6 mm
    return -(val - 0.4)


def pat_slots(pot, th, Z, r, s):
    """Air-pruning style: narrow wavy vertical slots with pointed ends."""
    pitch = pot.circ / pot.count(SLOT_NC_REF)
    rowh = 22.0
    S, ZZ = unroll(pot, th, Z)
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
    "voronoi_taper": Pattern("2d", pat_voronoi_taper, "organic cells flaring outward like funnels (default)"),
    "voronoi": Pattern("2d", pat_voronoi, "organic cells, straight holes"),
    "hex": Pattern("2d", pat_hex, "warped honeycomb, pointy-top cells"),
    "coral": Pattern("2d", pat_coral, "reaction-diffusion labyrinth (slow first run, cached)"),
    "slots": Pattern("2d", pat_slots, "narrow wavy vertical slots, air-pruning style"),
    "gyroid": Pattern("3d", pat_gyroid, "3D lattice, no straight line of sight"),
    "diamond": Pattern("3d", pat_diamond, "3D lattice, straighter 45-degree channels"),
}
