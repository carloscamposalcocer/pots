"""
Wall patterns.

Each pattern is `fn(pot, th, Z, r, s)` and returns a field for points in the
wall, where s goes from 0 on the soil side to 1 outside. Two kinds:
  - "3d" lattices (gyroid, diamond, weave): the field is the wall MATERIAL
    (negative = solid). Tortuous channels, no straight line of sight.
  - "2d" perforations (voronoi, hex, slots, lattice, drops, ...): the field is
    the HOLE (negative = open), cut through the wall. Every hole roof is either
    a bridge no longer than the hole width or a slope of at least SLOPE.

Sloped roofs are drawn at SLOPE (55 degrees) in the unrolled (s, z) plane.
The real radius is 0.875 to 1.125 x pot.r0, which stretches s, so that keeps
them at 45 degrees or steeper everywhere on the pot. Hole shapes never close
into loops, so the solid wall stays one piece.

All patterns wrap seamlessly around the pot: integer repeat counts in theta,
and warp terms with integer frequency in theta. Cell counts are given at the
reference height and scaled with `pot.count`, so holes keep their size in mm.
"""
from functools import lru_cache
from typing import Callable, NamedTuple

import numpy as np

from .geometry import PATTERN_GAP
from .sdf import CELL, T_IN, T_OUT, gyroid

GYROID_NC_REF = 48
DIAMOND_NC_REF = 44
VOR_NC_REF = 76
HEX_NC_REF = 84
SLOT_NC_REF = 60
LATTICE_NC_REF = 72
LOUVER_NC_REF = 29
DROP_NC_REF = 70
SPIRAL_NC_REF = 80    # counted in pairs: neighbouring helices alternate
ISOGRID_NC_REF = 54
CHEVRON_NC_REF = 30
WEAVE_NC_REF = 56
VOR_H = 5.2           # voronoi row height (mm), fixed
VOR_JITTER = 0.30     # seed jitter, fraction of a cell
STRUT = 1.8           # strut width of the fixed-strut 2D patterns
SLOPE = np.deg2rad(55)  # sloped roofs, from horizontal in the unrolled plane


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


def hex_edge(pot, th, Z, shift=0.0):
    """Hex metric (mm from the cell centre) of the warped honeycomb, and the
    apothem. The pattern is moved down by `shift`."""
    a = pot.circ / pot.count(HEX_NC_REF)   # column pitch (flat-to-flat)
    rc = a / np.sqrt(3)                  # circumradius
    b = 3 * rc
    S, ZZ = unroll(pot, th, Z)
    S = S + 1.6 * np.sin(2 * np.pi * Z / 38 + 3 * th)
    ZZ = ZZ + 1.0 * np.sin(5 * th) + shift
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
    return hexd, a / 2


def pat_hex(pot, th, Z, r, s):
    """Warped honeycomb, vertex pointing up (self-supporting roofs)."""
    hexd, apothem = hex_edge(pot, th, Z)
    return hexd - (apothem - STRUT / 2)


HEX_WARP_EXTRA = 0.2  # mm added to hex_taper struts


def pat_hex_taper(pot, th, Z, r, s):
    """The honeycomb with the voronoi_taper funnel: struts thin from strut_in
    to strut_out through the wall. The roof edges have a vertical normal
    component of sqrt(3)/2, so the pattern moves down by recession / (sqrt(3)/2)
    to keep every roof where it is. HEX_WARP_EXTRA makes up for the warp,
    which shears the cells and thins some struts."""
    strut = pot.strut_in + (pot.strut_out - pot.strut_in) * s + HEX_WARP_EXTRA
    shift = (pot.strut_in - strut) / 2 / (np.sqrt(3) / 2)
    hexd, apothem = hex_edge(pot, th, Z, shift)
    return hexd - (apothem - strut / 2)


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


def stripes(u, p):
    """Distance from u to the nearest multiple of p."""
    return np.abs(np.mod(u + p / 2, p) - p / 2)


def pat_lattice(pot, th, Z, r, s):
    """Diamond trellis: two sets of helical strips crossing at +-SLOPE.
    The holes are diamonds with pointed tops, so there are no bridges."""
    p = pot.circ / pot.count(LATTICE_NC_REF)
    S, ZZ = unroll(pot, th, Z)
    c = 1 / np.tan(SLOPE)
    du = stripes(S + ZZ * c, p) * np.sin(SLOPE)     # normal distance to each strip
    dv = stripes(S - ZZ * c, p) * np.sin(SLOPE)
    return STRUT / 2 - np.minimum(du, dv)


LOUVER_PITCH = 5.0    # row spacing (mm)
LOUVER_H = 2.4        # slit height (mm)
LOUVER_RIB = 2.0      # rib between slits in a row (mm)
LOUVER_DROP = 1.2     # the slit drops this much per mm of wall (50 degrees)


def pat_louvers(pot, th, Z, r, s):
    """Gills: brick-staggered slits that run down and outward through the wall
    like shutter blades. The drop is more than the slit height, so there is
    no line of sight: soil stays in, rain runs off, air still flows. Only rows
    that fit whole inside the pattern band are cut."""
    p = pot.circ / pot.count(LOUVER_NC_REF)
    S, ZZ = unroll(pot, th, Z)
    drop = LOUVER_DROP * pot.wall
    Zs = ZZ + drop * s                         # the height this point has on the soil side
    k = np.floor(Zs / LOUVER_PITCH)
    zc = (k + 0.5) * LOUVER_PITCH
    lo, hi = pot.base + PATTERN_GAP + 0.5, pot.height - pot.rim - 0.5
    whole = (zc - LOUVER_H / 2 - drop >= lo) & (zc + LOUVER_H / 2 <= hi)
    px = np.mod(S + np.mod(k, 2) * p / 2, p) - p / 2
    hole = np.maximum(np.abs(Zs - zc) - LOUVER_H / 2, np.abs(px) - (p - LOUVER_RIB) / 2)
    return np.where(whole, hole, np.maximum(hole, 1.0))


DROP_ROW = 1.05       # row height, fraction of the pitch


def pat_drops(pot, th, Z, r, s):
    """Staggered teardrops with SLOPE pointed tops, flaring outward like
    voronoi_taper: the radius grows from the soil side out and the centre
    moves down by dR / cos(SLOPE), so the pointed roof stays put."""
    p = pot.circ / pot.count(DROP_NC_REF)
    rowh = p * DROP_ROW
    S, ZZ = unroll(pot, th, Z)
    strut = pot.strut_in + (pot.strut_out - pot.strut_in) * s
    R0 = (p - pot.strut_in) / 2
    R = (p - strut) / 2
    sb, cb = np.sin(SLOPE), np.cos(SLOPE)
    j0 = np.floor(ZZ / rowh)
    hole = np.full(S.shape, 1e9, np.float32)
    for dj in (-1, 0, 1):
        j = j0 + dj
        px = np.mod(S - np.mod(j, 2) * p / 2, p) - p / 2
        pz = ZZ - (j + 0.5) * rowh + (R - R0) / cb
        circle = np.hypot(px, pz) - R
        cap = np.maximum(np.abs(px) * sb + pz * cb - R, R * cb - pz)
        hole = np.minimum(hole, np.minimum(circle, cap))
    return hole


SPIRAL_ANGLE = np.deg2rad(60)
SPIRAL_W = 2.2        # slot width (mm)
SPIRAL_LEN = 16.0     # slot height (mm)
SPIRAL_GAP = 3.0      # solid between slots along one helix (mm)


def pat_spiral(pot, th, Z, r, s):
    """Slots along a many-start helix at 60 degrees, staggered between
    neighbouring helices. Slot ends are cut level (a short flat bridge)."""
    n = 2 * pot.count(SPIRAL_NC_REF / 2)
    p = pot.circ / n
    S, ZZ = unroll(pot, th, Z)
    u = S - ZZ / np.tan(SPIRAL_ANGLE)
    k = np.floor(u / p)
    du = (u - (k + 0.5) * p) * np.sin(SPIRAL_ANGLE)
    period = SPIRAL_LEN + SPIRAL_GAP
    zz = np.mod(ZZ + np.mod(k, 2) * period / 2, period) - period / 2
    return np.maximum(np.abs(du) - SPIRAL_W / 2, np.abs(zz) - SPIRAL_LEN / 2)


def pat_isogrid(pot, th, Z, r, s):
    """Triangles: level struts plus two sets at +-60 degrees. Upward
    triangles have pointed tops; downward ones have a short flat bridge."""
    a = pot.circ / pot.count(ISOGRID_NC_REF)      # triangle side
    h = a * np.sqrt(3) / 2
    S, ZZ = unroll(pot, th, Z)
    c = 1 / np.sqrt(3)                            # cot 60
    d0 = stripes(ZZ, h)
    d1 = stripes(S + ZZ * c, a) * np.sqrt(3) / 2
    d2 = stripes(S - ZZ * c, a) * np.sqrt(3) / 2
    return STRUT / 2 - np.minimum.reduce([d0, d1, d2])


CHEVRON_ARM = 8.0     # arm length (mm)
CHEVRON_W = 2.0       # slot width (mm)
CHEVRON_ROW = 6.8     # row spacing (mm)


def pat_chevrons(pot, th, Z, r, s):
    """Stacked arrowheads: two SLOPE arms meeting at the top, so every roof
    slopes or is a point."""
    p = pot.circ / pot.count(CHEVRON_NC_REF)
    S, ZZ = unroll(pot, th, Z)
    px = np.abs(np.mod(S, p) - p / 2)             # the chevron is symmetric
    bx, bz = CHEVRON_ARM * np.cos(SLOPE), -CHEVRON_ARM * np.sin(SLOPE)
    j0 = np.floor(ZZ / CHEVRON_ROW)
    hole = np.full(S.shape, 1e9, np.float32)
    for dj in (0, 1, 2):
        pz = ZZ - (j0 + dj) * CHEVRON_ROW          # apex of that row at pz = 0
        t = np.clip((px * bx + pz * bz) / CHEVRON_ARM ** 2, 0, 1)
        hole = np.minimum(hole, np.hypot(px - t * bx, pz - t * bz))
    return hole - CHEVRON_W / 2


WEAVE_W = 2.4         # ribbon width (mm)


def pat_weave(pot, th, Z, r, s):
    """Woven strips: the lattice's two strip sets, each a ribbon 0.6 wall
    thick that moves in and out through the wall so it passes over one
    crossing and under the next. The ribbons overlap at each crossing so
    they fuse into one piece."""
    p = pot.circ / pot.count(WEAVE_NC_REF)
    S, ZZ = unroll(pot, th, Z)
    c = 1 / np.tan(SLOPE)
    du = stripes(S + ZZ * c, p) * np.sin(SLOPE)
    dv = stripes(S - ZZ * c, p) * np.sin(SLOPE)
    dz = p / (2 * c)                              # height between crossings along a ribbon
    wave = 0.2 * pot.wall * np.cos(np.pi * ZZ / dz)
    rc = pot.r_out(Z) - pot.wall / 2
    t = 0.3 * pot.wall
    a = np.maximum(du - WEAVE_W / 2, np.abs(r - rc - wave) - t)
    b = np.maximum(dv - WEAVE_W / 2, np.abs(r - rc + wave) - t)
    return np.minimum(a, b)


BAND_H = 22.0         # height of the air-pruning slot band (mm)
BAND_SEP = 1.5        # solid ring between the two bands (mm)


def pat_bands(pot, th, Z, r, s):
    """One row of air-pruning slots just above the base, voronoi_taper above
    it, and a solid ring between them."""
    lo = pot.base + PATTERN_GAP
    split = lo + BAND_H
    slots = pat_slots(pot, th, Z - (lo + BAND_H / 2 - 11.0), r, s)   # centre the slot row in its band
    hole = np.where(Z < split, slots, pat_voronoi_taper(pot, th, Z, r, s))
    return np.maximum(hole, BAND_SEP / 2 - np.abs(Z - split))


PATTERNS = {
    "voronoi_taper": Pattern("2d", pat_voronoi_taper, "organic cells flaring outward like funnels (default)"),
    "voronoi": Pattern("2d", pat_voronoi, "organic cells, straight holes"),
    "hex": Pattern("2d", pat_hex, "warped honeycomb, pointy-top cells"),
    "hex_taper": Pattern("2d", pat_hex_taper, "honeycomb flaring outward like voronoi_taper"),
    "drops": Pattern("2d", pat_drops, "staggered teardrops flaring outward"),
    "lattice": Pattern("2d", pat_lattice, "diamond trellis of crossing helical strips"),
    "isogrid": Pattern("2d", pat_isogrid, "triangle grid"),
    "louvers": Pattern("2d", pat_louvers, "gills sloping down and out, no line of sight"),
    "slots": Pattern("2d", pat_slots, "narrow wavy vertical slots, air-pruning style"),
    "spiral": Pattern("2d", pat_spiral, "slots on a many-start helix"),
    "chevrons": Pattern("2d", pat_chevrons, "stacked arrowhead slots"),
    "bands": Pattern("2d", pat_bands, "air-pruning slots at the base, voronoi_taper above"),
    "gyroid": Pattern("3d", pat_gyroid, "3D lattice, no straight line of sight"),
    "diamond": Pattern("3d", pat_diamond, "3D lattice, straighter 45-degree channels"),
    "weave": Pattern("3d", pat_weave, "woven strips passing over and under"),
}
