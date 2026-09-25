"""
Wall patterns.

Each pattern is `fn(pot, P, th, Z, r, s)` and returns a field for points in
the wall, where s goes from 0 on the soil side to 1 outside. P holds the
settings of every pattern (config/patterns.yaml, see `load_params`); a pattern
reads its own block, e.g. P.lattice.strut. Two kinds:
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
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, NamedTuple

import numpy as np

from .geometry import PATTERN_GAP
from .sdf import gyroid

PARAMS_FILE = Path(__file__).parent / "config" / "patterns.yaml"
SLOPE = np.deg2rad(55)  # sloped roofs, from horizontal in the unrolled plane


class Pattern(NamedTuple):
    kind: str                     # "3d" or "2d"
    fn: Callable
    description: str
    uses: tuple = ()              # other patterns whose settings it also reads


def load_params(params=None):
    """Pattern settings as nested attributes (P.lattice.strut). `params` is the
    `patterns` node of a composed config (DictConfig or dict); None loads the
    packaged defaults from config/patterns.yaml."""
    from omegaconf import DictConfig, OmegaConf
    if isinstance(params, SimpleNamespace):
        return params
    if params is None:
        params = OmegaConf.load(PARAMS_FILE).patterns
    if isinstance(params, DictConfig):
        params = OmegaConf.to_container(params, resolve=True)
    return SimpleNamespace(**{k: SimpleNamespace(**v) for k, v in params.items()})


def taper(c, s):
    """Strut width at depth s (0 soil side, 1 outside) of a tapered pattern."""
    return c.strut_in + (c.strut_out - c.strut_in) * s


def taper_shift(c, s, nz, extra=0.0):
    """How far down a tapered pattern is moved at depth s. nz is the vertical
    part of the roof edge normal; `extra` is added to every strut.

    center = 0: each roof edge stays where it is from the soil side out (it
    recedes by (strut_in - strut) / 2 along its normal), so holes only grow
    sideways and downward. center = 1: the outside is unchanged but the soil
    side moves down by the full recession, so each hole grows evenly all round
    and its roof rises toward the outside."""
    strut = taper(c, s)
    keep = (c.strut_in - strut - extra) / 2 / nz
    return keep + c.center * (strut - c.strut_out) / 2 / nz


# ---------------------------------------------------------------- 3D lattices
def pat_gyroid(pot, P, th, Z, r, s):
    c = P.gyroid
    return gyroid(th, Z, r, c.level_in + (c.level_out - c.level_in) * s, pot.count(c.cells), c.cell)


def pat_diamond(pot, P, th, Z, r, s):
    """Schwarz diamond TPMS: straighter 45-degree channels than the gyroid."""
    c = P.diamond
    n = pot.count(c.cells)
    x = th * n + 0.9 * np.sin(4 * th + 2 * np.pi * Z / 60)
    y = 2 * np.pi * Z / c.cell + 0.7 * np.sin(3 * th - 2 * np.pi * Z / 45)
    z = 2 * np.pi * r / c.cell
    d = (np.sin(x) * np.sin(y) * np.sin(z) + np.sin(x) * np.cos(y) * np.cos(z)
         + np.cos(x) * np.sin(y) * np.cos(z) + np.cos(x) * np.cos(y) * np.sin(z))
    level = c.level_in + (c.level_out - c.level_in) * s
    return (np.abs(d) - level) * c.cell / (2 * np.pi * 1.2)


# ------------------------------------------------------ 2D perforations
def unroll(pot, th, Z):
    """Cylinder -> flat (s, z) in mm, s measured at radius pot.r0."""
    return th * pot.r0, Z


@lru_cache(maxsize=8)
def _vor_jitter(nr, nc, jitter, seed):
    # periodic jittered seeds (fixed seed -> repeatable)
    return np.random.default_rng(seed).uniform(-jitter, jitter, (nr, nc, 2))


def voronoi_edge(pot, c, S, ZZ):
    """Approximate distance (mm) to the nearest Voronoi cell edge, (F2 - F1) / 2,
    for the cells, rows and jitter of settings block c."""
    nc = pot.count(c.cells)
    nr = int(pot.height / c.row_h) + 3
    w = pot.circ / nc
    jitter = _vor_jitter(nr, nc, c.jitter, c.seed)
    ci = np.floor(S / w).astype(int)
    cj = np.floor(ZZ / c.row_h).astype(int)
    f1 = np.full(S.shape, 1e9, np.float32)
    f2 = np.full(S.shape, 1e9, np.float32)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            i = ci + di
            j = np.clip(cj + dj + 1, 0, nr - 1)
            jit = jitter[j, i % nc]
            px = (i + 0.5 + jit[..., 0]) * w
            pz = (j - 1 + 0.5 + jit[..., 1]) * c.row_h
            d = np.hypot(S - px, ZZ - pz)
            f2 = np.where(d < f1, f1, np.minimum(f2, d))
            f1 = np.minimum(f1, d)
    return (f2 - f1) / 2


def pat_voronoi(pot, P, th, Z, r, s):
    """Organic cells: periodic jittered Voronoi, holes = shrunken cells."""
    c = P.voronoi
    S, ZZ = unroll(pot, th, Z)
    S = S + c.warp * np.sin(2 * np.pi * Z / 47)
    return c.strut / 2 - voronoi_edge(pot, c, S, ZZ)


def pat_voronoi_taper(pot, P, th, Z, r, s):
    """Voronoi cells shaped like funnels: small on the soil side, wide outside.
    Struts thin from strut_in to strut_out through the wall. The pattern is
    shifted down by the same amount each edge recedes, so every hole grows
    sideways and downward while its roof stays level (a plain short bridge,
    never a sagging sloped ceiling). `center` trades that for a centred hole
    (see taper_shift)."""
    c = P.voronoi_taper
    S, ZZ = unroll(pot, th, Z)
    S = S + c.warp * np.sin(2 * np.pi * Z / 47)
    strut = taper(c, s)
    return strut / 2 - voronoi_edge(pot, c, S, ZZ + taper_shift(c, s, 1.0))


def hex_edge(pot, c, th, Z, shift=0.0):
    """Hex metric (mm from the cell centre) of the warped honeycomb of
    settings block c, and the apothem. The pattern is moved down by `shift`."""
    a = pot.circ / pot.count(c.cells)    # column pitch (flat-to-flat)
    rc = a / np.sqrt(3)                  # circumradius
    b = 3 * rc
    S, ZZ = unroll(pot, th, Z)
    S = S + c.warp * np.sin(2 * np.pi * Z / 38 + 3 * th)
    ZZ = ZZ + c.warp_z * np.sin(5 * th) + shift
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


def pat_hex(pot, P, th, Z, r, s):
    """Warped honeycomb, vertex pointing up (self-supporting roofs)."""
    c = P.hex
    hexd, apothem = hex_edge(pot, c, th, Z)
    return hexd - (apothem - c.strut / 2)


def pat_hex_taper(pot, P, th, Z, r, s):
    """The honeycomb with the voronoi_taper funnel: struts thin from strut_in
    to strut_out through the wall. The roof edges have a vertical normal
    component of sqrt(3)/2, so the pattern moves down by recession / (sqrt(3)/2)
    to keep every roof where it is. strut_extra makes up for the warp, which
    shears the cells and thins some struts."""
    c = P.hex_taper
    strut = taper(c, s) + c.strut_extra
    shift = taper_shift(c, s, np.sqrt(3) / 2, c.strut_extra)
    hexd, apothem = hex_edge(pot, c, th, Z, shift)
    return hexd - (apothem - strut / 2)


def pat_slots(pot, P, th, Z, r, s):
    """Air-pruning style: narrow wavy vertical slots with pointed ends."""
    c = P.slots
    pitch = pot.circ / pot.count(c.cells)
    rowh = c.row_h
    S, ZZ = unroll(pot, th, Z)
    S = S + c.warp * np.sin(2 * np.pi * Z / 30 + 2 * th)
    row = np.floor(ZZ / rowh)
    Ss = S + (np.mod(row, 2)) * pitch / 2
    px = np.mod(Ss, pitch) - pitch / 2
    pz = np.mod(ZZ, rowh) - rowh / 2
    halfw, halfl = c.width / 2, c.length / 2
    # pointed-end slot: rectangle with 45-degree tips (diamond caps)
    tip = (np.abs(px) + np.abs(pz) - halfl) / np.sqrt(2)
    return np.maximum(np.abs(px) - halfw, tip)


def stripes(u, p):
    """Distance from u to the nearest multiple of p."""
    return np.abs(np.mod(u + p / 2, p) - p / 2)


def lattice_dist(pot, c, th, Z, shift=0.0):
    """Normal distance (mm) to the nearest strip of the diamond trellis of
    settings block c, moved down by `shift`."""
    p = pot.circ / pot.count(c.cells)
    S, ZZ = unroll(pot, th, Z)
    ZZ = ZZ + shift
    cot = 1 / np.tan(SLOPE)
    du = stripes(S + ZZ * cot, p) * np.sin(SLOPE)   # normal distance to each strip
    dv = stripes(S - ZZ * cot, p) * np.sin(SLOPE)
    return np.minimum(du, dv)


def pat_lattice(pot, P, th, Z, r, s):
    """Diamond trellis: two sets of helical strips crossing at +-SLOPE.
    The holes are diamonds with pointed tops, so there are no bridges."""
    c = P.lattice
    return c.strut / 2 - lattice_dist(pot, c, th, Z)


def pat_lattice_taper(pot, P, th, Z, r, s):
    """The trellis with the voronoi_taper funnel: strips thin from strut_in to
    strut_out through the wall. Every strip edge has a vertical normal
    component of cos(SLOPE), so the pattern moves down by recession / cos(SLOPE):
    the lower edge of each strip (the hole roof, up to the pointed top of each
    diamond) stays where it is, and the holes grow sideways and downward."""
    c = P.lattice_taper
    strut = taper(c, s)
    return strut / 2 - lattice_dist(pot, c, th, Z, taper_shift(c, s, np.cos(SLOPE)))


def pat_louvers(pot, P, th, Z, r, s):
    """Gills: brick-staggered slits that run down and outward through the wall
    like shutter blades. The drop is more than the slit height, so there is
    no line of sight: soil stays in, rain runs off, air still flows. Only rows
    that fit whole inside the pattern band are cut."""
    c = P.louvers
    p = pot.circ / pot.count(c.cells)
    S, ZZ = unroll(pot, th, Z)
    drop = c.drop * pot.wall
    Zs = ZZ + drop * s                         # the height this point has on the soil side
    k = np.floor(Zs / c.pitch)
    zc = (k + 0.5) * c.pitch
    lo, hi = pot.base + PATTERN_GAP + 0.5, pot.height - pot.rim - 0.5
    whole = (zc - c.slit_h / 2 - drop >= lo) & (zc + c.slit_h / 2 <= hi)
    px = np.mod(S + np.mod(k, 2) * p / 2, p) - p / 2
    hole = np.maximum(np.abs(Zs - zc) - c.slit_h / 2, np.abs(px) - (p - c.rib) / 2)
    return np.where(whole, hole, np.maximum(hole, 1.0))


def pat_drops(pot, P, th, Z, r, s):
    """Staggered teardrops with SLOPE pointed tops, flaring outward like
    voronoi_taper: the radius grows from the soil side out and the centre
    moves down by dR / cos(SLOPE), so the pointed roof stays put (unless
    `center` is set, see taper_shift)."""
    c = P.drops
    p = pot.circ / pot.count(c.cells)
    rowh = p * c.row
    S, ZZ = unroll(pot, th, Z)
    strut = taper(c, s)
    R = (p - strut) / 2
    shift = taper_shift(c, s, np.cos(SLOPE))
    sb, cb = np.sin(SLOPE), np.cos(SLOPE)
    j0 = np.floor(ZZ / rowh)
    hole = np.full(S.shape, 1e9, np.float32)
    for dj in (-1, 0, 1):
        j = j0 + dj
        px = np.mod(S - np.mod(j, 2) * p / 2, p) - p / 2
        pz = ZZ - (j + 0.5) * rowh + shift
        circle = np.hypot(px, pz) - R
        cap = np.maximum(np.abs(px) * sb + pz * cb - R, R * cb - pz)
        hole = np.minimum(hole, np.minimum(circle, cap))
    return hole


def pat_spiral(pot, P, th, Z, r, s):
    """Slots along a many-start helix, staggered between neighbouring
    helices. Slot ends are cut level (a short flat bridge). The helix count
    is even so the stagger closes around the pot."""
    c = P.spiral
    angle = np.deg2rad(c.angle)
    n = 2 * pot.count(c.cells / 2)
    p = pot.circ / n
    S, ZZ = unroll(pot, th, Z)
    u = S - ZZ / np.tan(angle)
    k = np.floor(u / p)
    du = (u - (k + 0.5) * p) * np.sin(angle)
    period = c.length + c.gap
    zz = np.mod(ZZ + np.mod(k, 2) * period / 2, period) - period / 2
    return np.maximum(np.abs(du) - c.width / 2, np.abs(zz) - c.length / 2)


def pat_isogrid(pot, P, th, Z, r, s):
    """Triangles: level struts plus two sets at +-60 degrees. Upward
    triangles have pointed tops; downward ones have a short flat bridge."""
    c = P.isogrid
    a = pot.circ / pot.count(c.cells)             # triangle side
    h = a * np.sqrt(3) / 2
    S, ZZ = unroll(pot, th, Z)
    cot = 1 / np.sqrt(3)                          # cot 60
    d0 = stripes(ZZ, h)
    d1 = stripes(S + ZZ * cot, a) * np.sqrt(3) / 2
    d2 = stripes(S - ZZ * cot, a) * np.sqrt(3) / 2
    return c.strut / 2 - np.minimum.reduce([d0, d1, d2])


def pat_chevrons(pot, P, th, Z, r, s):
    """Stacked arrowheads: two SLOPE arms meeting at the top, so every roof
    slopes or is a point."""
    c = P.chevrons
    p = pot.circ / pot.count(c.cells)
    S, ZZ = unroll(pot, th, Z)
    px = np.abs(np.mod(S, p) - p / 2)             # the chevron is symmetric
    bx, bz = c.arm * np.cos(SLOPE), -c.arm * np.sin(SLOPE)
    j0 = np.floor(ZZ / c.row_h)
    hole = np.full(S.shape, 1e9, np.float32)
    for dj in (0, 1, 2):
        pz = ZZ - (j0 + dj) * c.row_h             # apex of that row at pz = 0
        t = np.clip((px * bx + pz * bz) / c.arm ** 2, 0, 1)
        hole = np.minimum(hole, np.hypot(px - t * bx, pz - t * bz))
    return hole - c.width / 2


def pat_weave(pot, P, th, Z, r, s):
    """Woven strips: the lattice's two strip sets, each a ribbon 0.6 wall
    thick that moves in and out through the wall so it passes over one
    crossing and under the next. The ribbons overlap at each crossing so
    they fuse into one piece."""
    c = P.weave
    p = pot.circ / pot.count(c.cells)
    S, ZZ = unroll(pot, th, Z)
    cot = 1 / np.tan(SLOPE)
    du = stripes(S + ZZ * cot, p) * np.sin(SLOPE)
    dv = stripes(S - ZZ * cot, p) * np.sin(SLOPE)
    dz = p / (2 * cot)                            # height between crossings along a ribbon
    wave = 0.2 * pot.wall * np.cos(np.pi * ZZ / dz)
    rc = pot.r_out(Z) - pot.wall / 2
    t = 0.3 * pot.wall
    a = np.maximum(du - c.width / 2, np.abs(r - rc - wave) - t)
    b = np.maximum(dv - c.width / 2, np.abs(r - rc + wave) - t)
    return np.minimum(a, b)


def pat_bands(pot, P, th, Z, r, s):
    """One row of air-pruning slots just above the base, voronoi_taper above
    it, and a solid ring between them."""
    c = P.bands
    lo = pot.base + PATTERN_GAP
    split = lo + c.band_h
    centre = lo + c.band_h / 2 - P.slots.row_h / 2   # centre the slot row in its band
    slots = pat_slots(pot, P, th, Z - centre, r, s)
    hole = np.where(Z < split, slots, pat_voronoi_taper(pot, P, th, Z, r, s))
    return np.maximum(hole, c.sep / 2 - np.abs(Z - split))


PATTERNS = {
    "voronoi_taper": Pattern("2d", pat_voronoi_taper, "organic cells flaring outward like funnels (default)"),
    "voronoi": Pattern("2d", pat_voronoi, "organic cells, straight holes"),
    "hex": Pattern("2d", pat_hex, "warped honeycomb, pointy-top cells"),
    "hex_taper": Pattern("2d", pat_hex_taper, "honeycomb flaring outward like voronoi_taper"),
    "drops": Pattern("2d", pat_drops, "staggered teardrops flaring outward"),
    "lattice": Pattern("2d", pat_lattice, "diamond trellis of crossing helical strips"),
    "lattice_taper": Pattern("2d", pat_lattice_taper, "diamond trellis flaring outward like voronoi_taper"),
    "isogrid": Pattern("2d", pat_isogrid, "triangle grid"),
    "louvers": Pattern("2d", pat_louvers, "gills sloping down and out, no line of sight"),
    "slots": Pattern("2d", pat_slots, "narrow wavy vertical slots, air-pruning style"),
    "spiral": Pattern("2d", pat_spiral, "slots on a many-start helix"),
    "chevrons": Pattern("2d", pat_chevrons, "stacked arrowhead slots"),
    "bands": Pattern("2d", pat_bands, "air-pruning slots at the base, voronoi_taper above",
                     ("slots", "voronoi_taper")),
    "gyroid": Pattern("3d", pat_gyroid, "3D lattice, no straight line of sight"),
    "diamond": Pattern("3d", pat_diamond, "3D lattice, straighter 45-degree channels"),
    "weave": Pattern("3d", pat_weave, "woven strips passing over and under"),
}
