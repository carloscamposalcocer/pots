"""
Wall patterns.

Each pattern is `fn(pot, P, th, Z, r, s)` and returns a field for points in
the wall, where s goes from 0 on the soil side to 1 outside. P holds the
settings of every pattern (config/patterns.yaml, see `load_params`); a pattern
reads its own block, e.g. P.lattice.strut_in. Two kinds:
  - "3d" (weave): the field is the wall MATERIAL (negative = solid).
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

PARAMS_FILE = Path(__file__).parent / "config" / "patterns.yaml"
SLOPE = np.deg2rad(55)  # sloped roofs, from horizontal in the unrolled plane


class Pattern(NamedTuple):
    kind: str                     # "3d" or "2d"
    fn: Callable
    description: str
    uses: tuple = ()              # other patterns whose settings it also reads
    cutters: Callable = None      # exact hole outlines for solid.py, (pot, c, depths) -> outlines or None


def load_params(params=None):
    """Pattern settings as nested attributes (P.lattice.strut_in). `params` is the
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


def _vor_reach(D, strut, U):
    """Distance from a seed along the unit directions U (..., k, 2) to the
    edge of its hole: for each neighbour at offset d (D: (..., 1, j, 2)) the
    point p = t u where |p - d| - |p| = strut is at
    t = (|d|^2 - strut^2) / (2 (u . d + strut)); the hole ends at the nearest.
    A neighbour closer than the strut closes the hole (t = 0: F2 - F1 is
    at most |d| everywhere, and largest at the seed)."""
    num = (D * D).sum(-1) - strut ** 2
    ud = (U[..., :, None, :] * D).sum(-1) + strut
    t = np.where(ud > 1e-12, num / (2 * np.where(ud > 1e-12, ud, 1.0)), np.inf)
    return np.maximum(np.where(num > 0, t, 0.0).min(-1), 0.0)


def _vor_corners(D, strut, fine):
    """For each cell (offsets D: (n, j, 2)) at one strut width: its hole's
    corners, where the nearest neighbour changes, as a list of
    [(neighbour before, neighbour after, angle), ...] sorted by angle."""
    phi = np.linspace(0, 2 * np.pi, fine, endpoint=False)
    U = np.c_[np.cos(phi), np.sin(phi)]
    ud = (U[None, :, None, :] * D[:, None]).sum(-1) + strut
    num = (D * D).sum(-1)[:, None] - strut ** 2
    T = np.where(ud > 1e-12, num / (2 * np.where(ud > 1e-12, ud, 1.0)), np.inf)
    T = np.where(num > 0, T, 0.0)
    near = T.argmin(-1)                                          # (n, fine) nearest edge's neighbour
    cell, k = np.nonzero(near != np.roll(near, -1, 1))
    j1, j2 = near[cell, k], near[cell, (k + 1) % fine]
    a, b = phi[k], phi[k] + 2 * np.pi / fine
    for _ in range(30):                                          # bisect: where both neighbours tie
        m = (a + b) / 2
        u = np.c_[np.cos(m), np.sin(m)][:, None]
        first = _vor_reach(D[cell, j1][:, None, None], strut, u)[:, 0] < \
            _vor_reach(D[cell, j2][:, None, None], strut, u)[:, 0]
        a, b = np.where(first, m, a), np.where(first, b, m)
    out = [[] for _ in range(len(D))]
    for ci, x, y, ang in zip(cell, j1, j2, (a + b) / 2):
        out[ci].append((x, y, ang))
    return out


def _vor_sides(pot, per, step):
    """Sample angles (L, m) for one cell from its corners at each depth
    (`per`: [depth] -> [(j1, j2, angle)]), or None if the depths don't fit
    one order of sides. Every neighbour that borders the hole at some depth
    gets a side, in the order of the depth with the most sides; where it
    doesn't border it yet, its side is the corner between its neighbours,
    which is where it grows out of. Each side gets ceil(widest / step)
    points from its first corner on."""
    seqs = [[j2 for _, j2, _ in p] for p in per]                # the side after each corner
    master = max(seqs, key=len)
    if len(master) < 3 or len(set(master)) != len(master):
        return None
    pos = {j: i for i, j in enumerate(master)}
    for q in seqs:                                               # each must follow master's cyclic order
        if not q or any(j not in pos for j in q):
            return None
        idx = [pos[j] for j in q]
        k = idx.index(min(idx))
        if idx[k:] + idx[:k] != sorted(idx):
            return None
    n = len(master)
    starts = np.empty((len(per), n))
    spans = np.empty((len(per), n))
    for d, p in enumerate(per):
        start = {j2: a for _, j2, a in p}                        # side j2 starts at its corner
        end = {j1: a for j1, _, a in p}
        present = [j for j in master if j in start]
        for i, j in enumerate(master):
            if j in start:
                starts[d, i] = start[j]
                spans[d, i] = np.mod(end[j] - start[j], 2 * np.pi)
            else:                                                # the corner after the previous side there
                prev = next(master[(i - k) % n] for k in range(1, n + 1) if master[(i - k) % n] in start)
                starts[d, i] = end[prev]
                spans[d, i] = 0.0
        if len(present) == 1:
            return None
    pieces = np.maximum(1, np.ceil(spans.max(0) / step)).astype(int)
    return np.concatenate([starts[:, [i]] + spans[:, [i]] * np.arange(k) / k for i, k in enumerate(pieces)], 1)


def voronoi_cutters(pot, c, depths, step=np.deg2rad(24), fine=360, near=14):
    """The voronoi holes for solid.py: a list of (n, len(depths), m, 2)
    outlines in the unrolled (S, Z) plane, grouped by point count m. In
    voronoi_edge's warped coordinates, a hole is where (F2 - F1) exceeds the
    strut: from its seed, the distance to the edge at each angle comes from
    _vor_reach, and the sides are hyperbola arcs, one per neighbour. The
    outline has the exact corners at each depth and points along each side
    (_vor_sides); a cell whose depths don't fit that is sampled at fixed
    angles instead: the corners at the soil face, mid-wall and outer face
    and angles at most `step` apart. The points are then mapped back
    through the warp. Cells
    whose hole is closed at every depth are skipped; where a hole is closed
    at some depths only (a blind pocket), its outline there shrinks to the
    seed, where the field's hole closes."""
    nc, rh = pot.count(c.cells), c.row_h
    nr = int(pot.height / rh) + 3
    w = pot.circ / nc
    jit = _vor_jitter(nr, nc, c.jitter, c.seed)
    lo, hi = pot.base + PATTERN_GAP, pot.height - pot.rim
    depths = np.asarray(depths)
    strut = np.array([taper(c, s) for s in depths])
    shift = np.array([taper_shift(c, s, 1.0) for s in depths])
    # seeds as in voronoi_edge: row jr at z = (jr - 0.5 + jitter) rh
    def seed(jr, i):
        return np.stack([(i + 0.5 + jit[jr, i % nc, 0]) * w, (jr - 0.5 + jit[jr, i % nc, 1]) * rh], -1)
    margin = 2 * rh + np.abs(shift).max()
    rows = [jr for jr in range(nr) if lo - margin < (jr - 0.5) * rh < hi + margin]
    JR, I = [x.ravel() for x in np.meshgrid(rows, np.arange(nc), indexing="ij")]
    own = seed(JR, I)                                            # (n, 2)
    offs = [(dj, di) for dj in range(-2, 3) for di in range(-3, 4) if (dj, di) != (0, 0)]
    # neighbours in rows that don't exist are far away (the field doesn't have them either)
    D = np.stack([np.where(((JR + dj >= 0) & (JR + dj < nr))[:, None],
                           seed(np.clip(JR + dj, 0, nr - 1), I + di), 1e6) for dj, di in offs], 1)
    D = D - own[:, None]                                         # (n, j, 2)
    # only the nearest seeds can border a hole (a farther one's edge lies beyond theirs)
    D = np.take_along_axis(D, np.argsort((D * D).sum(-1), 1)[:, :near, None], 1)
    L = len(depths)
    corners = [_vor_corners(D, strut[d], fine) for d in range(L)]     # [depth][cell] -> [(j1, j2, angle)]

    groups = {}                                                  # point count -> [(cell, angles (L, m))]
    for ci in range(len(own)):
        ang = _vor_sides(pot, [corners[d][ci] for d in range(L)], step)
        if ang is None:
            base = [a for d in sorted({0, L // 2, L - 1}) for _, _, a in corners[d][ci]]
            base = np.unique(np.round(np.mod(base, 2 * np.pi), 6))
            if len(base) < 3:
                continue
            gaps = np.diff(np.r_[base, base[0] + 2 * np.pi])
            fill = [a0 + g * np.arange(1, int(np.ceil(g / step))) / np.ceil(g / step) for a0, g in zip(base, gaps)]
            ang = np.sort(np.mod(np.r_[base, np.concatenate(fill)], 2 * np.pi))
            ang = np.broadcast_to(ang, (L, len(ang)))
        groups.setdefault(ang.shape[1], []).append((ci, ang))

    out = []
    for m, cells in groups.items():
        idx = np.array([ci for ci, _ in cells])
        ang = np.stack([a for _, a in cells])                    # (g, L, m)
        U = np.stack([np.cos(ang), np.sin(ang)], -1)             # (g, L, m, 2)
        t = np.stack([_vor_reach(D[idx][:, None], strut[d], U[:, d]) for d in range(L)], 1)   # (g, L, m)
        # the first and last seed rows have no neighbours beyond them; their
        # holes run off there, outside the pattern band (the clip cuts them off)
        t = np.minimum(t, 3 * max(w, rh))
        # closed at a depth: a tiny outline at the seed (0.01 mm, far below what prints)
        some = (t.min(-1) > 0.01).any(1)
        t, U, idx = np.maximum(t[some], 0.01), U[some], idx[some]
        Sw = own[idx, None, None, 0] + t * U[..., 0]
        Zw = own[idx, None, None, 1] + t * U[..., 1]
        Z = Zw - shift[None, :, None]                            # voronoi_edge gets Z + shift ...
        S = Sw - c.warp * np.sin(2 * np.pi * Z / 47)             # ... and S + warp sin(2 pi Z / 47)
        keep = (Z.max((1, 2)) > lo) & (Z.min((1, 2)) < hi)
        out.append(np.stack([S, Z], -1)[keep])
    return out


def pat_voronoi(pot, P, th, Z, r, s):
    """Voronoi cells shaped like funnels: small on the soil side, wide outside.
    Struts thin from strut_in to strut_out through the wall. The pattern is
    shifted down by the same amount each edge recedes, so every hole grows
    sideways and downward while its roof stays level (a plain short bridge,
    never a sagging sloped ceiling). `center` trades that for a centred hole
    (see taper_shift)."""
    c = P.voronoi
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


def hex_cutters(pot, c, depths, per_edge=2):
    """The hex holes for solid.py: (n, len(depths), 6 per_edge, 2) outlines in
    the unrolled (S, Z) plane. In hex_edge's warped coordinates a hole is a
    regular pointy-top hexagon of apothem a/2 - strut/2 around each cell
    centre; `per_edge` points along each side are mapped back through the
    warp (by fixed-point iteration), which bends the sides like the field
    does. None if a hole closes up."""
    n = pot.count(c.cells)
    a = pot.circ / n
    b = 3 * a / np.sqrt(3)                       # row pitch of one sub-lattice (3 x circumradius)
    lo, hi = pot.base + PATTERN_GAP, pot.height - pot.rim
    strut = np.array([taper(c, s) + c.strut_extra for s in depths])
    shift = np.array([taper_shift(c, s, np.sqrt(3) / 2, c.strut_extra) for s in depths])
    R = a / 2 - strut / 2                        # hex metric of the hole edge
    if (R < 0.02).any():
        return None
    # unit hexagon (metric 1): top vertex first, clockwise, per_edge points per side
    v = np.array([(0, 2), (1, 1), (1, -1), (0, -2), (-1, -1), (-1, 1)]) * [1, 1 / np.sqrt(3)]
    t = np.arange(per_edge)[:, None] / per_edge
    hexagon = np.concatenate([v[i] * (1 - t) + v[(i + 1) % 6] * t for i in range(6)])
    # cell centres in warped coordinates: two sub-lattices, (ox + a/2 + i a, oz + b/2 + j b)
    margin = b + np.abs(shift).max() + c.warp_z + 2
    j = np.arange(np.floor((lo - margin) / b) - 1, np.ceil((hi + margin) / b) + 1)
    J, I = [x.ravel() for x in np.meshgrid(j, np.arange(n), indexing="ij")]
    cs = np.concatenate([a / 2 + I * a, a + I * a])
    cz = np.concatenate([b / 2 + J * b, b + J * b])
    keep = (cz > lo - margin) & (cz < hi + margin)
    cs, cz = cs[keep], cz[keep]
    # warped outline points (n, L, m) ...
    Sw = cs[:, None, None] + R[None, :, None] * hexagon[None, None, :, 0]
    Zw = cz[:, None, None] + R[None, :, None] * hexagon[None, None, :, 1]
    # ... mapped back: S' = S + warp sin(2 pi Z / 38 + 3 th), Z' = Z + warp_z sin(5 th) + shift
    S, Z = Sw.copy(), Zw - shift[None, :, None]
    for _ in range(60):
        th = S / pot.r0
        Z = Zw - shift[None, :, None] - c.warp_z * np.sin(5 * th)
        S = Sw - c.warp * np.sin(2 * np.pi * Z / 38 + 3 * th)
    th = S / pot.r0
    err = np.abs(S + c.warp * np.sin(2 * np.pi * Z / 38 + 3 * th) - Sw).max()
    if err > 1e-6:
        raise RuntimeError(f"hex warp inversion did not converge ({err:.2g} mm)")
    return np.stack([S, Z], -1)


def pat_hex(pot, P, th, Z, r, s):
    """Warped honeycomb, vertex pointing up, with the voronoi funnel: struts
    thin from strut_in to strut_out through the wall. The roof edges have a
    vertical normal component of sqrt(3)/2, so the pattern moves down by
    recession / (sqrt(3)/2) to keep every roof where it is. strut_extra makes up for the warp, which
    shears the cells and thins some struts."""
    c = P.hex
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
    """Diamond trellis: two sets of helical strips crossing at +-SLOPE, so the
    holes are diamonds with pointed tops and there are no bridges. It has the
    voronoi funnel: strips thin from strut_in to strut_out through the wall.
    Every strip edge has a vertical normal component of cos(SLOPE), so the
    pattern moves down by recession / cos(SLOPE):
    the lower edge of each strip (the hole roof, up to the pointed top of each
    diamond) stays where it is, and the holes grow sideways and downward."""
    c = P.lattice
    strut = taper(c, s)
    return strut / 2 - lattice_dist(pot, c, th, Z, taper_shift(c, s, np.cos(SLOPE)))


def lattice_cutters(pot, c, depths):
    """The lattice holes for solid.py: (n, len(depths), 4, 2) diamonds in the
    unrolled (S, Z) plane at each of `depths`. Each hole is the diamond between strips k, k+1 of one set
    (u = S + Z cot) and m, m+1 of the other (v = S - Z cot), minus half a
    strut on every side, moved down like lattice_dist. None if the holes
    close up somewhere (a loft can't do that; the field can)."""
    n = pot.count(c.cells)
    p = pot.circ / n
    cot = 1 / np.tan(SLOPE)
    lo, hi = pot.base + PATTERN_GAP, pot.height - pot.rim
    shift = np.array([taper_shift(c, s, np.cos(SLOPE)) for s in depths])
    w = np.array([taper(c, s) for s in depths]) / (2 * np.sin(SLOPE))   # half strut along u
    if (p / 2 - w < 0.01).any():
        return None
    # every (k, m) whose hole centre falls in one turn and near the pattern band
    m_ = abs(shift).max() + p / cot
    half = pot.circ / 2
    k = np.arange(np.floor((-half + (lo - m_) * cot) / p) - 1, np.ceil((half + (hi + m_) * cot) / p) + 1)
    m = np.arange(np.floor((-half - (hi + m_) * cot) / p) - 1, np.ceil((half - (lo - m_) * cot) / p) + 1)
    K, M = [a.ravel() for a in np.meshgrid(k, m)]
    uc, vc = (K + 0.5) * p, (M + 0.5) * p
    Zc = (uc - vc) / (2 * cot)
    # the centre is at S = (K + M + 1) p / 2: one turn is K + M + 1 in [-n, n)
    # (integers, so a hole on the seam isn't taken twice)
    j = K + M + 1
    keep = (j >= -n) & (j < n) & (Zc > lo - m_) & (Zc < hi + m_)
    uc, vc = uc[keep], vc[keep]
    out = np.empty((len(uc), len(depths), 4, 2))
    for i in range(len(depths)):
        a = p / 2 - w[i]                          # half the hole along u and v
        for j, (du, dv) in enumerate(((a, -a), (a, a), (-a, a), (-a, -a))):   # top, right, bottom, left
            u, v = uc + du, vc + dv
            out[:, i, j, 0] = (u + v) / 2
            out[:, i, j, 1] = (u - v) / (2 * cot) - shift[i]
    return out


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
    voronoi: the radius grows from the soil side out and the centre
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


def drops_cutters(pot, c, depths, arc=16):
    """The drops holes for solid.py: (n, len(depths), arc + 1, 2) teardrops in
    the unrolled (S, Z) plane. A teardrop is its circle (radius R) plus the
    cap between the two SLOPE lines tangent to it: the apex at R / cos(SLOPE)
    above the centre, then `arc` points around the bottom of the circle from
    one tangent point (90 - SLOPE degrees above the horizontal) to the
    other. None if a hole closes up."""
    n = pot.count(c.cells)
    p = pot.circ / n
    rowh = p * c.row
    lo, hi = pot.base + PATTERN_GAP, pot.height - pot.rim
    R = np.array([(p - taper(c, s)) / 2 for s in depths])
    if (R < 0.02).any():
        return None
    shift = np.array([taper_shift(c, s, np.cos(SLOPE)) for s in depths])
    a0 = np.pi / 2 - SLOPE                                   # tangent point, above the horizontal
    ang = np.linspace(a0, a0 - (np.pi + 2 * a0), arc)        # clockwise, under the centre, to the other one
    # points a little outside the circle, so the chords straddle it instead of cutting inside
    bulge = 2 / (1 + np.cos((ang[0] - ang[1]) / 2))
    unit = np.c_[np.cos(ang), np.sin(ang)] * bulge
    shape = np.concatenate([[[0, 1 / np.cos(SLOPE)]], unit])    # (arc + 1, 2), for R = 1
    margin = rowh + np.abs(shift).max() + p
    j = np.arange(np.floor((lo - margin) / rowh), np.ceil((hi + margin) / rowh) + 1)
    J, I = [x.ravel() for x in np.meshgrid(j, np.arange(n), indexing="ij")]
    Sc = np.mod(J, 2) * p / 2 + p / 2 + I * p                  # px = 0 in pat_drops
    Zc = (J + 0.5) * rowh
    keep = (Zc > lo - margin) & (Zc < hi + margin)
    Sc, Zc = Sc[keep], Zc[keep]
    out = shape[None, None] * R[None, :, None, None]           # (1, L, arc + 1, 2)
    out = out + np.stack([np.broadcast_to(Sc[:, None], (len(Sc), len(depths))),
                          Zc[:, None] - shift[None]], -1)[:, :, None]
    return out


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
    triangles have pointed tops; downward ones have a short flat bridge.
    It has the voronoi funnel: struts thin from strut_in to strut_out through
    the wall. The two kinds of roof edge have different normals (vertical
    part 1 for the level struts, cos 60 for the others), so each strut set is
    moved down by its own recession / nz: every lower strut edge (a hole
    roof) stays where it is and the holes grow sideways and downward. The
    strut sets can't also meet in clean vertices at every depth, so the
    shifts are measured from the soil side, where the holes hold the soil.
    With center = 1 no set moves at all: the struts just widen evenly and
    each hole stays a centred triangle."""
    c = P.isogrid
    a = pot.circ / pot.count(c.cells)             # triangle side
    h = a * np.sqrt(3) / 2
    S, ZZ = unroll(pot, th, Z)
    strut = taper(c, s)
    z0 = ZZ + taper_shift(c, s, 1.0) - taper_shift(c, 0.0, 1.0)     # level struts
    z60 = ZZ + taper_shift(c, s, 0.5) - taper_shift(c, 0.0, 0.5)    # 60-degree struts
    cot = 1 / np.sqrt(3)                          # cot 60
    d0 = stripes(z0, h)
    d1 = stripes(S + z60 * cot, a) * np.sqrt(3) / 2
    d2 = stripes(S - z60 * cot, a) * np.sqrt(3) / 2
    return strut / 2 - np.minimum.reduce([d0, d1, d2])


def isogrid_cutters(pot, c, depths):
    """The isogrid holes for solid.py: (n, len(depths), 3, 2) triangles in the
    unrolled (S, Z) plane. In the lattice (strut 0, no shift) the level lines
    are z = k h and the 60-degree ones u1 = S + z cot60 = m a and
    u2 = S - z cot60 = n a. Row k has upward triangles (floor z = k h, sides
    u2 = n a and u1 = (n + k + 1) a) and downward ones (roof z = (k + 1) h,
    sides u1 = m a and u2 = (m - k) a). Each side moves in by half a strut
    and each strut set down by its own shift, as in pat_isogrid. None if a
    triangle closes up."""
    N = pot.count(c.cells)
    a = pot.circ / N
    h = a * np.sqrt(3) / 2
    r3 = np.sqrt(3)
    lo, hi = pot.base + PATTERN_GAP, pot.height - pot.rim
    st = np.array([taper(c, s) for s in depths])
    sh0 = np.array([taper_shift(c, s, 1.0) - taper_shift(c, 0.0, 1.0) for s in depths])
    sh60 = np.array([taper_shift(c, s, 0.5) - taper_shift(c, 0.0, 0.5) for s in depths])
    w0, w60 = st / 2, st / r3                     # half strut: in z, and along u
    margin = h + np.abs(np.r_[sh0, sh60]).max()
    k = np.arange(np.floor((lo - margin) / h), np.ceil((hi + margin) / h) + 1)
    K, I = [x.ravel()[:, None] for x in np.meshgrid(k, np.arange(N), indexing="ij")]   # (T, 1)
    # upward: floor Z = k h + w0 - sh0; left side u2 = n a + w60; right side u1 = (n + k + 1) a - w60
    zb = K * h + w0 - sh0
    c1, c2 = (I + K + 1) * a - w60, I * a + w60
    q = (c1 - c2) / 2                             # at the apex, (Z + sh60) cot60 = q
    up = np.stack([np.stack([c2 + (zb + sh60) / r3, zb], -1),
                   np.stack([c1 - (zb + sh60) / r3, zb], -1),
                   np.stack([c2 + q, r3 * q - sh60], -1)], 2)
    # downward: roof Z = (k + 1) h - w0 - sh0; left side u1 = m a + w60; right side u2 = (m - k) a - w60
    zt = (K + 1) * h - w0 - sh0
    c1, c2 = I * a + w60, (I - K) * a - w60
    q = (c1 - c2) / 2
    down = np.stack([np.stack([c1 - (zt + sh60) / r3, zt], -1),
                     np.stack([c2 + (zt + sh60) / r3, zt], -1),
                     np.stack([c2 + q, r3 * q - sh60], -1)], 2)
    out = np.concatenate([up, down])              # (2T, L, 3, 2)
    z = out[..., 1]
    keep = (z.max((1, 2)) > lo) & (z.min((1, 2)) < hi)
    out = out[keep]
    size = np.abs(out[:, :, 0, 1] - out[:, :, 2, 1])   # height of each triangle at each depth
    if (size < 0.02).any():
        return None
    return out


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


PATTERNS = {
    "voronoi": Pattern("2d", pat_voronoi, "organic cells flaring outward like funnels (default)",
                       cutters=voronoi_cutters),
    "hex": Pattern("2d", pat_hex, "warped honeycomb, pointy-top cells, flaring outward", cutters=hex_cutters),
    "drops": Pattern("2d", pat_drops, "staggered teardrops flaring outward", cutters=drops_cutters),
    "lattice": Pattern("2d", pat_lattice, "diamond trellis of crossing helical strips, flaring outward",
                       cutters=lattice_cutters),
    "isogrid": Pattern("2d", pat_isogrid, "triangle grid, flaring outward", cutters=isogrid_cutters),
    "louvers": Pattern("2d", pat_louvers, "gills sloping down and out, no line of sight"),
    "slots": Pattern("2d", pat_slots, "narrow wavy vertical slots, air-pruning style"),
    "spiral": Pattern("2d", pat_spiral, "slots on a many-start helix"),
    "chevrons": Pattern("2d", pat_chevrons, "stacked arrowhead slots"),
    "weave": Pattern("3d", pat_weave, "woven strips passing over and under"),
}
