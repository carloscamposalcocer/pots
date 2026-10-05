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


def _vor_sides(per, step):
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


def voronoi_cutters(pot, c, depths, step=np.deg2rad(12), fine=360, near=14, face=0.1):
    """The voronoi holes for solid.py: a list of (n, len(depths), m, 2)
    outlines in the unrolled (S, Z) plane, grouped by point count m. In
    voronoi_edge's warped coordinates, a hole is where (F2 - F1) exceeds the
    strut: from its seed, the distance to the edge at each angle comes from
    _vor_reach, and the sides are hyperbola arcs, one per neighbour. The
    outline has the exact corners at each depth and points along each side
    (_vor_sides); a cell whose depths don't fit that is sampled at fixed
    angles instead: the corners at the soil face, mid-wall and outer face
    and angles at most `step` apart. The points are then mapped back
    through the warp. Cells whose hole is closed at every depth are
    skipped; where a hole is closed at some depths only (a blind pocket),
    its outline there shrinks to the seed, at the depth where the field's
    hole closes (third coordinate; at least `face` mm inside a wall face,
    or the tip would leave a sliver of a hole in it)."""
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
        ang = _vor_sides([corners[d][ci] for d in range(L)], step)
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
        # the hole closes where the strut reaches the nearest seed (_vor_reach);
        # the closed depths gather there, just apart (0.001 of their spacing)
        sd = np.broadcast_to(depths, t.shape[:2]).copy()        # (g, L)
        closed = t.max(-1) <= 0.01
        if closed.any() and c.strut_out != c.strut_in:
            sc = (np.sqrt((D[idx, 0] ** 2).sum(-1)) - c.strut_in) / (c.strut_out - c.strut_in)
            opens = np.sign(c.strut_in - c.strut_out)            # the side where it is open
            for f in (0.0, 1.0):
                sc = np.where(np.abs(sc - f) < face / pot.wall, f + opens * face / pot.wall, sc)
            sd = np.where(closed, sc[:, None] + 1e-3 * (sd - sc[:, None]), sd)
        Sw = own[idx, None, None, 0] + t * U[..., 0]
        Zw = own[idx, None, None, 1] + t * U[..., 1]
        Z = Zw - taper_shift(c, sd, 1.0)[..., None]              # voronoi_edge gets Z + shift ...
        S = Sw - c.warp * np.sin(2 * np.pi * Z / 47)             # ... and S + warp sin(2 pi Z / 47)
        keep = (Z.max((1, 2)) > lo) & (Z.min((1, 2)) < hi)
        out.append(np.stack([S, Z, np.broadcast_to(sd[..., None], S.shape)], -1)[keep])
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
    """Honeycomb (regular by default; `warp` / `warp_z` make it wavy), vertex
    pointing up, with the voronoi funnel: struts thin from strut_in to
    strut_out through the wall. The roof edges have a vertical normal
    component of sqrt(3)/2, so the pattern moves down by
    recession / (sqrt(3)/2) to keep every roof where it is. strut_extra
    makes up for a warp, which shears the cells and thins some struts."""
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
    unrolled (S, Z) plane at each of `depths`. Each hole is the diamond
    between strips k, k+1 of one set (u = S + Z cot) and m, m+1 of the
    other (v = S - Z cot), minus half a strut on every side, moved down
    like lattice_dist. None if the holes
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


LEAF_TIP = np.deg2rad(62)    # slope of the leaf tips; the sway (shear) may tilt them to SLOPE
LEAF_WAVE = 40.0             # sway wavelength up the pot (mm) ...
LEAF_TURNS = 3               # ... and the waves around it at H_REF (scaled, kept an integer: seamless)


@lru_cache(maxsize=8)
def _leaf_sizes(nr, nc, jitter, seed):
    # per-leaf scale in [1 - jitter, 1] (smaller only, so the struts never thin)
    return np.random.default_rng(seed).uniform(1 - jitter, 1, (nr, nc))


def _leaf_grid(pot, c):
    """Column pitch, row height, scale table and rows above/below a leaf centre
    that a leaf can reach, for settings block c."""
    n = pot.count(c.cells)
    p = pot.circ / n
    h = p * c.row
    sizes = _leaf_sizes(int(pot.height / h) + 6, n, c.jitter, c.seed)
    up = p / 2 / np.cos(LEAF_TIP) / h                 # apex above the centre, in rows
    down = (c.length + c.base) * p / h                # base below the centre, in rows
    return n, p, h, sizes, up, down


def leaf_hole(px, pz, R, rb, L, w):
    """Hole field of one leaf around its widest point (0, 0): the hull of a
    circle of radius R there, one of radius rb at (0, -L) (the base) and an
    apex where two LEAF_TIP lines tangent to the big circle meet, minus a
    keel of half-width w down the middle. Struts are made by shrinking both
    radii by half a strut: that moves every side in by exactly that much,
    and a negative rb gives the pointed base the shrunk sides meet in."""
    x = np.abs(px)
    y = pz + L                                         # from the base centre
    b = (rb - R) / L
    a = np.sqrt(1 - b * b)
    k = -b * x + a * y
    hull = np.where(k < 0, np.hypot(x, y) - rb,
                    np.where(k > a * L, np.hypot(x, pz) - R, a * x + b * y - rb))
    st, ct = np.sin(LEAF_TIP), np.cos(LEAF_TIP)
    tip = np.maximum(x * st + pz * ct - R, R * ct - pz)
    return np.maximum(np.minimum(hull, tip), w - x)


def _leaf_turns(pot):
    return max(1, round(LEAF_TURNS * pot.k))


def _leaf_warp(pot, c, S, Z):
    """The sway: S in the pattern's (warped) coordinates."""
    return S + c.warp * np.sin(2 * np.pi * Z / LEAF_WAVE + _leaf_turns(pot) * S / pot.r0)


def leaf_tip_slope(pot, c):
    """The shallowest leaf tip edge after the sway, in degrees from horizontal
    (in the unrolled plane, like SLOPE): the sway shears the leaves by up to
    2 pi warp / LEAF_WAVE and stretches them sideways by up to
    1 / (1 - warp turns / r0)."""
    shear = 2 * np.pi * c.warp / LEAF_WAVE
    stretch = 1 - c.warp * _leaf_turns(pot) / pot.r0
    run = (np.cos(LEAF_TIP) + shear * np.sin(LEAF_TIP)) / stretch
    return float(np.degrees(np.arctan2(np.sin(LEAF_TIP), run)))


def pat_leaves(pot, P, th, Z, r, s):
    """Echeveria leaves: plump, broadest high up, with a short sharp LEAF_TIP
    point, a keel down the middle and a narrow rounded base, in staggered
    rows so each tip points up between two leaves of the row above. The
    leaves differ a little in size (jitter) and sway in slow waves around
    the pot (warp). Tapered like drops: struts (and the keel) thin from
    strut_in to strut_out and the pattern moves down by recession /
    cos(LEAF_TIP), so the pointed roofs stay put (unless `center`)."""
    c = P.leaves
    n, p, h, sizes, up, down = _leaf_grid(pot, c)
    S, ZZ = unroll(pot, th, Z)
    S = _leaf_warp(pot, c, S, Z)
    ZZ = ZZ + taper_shift(c, s, np.cos(LEAF_TIP))
    strut = taper(c, s)
    j0 = np.floor(ZZ / h)
    hole = np.full(S.shape, 1e9, np.float32)
    for dj in range(int(np.floor(-0.5 - up)), int(np.ceil(0.5 + down)) + 1):
        j = j0 + dj
        off = np.mod(j, 2) * p / 2
        i = np.floor((S - off) / p)
        sc = sizes[np.clip(j + 2, 0, len(sizes) - 1).astype(int), i.astype(int) % n]
        px = S - off - (i + 0.5) * p
        pz = ZZ - (j + 0.5) * h
        d = leaf_hole(px, pz, sc * p / 2 - strut / 2, sc * c.base * p - strut / 2, sc * c.length * p,
                      c.rib * strut / 2)
        hole = np.minimum(hole, d)
    return hole


def leaves_cutters(pot, c, depths, arc=32, keel=8):
    """The leaves holes for solid.py: (n, len(depths), arc + keel + 1, 2)
    outlines in the unrolled (S, Z) plane, one per half leaf (the keel
    splits it). Each half is convex: its two corners on the keel, found by
    bisection along it, `arc` points around the outside, at angles from a
    point on the widest line evenly spaced between the corners', each found
    by bisection along its ray on leaf_hole, and `keel` points up the keel.
    Mapped back through the shift and the sway (fixed-point iteration),
    which bends the keel. None if a half closes up."""
    n, p, h, sizes, up, down = _leaf_grid(pot, c)
    lo, hi = pot.base + PATTERN_GAP, pot.height - pot.rim
    depths = np.asarray(depths)
    strut = np.array([taper(c, s) for s in depths])
    shift = np.array([taper_shift(c, s, np.cos(LEAF_TIP)) for s in depths])
    margin = (up + down + 1) * h + np.abs(shift).max() + c.warp
    J = np.arange(np.floor((lo - margin) / h), np.ceil((hi + margin) / h) + 1)
    J, I = [x.ravel() for x in np.meshgrid(J, np.arange(n), indexing="ij")]
    sc = sizes[np.clip(J + 2, 0, len(sizes) - 1).astype(int), I][:, None, None]     # (g, 1, 1)
    R = sc * p / 2 - strut[None, :, None] / 2                                    # (g, L, 1)
    rb = sc * c.base * p - strut[None, :, None] / 2
    Lb = sc * c.length * p
    w = c.rib * strut[None, :, None] / 2
    if (R - w < 0.05).any():
        return None

    def bisect(x0, dx, dy, keel, steps=36):
        """Distance from (x0, 0) (inside) along (dx, dy) to the edge of the
        leaf, with or without its keel (to 4 p / 2^steps)."""
        shape = np.broadcast(x0, dx, dy).shape
        a, b = np.zeros(shape), np.full(shape, 4 * p)
        for _ in range(steps):
            m = (a + b) / 2
            ok = leaf_hole(x0 + m * dx, m * dy, R, rb, Lb, w if keel else -np.inf) < 0
            a, b = np.where(ok, m, a), np.where(ok, b, m)
        return (a + b) / 2

    # the right half; the left one is its mirror image
    yt = bisect(w, 0.0, 1.0, False)                        # keel corners, (g, L, 1)
    yb = -bisect(w, 0.0, -1.0, False)
    cx = (w + R) / 2                                       # ray origin, on the widest line
    a0 = np.arctan2(yt, w - cx)                            # top corner: up and left, in (90, 180) deg
    a1 = np.arctan2(yb, w - cx)                            # bottom corner: down and left, in (-180, -90)
    # clockwise round the right side: a fine pass, then `arc` angles evenly spaced along it
    fine = a0 + (a1 - a0) * np.linspace(0, 1, 4 * arc + 1)
    d = bisect(cx, np.cos(fine), np.sin(fine), True, steps=16)
    px, py = cx + d * np.cos(fine), d * np.sin(fine)
    along = np.concatenate([np.zeros(d.shape[:-1] + (1,)),
                            np.cumsum(np.hypot(np.diff(px, axis=-1), np.diff(py, axis=-1)), -1)], -1)
    want = along[..., -1:] * np.arange(1, arc + 1) / (arc + 1)
    k = np.clip((along[..., None, :] <= want[..., None]).sum(-1) - 1, 0, 4 * arc - 1)
    at = lambda a: np.take_along_axis(a, k, -1)
    t = (want - at(along)) / at(np.diff(along, axis=-1).clip(1e-12))
    ang = at(fine) + t * at(np.diff(fine, axis=-1))
    dx, dy = np.cos(ang), np.sin(ang)
    d = bisect(cx, dx, dy, True)
    up_keel = yb + (yt - yb) * np.arange(keel) / keel      # bottom corner and up the keel (the sway bends it)
    x = np.concatenate([np.broadcast_to(w, yt.shape), cx + d * dx, np.broadcast_to(w, up_keel.shape)], -1)
    y = np.concatenate([yt, d * dy, up_keel], -1)          # (g, L, arc + keel + 1)
    off = np.mod(J, 2) * p / 2
    Sc = (off + (I + 0.5) * p)[:, None, None]
    Zc = ((J + 0.5) * h)[:, None, None]
    out = []
    for sign in (1, -1):
        Sw = Sc + sign * x
        Zp = Zc + y
        Z = Zp - shift[None, :, None]
        S = Sw.copy()
        for _ in range(60):                                # S + sway(S, Z) = Sw
            S = Sw - c.warp * np.sin(2 * np.pi * Z / LEAF_WAVE + _leaf_turns(pot) * S / pot.r0)
        err = np.abs(_leaf_warp(pot, c, S, Z) - Sw).max()
        if err > 1e-6:
            raise RuntimeError(f"leaves sway inversion did not converge ({err:.2g} mm)")
        out.append(np.stack([S, Z], -1))
    out = np.concatenate(out)
    z = out[..., 1]
    return out[(z.max((1, 2)) > lo) & (z.min((1, 2)) < hi)]


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


# ------------------------------------------- isometric cubes (lozenge tilings)
# The cube patterns tile the wall with 60-degree rhombi (lozenges) on a
# triangular lattice of side a whose edges are vertical and at +-30 degrees,
# so every hole roof is a 30-degree tent or slope, as on hex. In lattice
# coordinates a point is p = x e1 + y e2 with e1 = (a sqrt3/2, a/2) (the
# 30-degree edge) and e2 = (0, a). Cell (i, j) is split by its 150-degree
# diagonal into a right-pointing triangle t0 (x + y < 1) and a left-pointing
# t1. Each lozenge is a t0 plus one neighbour, named by its kind:
#   V: the t1 left of it, t1(i-1, j): a wide diamond (a cube's top face);
#      edges e1, e2 - e1 from corner (i, j)
#   A: the t1 of its own cell, t1(i, j); edges e1, e2 (rising sides)
#   B: the t1 below it, t1(i, j-1); edges e1 - e2, e2 (falling sides)
# A tiling is the kind of every t0 (it fixes every t1 too). Going once
# around the pot is N columns, and (i + N, j - N/2) is the same point
# (N even), so tilings are stored for i in [0, N) and rows j0 .. j0 + J.
# A hole is one lozenge or several of one kind (an L of three on blocks),
# drawn as up to two overlapping parallelograms; a strut runs only round
# the outside of each hole.
LOZ_V, LOZ_A, LOZ_B = 0, 1, 2
LOZ_EDGES = {LOZ_V: ((1, 0), (-1, 1)), LOZ_A: ((1, 0), (0, 1)), LOZ_B: ((1, -1), (0, 1))}
LOZ_PARTNER = {LOZ_V: (-1, 0), LOZ_A: (0, 0), LOZ_B: (0, -1)}     # its t1, from (i, j)


class Lozenges(NamedTuple):
    """A periodic hole layout on the lattice: N columns, side a, rows from j0;
    t0, t1: (N, J) hole index of each triangle; pieces: (H, 2, 3, 2) per hole
    two parallelograms as (centre, half edge A, half edge B) in (S, Z) mm."""
    N: int
    a: float
    j0: int
    t0: np.ndarray
    t1: np.ndarray
    pieces: np.ndarray


def _loz_basis(a):
    return np.array([a * np.sqrt(3) / 2, a / 2]), np.array([0.0, a])


def _loz_wrap(N, j0, J, i, j):
    """Canonical (i, row) array indices of lattice cells (i, j), and whether in range."""
    ii = np.mod(i, N)
    jj = j + (i - ii) // N * (N // 2) - j0
    return ii, jj, (jj >= 0) & (jj < J)


def _loz_rows(pot, N, a, margin):
    """Rows j0 and count J covering the pattern band (plus margin mm) at every column."""
    lo, hi = pot.base + PATTERN_GAP - margin, pot.height - pot.rim + margin
    j0 = int(np.floor(lo / a - N / 2)) - 2
    return j0, int(np.ceil(hi / a)) + 2 - j0


def _loz_layout(N, a, j0, kind, groups=None):
    """Lozenges from the kind of every t0 (N, J); `groups` (N, J) optionally
    gives each t0 a group label (holes of several lozenges of one kind: up to
    an L of three), else every lozenge is its own hole."""
    J = kind.shape[1]
    e1, e2 = _loz_basis(a)
    I, Jr = np.meshgrid(np.arange(N), np.arange(J), indexing="ij")
    j = Jr + j0
    t0 = np.full((N, J), -1)
    t1 = np.full((N, J), -1)
    key = groups if groups is not None else I * J + Jr
    labels, hole = np.unique(key.ravel(), return_inverse=True)
    t0[:] = hole.reshape(N, J)
    for k, (di, dj) in LOZ_PARTNER.items():
        m = kind == k
        ii, jj, ok = _loz_wrap(N, j0, J, I[m] + di, j[m] + dj)
        t1[ii[ok], jj[ok]] = t0[m][ok]
    # pieces: per hole, its lozenges (corner + edge vectors), merged into arms
    pieces = np.zeros((len(labels), 2, 3, 2))
    members = [[] for _ in labels]
    for (i, jr), h in np.ndenumerate(t0):
        members[h].append((i, jr + j0, kind[i, jr]))
    for h, mem in enumerate(members):
        k = mem[0][2]
        E1, E2 = (u[0] * e1 + u[1] * e2 for u in LOZ_EDGES[k])
        # lozenge centres, unwrapped next to the first one
        P = np.array([[i, j] for i, j, _ in mem], float)
        turn = np.round((P[0, 0] - P[:, 0]) / N)                # (i + N, j - N/2) is the same cell
        P += turn[:, None] * [N, -(N // 2)]
        C = P[:, :1] * e1 + P[:, 1:] * e2 + (E1 + E2) / 2
        if len(mem) == 1:
            pieces[h] = [C[0], E1 / 2, E2 / 2]
            continue
        # an L: the corner lozenge touches both others; each arm is the corner plus one
        d = C[:, None] - C[None]
        dist = np.hypot(d[..., 0], d[..., 1])
        side = np.isclose(dist, np.hypot(*E1)) | np.isclose(dist, np.hypot(*E2))
        corner = int(np.argmax(side.sum(1)))
        others = [m for m in range(len(mem)) if m != corner]
        if len(mem) > 3 or len(others) != side[corner].sum():
            raise ValueError("a hole must be one lozenge or an L of up to three")
        for p, m in enumerate((others * 2)[:2]):
            along1 = abs(np.dot(C[m] - C[corner], E1)) > abs(np.dot(C[m] - C[corner], E2))
            pieces[h, p] = [(C[m] + C[corner]) / 2, E1 if along1 else E1 / 2, E2 / 2 if along1 else E2]
    return Lozenges(N, a, j0, t0, t1, pieces)


def _normal(v):
    return np.stack([-v[..., 1], v[..., 0]], -1) / np.hypot(v[..., 0], v[..., 1])[..., None]


def _para_hole(q, A, B, strut):
    """Hole field of the parallelogram with half edges A, B at offset q from its centre."""
    nA, nB = _normal(A), _normal(B)
    dot = lambda u, v: (u * v).sum(-1)
    return np.maximum(np.abs(dot(nB, q)) - np.abs(dot(nB, A)),
                      np.abs(dot(nA, q)) - np.abs(dot(nA, B))) + strut / 2


def loz_hole(pot, L, S, ZZ, strut):
    """Hole field of the lozenge layout L at unrolled (S, ZZ)."""
    J = L.t0.shape[1]
    x = S / (L.a * np.sqrt(3) / 2)
    y = ZZ / L.a - x / 2
    i, j = np.floor(x), np.floor(y)
    upper = (x - i) + (y - j) >= 1
    ii, jj, ok = _loz_wrap(L.N, L.j0, J, i.astype(int), j.astype(int))
    jj = np.clip(jj, 0, J - 1)
    h = np.where(upper, L.t1[ii, jj], L.t0[ii, jj])
    ok &= h >= 0
    pc = L.pieces[np.where(ok, h, 0)]                        # (n, 2, 3, 2)
    q = np.stack([S, ZZ], -1)[:, None] - pc[:, :, 0]
    q[..., 0] = np.mod(q[..., 0] + pot.circ / 2, pot.circ) - pot.circ / 2
    f = _para_hole(q, pc[:, :, 1], pc[:, :, 2], np.asarray(strut)[..., None]).min(1)
    return np.where(ok, f, 1.0)


def _loz_frames(pieces):
    """Per hole: the centre of its corner lozenge, its half edges E1/2, E2/2
    and the sides its legs go to along them (s1, s2: +-1, or 0 for a single
    lozenge), from the two pieces of each hole (see _loz_layout)."""
    (C1, A1, B1), (C2, A2, B2) = pieces[:, 0].transpose(1, 0, 2), pieces[:, 1].transpose(1, 0, 2)
    norm = lambda v: np.hypot(v[:, 0], v[:, 1])
    first = norm(A1) > 1.5 * norm(A2)                            # piece 0 is the arm along E1
    Ca, Cb = np.where(first[:, None], C1, C2), np.where(first[:, None], C2, C1)
    Ea = np.where(first[:, None], A2, A1)                        # E1 / 2: the shorter A
    Eb = np.where(first[:, None], B1, B2)                        # E2 / 2: the shorter B
    single = np.all(np.isclose(pieces[:, 0], pieces[:, 1]), axis=(1, 2))
    Ea, Eb = np.where(single[:, None], A1, Ea), np.where(single[:, None], B1, Eb)
    # Ca = corner + s1 Ea, Cb = corner + s2 Eb
    s = np.linalg.solve(np.stack([Ea, -Eb], -1), (Ca - Cb)[..., None])[..., 0]
    s = np.where(single[:, None], 0, np.round(s))
    corner = np.where(single[:, None], C1, Ca - s[:, :1] * Ea)
    return corner, Ea, Eb, s[:, 0], s[:, 1]


def _l_outline(da, db, extra=7):
    """An L of three unit lozenges in the (alpha, beta) frame of its corner
    lozenge [-1, 1]^2, legs toward +alpha and +beta, shrunk by da, db; with
    `extra` - 1 points along each outer side of the corner lozenge, so the
    mean of the points (where solid.py centres the cap fans) lies in the
    corner lozenge, which sees the whole L. da, db: (L,) -> (L, 6 + 2 extra, 2)."""
    a0, a1, a3 = -1 + da, 1 - da, 3 - da
    b0, b1, b3 = -1 + db, 1 - db, 3 - db
    t = np.arange(1, extra) / extra
    one = np.ones_like(da)
    pts = [np.stack([a0, b0], -1)]
    pts += [np.stack([a0 + (a1 - a0) * k, b0], -1) for k in t]               # along the bottom
    pts += [np.stack(p, -1) for p in ((a3, b0), (a3, b1), (a1, b1), (a1, b3), (a0, b3), (a0, b1 * one))]
    pts += [np.stack([a0, b1 + (b0 - b1) * k], -1) for k in t]               # down the left side
    return np.stack(pts, -2)


def loz_cutters(pot, c, depths, L):
    """The lozenge holes for solid.py: (n, len(depths), 4, 2) parallelograms
    for single lozenges and (n, len(depths), 20, 2) outlines for Ls (see
    _l_outline), each shrunk by half a strut on every side and moved down
    like the field. None if a hole closes up."""
    lo, hi = pot.base + PATTERN_GAP, pot.height - pot.rim
    strut = np.array([taper(c, s) for s in depths])
    shift = np.array([taper_shift(c, s, np.sqrt(3) / 2) for s in depths])
    corner, Ea, Eb, s1, s2 = _loz_frames(L.pieces)
    na, nb = _normal(Ea), _normal(Eb)
    da = strut[None] / 2 / np.abs((nb * Ea).sum(-1))[:, None]   # (n, L): half a strut in units of Ea
    db = strut[None] / 2 / np.abs((na * Eb).sum(-1))[:, None]
    if (da > 0.99).any() or (db > 0.99).any():
        return None
    corner = corner + (np.mod(corner[:, :1] + pot.circ / 2, pot.circ) - pot.circ / 2 - corner[:, :1]) * [1, 0]
    out = []
    single = s1 == 0
    for group in (single, ~single):
        if not group.any():
            continue
        if group is single:
            sq = np.array([(1, 1), (-1, 1), (-1, -1), (1, -1)])
            ab = sq[None, None] * np.stack([1 - da[group], 1 - db[group]], -1)[:, :, None]
            sa, sb = np.ones(group.sum()), np.ones(group.sum())
        else:
            ab = np.stack([_l_outline(da[k], db[k]) for k in np.flatnonzero(group)])
            sa, sb = s1[group], s2[group]
        o = (corner[group, None, None] + (ab[..., :1] * sa[:, None, None, None]) * Ea[group, None, None]
             + (ab[..., 1:] * sb[:, None, None, None]) * Eb[group, None, None])
        o[..., 1] -= shift[None, :, None]
        z = o[..., 1]
        out.append(o[(z.max((1, 2)) > lo) & (z.min((1, 2)) < hi)])
    return out


@lru_cache(maxsize=16)
def _cubes_layout(N, j0, J):
    """Tumbling blocks: a cube on every third lattice point, (i - j) = 0 mod 3,
    with its top face (V) above it, its left face (B) at (i-1, j) and its
    right face (A) at (i, j-1)."""
    I, Jr = np.meshgrid(np.arange(N), np.arange(J) + j0, indexing="ij")
    return np.array([LOZ_V, LOZ_A, LOZ_B])[np.mod(I - Jr, 3)]


def _cubes(pot, c):
    N = 2 * pot.count(c.cells)                    # two columns per cube
    a = pot.circ / (N * np.sqrt(3) / 2)
    j0, J = _loz_rows(pot, N, a, 3 * a + 5)
    return _loz_cached(N, a, j0, J, "cubes")


def _voxel(pot, c):
    N = 2 * pot.count(c.cells)
    a = pot.circ / (N * np.sqrt(3) / 2)
    j0, J = _loz_rows(pot, N, a, 3 * a + 5)
    return _loz_cached(N, a, j0, J, "voxel", int(c.sweeps), int(c.seed))


def _blocks(pot, c):
    N = 6 * pot.count(c.cells / 2)                # a repeat is six columns and holds two blocks
    a = pot.circ / (N * np.sqrt(3) / 2)
    j0, J = _loz_rows(pot, N, a, 6 * a + 5)
    return _loz_cached(N, a, j0, J, "blocks")


@lru_cache(maxsize=16)
def _loz_cached(N, a, j0, J, name, sweeps=0, seed=0):
    if name == "cubes":
        return _loz_layout(N, a, j0, _cubes_layout(N, j0, J))
    if name == "voxel":
        return _loz_layout(N, a, j0, _voxel_layout(N, j0, J, sweeps, seed))
    return _loz_layout(N, a, j0, *_blocks_layout(N, j0, J))


def _voxel_layout(N, j0, J, sweeps, seed):
    """Random stacked cubes: the tumbling blocks reshuffled by `sweeps` rounds
    of random cube flips. Round lattice point C the three lozenges at t0(C),
    t0(C - e1), t0(C - e2) are either (V, B, A) (a cube) or (B, A, V) (a
    hollow corner); flipping swaps them, which keeps a valid tiling. Points
    of one class (i - j) mod 3 share no lozenge, so a class flips at once.
    Only points whose three cells are inside the rows flip, so it stays the
    plain tumbling blocks beyond them."""
    kind = _cubes_layout(N, j0, J).copy()
    rng = np.random.default_rng(seed)
    I, Jr = np.meshgrid(np.arange(N), np.arange(J), indexing="ij")
    cells = [_loz_wrap(N, j0, J, I + di, Jr + j0 + dj) for di, dj in ((0, 0), (-1, 0), (0, -1))]
    ok = cells[0][2] & cells[1][2] & cells[2][2]
    cls = np.mod(I - (Jr + j0), 3)
    for _ in range(sweeps):
        for k in range(3):
            pts = ok & (cls == k) & (rng.random((N, J)) < 0.5)
            got = [kind[ci[pts], cj[pts]] for ci, cj, _ in cells]
            cube = (got[0] == LOZ_V) & (got[1] == LOZ_B) & (got[2] == LOZ_A)
            hollow = (got[0] == LOZ_B) & (got[1] == LOZ_A) & (got[2] == LOZ_V)
            new = [np.where(cube, b, np.where(hollow, v, g)) for g, v, b in
                   zip(got, (LOZ_V, LOZ_B, LOZ_A), (LOZ_B, LOZ_A, LOZ_V))]
            for (ci, cj, _), n in zip(cells, new):
                kind[ci[pts], cj[pts]] = n
    return kind


# blocks: a cube of side 2 lattice units with its top-back, bottom-left and
# bottom-right eighths cut away; its three faces are Ls of three lozenges
# (offsets from the block's point C, which sits on every (3m, 3n)).
BLOCK_FACES = {LOZ_V: ((0, 0), (1, 0), (-1, 1)),       # top face, a V
               LOZ_B: ((-1, -1), (-2, 1), (-1, 0)),    # left face, leg down on its right
               LOZ_A: ((0, -2), (0, -1), (1, -1))}     # right face, leg down on its left


def _blocks_layout(N, j0, J):
    """The notched blocks: kind and group label of every t0."""
    kind = np.full((N, J), -1)
    group = np.full((N, J), -1)
    I, Jr = np.meshgrid(np.arange(N), np.arange(J) + j0, indexing="ij")
    C = (np.mod(I, 3) == 0) & (np.mod(Jr, 3) == 0)
    for f, (k, offs) in enumerate(BLOCK_FACES.items()):
        for di, dj in offs:
            ii, jj, ok = _loz_wrap(N, j0, J, I[C] + di, Jr[C] + dj)
            kind[ii[ok], jj[ok]] = k
            gi, gj, _ = _loz_wrap(N, j0, J, I[C], Jr[C])         # label: the block's own cell and face
            group[ii[ok], jj[ok]] = (gi * J + gj)[ok] * 3 + f
    # cells whose block lies beyond the rows: plain lozenges of the cube tiling
    # (they are outside the pattern band, the clip cuts them off)
    rest = kind < 0
    kind[rest] = _cubes_layout(N, j0, J)[rest]
    group[rest] = -1 - np.arange(rest.sum())
    return kind, group


def _loz_pattern(layout, name):
    def fn(pot, P, th, Z, r, s):
        c = getattr(P, name)
        S, ZZ = unroll(pot, th, Z)
        return loz_hole(pot, layout(pot, c), S, ZZ + taper_shift(c, s, np.sqrt(3) / 2), taper(c, s))

    def cutters(pot, c, depths):
        return loz_cutters(pot, c, depths, layout(pot, c))
    return fn, cutters


pat_cubes, cubes_cutters = _loz_pattern(_cubes, "cubes")
pat_voxel, voxel_cutters = _loz_pattern(_voxel, "voxel")
pat_blocks, blocks_cutters = _loz_pattern(_blocks, "blocks")


# ---------------------------------------------------------------- romanesco
GOLDEN = np.pi * (3 - np.sqrt(5))       # golden angle: the florets' spiral
ROOF = np.cos(SLOPE)                    # flat roof of the round holes, at this x R above the centre


@lru_cache(maxsize=8)
def _florets(circ, height, n, row, spacing, small, gap, seed):
    """Seeds (S, Z) of the romanesco holes, the soil-side radius of each
    one's round cap, and a KD-tree on the seeds (periodic around the pot).
    Florets sit on a staggered grid, n around; each is a golden-angle (Vogel)
    spiral of seeds `spacing` mm apart on average, turned at random and
    mirrored on every other floret, keeping only the seeds nearer its own
    centre than any other floret's. Where two florets meet, seeds closer
    than `gap` are thinned out (the one nearer its floret's edge goes), so
    every hole opens on the soil side. The cap radius is g spacing / 2, with
    g from `small` at the floret's centre to 1 at its edge."""
    from scipy.spatial import cKDTree
    w = circ / n
    h = w * row
    rng = np.random.default_rng(seed)
    rows = np.arange(-2, int(height / h) + 3)
    J, I = [a.ravel() for a in np.meshgrid(rows, np.arange(n), indexing="ij")]
    cs = np.mod((I + 0.5 + 0.5 * np.mod(J, 2)) * w, circ)
    cz = (J + 0.5) * h
    c = spacing * np.sqrt(np.sqrt(3) / (2 * np.pi))       # Vogel's scale for that mean spacing
    k = np.arange(int((0.9 * w / c) ** 2) + 1)
    rho, phi = c * np.sqrt(k + 0.5), k * GOLDEN
    ang = np.where(np.mod(I + J, 2), 1, -1)[:, None] * phi + rng.uniform(0, 2 * np.pi, len(cs))[:, None]
    S = np.mod(cs[:, None] + rho * np.cos(ang), circ).ravel()
    Zs = (cz[:, None] + rho * np.sin(ang)).ravel()
    box, up = [circ, 1e6], 1e5                            # Z moved up: the tree's box starts at 0
    _, near = cKDTree(np.c_[cs, cz + up], boxsize=box).query(np.c_[S, Zs + up])
    keep = near == np.repeat(np.arange(len(cs)), len(k))
    pts = np.c_[S[keep], Zs[keep] + up]
    u = np.clip(np.tile(rho, len(cs))[keep] / (0.5 * w), 0, 1)
    while True:
        pairs = cKDTree(pts, boxsize=box).query_pairs(gap, output_type="ndarray")
        if not len(pairs):
            break
        drop = np.unique(np.where(u[pairs[:, 0]] >= u[pairs[:, 1]], pairs[:, 0], pairs[:, 1]))
        pts, u = np.delete(pts, drop, 0), np.delete(u, drop)
    return pts - [0, up], (small + (1 - small) * u) * spacing / 2, cKDTree(pts, boxsize=box)


def _romanesco(pot, c):
    return _florets(pot.circ, pot.height, pot.count(c.cells), c.row, c.spacing, c.small, c.strut_in + 1.0, c.seed)


def round_hole(px, pz, R):
    """A round hole with a flat roof where the circle's slope drops below SLOPE."""
    return np.maximum(np.hypot(px, pz) - R, pz - R * ROOF)


def pat_romanesco(pot, P, th, Z, r, s):
    """Romanesco florets: holes on golden-angle spirals, small and round at
    each floret's centre (the tip of the cone), growing toward its edge
    into the cells of the spiral's Voronoi diagram; florets on a staggered
    grid, turned at random and alternately mirrored. Each hole is its
    seed's Voronoi cell (struts at least `strut` wide, as on voronoi) cut
    to a round cap, which has a flat roof where the circle gets flatter
    than SLOPE. Tapered like voronoi: struts thin from strut_in to
    strut_out through the wall, the caps grow by as much, and the pattern
    moves down to keep the roofs level (unless `center`)."""
    c = P.romanesco
    pts, R, tree = _romanesco(pot, c)
    S, ZZ = unroll(pot, th, Z)
    ZZ = ZZ + taper_shift(c, s, 1.0)
    strut = taper(c, s)
    d, idx = tree.query(np.c_[np.mod(S, pot.circ), ZZ + 1e5], k=2)
    i = idx[:, 0]
    px = np.mod(S - pts[i, 0] + pot.circ / 2, pot.circ) - pot.circ / 2
    cap = round_hole(px, ZZ - pts[i, 1], R[i] + (c.strut_in - strut) / 2)
    return np.maximum(strut / 2 - (d[:, 1] - d[:, 0]) / 2, cap)


def _by_turning(P, m):
    """Indices (n, m) of m of the points of closed outlines P (n, L, k, 2) to
    keep, the same at every depth: spread by the outline's turning (summed
    over the depths) half and evenly half, so the corners keep a point each
    and no stretch goes without."""
    a = np.roll(P, -1, 2) - P
    ang = np.arctan2(a[..., 1], a[..., 0])
    turn = np.abs(np.mod(ang - np.roll(ang, 1, 2) + np.pi, 2 * np.pi) - np.pi).sum(1)    # (n, k) at each point
    k = P.shape[2]
    w = 0.5 * turn / turn.sum(1, keepdims=True) + 0.5 / k
    cum = np.cumsum(w, 1)
    idx = np.stack([np.searchsorted(cw, (np.arange(m) + 0.5) / m) for cw in cum])
    idx = np.maximum.accumulate(idx - np.arange(m), 1) + np.arange(m)     # strictly increasing
    return np.minimum(idx, k - m + np.arange(m))


def romanesco_cutters(pot, c, depths, arc=28, fine=240, near=12):
    """The romanesco holes for solid.py: (n, len(depths), arc, 2) outlines in
    the unrolled (S, Z) plane. At `fine` angles round each seed the edge is
    the nearer of the Voronoi hole's (_vor_reach, from the `near` nearest
    seeds) and the round cap with its flat roof; of those, the `arc` that
    follow it best at every depth (_by_turning: the corners and evenly
    round the rest)."""
    pts, R, tree = _romanesco(pot, c)
    lo, hi = pot.base + PATTERN_GAP, pot.height - pot.rim
    depths = np.asarray(depths)
    strut = np.array([taper(c, s) for s in depths])
    shift = np.array([taper_shift(c, s, 1.0) for s in depths])
    reach = R.max() + c.strut_in
    sel = np.flatnonzero((pts[:, 1] + reach - shift.min() > lo) & (pts[:, 1] - reach - shift.max() < hi))
    _, nb = tree.query(pts[sel] + [0, 1e5], k=near + 1)
    D = pts[nb[:, 1:]] - pts[sel, None]                          # (n, j, 2) offsets to the neighbours
    D[..., 0] = np.mod(D[..., 0] + pot.circ / 2, pot.circ) - pot.circ / 2
    phi = np.linspace(0, 2 * np.pi, fine, endpoint=False)
    U = np.c_[np.cos(phi), np.sin(phi)]
    t = np.stack([_vor_reach(D[:, None], st, np.broadcast_to(U, (len(sel), fine, 2))) for st in strut], 1)
    Rc = R[sel, None] + (c.strut_in - strut[None]) / 2           # (n, L) cap radius
    roof = np.where(U[:, 1] > 1e-9, ROOF / np.where(U[:, 1] > 1e-9, U[:, 1], 1), np.inf)
    t = np.minimum(t, Rc[..., None] * np.minimum(1, roof))       # (n, L, fine)
    if (t.min(-1) < 0.02).any():
        return None
    P = t[..., None] * U                                         # around each seed
    P = np.take_along_axis(P, _by_turning(P, arc)[:, None, :, None], 2)
    S = np.mod(pts[sel, 0] + pot.circ / 2, pot.circ) - pot.circ / 2
    P[..., 0] += S[:, None, None]
    P[..., 1] += pts[sel, 1, None, None] - shift[None, :, None]
    return P


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
    "hex": Pattern("2d", pat_hex, "regular honeycomb, pointy-top cells, flaring outward", cutters=hex_cutters),
    "drops": Pattern("2d", pat_drops, "staggered teardrops flaring outward", cutters=drops_cutters),
    "lattice": Pattern("2d", pat_lattice, "diamond trellis of crossing helical strips, flaring outward",
                       cutters=lattice_cutters),
    "isogrid": Pattern("2d", pat_isogrid, "triangle grid, flaring outward", cutters=isogrid_cutters),
    "cubes": Pattern("2d", pat_cubes, "tumbling blocks: isometric cubes of three rhombi, flaring outward",
                     cutters=cubes_cutters),
    "voxel": Pattern("2d", pat_voxel, "random stacked isometric cubes, flaring outward", cutters=voxel_cutters),
    "blocks": Pattern("2d", pat_blocks, "notched isometric blocks, faces in L shapes, flaring outward",
                      cutters=blocks_cutters),
    "romanesco": Pattern("2d", pat_romanesco, "romanesco florets: round holes on golden spirals, flaring outward",
                         cutters=romanesco_cutters),
    "leaves": Pattern("2d", pat_leaves, "echeveria leaves with a keel, staggered and swaying, flaring outward",
                      cutters=leaves_cutters),
    "louvers": Pattern("2d", pat_louvers, "gills sloping down and out, no line of sight"),
    "slots": Pattern("2d", pat_slots, "narrow wavy vertical slots, air-pruning style"),
    "spiral": Pattern("2d", pat_spiral, "slots on a many-start helix"),
    "chevrons": Pattern("2d", pat_chevrons, "stacked arrowhead slots"),
    "weave": Pattern("3d", pat_weave, "woven strips passing over and under"),
}
