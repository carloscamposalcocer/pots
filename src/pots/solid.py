"""
Exact pot mesh for patterns whose holes can be outlined (lattice, isogrid,
drops, hex, voronoi), built with manifold3d instead of marching cubes: the
plain pot is a revolved profile and the holes are lofted cutters,
subtracted in one go. Flat faces stay flat and edges stay sharp, with a
small fraction of the triangles.

A pattern opts in with `Pattern.cutters(pot, c, depths)`: it returns the
hole outlines in the unrolled (S, Z) plane (mm, S = theta * r0), one per
depth s (0 soil side, 1 outside) in `depths`, as an (n, len(depths), m, 2)
array or a list of such arrays (holes with different point counts), or
None when it can't (holes that close up inside the wall). Each outline must
be star-shaped around its centroid, and point i of one outline must be
point i of the next: the hole is the loft through its outlines. The holes
may touch or overlap (they are then merged first, which is slower).

The first and last depth lie PAST mm beyond the wall faces (the outlines
are extended past the faces, where they change nothing), so no cutter
vertex sits on a wall face. A straight line between two depths strays from
the true path through the curved wall (it leans toward the bigger radius,
~0.05 mm across the whole wall); LAYERS depths keep that under ~0.01 mm.
A straight outline edge is a chord of the curved wall, around the pot and
(on the curved shapes) up it; the edges are split until that chord strays
less than CHORD from the wall (only small or curved pots need it).
"""
import logging
import time

import numpy as np
import trimesh

from .geometry import CHAMFER, PATTERN_GAP

log = logging.getLogger(__name__)

TOL = 0.01            # max deviation (mm) of the revolved surfaces from the true shape
PAST = 0.3            # the cutters reach this far past each wall face (mm)
FILLET = 0.8          # 45 deg chamfer legs in the inside corners of wall, base and cup
                      # (they stand in for the field's 1 mm blends); shorter than
                      # PATTERN_GAP, so they end below the first holes
LAYERS = 5            # depths through the wall at which the hole outlines are given
CHORD = 0.03          # max gap (mm) between a hole edge and the curved wall it follows


def depths(pot):
    """The depths s at which the patterns give their hole outlines."""
    return np.linspace(-PAST / pot.wall, 1 + PAST / pot.wall, LAYERS)


def _ccw(pts):
    """The polygon counter-clockwise: CrossSection's fill rule reads a clockwise one as a hole."""
    pts = np.asarray(pts, float)
    x, y = pts[:, 0], pts[:, 1]
    area = np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
    return pts if area > 0 else pts[::-1]


def _profile(pot):
    """The pot without its pattern, as (r, z) polygons to revolve."""
    from manifold3d import CrossSection
    H, b = pot.height, pot.base
    z = np.linspace(0, H, int(np.ceil(H / 0.25)) + 1)
    wall = np.concatenate([np.c_[pot.r_in(z), z], np.c_[pot.r_out(z), z][::-1]])
    co = lambda zz: float(pot.cup_ri(zz)) + pot.cup_wall
    ci = lambda zz: float(pot.cup_ri(zz))
    base = [(0, 0), (co(0), 0), (co(b), b), (0, b)]
    cup = [(ci(0), 0), (co(0), 0), (co(pot.cup_h), pot.cup_h), (ci(pot.cup_h), pot.cup_h)]
    F = FILLET
    ri, ro = float(pot.r_in(b)), float(pot.r_out(b))
    corners = [
        [(ri - F, b), (ri, b), (float(pot.r_in(b + F)), b + F)],       # soil side
        [(ro, b), (ro + F, b), (float(pot.r_out(b + F)), b + F)],      # moat, pot side
        [(ci(b) - F, b), (ci(b), b), (ci(b + F), b + F)],             # moat, cup side
    ]
    sec = CrossSection([_ccw(s) for s in [wall, base, cup] + corners])    # their union
    # 45 deg chamfer on the bottom outer edge (elephant foot)
    c0 = co(0)
    chamfer = CrossSection([_ccw([(c0 - CHAMFER, 0), (c0 + 5, 0), (c0 + 5, CHAMFER + 5)])])
    return (sec - chamfer).simplify(TOL / 2)


def _segments(r):
    """Segments around a circle of radius r for a chord error below TOL."""
    return int(np.ceil(np.pi / np.arccos(1 - TOL / r)))


TURNS = ((np.sqrt(5) - 1) / 2, np.sqrt(2) - 1, np.pi - 3)   # see _revolve


def _revolve(section, r, turn=TURNS[0]):
    """Revolve `section` with a chord error below TOL at radius r, turned by
    `turn` (an irrational part) of a segment: the pattern repeats a whole
    number of times around, and a facet edge lined up (nearly) exactly with
    a hole edge makes faces that touch (edges shared by 4 faces once the
    vertices are merged). build_solid tries the next turn if that happens."""
    from manifold3d import Manifold
    n = _segments(r)
    return Manifold.revolve(section, n).rotate([0, 0, 360 / n * turn])


def _clip(pot, turn=TURNS[0]):
    """Where holes may be cut: the pattern band, minus the soil-side skin."""
    from manifold3d import CrossSection
    lo, hi = pot.base + PATTERN_GAP, pot.height - pot.rim
    R = pot.r_max + 5
    band = CrossSection([_ccw([(0, lo), (R, lo), (R, hi), (0, hi)])])
    if pot.skin > 0 and pot.skin_z < hi:
        z = np.linspace(max(pot.skin_z, lo), hi, 200)
        skin = np.concatenate([[(0, z[0])], np.c_[pot.r_in(z) + pot.skin, z], [(0, z[-1])]])
        band = band - CrossSection([_ccw(skin)])
    return _revolve(band.simplify(TOL / 2), R, turn)


def _split(pot, outlines):
    """Pieces per outline edge for a chord error below CHORD: the sag of a
    chord of length a on a curve of radius R is a^2 / 8R, around the pot
    (R = the smallest inner radius) and up the wall (1/R = the largest |r''|)."""
    d = np.abs(np.roll(outlines, -1, 2) - outlines).reshape(-1, 2)
    if not len(d):
        return 1
    z = np.linspace(0, pot.height, 400)
    r = pot.r_in(z)
    bend = np.abs(np.gradient(np.gradient(r, z), z)).max()
    sag = max(d[:, 0].max() ** 2 / (8 * r.min()), d[:, 1].max() ** 2 * bend / 8)
    return max(1, int(np.ceil(np.sqrt(sag / CHORD))))


def _lofts(pot, outlines, s):
    """All holes as one triangle mesh of disjoint lofts: (vertices, faces)."""
    n, L, m, _ = outlines.shape
    # counter-clockwise in (S, Z), which is the view from outside
    x, y = outlines[:, 0, :, 0], outlines[:, 0, :, 1]
    cw = (x * np.roll(y, -1, 1) - y * np.roll(x, -1, 1)).sum(1) < 0
    outlines[cw] = outlines[cw, :, ::-1]
    # split every edge so the chords follow the curved wall
    split = _split(pot, outlines)
    t = np.arange(split) / split
    nxt = np.roll(outlines, -1, 2)
    outlines = (outlines[..., None, :] * (1 - t[:, None]) + nxt[..., None, :] * t[:, None]).reshape(n, L, m * split, 2)
    m *= split
    S, Z = outlines[..., 0], outlines[..., 1]
    r = pot.r_in(Z) + s[None, :, None] * pot.wall
    th = S / pot.r0
    ring = np.stack([r * np.cos(th), r * np.sin(th), Z], -1)             # (n, L, m, 3)
    caps = np.stack([ring[:, 0].mean(1), ring[:, -1].mean(1)], 1)       # (n, 2, 3)
    per = L * m + 2
    verts = np.concatenate([ring.reshape(n, L * m, 3), caps], 1).reshape(-1, 3)
    j, i = np.meshgrid(np.arange(L - 1), np.arange(m), indexing="ij")
    a, b = j * m + i, j * m + (i + 1) % m
    c, d = b + m, a + m
    side = np.concatenate([np.stack([a, b, c], -1), np.stack([a, c, d], -1)]).reshape(-1, 3)
    i = np.arange(m)
    inner = np.stack([np.full(m, L * m), (i + 1) % m, i], -1)            # faces the soil
    outer = np.stack([np.full(m, L * m + 1), (L - 1) * m + i, (L - 1) * m + (i + 1) % m], -1)
    one = np.concatenate([side, inner, outer])
    faces = (one[None] + per * np.arange(n)[:, None, None]).reshape(-1, 3)
    return verts, faces


def _groups(outlines):
    """The outlines as a list of non-empty (n, L, m, 2) float arrays."""
    if outlines is None:
        return []
    if not isinstance(outlines, (list, tuple)):
        outlines = [outlines]
    return [np.array(g, float) for g in outlines if len(g)]


def _disjoint(pot, groups):
    """True if no two holes overlap at any depth, around the seam too:
    then the union of the outlines has the sum of their areas."""
    from manifold3d import CrossSection
    L = groups[0].shape[1]
    for d in sorted({0, L // 2, L - 1}):
        polys, total = [], 0.0
        for g in groups:
            P = g[:, d]                                       # (n, m, 2)
            x, y = P[..., 0], P[..., 1]
            area = (x * np.roll(y, -1, 1) - y * np.roll(x, -1, 1)).sum(1) / 2
            P = np.where((area < 0)[:, None, None], P[:, ::-1], P)     # counter-clockwise
            # copies across the seam, so holes there are checked against each other
            edge = np.abs(x).max(1) > pot.circ / 2 - 20
            P = np.concatenate([P, P[edge] - (pot.circ, 0), P[edge] + (pot.circ, 0)])
            total += np.abs(area).sum() + 2 * np.abs(area[edge]).sum()
            polys += list(P)
        if CrossSection(polys).area() < total * (1 - 1e-7) - 1e-6:
            return False
    return True


def build_solid(pot, outlines):
    """The pot with holes `outlines`, given at depths(pot) (see the module
    doc), as a trimesh."""
    from manifold3d import Manifold, Mesh, OpType
    t0 = time.perf_counter()
    s = depths(pot)
    groups = _groups(outlines)
    n = sum(len(g) for g in groups)
    holes = None
    if groups:
        lofts = [_lofts(pot, g, s) for g in groups]
        if _disjoint(pot, groups):
            off = np.cumsum([0] + [len(v) for v, _ in lofts])
            verts = np.concatenate([v for v, _ in lofts])
            faces = np.concatenate([f + o for (_, f), o in zip(lofts, off)])
            holes = Manifold(Mesh(verts.astype(np.float32), faces.astype(np.uint32)))
        else:
            log.info("some holes touch; merging them first")
            parts = []
            for (v, f), g in zip(lofts, groups):
                nv, nf = len(v) // len(g), len(f) // len(g)
                parts += [Manifold(Mesh(v[k * nv:(k + 1) * nv].astype(np.float32),
                                        (f[k * nf:(k + 1) * nf] - k * nv).astype(np.uint32)))
                          for k in range(len(g))]
            holes = Manifold.batch_boolean(parts, OpType.Add)
        if holes.status().name != "NoError":
            raise RuntimeError(f"hole cutters are not a valid solid: {holes.status()}")
    log.info("%d holes", n)
    profile = _profile(pot)
    for turn in TURNS:
        body = _revolve(profile, pot.r_max, turn)
        if holes is not None:
            body = body - (holes ^ _clip(pot, turn))
        m = body.to_mesh()
        mesh = trimesh.Trimesh(m.vert_properties[:, :3], m.tri_verts, process=True)
        if mesh.is_watertight:
            break
        log.debug("facets lined up with a hole edge (pinched edges); turning the revolve")
    else:
        log.warning("solid mesh is not watertight after %d tries", len(TURNS))
    log.info("solid mesh: %s faces in %.1f s", f"{len(mesh.faces):,}", time.perf_counter() - t0)
    return mesh
