"""
Breathing plant pot v2 + plain drip cup. Two separate prints, no supports.

POT : graded gyroid lattice wall from the build plate up to a solid rim,
      perforated floor (vertical pores, printed flat on the bed).
CUP : plain smooth saucer-cup that catches drips and acts as the water
      reserve. Six low ribs inside lift the pot a few mm so drained water
      spreads underneath; once the level rises above the ribs, the soil
      wicks it back up through the perforated floor.

All dimensions in mm. Negative field = solid.
Usage:  python3 pot_v2.py [voxel_mm]      -> pot_v2_raw.stl, cup_v2_raw.stl
"""
import logging
import sys
import time

import numpy as np
from skimage.measure import marching_cubes
import trimesh

log = logging.getLogger(__name__)

# ---------------- pot ----------------
# H drives the size: set_height() rescales every proportion (radii, cup
# height, cells around the circumference) by k = H / H_REF. Print-physics
# values (wall, rim, floor, struts, hole size, blends) stay fixed in mm.
H_REF = 130.0
R_BOT_REF = 56.0      # outer radius at the bottom, at H_REF
R_TOP_REF = 72.0      # outer radius at the top, at H_REF
N_THETA_REF = 48      # gyroid cells around, at H_REF
H = H_REF
K = 1.0               # H / H_REF
R_BOT = R_BOT_REF     # outer radius at the bottom
R_TOP = R_TOP_REF     # outer radius at the top
WALL = 6.0            # lattice wall thickness
FLOOR = 2.4           # floor thickness
RIM = 5.0             # solid rim at the top
CELL = 8.0            # gyroid cell size
N_THETA = N_THETA_REF # cells around the circumference (integer -> seamless)
T_IN, T_OUT = 1.00, 0.62   # gyroid level: dense on the soil side, open outside
FLOOR_LEVEL = 1.05    # floor pore level (higher = smaller pores)
FLOOR_EDGE = 3.0      # solid ring where the floor meets the lattice (strength)

# ---------------- cup ----------------
CUP_H = 28.0          # height of the cup
CUP_GAP = 5.0         # radial clearance at the bottom between pot and cup
CUP_FLARE = 3.0       # extra inner radius at the top of the cup (gentle flare)
CUP_WALL = 2.4
CUP_FLOOR = 2.4
RIB_H = 3.0           # ribs lift the pot this much above the cup floor
RIB_W = 2.4
N_RIBS = 6

BLEND = 1.5
VOX = 0.3             # default voxel size (mm); make_pot.py passes its own


def set_height(h):
    """Scale the pot to height h (mm), keeping its proportions."""
    global H, K, R_BOT, R_TOP, N_THETA
    H = float(h)
    K = H / H_REF
    R_BOT, R_TOP = R_BOT_REF * K, R_TOP_REF * K
    N_THETA = max(3, round(N_THETA_REF * K))     # keeps cell size ~constant


def r_out(z):
    return R_BOT + (R_TOP - R_BOT) * np.clip(z, 0, H) / H


def r_in(z):
    return r_out(z) - WALL


def smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b * (1 - h) + a * h - k * h * (1 - h)


def gyroid(theta, z, r, level):
    """Cylindrical gyroid with a low-frequency organic warp; every warp term
    has an integer frequency in theta so the pattern closes seamlessly."""
    u = theta * N_THETA
    v = 2 * np.pi * z / CELL
    w = 2 * np.pi * r / CELL
    u = u + 1.1 * np.sin(3 * theta + 2 * np.pi * z / 55.0) + 0.5 * np.sin(7 * theta - 2 * np.pi * z / 31.0)
    v = v + 0.9 * np.sin(5 * theta + 2 * np.pi * z / 70.0) + 0.4 * np.cos(2 * theta + 2 * np.pi * z / 23.0)
    g = np.sin(u) * np.cos(v) + np.sin(v) * np.cos(w) + np.sin(w) * np.cos(u)
    return (np.abs(g) - level) * CELL / (2 * np.pi * 1.4)


def floor_gyroid(X, Y, level, z0=1.3):
    """Same gyroid family in plain x/y coordinates (uniform pores, no pinch
    at the centre), with its own gentle warp, sliced at z0."""
    u = 2 * np.pi * X / CELL + 1.6 * np.sin(2 * np.pi * Y / 37.0)
    v = 2 * np.pi * Y / CELL + 1.6 * np.sin(2 * np.pi * X / 41.0)
    w = 2 * np.pi * z0 / CELL
    g = np.sin(u) * np.cos(v) + np.sin(v) * np.cos(w) + np.sin(w) * np.cos(u)
    return (np.abs(g) - level) * CELL / (2 * np.pi * 1.4)


def pot_field(X, Y, Z):
    r = np.sqrt(X * X + Y * Y)
    th = np.arctan2(Y, X)
    ro, ri = r_out(Z), r_in(Z)

    # lattice wall, from the bed to the rim
    shell = np.maximum.reduce([r - ro, ri - r, -Z, Z - H])
    s = np.clip((r - ri) / WALL, 0, 1)
    dg = gyroid(th, Z, r, T_IN + (T_OUT - T_IN) * s)
    rim = (H - RIM) - Z                          # <0 inside the rim band
    wall = np.maximum(shell, smin(dg, rim, BLEND))

    # perforated floor: the gyroid pattern taken at a fixed height and
    # extruded vertically, so every pore is a straight vertical hole
    floor_env = np.maximum.reduce([r - ri - 0.5, -Z, Z - FLOOR])
    pores = floor_gyroid(X, Y, FLOOR_LEVEL)
    edge = (ri - FLOOR_EDGE) - r                 # <0 in the edge ring -> solid there
    floor = np.maximum(floor_env, smin(pores, edge, 0.6))

    body = smin(wall, floor, 0.8)

    # 45 deg chamfer on the bottom outer edge against elephant foot
    body = np.maximum(body, ((r - ro) + (0.6 - Z)) / np.sqrt(2))
    return body


def cup_ri(z):
    return R_BOT + CUP_GAP + CUP_FLARE * np.clip(z, 0, CUP_H) / CUP_H


def cup_field(X, Y, Z):
    r = np.sqrt(X * X + Y * Y)
    th = np.arctan2(Y, X)
    ci = cup_ri(Z)
    co = ci + CUP_WALL
    outer = np.maximum.reduce([r - co, -Z, Z - CUP_H])
    cavity = np.maximum(r - ci, CUP_FLOOR - Z)
    cup = np.maximum(outer, -cavity)

    # radial ribs on the cup floor
    ribs = np.full_like(X, 1e3)
    for i in range(N_RIBS):
        a = 2 * np.pi * i / N_RIBS
        across = np.abs(-X * np.sin(a) + Y * np.cos(a)) - RIB_W / 2
        along = X * np.cos(a) + Y * np.sin(a)
        rib = np.maximum.reduce([across, 12.0 - along, along - (ci - 0.2), Z - (CUP_FLOOR + RIB_H), -Z])
        ribs = np.minimum(ribs, rib)
    cup = smin(cup, ribs, 0.6)

    # chamfer on the bottom outer edge
    cup = np.maximum(cup, ((r - co) + (0.6 - Z)) / np.sqrt(2))
    return cup


def build(fn, rmax, zmax, vox, out):
    pad = 2.0
    xs = np.arange(-rmax - pad, rmax + pad + vox, vox, dtype=np.float32)
    zs = np.arange(-pad, zmax + pad + vox, vox, dtype=np.float32)
    X2, Y2 = np.meshgrid(xs, xs, indexing="ij")
    SL = 24
    n_slabs = (len(zs) - 2) // SL + 1 if len(zs) > 1 else 0
    log.info(
        "grid %d x %d x %d (rmax=%.1f mm, zmax=%.1f mm, voxel=%.3f mm, %d slabs)",
        len(xs), len(xs), len(zs), rmax, zmax, vox, n_slabs,
    )
    t0 = time.perf_counter()
    verts, faces, off, k0, slab = [], [], 0, 0, 0
    while k0 < len(zs) - 1:
        k1 = min(k0 + SL, len(zs) - 1)
        zz = zs[k0:k1 + 1]
        slab += 1
        log.info(
            "slab %d/%d  z=%.1f–%.1f mm",
            slab, n_slabs, float(zz[0]), float(zz[-1]),
        )
        X = np.repeat(X2[:, :, None], len(zz), 2)
        Y = np.repeat(Y2[:, :, None], len(zz), 2)
        Z = np.broadcast_to(zz, X.shape).astype(np.float32)
        F = fn(X, Y, Z).astype(np.float32)
        if F.min() < 0 < F.max():
            v, f, _, _ = marching_cubes(F, 0.0)
            v[:, 2] += k0                     # index space -> exact seams
            verts.append(v); faces.append(f + off); off += len(v)
            log.debug("  marching cubes: %s verts, %s faces", f"{len(v):,}", f"{len(f):,}")
        else:
            log.debug("  skipped empty slab")
        k0 = k1
    if not verts:
        raise RuntimeError("field never crossed zero; nothing to mesh")
    m = trimesh.Trimesh(np.vstack(verts), np.vstack(faces), process=False)
    m.merge_vertices(digits_vertex=3)
    m.vertices = m.vertices * vox + np.array([xs[0], xs[0], zs[0]])
    m.update_faces(m.nondegenerate_faces())
    m.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(m)
    log.info("writing %s  (%s faces)", out, f"{len(m.faces):,}")
    m.export(out)
    log.info("meshed in %.1f s", time.perf_counter() - t0)
    return m


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-5s  %(message)s", datefmt="%H:%M:%S")
    vox = float(sys.argv[1]) if len(sys.argv) > 1 else VOX
    for name, fn, rmax, zmax in [("cup", cup_field, R_BOT + CUP_GAP + CUP_FLARE + CUP_WALL, CUP_H),
                                 ("pot", pot_field, R_TOP, H)]:
        m = build(fn, rmax, zmax, vox, f"{name}_v2_raw.stl")
        log.info("%s  faces=%s  volume=%.1f cm3", name, f"{len(m.faces):,}", m.volume / 1000)
