"""
Generate the pot: STL + 3MF + preview image, in one command.

Needs pot_v2.py, pot_v3.py and patterns.py in the same folder.

Examples
    python make_pot.py                          # tapered Voronoi, full quality
    python make_pot.py --pattern coral          # another pattern
    python make_pot.py --draft                  # fast low-res test (~15 s)
    python make_pot.py --height 65              # half-size pot, same shape
    python make_pot.py --list                   # show available patterns
    python make_pot.py -v                       # debug: per-slab marching cubes

Outputs (in --out, default ./output):
    pot_<pattern>.stl      print file
    pot_<pattern>.3mf      same mesh, ~5x smaller file
    pot_<pattern>.png      preview: outside view, cut-away, wall faces
"""
import argparse
import logging
import os
import sys
import time

import numpy as np
import trimesh

import patterns
from patterns import PATTERNS, make_field, build, r_out, WALL, P2, P3

log = logging.getLogger("pots")


def clean(mesh, target_faces):
    """Decimate, drop floating specks, make watertight."""
    import fast_simplification
    import pymeshfix
    t0 = time.perf_counter()
    n0 = len(mesh.faces)
    if n0 > target_faces:
        log.info("decimating %s -> %s faces", f"{n0:,}", f"{target_faces:,}")
        v, f = fast_simplification.simplify(mesh.vertices.astype(np.float32), mesh.faces,
                                            target_reduction=1 - target_faces / n0)
        mesh = trimesh.Trimesh(v, f)
        log.info("decimated to %s faces", f"{len(mesh.faces):,}")
    else:
        log.info("skipping decimation (%s faces <= budget %s)", f"{n0:,}", f"{target_faces:,}")
    parts = mesh.split(only_watertight=False)
    mesh = max(parts, key=lambda p: len(p.faces))
    log.info("kept largest of %d components (%s faces)", len(parts), f"{len(mesh.faces):,}")
    log.info("repairing mesh")
    fix = pymeshfix.MeshFix(mesh.vertices, mesh.faces)
    fix.repair(joincomp=False, remove_smallest_components=True)
    mesh = trimesh.Trimesh(fix.points, fix.faces)
    trimesh.repair.fix_normals(mesh)
    mesh.apply_translation([0, 0, -mesh.bounds[0, 2]])        # sit on z = 0
    log.info("cleaned in %.1f s", time.perf_counter() - t0)
    return mesh


def preview(mesh, field, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import fast_simplification

    t0 = time.perf_counter()
    # light mesh for drawing
    v, f = fast_simplification.simplify(mesh.vertices.astype(np.float32), mesh.faces,
                                        target_reduction=max(0.0, 1 - 120000 / len(mesh.faces)))
    m = trimesh.Trimesh(v, f)
    light = np.array([0.4, -0.6, 0.7]); light /= np.linalg.norm(light)

    def draw(ax, tris, normals, elev, azim):
        sh = np.clip(normals @ light, 0, 1) * 0.75 + 0.2
        col = np.stack([0.30 * sh + 0.04, 0.52 * sh + 0.04, 0.36 * sh + 0.04, np.ones_like(sh)], 1).clip(0, 1)
        ax.add_collection3d(Poly3DCollection(tris, facecolors=col, edgecolors="none"))
        lim = P2.R_TOP + 4
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-10, P2.H + 10)
        ax.set_box_aspect((1, 1, 1)); ax.view_init(elev, azim); ax.set_axis_off()

    fig = plt.figure(figsize=(18, 11))
    ax = fig.add_subplot(2, 3, 1, projection="3d")
    draw(ax, m.triangles, m.face_normals, 16, -60); ax.set_title("Outside")
    keep = m.triangles_center[:, 1] > 0                      # back half only -> cut-away
    ax = fig.add_subplot(2, 3, 2, projection="3d")
    draw(ax, m.triangles[keep], m.face_normals[keep], 15, -90); ax.set_title("Cut-away")

    # vertical section straight from the field
    ax = fig.add_subplot(2, 3, 3)
    s = np.arange(-P2.R_TOP - 4, P2.R_TOP + 4, 0.15); z = np.arange(-1, P2.H + 1, 0.15)
    S, Z = np.meshgrid(s, z)
    F = field(S.astype(np.float32), np.full(S.shape, 0.7, np.float32), Z.astype(np.float32)) < 0
    ax.imshow(F, origin="lower", extent=[s[0], s[-1], z[0], z[-1]], cmap="Greys")
    ax.set_aspect("equal"); ax.set_title("Vertical section (mm)")

    # both wall faces at true scale: a 70 x 50 mm window at mid-height, cut
    # down on small pots so it stays inside the patterned band
    wz = min(50.0, P2.H - P3.BASE - P2.RIM - 4.0)
    ww = min(70.0, np.pi * P2.R_BOT)
    z0 = P2.H / 2 - wz / 2
    zc = np.arange(z0, z0 + wz, 0.1); ang = np.arange(0, ww, 0.1)
    A, Zg = np.meshgrid(ang, zc); TH = A / r_out(Zg)
    for k, (frac, lab) in enumerate(((0.02, "Outside face"), (0.98, "Soil-side face"))):
        R = r_out(Zg) - WALL * frac
        open_ = field((R * np.cos(TH)).astype(np.float32), (R * np.sin(TH)).astype(np.float32),
                      Zg.astype(np.float32)) >= 0
        img = np.ones(open_.shape + (3,)); img[~open_] = [0.26, 0.45, 0.32]
        ax = fig.add_subplot(2, 3, 4 + k)
        ax.imshow(img, origin="lower", extent=[0, ww, z0, z0 + wz]); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{lab}: {open_.mean() * 100:.0f}% open ({ww:.0f} x {wz:.0f} mm, true scale)")
    fig.suptitle(title, fontsize=15)
    plt.tight_layout(); plt.savefig(path, dpi=85); plt.close(fig)
    log.info("preview saved %s in %.1f s", path, time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pattern", default="voronoi_taper", choices=sorted(PATTERNS))
    ap.add_argument("--height", type=float, default=P2.H_REF,
                    help=f"pot height in mm (default {P2.H_REF:g}); radii and cup scale with it, "
                         "wall/rim/base/struts/hole size stay fixed")
    ap.add_argument("--voxel", type=float, default=0.3, help="mesh resolution in mm (smaller = finer, slower)")
    ap.add_argument("--faces", type=int, default=900_000, help="triangle budget of the final mesh")
    ap.add_argument("--draft", action="store_true", help="quick low-res run (voxel 0.6, 300k faces)")
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--out", default="output")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true", help="debug logging (per-slab marching cubes, etc.)")
    a = ap.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if a.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-5s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("trimesh").setLevel(logging.WARNING)
    if a.list:
        print("\n".join(sorted(PATTERNS))); return
    if a.draft:
        a.voxel, a.faces = 0.6, 300_000
        log.info("draft mode: voxel=%.2f mm, face budget=%s", a.voxel, f"{a.faces:,}")

    min_h = P3.BASE + P2.RIM + 10.0
    if a.height < min_h:
        ap.error(f"--height must be at least {min_h:g} mm (base + rim + some pattern)")
    patterns.set_height(a.height)
    log.info("height %.1f mm (x%.3f): radius %.1f -> %.1f mm, cup %.1f mm tall",
             P2.H, P2.K, P2.R_BOT, P2.R_TOP, P3.CUP_H)

    os.makedirs(a.out, exist_ok=True)
    base = os.path.join(a.out, f"pot_{a.pattern}")
    field = make_field(a.pattern)
    t = time.perf_counter()
    log.info("[1/3] meshing '%s' at %.3f mm", a.pattern, a.voxel)
    raw = build(field, P2.R_TOP + P3.LIP + P3.CUP_WALL + 1.0, P2.H, a.voxel, base + "_raw.stl")
    os.remove(base + "_raw.stl")
    log.info("[2/3] cleaning %s faces", f"{len(raw.faces):,}")
    m = clean(raw, a.faces)
    size = np.ptp(m.bounds, axis=0)
    log.info(
        "mesh  faces=%s  watertight=%s  size=%.1f x %.1f x %.1f mm  volume=%.0f cm3",
        f"{len(m.faces):,}", m.is_watertight, size[0], size[1], size[2], m.volume / 1000,
    )
    log.info("exporting %s.stl and %s.3mf", base, base)
    m.export(base + ".stl"); m.export(base + ".3mf")
    if not a.no_preview:
        log.info("[3/3] rendering preview")
        preview(m, field, base + ".png", f"pot_{a.pattern}")
    else:
        log.info("[3/3] skipped preview")
    outs = [f"{base}.stl", f"{base}.3mf"]
    if not a.no_preview:
        outs.append(f"{base}.png")
    log.info("done in %.0f s -> %s", time.perf_counter() - t, "  ".join(outs))


if __name__ == "__main__":
    main()
