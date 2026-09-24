"""
Generate the pot: STL + 3MF + preview image, in one command.

Needs pot_v2.py, pot_v3.py, patterns.py and the conf/ folder (Hydra config).
Every setting has a default in conf/config.yaml; pass only what you change.

Examples
    python make_pot.py                          # tapered Voronoi, 130 mm, full quality
    python make_pot.py quality=draft            # fast low-res test
    python make_pot.py height=65                # half-size pot, same shape
    python make_pot.py pattern=coral            # another pattern
    python make_pot.py design.wall=4            # thinner wall (mm, does not scale)
    python make_pot.py height=65 --show         # print the resolved settings, don't build
    python make_pot.py -v                       # debug: per-slab marching cubes

Outputs (in `out`, default output/<pattern>_h<height>/):
    pot_<pattern>.stl      print file
    pot_<pattern>.3mf      same mesh, ~5x smaller file
    pot_<pattern>.png      preview: outside view, cut-away, wall faces
    config.yaml            the exact settings used (re-run: copy values back as overrides)
"""
import logging
import os
import sys
import time

import hydra
import numpy as np
import trimesh
from omegaconf import OmegaConf

import patterns
from patterns import PATTERNS, make_field, build, r_out, P2, P3

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
        R = r_out(Zg) - P2.WALL * frac
        open_ = field((R * np.cos(TH)).astype(np.float32), (R * np.sin(TH)).astype(np.float32),
                      Zg.astype(np.float32)) >= 0
        img = np.ones(open_.shape + (3,)); img[~open_] = [0.26, 0.45, 0.32]
        ax = fig.add_subplot(2, 3, 4 + k)
        ax.imshow(img, origin="lower", extent=[0, ww, z0, z0 + wz]); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{lab}: {open_.mean() * 100:.0f}% open ({ww:.0f} x {wz:.0f} mm, true scale)")
    fig.suptitle(title, fontsize=15)
    plt.tight_layout(); plt.savefig(path, dpi=85); plt.close(fig)
    log.info("preview saved %s in %.1f s", path, time.perf_counter() - t0)


def apply_config(cfg):
    """Push the config into the geometry modules; returns an error message or None."""
    if cfg.pattern not in PATTERNS:
        return f"unknown pattern '{cfg.pattern}'; choose from: {', '.join(sorted(PATTERNS))}"
    d = cfg.design
    P2.WALL, P2.RIM = float(d.wall), float(d.rim)
    P3.BASE, P3.CUP_WALL = float(d.base), float(d.cup_wall)
    patterns.STRUT_IN, patterns.STRUT_OUT = float(d.strut_in), float(d.strut_out)
    min_h = P3.BASE + P2.RIM + 10.0
    if cfg.height < min_h:
        return f"height must be at least {min_h:g} mm (base + rim + some pattern)"
    if d.strut_out < 1.1:
        log.warning("design.strut_out=%.2f mm is below ~1.1 mm, too thin for a 0.4 mm nozzle", d.strut_out)
    patterns.set_height(cfg.height)
    return None


def load_config(overrides):
    # Compose API instead of @hydra.main: hydra-core 1.3's own argparse CLI
    # crashes on Python 3.14. Overrides use the same key=value syntax.
    with hydra.initialize(config_path="conf", version_base="1.3"):
        return hydra.compose("config", overrides=overrides)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if {"-h", "--help"} & set(argv):
        print(__doc__); return
    flags = {a for a in argv if a.startswith("-")}
    unknown = flags - {"-v", "--verbose", "--show"}
    if unknown:
        raise SystemExit(f"unknown option(s): {' '.join(sorted(unknown))}  (settings are key=value, see --help)")
    try:
        cfg = load_config([a for a in argv if not a.startswith("-")])
    except hydra.errors.HydraException as e:
        raise SystemExit(f"bad setting: {e}".splitlines()[0] + "  (see --show for valid keys)")
    if "--show" in flags:
        print(OmegaConf.to_yaml(cfg, resolve=True), end=""); return
    logging.basicConfig(
        level=logging.DEBUG if flags & {"-v", "--verbose"} else logging.INFO,
        format="%(asctime)s  %(levelname)-5s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("trimesh").setLevel(logging.WARNING)
    err = apply_config(cfg)
    if err:
        log.error(err)
        raise SystemExit(2)
    q = cfg.quality
    log.info("height %.1f mm (x%.3f): radius %.1f -> %.1f mm, cup %.1f mm tall, wall %.1f mm",
             P2.H, P2.K, P2.R_BOT, P2.R_TOP, P3.CUP_H, P2.WALL)

    os.makedirs(cfg.out, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(cfg.out, "config.yaml"), resolve=True)
    base = os.path.join(cfg.out, f"pot_{cfg.pattern}")
    field = make_field(cfg.pattern)
    t = time.perf_counter()
    log.info("[1/3] meshing '%s' at %.3f mm", cfg.pattern, q.voxel)
    raw = build(field, P2.R_TOP + P3.LIP + P3.CUP_WALL + 1.0, P2.H, q.voxel, base + "_raw.stl")
    os.remove(base + "_raw.stl")
    log.info("[2/3] cleaning %s faces", f"{len(raw.faces):,}")
    m = clean(raw, q.faces)
    size = np.ptp(m.bounds, axis=0)
    log.info(
        "mesh  faces=%s  watertight=%s  size=%.1f x %.1f x %.1f mm  volume=%.0f cm3",
        f"{len(m.faces):,}", m.is_watertight, size[0], size[1], size[2], m.volume / 1000,
    )
    log.info("exporting %s.stl and %s.3mf", base, base)
    m.export(base + ".stl"); m.export(base + ".3mf")
    outs = [f"{base}.stl", f"{base}.3mf"]
    if cfg.preview:
        log.info("[3/3] rendering preview")
        preview(m, field, base + ".png", f"pot_{cfg.pattern}  (H {P2.H:g} mm)")
        outs.append(f"{base}.png")
    else:
        log.info("[3/3] skipped preview")
    log.info("done in %.0f s -> %s", time.perf_counter() - t, "  ".join(outs))


if __name__ == "__main__":
    main()
