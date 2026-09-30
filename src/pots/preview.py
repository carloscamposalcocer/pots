"""PNG preview: outside view, cut-away, vertical section and both wall faces."""
import logging
import time

import numpy as np
import trimesh

from .mesh import decimate
from .metrics import INNER, OUTER, face_window, sample_face

log = logging.getLogger(__name__)

PREVIEW_FACES = 120_000
MAX_EDGE = 5.0        # mm; longer triangle edges are split before drawing (see drawable)
LIGHT = np.array([0.4, -0.6, 0.7]) / np.linalg.norm([0.4, -0.6, 0.7])


def drawable(mesh, faces):
    """The mesh for drawing: decimated to `faces`, then with every edge longer
    than MAX_EDGE split. matplotlib has no depth buffer, it draws triangles
    in the order of their centres' depth, and the long thin triangles of an
    exact (solid.py) mesh then show through as streaks."""
    m = decimate(mesh, faces)
    v, f = trimesh.remesh.subdivide_to_size(m.vertices, m.faces, max_edge=MAX_EDGE, max_iter=10)
    return trimesh.Trimesh(v, f, process=False)


def draw_mesh(ax, tris, normals, pot, elev, azim, pad=4, tight=False):
    """Shaded triangles on a 3D axis; `tight` fits the box to the pot instead of a cube."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    sh = np.clip(normals @ LIGHT, 0, 1) * 0.75 + 0.2
    col = np.stack([0.30 * sh + 0.04, 0.52 * sh + 0.04, 0.36 * sh + 0.04, np.ones_like(sh)], 1).clip(0, 1)
    ax.add_collection3d(Poly3DCollection(tris, facecolors=col, edgecolors="none"))
    lim = pot.r_wide + pad
    zpad = 2 if tight else 10
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-zpad, pot.height + zpad)
    ax.set_box_aspect((1, 1, (pot.height + 2 * zpad) / (2 * lim)) if tight else (1, 1, 1))
    ax.view_init(elev, azim); ax.set_axis_off()


def render(mesh, field, pot, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t0 = time.perf_counter()
    m = drawable(mesh, PREVIEW_FACES)          # light mesh for drawing
    fig = plt.figure(figsize=(18, 11))
    ax = fig.add_subplot(2, 3, 1, projection="3d")
    draw_mesh(ax, m.triangles, m.face_normals, pot, 16, -60); ax.set_title("Outside")
    keep = m.triangles_center[:, 1] > 0                      # back half only -> cut-away
    ax = fig.add_subplot(2, 3, 2, projection="3d")
    draw_mesh(ax, m.triangles[keep], m.face_normals[keep], pot, 15, -90); ax.set_title("Cut-away")

    # vertical section straight from the field
    ax = fig.add_subplot(2, 3, 3)
    s = np.arange(-pot.r_max, pot.r_max, 0.15); z = np.arange(-1, pot.height + 1, 0.15)
    S, Z = np.meshgrid(s, z)
    F = field(S.astype(np.float32), np.full(S.shape, 0.7, np.float32), Z.astype(np.float32)) < 0
    ax.imshow(F, origin="lower", extent=[s[0], s[-1], z[0], z[-1]], cmap="Greys")
    ax.set_aspect("equal"); ax.set_title("Vertical section (mm)")

    # both wall faces at true scale
    win = face_window(pot)
    ww, wz = win.arc[-1] + win.px, win.z[-1] + win.px - win.z[0]
    for k, (frac, lab) in enumerate(((OUTER, "Outside face"), (INNER, "Soil-side face"))):
        open_ = sample_face(field, pot, frac, win)
        img = np.ones(open_.shape + (3,)); img[~open_] = [0.26, 0.45, 0.32]
        ax = fig.add_subplot(2, 3, 4 + k)
        ax.imshow(img, origin="lower", extent=win.extent); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{lab}: {open_.mean() * 100:.0f}% open ({ww:.0f} x {wz:.0f} mm, true scale)")
    fig.suptitle(title, fontsize=15)
    plt.tight_layout(); plt.savefig(path, dpi=85); plt.close(fig)
    log.info("preview saved %s in %.1f s", path, time.perf_counter() - t0)


def render_card(mesh, field, pot, path, title):
    """Small gallery image: outside view + a 40 x 30 mm true-scale swatch of the outer face."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = drawable(mesh, 2 * PREVIEW_FACES)
    fig = plt.figure(figsize=(8, 4.2))
    draw_mesh(fig.add_subplot(1, 2, 1, projection="3d"), m.triangles, m.face_normals, pot, 14, -60, pad=2, tight=True)
    win = face_window(pot, width=40.0, height=30.0)
    open_ = sample_face(field, pot, OUTER, win)
    img = np.ones(open_.shape + (3,)); img[~open_] = [0.26, 0.45, 0.32]
    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(img, origin="lower", extent=win.extent); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"outside face, {open_.mean() * 100:.0f}% open (40 x 30 mm)", fontsize=10)
    fig.suptitle(title, fontsize=14)
    plt.tight_layout(); plt.savefig(path, dpi=80); plt.close(fig)
    log.info("card saved %s", path)


POT_GREY, WATER_BLUE, SOIL_BROWN, INK, MUTED = "#8f9a94", "#cfe2f2", "#eee4d6", "#333333", "#777777"


def volumes_ml(pot, n=400):
    """(soil in L, water in ml): the inside of the pot above the base, and the
    moat between the pot and the cup filled to the lip."""
    z = np.linspace(pot.base, pot.height, n)
    soil = np.trapezoid(np.pi * pot.r_in(z) ** 2, z)
    z = np.linspace(pot.base, pot.cup_h, n)
    water = np.trapezoid(np.pi * (pot.cup_ri(z) ** 2 - pot.r_out(z) ** 2).clip(0), z)
    return soil / 1e6, water / 1000


def render_shapes(pot, path, meshes=None):
    """Every pot shape side by side as a vertical section through the middle:
    wall, base, cup, soil and the water the moat holds, drawn straight from
    the Pot geometry (the wall solid, without the pattern). `meshes`
    ({shape: mesh}) adds a row of outside views above the sections."""
    import dataclasses

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    from .geometry import SHAPES

    names = list(SHAPES)
    rows = 2 if meshes else 1
    fig = plt.figure(figsize=(3.1 * len(names), 4.6 + 3.4 * (rows - 1)))
    lim = pot.r_max
    for col, name in enumerate(names):
        p = dataclasses.replace(pot, shape=name)
        if meshes:
            m = drawable(meshes[name], PREVIEW_FACES)
            ax3 = fig.add_subplot(rows, len(names), col + 1, projection="3d")
            draw_mesh(ax3, m.triangles, m.face_normals, p, 14, -60, pad=2, tight=True)
            ax3.set_title(name, fontsize=12, color=INK)
        ax = fig.add_subplot(rows, len(names), len(names) * (rows - 1) + col + 1)
        z = np.linspace(0, p.height, 300)
        zc = np.linspace(0, p.cup_h, 100)
        soil, water = volumes_ml(p)
        zs = z[z >= p.base]
        ax.fill_betweenx(zs, -p.r_in(zs), p.r_in(zs), color=SOIL_BROWN, lw=0)
        for side in (-1, 1):
            zw = zc[zc >= p.base]
            ax.fill_betweenx(zw, side * p.r_out(zw), side * p.cup_ri(zw), color=WATER_BLUE, lw=0)
            ax.fill_betweenx(z, side * p.r_in(z), side * p.r_out(z), color=POT_GREY, lw=0)
            ax.fill_betweenx(zc, side * p.cup_ri(zc), side * (p.cup_ri(zc) + p.cup_wall), color=POT_GREY, lw=0)
        ax.fill_between([-p.cup_ri(0) - p.cup_wall, p.cup_ri(0) + p.cup_wall], 0, p.base, color=POT_GREY, lw=0)
        ax.set_xlim(-lim, lim); ax.set_ylim(0, p.height + 2)
        ax.set_aspect("equal")
        label = f"{soil:.2f} L soil, {water:.0f} ml water"
        ax.set_title(label if meshes else f"{name}\n{label}", fontsize=10 if meshes else 11, color=INK)
        ax.tick_params(labelsize=8, colors=MUTED)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xlabel("mm", fontsize=8, color=MUTED)
        if col:
            ax.set_yticklabels([])
    fig.legend(handles=[Patch(color=POT_GREY, label="pot + cup (pattern not drawn)"),
                        Patch(color=SOIL_BROWN, label="soil"),
                        Patch(color=WATER_BLUE, label="water reserve, cup filled to the lip")],
               loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle(f"Pot shapes, shape=...  (H {pot.height:g} mm, vertical sections at the same scale)",
                 fontsize=14, color=INK)
    plt.tight_layout(rect=(0, 0.06 / rows, 1, 1)); plt.savefig(path, dpi=90); plt.close(fig)
    log.info("shape sheet saved %s", path)
