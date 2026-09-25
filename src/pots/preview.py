"""PNG preview: outside view, cut-away, vertical section and both wall faces."""
import logging
import time

import numpy as np
import trimesh

from .mesh import decimate
from .metrics import INNER, OUTER, face_window, sample_face

log = logging.getLogger(__name__)

PREVIEW_FACES = 120_000
LIGHT = np.array([0.4, -0.6, 0.7]) / np.linalg.norm([0.4, -0.6, 0.7])


def draw_mesh(ax, tris, normals, pot, elev, azim, pad=4, tight=False):
    """Shaded triangles on a 3D axis; `tight` fits the box to the pot instead of a cube."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    sh = np.clip(normals @ LIGHT, 0, 1) * 0.75 + 0.2
    col = np.stack([0.30 * sh + 0.04, 0.52 * sh + 0.04, 0.36 * sh + 0.04, np.ones_like(sh)], 1).clip(0, 1)
    ax.add_collection3d(Poly3DCollection(tris, facecolors=col, edgecolors="none"))
    lim = pot.r_top + pad
    zpad = 2 if tight else 10
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-zpad, pot.height + zpad)
    ax.set_box_aspect((1, 1, (pot.height + 2 * zpad) / (2 * lim)) if tight else (1, 1, 1))
    ax.view_init(elev, azim); ax.set_axis_off()


def render(mesh, field, pot, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t0 = time.perf_counter()
    m = decimate(mesh, PREVIEW_FACES)          # light mesh for drawing

    def draw(ax, tris, normals, elev, azim):
        draw_mesh(ax, tris, normals, pot, elev, azim)

    fig = plt.figure(figsize=(18, 11))
    ax = fig.add_subplot(2, 3, 1, projection="3d")
    draw(ax, m.triangles, m.face_normals, 16, -60); ax.set_title("Outside")
    keep = m.triangles_center[:, 1] > 0                      # back half only -> cut-away
    ax = fig.add_subplot(2, 3, 2, projection="3d")
    draw(ax, m.triangles[keep], m.face_normals[keep], 15, -90); ax.set_title("Cut-away")

    # vertical section straight from the field
    ax = fig.add_subplot(2, 3, 3)
    s = np.arange(-pot.r_top - 4, pot.r_top + 4, 0.15); z = np.arange(-1, pot.height + 1, 0.15)
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

    m = decimate(mesh, 2 * PREVIEW_FACES)
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
