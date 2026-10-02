"""PNG preview: outside view, cut-away, vertical section and both wall faces.

The 3D views are drawn by a small numpy z-buffer renderer (`rasterize` +
`shade`) rather than matplotlib's mplot3d, which shades every triangle flat
and sorts them by centre depth (no depth buffer): the pot looked faceted and
long thin triangles showed through as streaks. The rendered image is then
placed on an ordinary 2D axis with imshow.
"""
import logging
import time

import numpy as np

from .mesh import decimate
from .metrics import INNER, OUTER, face_window, sample_face

log = logging.getLogger(__name__)

PREVIEW_FACES = 1_500_000        # meshes above this are decimated before drawing
CREASE = np.cos(np.radians(25))  # corners whose smooth normal strays further stay flat (sharp edges)
SS = 2                           # supersampling factor (anti-aliasing)
CHUNK = 4_000_000                # pixel fragments rasterized at once (roughly)

# terracotta palette
CLAY = np.array([0.80, 0.44, 0.29])          # rendered pot (sRGB, before lighting)
CLAY_HEX, PAPER = "#c0643c", "#f7f0e6"       # flat pot colour, holes / background
SHADOW = np.array([0.36, 0.24, 0.18])
WATER_BLUE, SOIL_BROWN, INK, MUTED = "#bcd7e8", "#e8d9c4", "#3b2a22", "#8a7a70"


def view_basis(elev, azim):
    """(eye, right, up) unit vectors of matplotlib's view_init(elev, azim)."""
    e, a = np.radians(elev), np.radians(azim)
    eye = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    right = np.array([-np.sin(a), np.cos(a), 0.0])
    return eye, right, np.cross(eye, right)


def corner_normals(mesh):
    """(F, 3, 3) normal at each face corner: the vertex normal (smooth shading),
    or the face normal where the two differ by more than CREASE (sharp edges)."""
    fn = mesh.face_normals
    vn = mesh.vertex_normals[mesh.faces]
    sharp = np.einsum("fkc,fc->fk", vn, fn) < CREASE
    return np.where(sharp[..., None], fn[:, None, :], vn)


def frame(box, scale):
    """Image size (W, H) at SS x resolution for a view box in mm."""
    s = scale * SS
    return int(np.ceil((box[1] - box[0]) * s)), int(np.ceil((box[3] - box[2]) * s))


def rasterize(mesh, elev, azim, box, scale):
    """Z-buffer render of `mesh` seen from (elev, azim), orthographic. `box` =
    (u0, u1, v0, v1) in mm on the view plane, `scale` px per mm. Returns
    (depth, normal) images at SS x that resolution; depth is -inf off the mesh."""
    eye, right, up = view_basis(elev, azim)
    u0, u1, v0, v1 = box
    s = scale * SS
    W, H = frame(box, scale)
    V = mesh.vertices
    x, y, d = (V @ right - u0) * s - 0.5, (v1 - V @ up) * s - 0.5, V @ eye   # pixel centres at integers
    F = np.asarray(mesh.faces)
    cn = corner_normals(mesh)
    tx, ty, td = x[F], y[F], d[F]
    area = (tx[:, 1] - tx[:, 0]) * (ty[:, 2] - ty[:, 0]) - (tx[:, 2] - tx[:, 0]) * (ty[:, 1] - ty[:, 0])
    y0 = np.clip(np.ceil(ty.min(1)), 0, H).astype(int)
    y1 = np.clip(np.floor(ty.max(1)) + 1, 0, H).astype(int)
    tri = np.flatnonzero((np.abs(area) > 1e-12) & (y1 > y0) & (tx.max(1) >= 0) & (tx.min(1) < W))

    zbuf = np.full(W * H, -np.inf)
    nbuf = np.zeros((W * H, 3))
    rows = np.cumsum(y1[tri] - y0[tri])
    start = 0
    while start < len(tri):                 # chunks of about CHUNK / 8 scanline spans
        stop = max(start + 1, np.searchsorted(rows, (rows[start - 1] if start else 0) + CHUNK // 8))
        t = tri[start:stop]
        start = stop
        # one entry per (triangle, pixel row) it covers
        nr = y1[t] - y0[t]
        t = np.repeat(t, nr)
        py = y0[t] + np.arange(len(t)) - np.repeat(np.cumsum(nr) - nr, nr)
        X, Y, A = tx[t], ty[t], area[t]
        # barycentrics along the row are linear in x: w = a + b x; w >= 0 bounds the span
        b0, a0 = (Y[:, 1] - Y[:, 2]) / A, (X[:, 1] * (Y[:, 2] - py) - X[:, 2] * (Y[:, 1] - py)) / A
        b1, a1 = (Y[:, 2] - Y[:, 0]) / A, (X[:, 2] * (Y[:, 0] - py) - X[:, 0] * (Y[:, 2] - py)) / A
        b2, a2 = -b0 - b1, 1 - a0 - a1
        lo = np.zeros(len(t)); hi = np.full(len(t), W - 1.0)
        for a, b in ((a0, b0), (a1, b1), (a2, b2)):
            with np.errstate(divide="ignore", invalid="ignore"):
                r = -a / b
            lo = np.where(b > 0, np.maximum(lo, r - 1e-7), lo)
            hi = np.where(b < 0, np.minimum(hi, r + 1e-7), hi)
            hi = np.where((b == 0) & (a < 0), -1, hi)
        xs, xe = np.ceil(lo).astype(int), np.floor(hi).astype(int) + 1
        n = (xe - xs).clip(0)
        k = np.repeat(np.arange(len(t)), n)
        if not len(k):
            continue
        px = xs[k] + np.arange(len(k)) - np.repeat(np.cumsum(n) - n, n)
        w0, w1 = a0[k] + b0[k] * px, a1[k] + b1[k] * px
        w = np.stack([w0, w1, 1 - w0 - w1], 1)
        tt, pix = t[k], py[k] * W + px
        dep = np.einsum("nk,nk->n", w, td[tt])
        o = np.lexsort((-dep, pix))                               # nearest first per pixel
        o = o[np.r_[True, pix[o][1:] != pix[o][:-1]]]
        o = o[dep[o] > zbuf[pix[o]]]
        zbuf[pix[o]] = dep[o]
        nbuf[pix[o]] = np.einsum("nk,nkc->nc", w[o], cn[tt[o]])
    return zbuf.reshape(H, W), nbuf.reshape(H, W, 3)


def ground_shadow(pot, elev, box, scale):
    """Alpha (H, W) of a soft contact shadow under the pot: its footprint disc
    on the ground plane z = 0, seen from `elev`, blurred."""
    from scipy.ndimage import gaussian_filter

    W, H = frame(box, scale)
    s = scale * SS
    u = box[0] + (np.arange(W) + 0.5) / s
    v = box[3] - (np.arange(H) + 0.5) / s
    R = pot.r_wide
    # a ground point (x, y, 0) sits at u = its sideways coordinate, v = -sin(elev) * its depth
    disc = u[None, :] ** 2 + ((v[:, None] + 0.04 * R) / np.sin(np.radians(elev))) ** 2 < (0.98 * R) ** 2
    return 0.4 * gaussian_filter(disc.astype(float), 1.8 * s)


def shade(depth, normal, elev, azim, scale, shadow=None, color=CLAY):
    """RGBA image of a rasterized mesh, downsampled by SS: soft key + fill
    light, a little specular, and screen-space ambient occlusion so the holes
    read as holes. `shadow` is the alpha of a ground shadow drawn behind it."""
    from scipy.ndimage import gaussian_filter

    eye, right, up = view_basis(elev, azim)
    hit = np.isfinite(depth)
    n = normal / np.linalg.norm(normal, axis=-1, keepdims=True).clip(1e-9)
    n = np.where((n @ eye < 0)[..., None], -n, n)                   # two-sided (cut-away)
    key = -0.45 * right + 0.75 * up + 0.85 * eye; key /= np.linalg.norm(key)
    fill = 0.8 * right - 0.1 * up + 0.6 * eye; fill /= np.linalg.norm(fill)
    half = key + eye; half /= np.linalg.norm(half)
    light = 0.30 + 0.62 * (n @ key).clip(0) + 0.20 * (n @ fill).clip(0)
    spec = 0.12 * (n @ half).clip(0) ** 30

    # ambient occlusion: darker where the surface lies behind its surroundings
    s = scale * SS
    D = np.where(hit, depth, depth[hit].min() if hit.any() else 0.0)
    ao = sum(w * ((gaussian_filter(D, sigma * s) - D) / mm).clip(0, 1)
             for sigma, mm, w in ((0.6, 1.0, 0.6), (2.5, 4.0, 0.4)))
    light = light * (1 - 0.7 * ao)

    rgb = (color * light[..., None] + spec[..., None]).clip(0, 1)
    a = hit.astype(float)
    pre = rgb * a[..., None]                                           # premultiplied
    if shadow is not None:
        sa = shadow * (1 - a)
        pre, a = pre + SHADOW * sa[..., None], a + sa
    img = np.concatenate([pre, a[..., None]], -1)
    h, w = img.shape[0] // SS * SS, img.shape[1] // SS * SS
    img = img[:h, :w].reshape(h // SS, SS, w // SS, SS, 4).mean((1, 3))
    img[..., :3] /= img[..., 3:].clip(1e-9)
    return img.clip(0, 1)


def draw_mesh(ax, mesh, pot, elev, azim, pad=4, tight=False, px=900, shadow=True):
    """Render `mesh` onto a 2D axis: z-buffered, smooth-shaded terracotta.
    `tight` fits the frame to the pot; `px` is the rendered width in pixels."""
    t0 = time.perf_counter()
    if len(mesh.faces) > PREVIEW_FACES:
        mesh = decimate(mesh, PREVIEW_FACES)
    e = np.radians(elev)
    R = pot.r_wide + pad
    zpad = 2 if tight else 10
    box = (-R, R, -R * np.sin(e) - zpad * np.cos(e) - (4 if shadow else 0), pot.height * np.cos(e) + R * np.sin(e) + zpad)
    scale = px / (2 * R)
    depth, normal = rasterize(mesh, elev, azim, box, scale)
    sh = ground_shadow(pot, elev, box, scale) if shadow else None
    ax.imshow(shade(depth, normal, elev, azim, scale, sh), extent=box, interpolation="antialiased")
    ax.set_xlim(box[0], box[1]); ax.set_ylim(box[2], box[3]); ax.set_aspect("equal")
    ax.set_axis_off()
    log.debug("rendered %d faces in %.1f s", len(mesh.faces), time.perf_counter() - t0)


def face_image(open_):
    """RGB swatch of a wall face: holes in PAPER, material in CLAY_HEX."""
    from matplotlib.colors import to_rgb
    img = np.empty(open_.shape + (3,))
    img[open_], img[~open_] = to_rgb(PAPER), to_rgb(CLAY_HEX)
    return img


def style(ax, title, **kw):
    ax.set_title(title, color=INK, **kw)
    for sp in ax.spines.values():
        sp.set_color(MUTED)


def render(mesh, field, pot, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    t0 = time.perf_counter()
    fig = plt.figure(figsize=(18, 11), facecolor="white")
    ax = fig.add_subplot(2, 3, 1)
    draw_mesh(ax, mesh, pot, 16, -60); style(ax, "Outside")
    back = mesh.submesh([np.flatnonzero(mesh.triangles_center[:, 1] > 0)], append=True)  # cut-away
    ax = fig.add_subplot(2, 3, 2)
    draw_mesh(ax, back, pot, 15, -90, shadow=False); style(ax, "Cut-away")

    # vertical section straight from the field
    ax = fig.add_subplot(2, 3, 3)
    s = np.arange(-pot.r_max, pot.r_max, 0.15); z = np.arange(-1, pot.height + 1, 0.15)
    S, Z = np.meshgrid(s, z)
    F = field(S.astype(np.float32), np.full(S.shape, 0.7, np.float32), Z.astype(np.float32)) < 0
    ax.imshow(F, origin="lower", extent=[s[0], s[-1], z[0], z[-1]], cmap=ListedColormap([PAPER, CLAY_HEX]))
    ax.set_aspect("equal"); ax.tick_params(colors=MUTED); style(ax, "Vertical section (mm)")

    # both wall faces at true scale
    win = face_window(pot)
    ww, wz = win.arc[-1] + win.px, win.z[-1] + win.px - win.z[0]
    for k, (frac, lab) in enumerate(((OUTER, "Outside face"), (INNER, "Soil-side face"))):
        open_ = sample_face(field, pot, frac, win)
        ax = fig.add_subplot(2, 3, 4 + k)
        ax.imshow(face_image(open_), origin="lower", extent=win.extent); ax.set_xticks([]); ax.set_yticks([])
        style(ax, f"{lab}: {open_.mean() * 100:.0f}% open ({ww:.0f} x {wz:.0f} mm, true scale)")
    fig.suptitle(title, fontsize=15, color=INK)
    plt.tight_layout(); plt.savefig(path, dpi=85); plt.close(fig)
    log.info("preview saved %s in %.1f s", path, time.perf_counter() - t0)


def render_card(mesh, field, pot, path, title):
    """Small gallery image: outside view + a 40 x 30 mm true-scale swatch of the outer face."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 4.2), facecolor="white")
    draw_mesh(fig.add_subplot(1, 2, 1), mesh, pot, 14, -60, pad=4, tight=True, px=700)
    win = face_window(pot, width=40.0, height=30.0)
    open_ = sample_face(field, pot, OUTER, win)
    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(face_image(open_), origin="lower", extent=win.extent); ax.set_xticks([]); ax.set_yticks([])
    style(ax, f"outside face, {open_.mean() * 100:.0f}% open (40 x 30 mm)", fontsize=10)
    fig.suptitle(title, fontsize=14, color=INK)
    plt.tight_layout(); plt.savefig(path, dpi=100); plt.close(fig)
    log.info("card saved %s", path)


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
    fig = plt.figure(figsize=(3.1 * len(names), 4.6 + 3.4 * (rows - 1)), facecolor="white")
    lim = pot.r_max
    for col, name in enumerate(names):
        p = dataclasses.replace(pot, shape=name)
        if meshes:
            ax3 = fig.add_subplot(rows, len(names), col + 1)
            draw_mesh(ax3, meshes[name], p, 14, -60, pad=5, tight=True, px=500)
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
            ax.fill_betweenx(z, side * p.r_in(z), side * p.r_out(z), color=CLAY_HEX, lw=0)
            ax.fill_betweenx(zc, side * p.cup_ri(zc), side * (p.cup_ri(zc) + p.cup_wall), color=CLAY_HEX, lw=0)
        ax.fill_between([-p.cup_ri(0) - p.cup_wall, p.cup_ri(0) + p.cup_wall], 0, p.base, color=CLAY_HEX, lw=0)
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
    fig.legend(handles=[Patch(color=CLAY_HEX, label="pot + cup (pattern not drawn)"),
                        Patch(color=SOIL_BROWN, label="soil"),
                        Patch(color=WATER_BLUE, label="water reserve, cup filled to the lip")],
               loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle(f"Pot shapes, shape=...  (H {pot.height:g} mm, vertical sections at the same scale)",
                 fontsize=14, color=INK)
    plt.tight_layout(rect=(0, 0.06 / rows, 1, 1)); plt.savefig(path, dpi=90); plt.close(fig)
    log.info("shape sheet saved %s", path)
