"""The optional wall mount: where the pads sit, that the rail's keys really
fit all the way down both sockets, that the rail prints in one piece, and
that the clips only touch their own side of the pot."""
import numpy as np
import pytest

from pots import Pot, generate, make_field
from pots.clips import GAP, LEAD, SLOP, WEB, Clips
from pots.geometry import SHAPES
from pots.metrics import overhang_share

SIZES = [(50, 4.0), (65, 4.0), (130, 6.0)]


def envelope(pot):
    """Widest radius of the pot and cup: nothing but the rail may reach past it."""
    z = np.linspace(0, pot.height, 601)
    return max(float(pot.r_out(z).max()), pot.cup_lip + pot.cup_wall)


def place(c, u, v, z):
    """(X, Y, Z) of points given in the pads' own (out, along, up) frame."""
    a = np.radians(c.at)
    u, v, z = np.broadcast_arrays(*(np.asarray(x, float) for x in (u, v, z)))
    return [x.astype(np.float32) for x in
            (u * np.cos(a) - v * np.sin(a), u * np.sin(a) + v * np.cos(a), z)]


def key_points(c, low, eps=0.15, n=7):
    """A grid of points strictly inside one of the rail's keys, in the rail's
    own frame, over the part of it that goes into a socket."""
    lo, hi = c.low_key if low else c.high_key
    lo, hi = lo - c.rail_lo, hi - c.rail_lo
    if low:
        hi -= LEAD                                   # the lead-in on its tip
    out = []
    for z in np.linspace(c.plate + eps, c.plate + c.grip - eps, n):
        t = z - c.plate
        lean = 0.0 if low else t                     # the upper key leans with its socket
        for y in np.linspace(lo + lean + eps, hi + lean - eps, n):
            for x in np.linspace(-(c.key / 2 + t - eps), c.key / 2 + t - eps, n):
                out.append((x, y, z))
    return np.array(out).T


def seated(c, f, pts, offsets=9):
    """The field at `pts` (rail frame) all the way down the slide: the pot
    starts `slide` mm high, so its own z runs from z - slide to z."""
    x, y, z = c.world(*pts)
    return min(f(*(a.astype(np.float32) for a in (x, y, z - d))).min()
               for d in np.linspace(0, c.slide, 13))


@pytest.mark.parametrize("h, wall", SIZES)
@pytest.mark.parametrize("shape", SHAPES)
def test_pads_are_thick_enough_and_stand_clear(h, wall, shape):
    """Both sockets keep WEB of material to the pot's surface, nothing but the
    rail comes near the wall, and the rail stays hidden behind the pot."""
    pot = Pot(height=h, wall=wall, shape=shape)
    c = Clips(pot)
    floor = c.x_wall - c.depth                        # the deepest the socket cuts
    high = np.linspace(c.slot_lo, c.top_z, 400)
    assert floor - float(pot.r_out(high).max()) >= WEB - 1e-9
    low = np.linspace(0, c.low_top, 400)
    assert floor - float((pot.cup_ri(low) + pot.cup_wall).max()) >= WEB - 1e-9
    assert c.low_top < pot.cup_h                      # the lower pad stays on the cup
    assert pot.base < c.slot_lo < c.mouth < c.seat < c.top_z
    assert c.x_wall >= envelope(pot) + GAP - 1e-9
    assert 0 < c.rail_lo and c.rail_hi < c.top_z      # behind the cup and the rim
    assert c.strip <= c.width                         # and behind the pads
    assert c.spacing == pytest.approx(c.screw_hi - c.screw_lo)


def test_a_pot_too_short_for_the_mount_is_refused():
    with pytest.raises(ValueError, match="taller pot"):
        Clips(Pot(height=30, wall=4.0))


def test_the_slide_is_clamped_to_the_cup():
    """A short cup shortens the slide instead of running the lower key off it."""
    pot = Pot(height=50, wall=4.0)
    small, big = Clips(pot), Clips(Pot(height=130))
    assert small.slide < small.travel and small.low_top < pot.cup_h
    assert big.slide == big.travel


def test_bad_clip_sizes():
    pot = Pot(height=65, wall=4.0)
    with pytest.raises(ValueError, match="wider than"):
        Clips(pot, screw=5.0, head=5.0)
    with pytest.raises(ValueError, match="width must be at least"):
        Clips(pot, width=6.0)
    with pytest.raises(ValueError, match="width must be at most"):
        Clips(pot, width=60.0)


@pytest.mark.parametrize("h, wall", SIZES)
def test_a_key_slides_down_both_sockets(h, wall):
    """Both keys sweep an open path all the way down, and the pot ends up
    sitting on the end of the upper one."""
    pot = Pot(height=h, wall=wall, skin=1.6, skin_frac=0.67)
    c = Clips(pot)
    f = make_field(pot, "voronoi", clips=c)
    for low in (False, True):
        assert seated(c, f, key_points(c, low)) >= 0
    # the chisel the pot hangs from: material just past the upper key's end
    X, Y, Z = key_points(c, False)
    end = Y > (c.high_key[1] - c.rail_lo) - 1.0 + (Z - c.plate)
    x, y, z = c.world(X[end], Y[end] + 1.0, Z[end])
    assert f(*(a.astype(np.float32) for a in (x, y, z))).max() < 0


@pytest.mark.parametrize("h, wall", SIZES)
def test_the_rail_clears_the_pot_all_the_way_down(h, wall):
    """Nothing on the rail but its keys and its face ever touches the pot."""
    pot = Pot(height=h, wall=wall, skin=1.6, skin_frac=0.67)
    c = Clips(pot)
    f = make_field(pot, "voronoi", clips=c)
    rail = c.rail()
    v = np.vstack([rail.vertices, rail.triangles.mean(axis=1)])
    assert seated(c, f, v.T) >= -0.01        # 0 where its face slides on the pads


def test_the_rail_prints_in_one_piece():
    """One watertight body, no overhang past 45 degrees in its print
    orientation, and a dovetail the socket's mouth cannot let go of."""
    c = Clips(Pot(height=100))
    rail = c.rail()
    assert rail.is_watertight and rail.body_count == 1
    assert overhang_share(rail) == 0
    size = np.ptp(rail.bounds, axis=0)
    assert rail.bounds[0, 2] == 0                     # the wall face is on the bed
    assert size == pytest.approx([c.strip, c.rail_len, c.plate + c.grip])
    face = rail.vertices[rail.vertices[:, 2] > c.plate + c.grip - 1e-6]
    assert np.ptp(face[:, 0]) == pytest.approx(c.key + 2 * c.grip)
    assert np.ptp(face[:, 0]) > c.slot_w              # it cannot pull out
    # the screw heads sink in flush, the upper hole round and the lower a slot
    for z, want in ((c.screw_hi, c.sink), (c.screw_lo, c.sink + 2 * SLOP)):
        y, v = z - c.rail_lo, rail.vertices
        rim = v[(np.abs(v[:, 2] - c.plate) < 1e-6) & (np.abs(v[:, 1] - y) < want / 2 + 0.3)]
        assert np.ptp(rim[:, 1]) == pytest.approx(want, abs=0.05)
        assert np.ptp(rim[:, 0]) == pytest.approx(c.sink, abs=0.05)


def test_the_chamfer_breaks_every_outside_edge_of_the_pads():
    """Each of the three edge families is cut, the socket is not touched, and
    a chamfer that would reach the socket is refused."""
    pot = Pot(height=65, wall=4.0)
    sharp, broken = Clips(pot, chamfer=0.0), Clips(pot, chamfer=1.5)
    f0, f1 = (make_field(pot, "voronoi", clips=c) for c in (sharp, broken))
    e, half = 0.2, broken.width / 2
    for top in (broken.top_z, broken.low_top):
        u = [broken.x_wall - e] * 3
        beside = (broken.floor_w / 2 + half) / 2   # beside the socket, open at the top
        v = [beside, half - e, half - e]           # back-top, back-side, all three
        z = [top - e, top - 4, top - e]
        p = place(broken, u, v, z)
        assert f0(*p).max() < 0 and f1(*p).min() >= 0
    with pytest.raises(ValueError, match="chamfer must be between"):
        Clips(pot, chamfer=broken.max_chamfer + 0.1)


def test_the_clips_only_add_to_their_own_side():
    """The pads are half-space slabs; they must not claim material across the
    pot (which they would if `u` were left unbounded)."""
    pot = Pot(height=65, wall=4.0, skin=1.6, skin_frac=0.67)
    c = Clips(pot)
    plain, clipped = (make_field(pot, "voronoi", clips=k) for k in (None, c))
    rng = np.random.default_rng(0)
    th = rng.uniform(-np.pi, np.pi, 40_000)
    z = rng.uniform(0, pot.height, 40_000)
    r = rng.uniform(0.5 * pot.r_bot, c.r_max, 40_000)
    p = [a.astype(np.float32) for a in (r * np.cos(th), r * np.sin(th), z)]
    u, v = c.local(p[0], p[1])
    same = (u < 0) | (np.abs(v) > c.width / 2 + 0.01)
    assert np.array_equal(plain(*p)[same] < 0, clipped(*p)[same] < 0)
    assert (clipped(*p) < 0).sum() > (plain(*p) < 0).sum()        # and they do add material


@pytest.mark.parametrize("shape", ["tapered", "barrel"])
def test_a_clipped_pot_builds_clean(shape):
    """Watertight, one body (no pocket sealed between a pad and the skin), no
    new overhangs, and the exact mesh matches the field."""
    pot = Pot(height=65, wall=4.0, shape=shape, skin=1.6, skin_frac=0.67)
    c = Clips(pot)
    mesh, _ = generate(pot, "lattice", voxel=0.6, faces=200_000, clips=c)
    assert mesh.is_watertight and mesh.body_count == 1
    assert overhang_share(mesh) < 0.02
    assert mesh.bounds[0, 0] == pytest.approx(-c.x_wall, abs=0.05)     # at=180: the pads face -x
    sdf, _ = generate(pot, "lattice", voxel=0.6, faces=200_000, clips=c, mesher="sdf")
    assert sdf.is_watertight and sdf.body_count == 1
    assert mesh.volume == pytest.approx(sdf.volume, rel=0.01)


@pytest.mark.slow
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("h, wall", SIZES[:2])
def test_every_shape_and_size_builds_with_clips(shape, h, wall):
    pot = Pot(height=h, wall=wall, shape=shape, skin=1.6, skin_frac=0.67)
    clips = Clips(pot)
    mesh, _ = generate(pot, "voronoi", voxel=0.6, faces=200_000, clips=clips)
    assert mesh.is_watertight and mesh.body_count == 1
    rail = clips.rail()
    assert rail.is_watertight and rail.body_count == 1
