"""The optional wall mount: where the pads sit, that a screw really fits all
the way down both slots, and that the clips only touch their own side."""
import numpy as np
import pytest

from pots import Pot, generate, make_field
from pots.clips import STANDOFF, WEB, Clips
from pots.geometry import SHAPES
from pots.metrics import overhang_share

SIZES = [(50, 4.0), (65, 4.0), (130, 6.0)]


def envelope(pot):
    """Widest radius of the pot and cup: nothing but the pads may reach the wall."""
    z = np.linspace(0, pot.height, 601)
    return max(float(pot.r_out(z).max()), pot.cup_lip + pot.cup_wall)


def place(c, u, v, z):
    """(X, Y, Z) of points given in the pads' own (out, along, up) frame."""
    a = np.radians(c.at)
    u, v, z = np.broadcast_arrays(*(np.asarray(x, float) for x in (u, v, z)))
    return [x.astype(np.float32) for x in
            (u * np.cos(a) - v * np.sin(a), u * np.sin(a) + v * np.cos(a), z)]


def screw(c, u0, u1, dia, z0, steps=9, eps=0.05):
    """Points swept by a screw of diameter `dia` between depths u0 and u1 as the
    pot slides down onto it: its axis runs from z0 (seated) to z0 - slide."""
    u = np.linspace(u0 + eps, u1 - eps, 3)
    rho = np.array([0.0, 0.5, 1.0]) * (dia / 2 - eps)
    ang = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    z = z0 - np.linspace(0, c.slide, steps)
    U, R, A, Z = np.meshgrid(u, rho, ang, z, indexing="ij")
    return place(c, U, R * np.cos(A), Z + R * np.sin(A))


@pytest.mark.parametrize("h, wall", SIZES)
@pytest.mark.parametrize("shape", SHAPES)
def test_pads_are_thick_enough_and_stand_clear(h, wall, shape):
    """Both slots keep WEB of material to the pot's surface, and nothing but
    the two pads comes near the wall."""
    pot = Pot(height=h, wall=wall, shape=shape)
    c = Clips(pot)
    floor = c.x_wall - c.depth                        # the deepest the slot cuts
    high = np.linspace(c.slot_lo, c.top_z, 400)
    assert floor - float(pot.r_out(high).max()) >= WEB - 1e-9
    low = np.linspace(0, c.low_top, 400)
    assert floor - float((pot.cup_ri(low) + pot.cup_wall).max()) >= WEB - 1e-9
    assert c.low_top < pot.cup_h                      # the lower pad stays on the cup
    assert pot.base < c.slot_lo < c.mouth < c.seat < c.top_z
    assert c.x_wall >= envelope(pot) + STANDOFF - 1e-9
    assert c.spacing == pytest.approx(c.high_screw - c.low_screw)


def test_a_pot_too_short_for_the_screws_is_refused():
    with pytest.raises(ValueError, match="taller pot"):
        Clips(Pot(height=30, wall=4.0))


def test_the_slide_is_clamped_to_the_cup():
    """A short cup shortens the slide instead of running the lower pad off it."""
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
def test_a_screw_slides_down_both_slots_and_stays_trapped(h, wall):
    """Shank and head sweep an open path from the mouth to the seat; the lips
    stay solid beside the head, and the seat stays solid above the shank."""
    pot = Pot(height=h, wall=wall, skin=1.6, skin_frac=0.67)
    c = Clips(pot)
    f = make_field(pot, "voronoi", clips=c)
    x = c.x_wall
    for z0 in (c.high_screw, c.low_screw):
        assert f(*screw(c, x - c.lip, x, c.screw, z0)).min() >= 0        # shank
        assert f(*screw(c, x - c.depth, x - c.lip, c.head, z0)).min() >= 0   # head
        # the lips beside the head, between the slot and the recess
        v = np.linspace(c.slot_w / 2 + 0.3, c.head_w / 2 - 0.3, 4)
        v = np.concatenate([v, -v])
        u = np.linspace(x - c.lip + 0.2, x - 0.2, 3)
        zz = z0 + np.linspace(-c.slide, 0, 5)
        assert f(*place(c, *np.meshgrid(u, v, zz, indexing="ij"))).max() < 0
    # the seat the pot hangs from
    u = np.linspace(x - c.lip + 0.2, x - 0.2, 3)
    v = np.linspace(-c.screw / 2, c.screw / 2, 5)
    z = c.seat + np.linspace(0.2, 0.8, 3)
    assert f(*place(c, *np.meshgrid(u, v, z, indexing="ij"))).max() < 0


def test_the_chamfer_breaks_every_outside_edge_of_the_pads():
    """Each of the three edge families is cut, the slot is not touched, and
    a chamfer that would reach the slot is refused."""
    pot = Pot(height=65, wall=4.0)
    sharp, broken = Clips(pot, chamfer=0.0), Clips(pot, chamfer=1.5)
    f0, f1 = (make_field(pot, "voronoi", clips=c) for c in (sharp, broken))
    e, half = 0.2, broken.width / 2
    for top in (broken.top_z, broken.low_top):
        u = [broken.x_wall - e] * 3
        beside = (broken.head_w / 2 + half) / 2   # beside the slot, which is open at the top
        v = [beside, half - e, half - e]          # back-top, back-side, all three
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
    mesh, _ = generate(pot, "voronoi", voxel=0.6, faces=200_000, clips=Clips(pot))
    assert mesh.is_watertight and mesh.body_count == 1
