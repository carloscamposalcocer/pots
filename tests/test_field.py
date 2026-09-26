import numpy as np
import pytest

from pots import PATTERNS, Pot, make_field
from pots.metrics import face_window, wall_metrics
from pots.patterns import load_params


def reference_params():
    """The pattern settings with the tapered struts at the reference 2.2 / 1.1 mm,
    so the metric tests check the pattern shapes, not the struts tuned in
    patterns.yaml."""
    P = load_params()
    for c in vars(P).values():
        if hasattr(c, "strut_in"):
            c.strut_in, c.strut_out = 2.2, 1.1
    return P


def ring(pot, frac, z, th):
    r = pot.r_out(z) - pot.wall * frac
    return (r * np.cos(th)).astype(np.float32), (r * np.sin(th)).astype(np.float32), \
        np.full_like(th, z, dtype=np.float32)


@pytest.mark.parametrize("name", PATTERNS)
@pytest.mark.parametrize("h", [65, 130])
def test_seamless_around_the_pot(name, h):
    """The field just below theta = +pi matches the field just above -pi."""
    pot = Pot(height=h, wall=4.0)
    f = make_field(pot, name)
    eps = 1e-5
    for z in np.linspace(pot.base + 2, pot.height - pot.rim - 1, 25):
        for frac in (0.02, 0.5, 0.98):
            a = f(*ring(pot, frac, z, np.array([np.pi - eps])))
            b = f(*ring(pot, frac, z, np.array([-np.pi + eps])))
            assert a == pytest.approx(b, abs=1e-2)


@pytest.mark.parametrize("name", PATTERNS)
def test_solid_base_rim_and_cup(name):
    pot = Pot(height=65, wall=4.0)
    f = make_field(pot, name)
    th = np.linspace(-np.pi, np.pi, 720, endpoint=False)
    assert (f(*ring(pot, 0.5, pot.base / 2, th)) < 0).all()                 # base
    assert (f(*ring(pot, 0.5, pot.height - pot.rim / 2, th)) < 0).all()     # rim
    z = pot.cup_h / 2
    r = pot.cup_ri(z) + pot.cup_wall / 2
    x, y = (r * np.cos(th)).astype(np.float32), (r * np.sin(th)).astype(np.float32)
    assert (f(x, y, np.full_like(x, z)) < 0).all()                           # cup wall
    assert (f(np.zeros(1, np.float32), np.zeros(1, np.float32), np.float32([pot.height / 2])) > 0).all()


@pytest.mark.parametrize("name", PATTERNS)
def test_flat_bottom(name):
    """Nothing sits below z = 0 (the blends used to bulge 0.25 mm under the
    wall and cup, lifting the rest of the base off the bed), and the base is
    solid right down to it across the whole footprint."""
    pot = Pot(height=65, wall=4.0)
    f = make_field(pot, name)
    r = np.linspace(0, pot.cup_ri(0) + pot.cup_wall - 1.0, 400).astype(np.float32)
    y = np.zeros_like(r)
    assert (f(r, y, np.full_like(r, -0.05)) > 0).all()
    assert (f(r, y, np.full_like(r, 0.05)) < 0).all()


@pytest.mark.parametrize("name", ["voronoi_taper", "hex_taper", "drops", "lattice_taper"])
def test_tapered_patterns_open_outward(name):
    pot = Pot()
    m = wall_metrics(make_field(pot, name, reference_params()), pot, face_window(pot, 40, 30))
    assert m.outer_open > m.inner_open > 0.15
    assert m.outer_hole > m.inner_hole
    assert 2.0 < m.inner_hole < 5.0


@pytest.mark.parametrize("name", [n for n, p in PATTERNS.items() if p.kind == "2d"])
def test_2d_patterns_are_open_but_hold_soil(name):
    pot = Pot(height=65, wall=4.0)
    m = wall_metrics(make_field(pot, name, reference_params()), pot, face_window(pot, 30, 25))
    assert m.outer_open > 0.15 and m.inner_open > 0.15
    assert m.outer_hole < 5.5 and m.inner_hole < 5.5


def test_louvers_have_no_line_of_sight():
    pot = Pot()
    m = wall_metrics(make_field(pot, "louvers"), pot, face_window(pot, 40, 30))
    assert m.through < 0.01
    assert m.outer_open > 0.3


@pytest.mark.parametrize("name", PATTERNS)
def test_skin_closes_the_soil_side_only_above_skin_z(name):
    pot = Pot(height=65, wall=4.0, skin=1.6, skin_frac=0.67)
    f = make_field(pot, name)
    th = np.linspace(-np.pi, np.pi, 2000, endpoint=False)
    top = np.linspace(pot.skin_z + 0.5, pot.height - pot.rim - 0.5, 12)
    low = np.linspace(pot.base + 2, pot.skin_z - 1, 12)
    inner_top = np.concatenate([f(*ring(pot, 0.98, z, th)) for z in top])
    outer_top = np.concatenate([f(*ring(pot, 0.02, z, th)) for z in top])
    inner_low = np.concatenate([f(*ring(pot, 0.98, z, th)) for z in low])
    assert (inner_top < 0).all()          # closed skin on the soil side
    assert (outer_top > 0).any()          # the pattern still shows outside
    assert (inner_low > 0).any()          # open below the skin
