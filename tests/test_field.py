import numpy as np
import pytest

from pots import PATTERNS, Pot, make_field
from pots.metrics import face_window, wall_metrics


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
    m = wall_metrics(make_field(pot, name), pot, face_window(pot, 40, 30))
    assert m.outer_open > m.inner_open > 0.15
    assert m.outer_hole > m.inner_hole
    assert 2.0 < m.inner_hole < 5.0


@pytest.mark.parametrize("name", [n for n, p in PATTERNS.items() if p.kind == "2d"])
def test_2d_patterns_are_open_but_hold_soil(name):
    pot = Pot(height=65, wall=4.0)
    m = wall_metrics(make_field(pot, name), pot, face_window(pot, 30, 25))
    assert m.outer_open > 0.15 and m.inner_open > 0.15
    assert m.outer_hole < 5.5 and m.inner_hole < 5.5


def test_louvers_have_no_line_of_sight():
    pot = Pot()
    m = wall_metrics(make_field(pot, "louvers"), pot, face_window(pot, 40, 30))
    assert m.through < 0.01
    assert m.outer_open > 0.3
