import numpy as np
import pytest

from pots import PATTERNS, Pot, make_field
from pots.metrics import wall_metrics

FAST = [n for n in PATTERNS if n != "coral"]      # coral needs its slow texture


def ring(pot, frac, z, th):
    r = pot.r_out(z) - pot.wall * frac
    return (r * np.cos(th)).astype(np.float32), (r * np.sin(th)).astype(np.float32), \
        np.full_like(th, z, dtype=np.float32)


@pytest.mark.parametrize("name", FAST)
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


@pytest.mark.parametrize("name", FAST)
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


def test_voronoi_taper_opens_outward():
    pot = Pot()
    m = wall_metrics(make_field(pot, "voronoi_taper"), pot)
    assert m.outer_open > m.inner_open > 0.15
    assert m.outer_hole > m.inner_hole
    assert 2.0 < m.inner_hole < 5.0
