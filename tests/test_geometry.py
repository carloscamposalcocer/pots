import numpy as np
import pytest

from pots.geometry import H_REF, LIP, SHAPES, Pot


def test_reference_size():
    pot = Pot()
    assert pot.k == 1.0
    assert (pot.r_bot, pot.r_top) == (56.0, 72.0)
    assert pot.cup_h == 28.0


def test_shape_scales_but_print_sizes_do_not():
    ref, half = Pot(), Pot(height=H_REF / 2)
    assert half.r_bot == pytest.approx(ref.r_bot / 2)
    assert half.r_top == pytest.approx(ref.r_top / 2)
    assert half.cup_h == pytest.approx(ref.cup_h / 2)
    assert (half.wall, half.rim, half.base) == (ref.wall, ref.rim, ref.base)


@pytest.mark.parametrize("h", [30, 65, 130, 200])
def test_cup_lip_sits_outside_pot_top(h):
    pot = Pot(height=h)
    assert pot.cup_ri(pot.cup_h) == pytest.approx(pot.r_top + LIP)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("h", [18, 50, 130, 200])
def test_every_shape_prints_and_keeps_the_patterns_valid(shape, h):
    pot = Pot(height=h, shape=shape)
    z = np.linspace(0, h, 2000)
    r = pot.r_out(z)
    # the radius range the patterns are designed for (r / r0 = 0.875 to 1.125)
    assert r.min() >= pot.r_min - 1e-9 and r.max() <= pot.r_wide + 1e-9
    assert np.abs(np.diff(r) / np.diff(z)).max() < np.tan(np.deg2rad(25))   # gentle walls


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("h", [18, 50, 130, 200])
def test_cup_follows_every_shape(shape, h):
    pot = Pot(height=h, shape=shape)
    above = np.linspace(pot.cup_h, h, 500)
    assert pot.cup_ri(pot.cup_h) >= pot.r_out(above).max() + LIP - 1e-9     # catches drips
    z = np.linspace(0, pot.cup_h, 400)
    assert (pot.cup_ri(z) - pot.r_out(z)).min() >= 0.9 * pot.cup_gap       # moat stays open
    assert np.abs(np.diff(pot.cup_ri(z)) / np.diff(z)).max() < 1.0         # support-free
    assert pot.r_max > pot.cup_ri(0) + pot.cup_wall and pot.r_max > pot.cup_lip + pot.cup_wall


def test_unknown_shape_is_rejected():
    with pytest.raises(ValueError, match="unknown shape"):
        Pot(shape="vase")


def test_counts_are_integers():
    assert Pot(height=65).count(76) == 38
    assert Pot(height=20).count(76) == 12
    assert isinstance(Pot(height=77.7).count(76), int)


def test_too_short_is_rejected():
    with pytest.raises(ValueError, match="at least"):
        Pot(height=10)


def test_radius_is_tapered_and_clamped():
    pot = Pot()
    assert pot.r_out(np.array([-5.0, 0.0, pot.height, pot.height + 5])).tolist() == [56, 56, 72, 72]
