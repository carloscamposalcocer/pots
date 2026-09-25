import numpy as np
import pytest

from pots.geometry import H_REF, LIP, Pot


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
    assert (half.wall, half.rim, half.base, half.strut_out) == (ref.wall, ref.rim, ref.base, ref.strut_out)


@pytest.mark.parametrize("h", [30, 65, 130, 200])
def test_cup_lip_sits_outside_pot_top(h):
    pot = Pot(height=h)
    assert pot.cup_ri(pot.cup_h) == pytest.approx(pot.r_top + LIP)


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
