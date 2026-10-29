"""Wall holder: a printable part that fits the unchanged pot, and a way to
get the pot in and out of it."""
import numpy as np
import pytest

from pots.cli import holder_from_config, load_config, pot_from_config
from pots.geometry import SHAPES, Pot
from pots.holder import Holder
from pots.metrics import overhang_share


def plain(pot):
    """The pot without its pattern (the holder never reaches the patterned band)."""
    from manifold3d import Manifold

    from pots.solid import _profile, _segments
    return Manifold.revolve(_profile(pot), _segments(pot.r_max))


def pot_for(h, shape):
    return Pot(height=h, shape=shape, wall=6 if h > 60 else 4, skin=1.6, skin_frac=0.67)


def place(holder, pot_m, lift, angle, slide, away):
    """The pot moved out of the holder: lifted straight up, its bottom swung
    out about the finger's tip by `angle` degrees, slid down its own (tilted)
    axis and moved away from the wall."""
    a = np.radians(angle)
    f, tip = holder.finger, holder.tip
    m = pot_m.translate([-f, 0, lift - tip]).rotate([0, angle, 0]).translate([f, 0, tip])
    return m.translate([-slide * np.sin(a) - away, 0, -slide * np.cos(a)])


def removal(holder, angle, frac, n=8):
    """Poses taking the pot out: lift `frac` of the room above the rim, swing,
    slide until the rim is clear of the finger, move away."""
    lift, slide = holder.lift * frac, holder.drop + holder.lift + 0.5
    return ([(lift * t, 0, 0, 0) for t in np.linspace(0, 1, 3)]
            + [(lift, angle * t, 0, 0) for t in np.linspace(0, 1, n)[1:]]
            + [(lift, angle, slide * t, 0) for t in np.linspace(0, 1, n)[1:]]
            + [(lift, angle, slide, 20 * t) for t in np.linspace(0, 1, 4)[1:]])


def swing_needed(holder, angles=np.arange(2, 21, 2.0)):
    """The smallest swing (degrees) that gets the pot out without touching the holder."""
    hm, pm = holder.manifold(), plain(holder.pot)
    for angle in angles:
        for frac in (0.3, 0.6, 0.9):
            if all((hm ^ place(holder, pm, *p)).volume() < 0.01 for p in removal(holder, angle, frac)):
                return angle
    return None


def check(holder):
    m = holder.mesh()
    assert m.is_watertight and m.body_count == 1
    assert overhang_share(m) == 0
    assert np.allclose(m.bounds[0], 0) and m.bounds[1][2] == pytest.approx(holder.width)
    assert (holder.manifold() ^ plain(holder.pot)).volume() < 1e-3      # clear of the pot


@pytest.mark.parametrize("h", [50, 100])
def test_holder_fits(h):
    check(Holder(pot_for(h, "tapered")))


@pytest.mark.slow
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("h", [50, 65, 130])
def test_holder_fits_every_shape(h, shape):
    holder = Holder(pot_for(h, shape))
    check(holder)
    assert swing_needed(holder) is not None


def test_holder_holds():
    """The ledge is under the back of the base, the finger inside the rim,
    and the plate behind the pot; the pot touches none of them."""
    holder = Holder(pot_for(100, "tapered"))
    pot = holder.pot
    from manifold3d import Manifold
    m = holder.manifold()
    dot = Manifold.cube([0.2] * 3, center=True)
    inside = np.array([(m ^ dot.translate(p)).volume() > 0.007 for p in [
        [float(holder.env(0.0)) - holder.ledge / 2, 0, -1],             # ledge, under the base
        [holder.finger - holder.thick / 2, 0, pot.height - 2],         # finger, inside the rim
        [float(pot.r_out(pot.height)) + 1, 0, holder.top - 1],         # bridge, over the rim
        [holder.x_wall - 1, 0, pot.height / 2],                        # plate
    ]])
    assert inside.all()
    assert holder.finger < float(pot.r_in(pot.height)) - holder.clear + 1e-9
    assert holder.face > holder.env_max + holder.clear


@pytest.mark.parametrize("h", [50, 100])
def test_pot_comes_out(h):
    """The pot comes out with a modest swing of the bottom (and goes in the same way back)."""
    assert swing_needed(Holder(pot_for(h, "tapered"))) <= (14 if h < 60 else 8)


def test_holder_bad_settings():
    pot = pot_for(100, "tapered")
    with pytest.raises(ValueError, match="width must be at least"):
        Holder(pot, width=10)
    with pytest.raises(ValueError, match="positive"):
        Holder(pot, groove=0)
    with pytest.raises(ValueError, match="deeper than"):
        Holder(pot, groove=7)
    with pytest.raises(ValueError, match="negative"):
        Holder(pot, clear=-0.1)
    with pytest.raises(ValueError, match="wider than"):
        Holder(pot, head=3)
    with pytest.raises(ValueError, match="too short"):
        Holder(Pot(height=20, wall=3))


def test_holder_config():
    cfg = load_config([])
    assert holder_from_config(cfg, pot_from_config(cfg)) is None
    cfg = load_config(["holder.enable=true", "holder.width=18", "size=small"])
    holder = holder_from_config(cfg, pot_from_config(cfg))
    assert holder.width == 18 and holder.pot.height == 50
    assert set(cfg.holder) - {"enable"} == set(Holder.__dataclass_fields__) - {"pot"}
