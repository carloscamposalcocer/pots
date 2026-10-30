"""Wall clip: a printable part that fits the unchanged pot, and a way to
get the pot in and out of it."""
import numpy as np
import pytest

from pots.cli import clip_from_config, load_config, pot_from_config
from pots.geometry import SHAPES, Pot
from pots.clip import Clip
from pots.metrics import overhang_share


def plain(pot):
    """The pot without its pattern (the clip never reaches the patterned band)."""
    from manifold3d import Manifold

    from pots.solid import _profile, _segments
    return Manifold.revolve(_profile(pot), _segments(pot.r_max))


def pot_for(h, shape):
    return Pot(height=h, shape=shape, wall=6 if h > 60 else 4, skin=1.6, skin_frac=0.67)


def place(clip, pot_m, lift, angle, slide, away):
    """The pot moved out of the clip: lifted straight up, its bottom swung
    out about the finger's tip by `angle` degrees, slid down its own (tilted)
    axis and moved away from the wall."""
    a = np.radians(angle)
    f, tip = clip.finger, clip.tip
    m = pot_m.translate([-f, 0, lift - tip]).rotate([0, angle, 0]).translate([f, 0, tip])
    return m.translate([-slide * np.sin(a) - away, 0, -slide * np.cos(a)])


def removal(clip, angle, frac, n=8):
    """Poses taking the pot out: lift `frac` of the room above the rim, swing,
    slide until the rim is clear of the finger, move away."""
    lift, slide = clip.lift * frac, clip.drop + clip.lift + 0.5
    return ([(lift * t, 0, 0, 0) for t in np.linspace(0, 1, 3)]
            + [(lift, angle * t, 0, 0) for t in np.linspace(0, 1, n)[1:]]
            + [(lift, angle, slide * t, 0) for t in np.linspace(0, 1, n)[1:]]
            + [(lift, angle, slide, 20 * t) for t in np.linspace(0, 1, 4)[1:]])


def swing_needed(clip, angles=np.arange(2, 21, 2.0)):
    """The smallest swing (degrees) that gets the pot out without touching the clip."""
    hm, pm = clip.manifold(), plain(clip.pot)
    for angle in angles:
        for frac in (0.3, 0.6, 0.9):
            if all((hm ^ place(clip, pm, *p)).volume() < 0.01 for p in removal(clip, angle, frac)):
                return angle
    return None


def check(clip):
    m = clip.mesh()
    assert m.is_watertight and m.body_count == 1
    assert overhang_share(m) == 0
    assert np.allclose(m.bounds[0], 0) and m.bounds[1][2] == pytest.approx(clip.width)
    assert (clip.manifold() ^ plain(clip.pot)).volume() < 1e-3      # clear of the pot


@pytest.mark.parametrize("h", [50, 100])
def test_clip_fits(h):
    check(Clip(pot_for(h, "tapered")))


@pytest.mark.slow
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("h", [50, 65, 130])
def test_clip_fits_every_shape(h, shape):
    clip = Clip(pot_for(h, shape))
    check(clip)
    assert swing_needed(clip) is not None


def test_clip_holds():
    """The ledge is under the back of the base, the finger inside the rim,
    and the plate behind the pot; the pot touches none of them."""
    clip = Clip(pot_for(100, "tapered"))
    pot = clip.pot
    from manifold3d import Manifold
    m = clip.manifold()
    dot = Manifold.cube([0.2] * 3, center=True)
    inside = np.array([(m ^ dot.translate(p)).volume() > 0.007 for p in [
        [float(clip.env(0.0)) - clip.ledge / 2, 0, -1],             # ledge, under the base
        [clip.finger - clip.thick / 2, 0, pot.height - 2],         # finger, inside the rim
        [float(pot.r_out(pot.height)) + 1, 0, clip.top - 1],         # bridge, over the rim
        [clip.x_wall - 1, 0, pot.height / 2],                        # plate
    ]])
    assert inside.all()
    assert clip.finger < float(pot.r_in(pot.height)) - clip.clear + 1e-9
    assert clip.face > clip.env_max + clip.clear


@pytest.mark.parametrize("h", [50, 100])
def test_pot_comes_out(h):
    """The pot comes out with a modest swing of the bottom (and goes in the same way back)."""
    assert swing_needed(Clip(pot_for(h, "tapered"))) <= (14 if h < 60 else 8)


def test_clip_bad_settings():
    pot = pot_for(100, "tapered")
    with pytest.raises(ValueError, match="width must be at least"):
        Clip(pot, width=10)
    with pytest.raises(ValueError, match="positive"):
        Clip(pot, groove=0)
    with pytest.raises(ValueError, match="deeper than"):
        Clip(pot, groove=7)
    with pytest.raises(ValueError, match="negative"):
        Clip(pot, clear=-0.1)
    with pytest.raises(ValueError, match="wider than"):
        Clip(pot, head=3)
    with pytest.raises(ValueError, match="too short"):
        Clip(Pot(height=20, wall=3))


def test_clip_config():
    cfg = load_config([])
    assert clip_from_config(cfg, pot_from_config(cfg)) is None
    cfg = load_config(["clip.enable=true", "clip.width=18", "size=small"])
    clip = clip_from_config(cfg, pot_from_config(cfg))
    assert clip.width == 18 and clip.pot.height == 50
    assert set(cfg.clip) - {"enable"} == set(Clip.__dataclass_fields__) - {"pot"}


def test_clip_drawing(tmp_path):
    """The dimensioned drawing renders (2D slices only, no mesh rendering)."""
    from pots.preview import render_clip_drawing
    path = tmp_path / "drawing.png"
    render_clip_drawing(Clip(pot_for(100, "tapered")), path)
    assert path.stat().st_size > 10_000
