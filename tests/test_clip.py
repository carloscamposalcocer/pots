"""Cup clip: a narrow strip whose hook has the shape of the back of the cup
wall, holding the unchanged (tapered) pot by its cup, tight."""
import numpy as np
import pytest
from test_holder import plain, pot_for

from pots.cli import clip_from_config, load_config, pot_from_config
from pots.clip import Clip
from pots.geometry import Pot
from pots.metrics import overhang_share


def check(clip):
    m = clip.mesh()
    assert m.is_watertight and m.body_count == 1
    assert overhang_share(m) == 0
    assert np.allclose(m.bounds[0], 0) and m.bounds[1][2] == pytest.approx(clip.width)
    pot = plain(clip.pot)
    assert (Clip(clip.pot, squeeze=0).manifold() ^ pot).volume() < 0.5    # touches, facets aside
    assert (clip.manifold() ^ pot).volume() > 1.0                           # grips


def give(clip, angles=np.arange(0, 21, 2.0)):
    """How far (mm) the clip has to give for the pot to come out, on the best
    swing: the deepest overlap of cup and clip in the centre section as the
    bottom swings out about the finger's tip, slides down along the cup's
    swung back wall and moves away (squeeze 0)."""
    from pots.solid import _profile
    p = clip.pot
    clip = Clip(p, **{k: getattr(clip, k) for k in Clip.__dataclass_fields__ if k != "pot"} | {"squeeze": 0.0})
    part, cup = clip.section() - clip.cutter(), _profile(p)
    slant = np.degrees(np.arctan((p.cup_ri(p.cup_h) - p.cup_ri(0.0)) / p.cup_h))
    f, tip, slide = clip.finger, clip.tip, clip.drop + clip.lift + 1.0

    def depth(a, s, away):
        b = np.radians(a + slant)
        m = cup.translate([-f, -tip]).rotate(-a).translate([f - s * np.sin(b) - away, tip - s * np.cos(b)])
        lo, hi = 0.0, 3.0
        if (part ^ m).area() < 1e-4:
            return 0.0
        for _ in range(12):
            d = (lo + hi) / 2
            lo, hi = (lo, d) if (part.offset(-d) ^ m).area() < 1e-4 else (d, hi)
        return hi

    t = np.linspace(0, 1, 7)[1:]
    return min(max([depth(a * u, 0, 0) for u in t] + [depth(a, slide * u, 0) for u in t]
                   + [depth(a, slide, 15 * u) for u in t]) for a in angles)


@pytest.mark.parametrize("h", [50, 100])
def test_clip_fits(h):
    check(Clip(pot_for(h, "tapered")))


@pytest.mark.slow
@pytest.mark.parametrize("h", [65, 130])
def test_clip_fits_other_sizes(h):
    check(Clip(pot_for(h, "tapered")))


def test_clip_holds():
    """The ledge is under the base, the finger inside the cup (clear of the
    pot wall), the bridge on the lip, the back behind the cup, the screws below."""
    clip = Clip(pot_for(100, "tapered"))
    p = clip.pot
    from manifold3d import Manifold
    m = clip.manifold()
    dot = Manifold.cube([0.2] * 3, center=True)
    lip = float(clip.env(p.cup_h))
    inside = np.array([(m ^ dot.translate(q)).volume() > 0.007 for q in [
        [float(clip.env(0.0)) - clip.ledge / 2, 0, -1],                  # ledge, under the base
        [float(clip.ri(clip.tip + 1)) - clip.fin / 2, 0, clip.tip + 1],  # finger, inside the cup
        [lip - 1, 0, clip.top - 1],                                      # bridge, on the lip
        [lip + 0.5, 0, p.cup_h - 1],                                     # back, behind the lip
        [clip.x_wall - 1, 0, (clip.screw_lo + clip.screw_hi) / 2],       # plate, between the screws
    ]])
    assert inside.all()
    assert clip.finger > float(p.cup_ri(clip.tip))                       # squeezes the cup's inside
    assert clip.ri(p.cup_h) - clip.fin > float(p.r_out(p.cup_h)) + clip.clear   # clear of the pot wall
    assert clip.screw_hi < clip.bottom and clip.under == p.cup_h


@pytest.mark.parametrize("h", [50, 100])
def test_pot_snaps_in(h):
    """The pot goes in and out with a swing of the bottom if the clip gives
    less than a millimetre (a press fit: the slot is the cup wall's shape)."""
    assert give(Clip(pot_for(h, "tapered"))) < (1.0 if h < 80 else 0.7)


def test_clip_bad_settings():
    pot = pot_for(100, "tapered")
    with pytest.raises(ValueError, match="width must be at least"):
        Clip(pot, width=10)
    with pytest.raises(ValueError, match="positive"):
        Clip(pot, clear=0)
    with pytest.raises(ValueError, match="squeeze"):
        Clip(pot, squeeze=-0.1)
    with pytest.raises(ValueError, match="wider than"):
        Clip(pot, head=3)
    with pytest.raises(ValueError, match="too far down"):
        Clip(pot, drop=20)
    with pytest.raises(ValueError, match="finger doesn't fit"):
        Clip(Pot(height=40, shape="straight", wall=4), width=16)    # a ~1.5 mm moat


def test_clip_config():
    cfg = load_config([])
    assert clip_from_config(cfg, pot_from_config(cfg)) is None
    cfg = load_config(["clip.enable=true", "clip.width=18", "size=small"])
    clip = clip_from_config(cfg, pot_from_config(cfg))
    assert clip.width == 18 and clip.pot.height == 50
    assert set(cfg.clip) - {"enable"} == set(Clip.__dataclass_fields__) - {"pot"}
