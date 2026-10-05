import logging

import numpy as np
import pytest

from pots import PATTERNS, Pot, make_field
from pots.cli import load_config, pot_from_config
from pots.patterns import load_params


def test_every_pattern_has_a_settings_block():
    P = load_params()
    assert set(vars(P)) == set(PATTERNS)
    for p in PATTERNS.values():
        assert all(u in PATTERNS for u in p.uses)


def test_config_carries_the_pattern_settings():
    cfg = load_config(["patterns.lattice.strut_in=2.5"])
    P = load_params(cfg.patterns)
    assert P.lattice.strut_in == 2.5
    assert P.voronoi.cells == load_params().voronoi.cells


@pytest.mark.parametrize("name, override", [
    ("isogrid", {"strut_in": 2.6}),
    ("voronoi", {"seed": 3}),
    ("lattice", {"strut_in": 2.6}),
])
def test_settings_change_the_field(name, override):
    pot = Pot(height=65, wall=4.0)
    rng = np.random.default_rng(0)
    th = rng.uniform(-np.pi, np.pi, 20_000)
    z = rng.uniform(pot.base + 2, 30, th.size)
    r = pot.r_out(z) - pot.wall * rng.uniform(0, 1, th.size)
    pts = [a.astype(np.float32) for a in (r * np.cos(th), r * np.sin(th), z)]
    params = vars(load_params()).copy()
    changed = {k: dict(vars(v)) for k, v in params.items()}
    changed[name].update(override)
    assert not np.allclose(make_field(pot, name)(*pts), make_field(pot, name, changed)(*pts))


def test_unknown_pattern_setting_is_rejected():
    from hydra.errors import HydraException
    with pytest.raises(HydraException):
        load_config(["patterns.lattice.nope=1"])


def test_warns_about_thin_struts(caplog):
    with caplog.at_level(logging.WARNING, logger="pots"):
        pot_from_config(load_config(["pattern=lattice", "patterns.lattice.strut_in=0.8"]))
    assert "too thin" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="pots"):
        pot_from_config(load_config(["pattern=lattice"]))       # a thin outer edge is fine
    assert "too thin" not in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="pots"):                # so is a thin soil-side one
        pot_from_config(load_config(["pattern=hex", "patterns.hex.strut_in=0.5", "patterns.hex.strut_out=3"]))
    assert "too thin" not in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="pots"):
        pot_from_config(load_config(["pattern=voronoi", "patterns.voronoi.strut_in=2.2", "patterns.voronoi.strut_out=3"]))
    assert "narrow outward" in caplog.text


@pytest.mark.parametrize("name", ["voronoi", "hex", "lattice", "isogrid"])
def test_center_lines_up_the_holes(name):
    """center=0 keeps each roof level (holes grow downward); center=1 puts
    the soil-side hole in the middle of its outside opening."""
    from scipy import ndimage
    pot = Pot()
    TH, Z = np.meshgrid(np.linspace(-0.3, 0.3, 500), np.linspace(40, 90, 500))
    for center in (0.0, 1.0):
        P = load_params()
        c = getattr(P, name)
        c.strut_in, c.strut_out, c.center = 2.2, 1.1, center
        fn = PATTERNS[name].fn
        inner = fn(pot, P, TH, Z, pot.r0, np.zeros_like(Z)) < 0
        outer = fn(pot, P, TH, Z, pot.r0, np.ones_like(Z)) < 0
        lo, _ = ndimage.label(outer)
        li, n = ndimage.label(inner)
        d_roof, d_mid = [], []
        for k in range(1, n + 1):
            mi = li == k
            om = lo == np.bincount(lo[mi]).argmax()
            if any(m[[0, -1]].any() or m[:, [0, -1]].any() for m in (mi, om)):
                continue
            d_roof.append(Z[om].max() - Z[mi].max())
            d_mid.append(Z[mi].mean() - Z[om].mean())
        level, centred = np.mean(d_roof) < 0.15, abs(np.mean(d_mid)) < 0.1
        assert (level, centred) == ((True, False) if center == 0 else (False, True))


def test_center_is_checked(caplog):
    with pytest.raises(ValueError, match="between 0 and 1"):
        pot_from_config(load_config(["patterns.voronoi.center=1.5"]))
    with caplog.at_level(logging.WARNING, logger="pots"):
        pot_from_config(load_config(["patterns.voronoi.center=1"]))
    assert "may sag" in caplog.text


def test_leaves_stay_apart():
    """Thin struts and no keel: still one hole per leaf (the tips fit under
    the leaves above, the rows don't merge)."""
    from scipy import ndimage
    pot = Pot(height=100, wall=6.0)
    P = load_params()
    c = P.leaves
    c.strut_in = c.strut_out = 0.3
    c.rib, c.jitter = 0.0, 0.0
    S, Z = np.meshgrid(np.arange(0, 60, 0.03), np.arange(30, 70, 0.03))
    hole = PATTERNS["leaves"].fn(pot, P, S / pot.r0, Z, pot.r0, np.zeros_like(Z)) < 0
    _, n = ndimage.label(hole)
    p = pot.circ / pot.count(c.cells)
    assert n >= 60 * 40 / (p * p * c.row)          # at least one per leaf in the window
    assert hole.mean() < 0.97


@pytest.mark.parametrize("h", [30, 50, 100, 130, 200])
def test_leaf_tips_stay_steep(h):
    """The sway shears the leaves; at the default warp the tips stay at SLOPE or steeper."""
    from pots.patterns import SLOPE, leaf_tip_slope
    assert leaf_tip_slope(Pot(height=h, wall=4.0), load_params().leaves) >= np.degrees(SLOPE)


def test_warns_about_leaf_sway_and_keel(caplog):
    with caplog.at_level(logging.WARNING, logger="pots"):
        pot_from_config(load_config(["pattern=leaves"]))
    assert not caplog.text
    with caplog.at_level(logging.WARNING, logger="pots"):
        pot_from_config(load_config(["pattern=leaves", "patterns.leaves.warp=2", "patterns.leaves.rib=0.3"]))
    assert "tilts the leaf tips" in caplog.text and "keel" in caplog.text
