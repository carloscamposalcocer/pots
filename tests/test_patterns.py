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
    cfg = load_config(["patterns.lattice_taper.strut_in=2.5"])
    P = load_params(cfg.patterns)
    assert P.lattice_taper.strut_in == 2.5
    assert P.voronoi_taper.cells == load_params().voronoi_taper.cells


@pytest.mark.parametrize("name, override", [
    ("lattice", {"strut": 2.4}),
    ("voronoi", {"seed": 3}),
    ("lattice_taper", {"strut_in": 2.6}),
    ("bands", {"slots": {"width": 1.6}}),
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
    block = override if "slots" in override else {name: override}
    for k, v in block.items():
        changed[k].update(v)
    assert not np.allclose(make_field(pot, name)(*pts), make_field(pot, name, changed)(*pts))


def test_unknown_pattern_setting_is_rejected():
    from hydra.errors import HydraException
    with pytest.raises(HydraException):
        load_config(["patterns.lattice.nope=1"])


def test_warns_about_thin_struts(caplog):
    with caplog.at_level(logging.WARNING, logger="pots"):
        pot_from_config(load_config(["pattern=lattice_taper", "patterns.lattice_taper.strut_out=0.8"]))
    assert "too thin" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="pots"):
        pot_from_config(load_config(["pattern=bands", "patterns.voronoi_taper.strut_out=3"]))
    assert "narrow outward" in caplog.text
