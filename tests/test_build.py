import logging

import numpy as np
import pytest

from pots import PATTERNS, Pot, generate
from pots.metrics import overhang_share
from pots.patterns import Pattern


def test_small_draft_pot_is_one_watertight_body(tmp_path):
    pot = Pot(height=30, wall=4.0)
    mesh, _ = generate(pot, "voronoi_taper", voxel=0.6, faces=60_000)
    assert mesh.is_watertight
    assert mesh.body_count == 1
    assert mesh.bounds[0, 2] == 0
    assert abs(mesh.bounds[1, 2] - pot.height) < 1.0
    mesh.export(tmp_path / "pot.3mf")


def test_loose_parts_are_reported(caplog, monkeypatch):
    """Ring-shaped holes cut out islands (what sank `coral`): the build warns."""
    def rings(pot, P, th, Z, r, s):
        p = pot.circ / 20
        d = np.hypot(np.mod(th * pot.r0, p) - p / 2, np.mod(Z, 8.0) - 4.0)
        return np.abs(d - 2.5) - 0.7
    monkeypatch.setitem(PATTERNS, "rings", Pattern("2d", rings, "test"))
    with caplog.at_level(logging.WARNING, logger="pots.mesh"):
        generate(Pot(height=40, wall=4.0), "rings", voxel=0.6, faces=60_000)
    assert "detached" in caplog.text


@pytest.mark.slow
@pytest.mark.parametrize("name", PATTERNS)
def test_pattern_has_no_loose_parts(name, caplog):
    """clean() keeps only the largest piece; a pattern whose holes close into
    loops leaves loose islands that get dropped (and gaps in the wall)."""
    pot = Pot(height=40, wall=4.0)
    with caplog.at_level(logging.WARNING, logger="pots.mesh"):
        mesh, _ = generate(pot, name, voxel=0.6, faces=60_000)
    assert "detached" not in caplog.text
    assert mesh.is_watertight and mesh.body_count == 1
    assert overhang_share(mesh) < 0.12      # hex, the worst of the existing ones, is ~0.09
