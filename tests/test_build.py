import logging

import numpy as np
import pytest

import pots.field
from pots import PATTERNS, Pot, generate, make_field
from pots.geometry import SHAPES
from pots.mesh import build
from pots.metrics import overhang_share
from pots.patterns import Pattern


def test_small_draft_pot_is_one_watertight_body(tmp_path):
    pot = Pot(height=30, wall=4.0)
    mesh, _ = generate(pot, "voronoi", voxel=0.6, faces=60_000)
    assert mesh.is_watertight
    assert mesh.body_count == 1
    assert mesh.bounds[0, 2] == 0
    assert abs(mesh.bounds[1, 2] - pot.height) < 1.0
    mesh.export(tmp_path / "pot.3mf")


@pytest.mark.parametrize("name", ["voronoi", "weave", "louvers"])
def test_narrow_band_field_meshes_the_same(name, monkeypatch):
    """The field skips the pattern (and the cup) far from the wall; that must
    not change the mesh. The raw mesh is also closed: no pinched edges."""
    pot = Pot(height=30, wall=4.0, shape="barrel")
    fast = build(make_field(pot, name), pot.r_max, pot.height, 0.6)
    monkeypatch.setattr(pots.field, "NEAR", np.inf)
    full = build(make_field(pot, name), pot.r_max, pot.height, 0.6)
    assert np.array_equal(fast.faces, full.faces)
    assert np.abs(fast.vertices - full.vertices).max() < 1e-4
    assert fast.is_watertight


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
    # hex: its pointy roofs are 30-degree slopes (short near-bridges), ~13% here
    assert overhang_share(mesh) < (0.15 if name == "hex" else 0.12)


@pytest.mark.slow
@pytest.mark.parametrize("shape", SHAPES)
def test_every_shape_builds_clean(shape, caplog):
    pot = Pot(height=40, wall=4.0, shape=shape)
    with caplog.at_level(logging.WARNING, logger="pots.mesh"):
        mesh, _ = generate(pot, "voronoi", voxel=0.6, faces=60_000)
    assert "detached" not in caplog.text
    assert mesh.is_watertight and mesh.body_count == 1
    assert overhang_share(mesh) < 0.12
