"""The exact (solid.py) builder: the same pot as the field, as a clean mesh."""
import numpy as np
import pytest

from pots import PATTERNS, Pot, generate, make_field
from pots.geometry import SHAPES
from pots.metrics import overhang_share
from pots.patterns import Pattern, load_params
from pots.solid import build_solid, depths

SOLID = [name for name, p in PATTERNS.items() if p.cutters is not None]


def field_hole_volume(pot, name, n=300_000, params=None):
    """Volume (mm3) the field leaves open in the wall between base and rim (Monte Carlo)."""
    f = make_field(pot, name, params)
    rng = np.random.default_rng(0)
    lo, hi = pot.base, pot.height - pot.rim
    z = rng.uniform(lo, hi, n)
    ri, ro = pot.r_in(z), pot.r_out(z)
    r = np.sqrt(rng.uniform(ri ** 2, ro ** 2))
    th = rng.uniform(-np.pi, np.pi, n)
    F = f(*(a.astype(np.float32) for a in (r * np.cos(th), r * np.sin(th), z)))
    return np.mean((F >= 0) * np.pi * (ro ** 2 - ri ** 2)) * (hi - lo)


def test_tapered_patterns_use_the_solid_builder():
    assert set(SOLID) == {"voronoi", "hex", "drops", "lattice", "isogrid"}


@pytest.mark.parametrize("name", SOLID)
@pytest.mark.parametrize("shape, skin", [("tapered", 0.0), ("hourglass", 1.6)])
def test_solid_matches_the_field(name, shape, skin):
    """Watertight, one body, on the bed, and the holes have the field's volume."""
    pot = Pot(height=40, wall=4.0, shape=shape, skin=skin, skin_frac=0.67 if skin else 0.0)
    holes = PATTERNS[name].cutters(pot, getattr(load_params(), name), depths(pot))
    mesh = build_solid(pot, holes)
    assert mesh.is_watertight and mesh.body_count == 1
    assert mesh.bounds[0, 2] == 0
    assert abs(mesh.bounds[1, 2] - pot.height) < 1e-6
    cut = build_solid(pot, []).volume - mesh.volume
    assert cut == pytest.approx(field_hole_volume(pot, name), rel=0.01)


def test_plain_pot_matches_the_field(monkeypatch):
    """No holes: the revolved body is the field's pot (its 1 mm blends are chamfers)."""
    monkeypatch.setitem(PATTERNS, "plain", Pattern("2d", lambda pot, P, th, Z, r, s: np.ones_like(Z), "test"))
    pot = Pot(height=40, wall=4.0, shape="bowl")
    sdf, _ = generate(pot, "plain", voxel=0.4, faces=400_000)
    assert build_solid(pot, []).volume == pytest.approx(sdf.volume, rel=0.005)


def test_auto_is_solid_and_sdf_is_marching_cubes():
    pot = Pot(height=30, wall=4.0)
    exact, _ = generate(pot, "lattice", voxel=0.6, faces=60_000)
    mc, _ = generate(pot, "lattice", voxel=0.6, faces=60_000, mesher="sdf")
    assert exact.is_watertight and mc.is_watertight
    assert overhang_share(exact) < 0.12
    assert exact.volume == pytest.approx(mc.volume, rel=0.02)
    assert len(exact.faces) != 60_000 and len(mc.faces) <= 60_000


def test_holes_that_close_fall_back_to_marching_cubes(caplog):
    """A strut wider than the cell closes the holes inside the wall: no loft, the field does it."""
    P = {k: vars(v) for k, v in vars(load_params()).items()}
    P["lattice"] = {**P["lattice"], "strut_in": 8.0, "strut_out": 1.1}
    with caplog.at_level("INFO", logger="pots.pipeline"):
        mesh, _ = generate(Pot(height=30, wall=4.0), "lattice", voxel=0.6, faces=60_000, params=P)
    assert "using marching cubes" in caplog.text
    assert mesh.is_watertight


def test_unknown_mesher():
    with pytest.raises(ValueError, match="unknown mesher"):
        generate(Pot(height=30, wall=4.0), "lattice", voxel=0.6, faces=60_000, mesher="nope")


def test_voronoi_blind_pockets():
    """A wide soil-side strut closes some cells there but not outside: those
    holes shrink to their seed inside the wall, as the field's do."""
    P = {k: vars(v) for k, v in vars(load_params()).items()}
    P["voronoi"] = {**P["voronoi"], "strut_in": 3.6}
    pot = Pot(height=40, wall=4.0)
    holes = PATTERNS["voronoi"].cutters(pot, load_params(P).voronoi, depths(pot))
    mesh = build_solid(pot, holes)
    assert mesh.is_watertight and mesh.body_count == 1
    f = make_field(pot, "voronoi", P)
    ring = pot.r_in(20.0) + 0.02 * pot.wall, np.linspace(-np.pi, np.pi, 4000)
    closed_inside = (f(*(a.astype(np.float32) for a in (ring[0] * np.cos(ring[1]), ring[0] * np.sin(ring[1]),
                                                        np.full(4000, 20.0)))) < 0).mean()
    assert closed_inside > 0.9                  # the soil face is nearly closed with this strut
    cut = build_solid(pot, []).volume - mesh.volume
    assert cut == pytest.approx(field_hole_volume(pot, "voronoi", params=P), rel=0.02)


@pytest.mark.slow
@pytest.mark.parametrize("name", SOLID)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("h", [30, 65])
def test_solid_every_shape_and_size(name, shape, h):
    pot = Pot(height=h, wall=4.0, shape=shape)
    holes = PATTERNS[name].cutters(pot, getattr(load_params(), name), depths(pot))
    mesh = build_solid(pot, holes)
    assert mesh.is_watertight and mesh.body_count == 1
    cut = build_solid(pot, []).volume - mesh.volume
    assert cut == pytest.approx(field_hole_volume(pot, name), rel=0.01)
