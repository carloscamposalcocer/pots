from pots import Pot, generate


def test_small_draft_pot_is_one_watertight_body(tmp_path):
    pot = Pot(height=30, wall=4.0)
    mesh, _ = generate(pot, "voronoi_taper", voxel=0.6, faces=60_000)
    assert mesh.is_watertight
    assert mesh.body_count == 1
    assert mesh.bounds[0, 2] == 0
    assert abs(mesh.bounds[1, 2] - pot.height) < 1.0
    mesh.export(tmp_path / "pot.3mf")
