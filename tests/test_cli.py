import pytest

from pots.cli import load_config, main, pot_from_config


def test_defaults_compose():
    cfg = load_config([])
    assert cfg.quality.voxel == 0.3
    assert cfg.out == f"output/{cfg.pattern}_h{cfg.height}"


def test_overrides():
    cfg = load_config(["height=65", "quality=draft", "design.wall=3.5"])
    pot = pot_from_config(cfg)
    assert (pot.height, pot.wall) == (65, 3.5)
    assert cfg.quality.voxel == 0.6


def test_bad_values():
    with pytest.raises(ValueError, match="unknown pattern"):
        pot_from_config(load_config(["pattern=nope"]))
    with pytest.raises(ValueError, match="at least"):
        pot_from_config(load_config(["height=5"]))


def test_show_and_bad_key(capsys):
    main(["height=65", "--show"])
    assert "height: 65" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="bad setting"):
        main(["design.nope=1"])
    with pytest.raises(SystemExit, match="unknown option"):
        main(["--nope"])


@pytest.mark.parametrize("size, height, wall", [("big", 130, 6.0), ("small", 50, 4.0)])
def test_size_presets(size, height, wall):
    pot = pot_from_config(load_config([f"size={size}"]))
    assert (pot.height, pot.wall) == (height, wall)


def test_overrides_beat_size_preset():
    pot = pot_from_config(load_config(["size=big", "height=100", "design.wall=5"]))
    assert (pot.height, pot.wall) == (100, 5)
