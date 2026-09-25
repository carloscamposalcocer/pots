import pytest

from pots.gallery import main


def test_gallery_writes_only_images(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    main(["hex", "height=30", "quality=draft", "--images", "pics"])
    assert [p.name for p in tmp_path.rglob("*") if p.is_file()] == ["hex.png"]


def test_gallery_bad_args():
    with pytest.raises(SystemExit, match="unknown pattern"):
        main(["nope"])
    with pytest.raises(SystemExit, match="not used"):
        main(["pattern=hex"])
    with pytest.raises(SystemExit, match="bad setting"):
        main(["design.nope=1"])
