"""
pots-gallery: render the README gallery image of every wall pattern.

For each pattern it builds the pot in memory and saves only
docs/patterns/<name>.png: the pot from outside and a 40 x 30 mm true-scale
swatch of the outer wall face with its open %. No STL, 3MF or config is
written; use `pots pattern=<name>` for those. Patterns are built one after
another (never in parallel: memory), well under a minute each at the default
quality.

Usage
    pots-gallery                         # every pattern -> docs/patterns/<name>.png
    pots-gallery hex slots               # only these
    pots-gallery quality=draft           # faster, coarser
    pots-gallery size=big --images pics  # the 130 mm pot, images in pics/
    pots-gallery height=90 design.wall=5

Takes the shape settings of `pots` (size, height, design.*, patterns.*, quality); `pattern`,
`out` and `preview` don't apply. The pot defaults to the config's size preset
(small). Mesh quality defaults to quality.voxel=0.35 quality.faces=400000,
between draft and full and fine enough for 1.1 mm struts to show;
quality=... or quality.* replaces that.
"""
import logging
import sys
import time
from pathlib import Path

from .cli import load_config, pot_from_config, setup_logging
from .patterns import PATTERNS

log = logging.getLogger("pots")

# used unless the command line sets the same key or its group (quality=draft)
DEFAULTS = ["quality.voxel=0.35", "quality.faces=400000"]
IMAGES = "docs/patterns"
IGNORED = {"pattern", "out", "preview"}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if {"-h", "--help"} & set(argv):
        print(__doc__); return
    images = Path(IMAGES)
    if "--images" in argv[:-1]:
        i = argv.index("--images")
        images, argv = Path(argv[i + 1]), argv[:i] + argv[i + 2:]
    names = [a for a in argv if "=" not in a]
    given = [a for a in argv if "=" in a]
    keys = {a.split("=")[0] for a in given}
    if keys & IGNORED:
        raise SystemExit(f"{', '.join(sorted(keys & IGNORED))} not used by pots-gallery; "
                         "name patterns as plain words (pots-gallery hex slots)")
    unknown = [n for n in names if n not in PATTERNS]
    if unknown:
        raise SystemExit(f"unknown pattern(s): {' '.join(unknown)}; choose from: {', '.join(PATTERNS)}")
    overrides = [d for d in DEFAULTS if not keys & {d.split("=")[0], d.split(".")[0]}] + given

    from hydra.errors import HydraException
    try:
        cfg = load_config(overrides)
        pot = pot_from_config(cfg)
    except HydraException as e:
        raise SystemExit(f"bad setting: {e}".splitlines()[0])
    except (ValueError, TypeError) as e:
        raise SystemExit(str(e))
    setup_logging(False)

    from .pipeline import generate
    from .preview import render_card

    images.mkdir(parents=True, exist_ok=True)
    log.info("pot H %g mm, wall %g mm, voxel %g mm -> %s", pot.height, pot.wall, cfg.quality.voxel, images)
    for name in names or list(PATTERNS):
        t = time.perf_counter()
        m, field = generate(pot, name, cfg.quality.voxel, cfg.quality.faces, cfg.patterns)
        if not m.is_watertight or m.body_count != 1:
            log.warning("%s: mesh is not a single watertight body", name)
        render_card(m, field, pot, images / f"{name}.png", f"{name}  (H {pot.height:g} mm)")
        log.info("%s done in %.0f s", name, time.perf_counter() - t)


if __name__ == "__main__":
    main()
