"""
pots: generate the plant pot as STL + 3MF + preview image.

Every setting has a default in the packaged conf/config.yaml; pass only what
you change, as key=value (Hydra syntax, nested keys use dots).

Examples
    pots                              # build with the config defaults
    pots quality=draft                # fast low-res check
    pots height=65                    # half-size pot, same shape
    pots pattern=coral                # another pattern
    pots design.wall=4                # thinner wall (mm, does not scale)
    pots height=65 --show             # print the resolved settings, don't build
    pots -v                           # debug logging (per-slab marching cubes)

Options
    --show          print the resolved config and exit
    -v, --verbose   debug logging
    -h, --help      this help

Outputs (in `out`, default output/<pattern>_h<height>/):
    pot_<pattern>.stl      print file
    pot_<pattern>.3mf      same mesh, ~5x smaller file
    pot_<pattern>.png      preview: outside view, cut-away, wall faces
    config.yaml            the exact settings used
"""
import logging
import sys
import time
from pathlib import Path

import numpy as np

from .geometry import Pot
from .patterns import PATTERNS

log = logging.getLogger("pots")

CONF_DIR = Path(__file__).parent / "conf"
FLAGS = {"-v", "--verbose", "--show"}


def usage():
    names = "\n".join(f"    {n:<15} {p.description}" for n, p in PATTERNS.items())
    return f"{__doc__}\nPatterns (pattern=...)\n{names}\n"


def load_config(overrides):
    # Compose API instead of @hydra.main: hydra-core 1.3's own argparse CLI
    # crashes on Python 3.14. Overrides use the same key=value syntax.
    import hydra
    with hydra.initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        return hydra.compose("config", overrides=overrides)


def pot_from_config(cfg):
    """Validated Pot for a config; raises ValueError with a user-facing message."""
    if cfg.pattern not in PATTERNS:
        raise ValueError(f"unknown pattern '{cfg.pattern}'; choose from: {', '.join(sorted(PATTERNS))}")
    pot = Pot(height=float(cfg.height), **{k: float(v) for k, v in cfg.design.items()})
    if pot.strut_out < 1.1:
        log.warning("design.strut_out=%.2f mm is below ~1.1 mm, too thin for a 0.4 mm nozzle", pot.strut_out)
    if pot.strut_out > pot.strut_in:
        log.warning("design.strut_out > design.strut_in: voronoi_taper holes will narrow outward")
    return pot


def setup_logging(verbose):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-5s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    for noisy in ("matplotlib", "trimesh", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def run(cfg):
    from omegaconf import OmegaConf

    from .metrics import wall_metrics
    from .pipeline import generate
    from .preview import render

    try:
        pot = pot_from_config(cfg)
    except (ValueError, TypeError) as e:
        log.error(e)
        raise SystemExit(2)
    q = cfg.quality
    log.info("height %.1f mm (x%.3f): radius %.1f -> %.1f mm, cup %.1f mm tall, wall %.1f mm",
             pot.height, pot.k, pot.r_bot, pot.r_top, pot.cup_h, pot.wall)

    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out / "config.yaml", resolve=True)
    base = out / f"pot_{cfg.pattern}"
    t = time.perf_counter()
    log.info("[1/3] building")
    m, field = generate(pot, cfg.pattern, q.voxel, q.faces)
    size = np.ptp(m.bounds, axis=0)
    log.info("mesh  faces=%s  watertight=%s  bodies=%d  size=%.1f x %.1f x %.1f mm  volume=%.0f cm3",
             f"{len(m.faces):,}", m.is_watertight, m.body_count, *size, m.volume / 1000)
    if not m.is_watertight or m.body_count != 1:
        log.warning("mesh is not a single watertight body; check it before printing")
    log.info("wall  %s", wall_metrics(field, pot))
    log.info("[2/3] exporting %s.stl and %s.3mf", base, base)
    m.export(base.with_suffix(".stl")); m.export(base.with_suffix(".3mf"))
    outs = [base.with_suffix(".stl"), base.with_suffix(".3mf")]
    if cfg.preview:
        log.info("[3/3] rendering preview")
        render(m, field, pot, base.with_suffix(".png"), f"pot_{cfg.pattern}  (H {pot.height:g} mm)")
        outs.append(base.with_suffix(".png"))
    else:
        log.info("[3/3] skipped preview")
    log.info("done in %.0f s -> %s", time.perf_counter() - t, "  ".join(map(str, outs)))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if {"-h", "--help"} & set(argv):
        print(usage()); return
    flags = {a for a in argv if a.startswith("-")}
    unknown = flags - FLAGS
    if unknown:
        raise SystemExit(f"unknown option(s): {' '.join(sorted(unknown))}  (settings are key=value, see --help)")
    from hydra.errors import HydraException
    from omegaconf import OmegaConf
    try:
        cfg = load_config([a for a in argv if not a.startswith("-")])
    except HydraException as e:
        raise SystemExit(f"bad setting: {e}".splitlines()[0] + "  (see --show for valid keys)")
    if "--show" in flags:
        print(OmegaConf.to_yaml(cfg, resolve=True), end=""); return
    setup_logging(bool(flags & {"-v", "--verbose"}))
    run(cfg)
