"""
pots: generate the plant pot as STL + 3MF + preview image.

Every setting has a default in the packaged config/config.yaml; pass only what
you change, as key=value (Hydra syntax, nested keys use dots).

Examples
    pots                              # quick draft build with the defaults (size=small)
    pots quality=full                 # print-quality mesh (the file you print)
    pots size=big                     # the 130 mm reference pot
    pots size=big height=100          # start from a preset, change one value
    pots pattern=hex                  # another pattern
    pots design.wall=4                # thinner wall (mm, does not scale)
    pots pattern=lattice_taper patterns.lattice_taper.strut_in=2.5
                                      # a pattern setting (config/patterns.yaml)
    pots height=65 --show             # print the resolved settings, don't build
    pots --all                        # every pattern, one after another
    pots -v                           # debug logging (per-slab marching cubes)

Options
    --all           build every pattern (sequentially), each in its own `out`
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

CONFIG_DIR = Path(__file__).parent / "config"
FLAGS = {"-v", "--verbose", "--show", "--all"}


def usage():
    names = "\n".join(f"    {n:<15} {p.description}" for n, p in PATTERNS.items())
    return f"{__doc__}\nPatterns (pattern=...)\n{names}\n"


def load_config(overrides):
    # Compose API instead of @hydra.main: hydra-core 1.3's own argparse CLI
    # crashes on Python 3.14. Overrides use the same key=value syntax.
    import hydra
    with hydra.initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        return hydra.compose("config", overrides=overrides)


def pot_from_config(cfg):
    """Validated Pot for a config; raises ValueError with a user-facing message."""
    if cfg.pattern not in PATTERNS:
        raise ValueError(f"unknown pattern '{cfg.pattern}'; choose from: {', '.join(sorted(PATTERNS))}")
    pot = Pot(height=float(cfg.height), **{k: float(v) for k, v in cfg.design.items()})
    check_pattern_params(cfg)
    return pot


# tapered patterns whose hole roofs are flat bridges (the others have 55-degree roofs)
BRIDGE_ROOFS = {"voronoi_taper", "hex_taper"}


def check_pattern_params(cfg):
    """Warn about pattern settings that won't print well."""
    for name in (cfg.pattern, *PATTERNS[cfg.pattern].uses):
        c = cfg.patterns[name]
        for key in ("strut", "strut_out"):
            if key in c and c[key] < 1.1:
                log.warning("patterns.%s.%s=%.2f mm is below ~1.1 mm, too thin for a 0.4 mm nozzle",
                            name, key, c[key])
        if "strut_in" in c and c.strut_out > c.strut_in:
            log.warning("patterns.%s.strut_out > strut_in: the holes will narrow outward", name)
        if "center" in c:
            if not 0 <= c.center <= 1:
                raise ValueError(f"patterns.{name}.center must be between 0 and 1")
            if c.center > 0 and name in BRIDGE_ROOFS:
                log.warning("patterns.%s.center=%g: the flat hole roofs rise toward the outside "
                            "and may sag (0 keeps them level)", name, c.center)


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

    from .metrics import overhang_share, wall_metrics
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
    m, field = generate(pot, cfg.pattern, q.voxel, q.faces, cfg.patterns)
    size = np.ptp(m.bounds, axis=0)
    log.info("mesh  faces=%s  watertight=%s  bodies=%d  size=%.1f x %.1f x %.1f mm  volume=%.0f cm3",
             f"{len(m.faces):,}", m.is_watertight, m.body_count, *size, m.volume / 1000)
    if not m.is_watertight or m.body_count != 1:
        log.warning("mesh is not a single watertight body; check it before printing")
    log.info("wall  %s", wall_metrics(field, pot))
    log.info("overhangs past 50 degrees (not bridges): %.1f%% of the surface", 100 * overhang_share(m))
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
    overrides = [a for a in argv if not a.startswith("-")]
    if "--all" in flags:
        run_all(overrides, "--show" in flags, bool(flags & {"-v", "--verbose"}))
        return
    from omegaconf import OmegaConf
    cfg = compose(overrides)
    if "--show" in flags:
        print(OmegaConf.to_yaml(cfg, resolve=True), end=""); return
    setup_logging(bool(flags & {"-v", "--verbose"}))
    run(cfg)


def compose(overrides):
    from hydra.errors import HydraException
    try:
        return load_config(overrides)
    except HydraException as e:
        raise SystemExit(f"bad setting: {e}".splitlines()[0] + "  (see --show for valid keys)")


def all_configs(overrides):
    """One config per pattern. If `out` doesn't depend on the pattern (out=foo),
    each pattern goes to its own subfolder, <out>/<pattern>."""
    if any(a.split("=")[0].lstrip("+~") == "pattern" for a in overrides):
        raise SystemExit("--all builds every pattern; drop pattern=...")
    cfgs = {n: compose(overrides + [f"pattern={n}"]) for n in PATTERNS}
    if len({c.out for c in cfgs.values()}) < len(cfgs):
        for n, c in cfgs.items():
            c.out = f"{c.out}/{n}"
    return cfgs


def run_all(overrides, show, verbose):
    """Build every pattern one after another (never in parallel: memory).
    A failing pattern is logged and skipped; exits 1 at the end if any failed."""
    cfgs = all_configs(overrides)
    if show:
        for n, c in cfgs.items():
            print(f"{n:<15} -> {c.out}")
        return
    setup_logging(verbose)
    failed = []
    t = time.perf_counter()
    for i, (n, c) in enumerate(cfgs.items(), 1):
        log.info("===== pattern %d/%d: %s =====", i, len(cfgs), n)
        try:
            run(c)
        except SystemExit:
            raise  # bad settings: the same for every pattern
        except Exception:
            log.exception("%s failed", n)
            failed.append(n)
    log.info("all patterns done in %.0f s; %d built, %d failed%s", time.perf_counter() - t,
             len(cfgs) - len(failed), len(failed), f": {' '.join(failed)}" if failed else "")
    if failed:
        raise SystemExit(1)
