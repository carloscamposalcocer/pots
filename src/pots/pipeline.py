"""Field -> clean, printable mesh."""
import logging

from .field import make_field
from .mesh import build, clean
from .patterns import PATTERNS, load_params

log = logging.getLogger(__name__)

MESHERS = ("auto", "sdf")


def generate(pot, pattern, voxel, faces, params=None, mesher="auto", clips=None):
    """Mesh `pot` with wall `pattern` (settings `params`, None = defaults) and
    the wall-mount pads `clips` (a `Clips`, or None for a plain pot).
    Returns (mesh, field); the field is always returned for the metrics.
    mesher: "auto" builds the exact mesh (solid.py) for patterns that have
    cutters, and marching cubes at `voxel` / `faces` for the others; "sdf"
    uses marching cubes for every pattern."""
    if mesher not in MESHERS:
        raise ValueError(f"unknown mesher '{mesher}'; choose from: {', '.join(MESHERS)}")
    field = make_field(pot, pattern, params, clips)
    cutters = PATTERNS[pattern].cutters
    if mesher == "auto" and cutters is not None:
        from .solid import build_solid, depths
        holes = cutters(pot, getattr(load_params(params), pattern), depths(pot))
        if holes is not None:
            log.info("building '%s' exactly (solid)", pattern)
            return build_solid(pot, holes, clips), field
        log.info("'%s' holes close up inside the wall with these settings; using marching cubes", pattern)
    log.info("meshing '%s' at %.3f mm", pattern, voxel)
    rmax = pot.r_max if clips is None else max(pot.r_max, clips.r_max)
    raw = build(field, rmax, pot.height, voxel)
    log.info("cleaning %s faces", f"{len(raw.faces):,}")
    return clean(raw, faces), field
