"""Field -> clean, printable mesh."""
import logging

from .field import make_field
from .mesh import build, clean

log = logging.getLogger(__name__)


def generate(pot, pattern, voxel, faces):
    """Mesh `pot` with wall `pattern`. Returns (mesh, field)."""
    field = make_field(pot, pattern)
    log.info("meshing '%s' at %.3f mm", pattern, voxel)
    raw = build(field, pot.r_max, pot.height, voxel)
    log.info("cleaning %s faces", f"{len(raw.faces):,}")
    return clean(raw, faces), field
