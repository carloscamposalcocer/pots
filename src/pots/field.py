"""
The whole pot as one implicit field (negative = solid): patterned wall,
solid rim, solid base, and the plain drip cup fused to the base.
"""
import numpy as np

from .geometry import CHAMFER, PATTERN_GAP
from .patterns import PATTERNS
from .sdf import cylindrical, smin

BLEND = 1.5           # blend between a 3D lattice and the solid rim/base band


def make_field(pot, pattern):
    """Return field(X, Y, Z) for `pot` with the wall pattern named `pattern`."""
    kind, fn, _ = PATTERNS[pattern]

    def field(X, Y, Z):
        r, th = cylindrical(X, Y)
        ro, ri = pot.r_out(Z), pot.r_in(Z)
        shell = np.maximum.reduce([r - ro, ri - r, -Z, Z - pot.height])
        s = np.clip((r - ri) / pot.wall, 0, 1)
        band_lo = Z - (pot.base + PATTERN_GAP)    # pattern starts just above the base
        band_hi = (pot.height - pot.rim) - Z
        band = np.minimum(band_lo, band_hi)       # >0 inside the pattern band
        if kind == "3d":
            mat = smin(fn(pot, th, Z, r, s), band, BLEND)   # solid outside the band
        else:
            hole = np.maximum(fn(pot, th, Z, r, s), -band)  # holes only inside the band
            mat = -hole
        wall = np.maximum(shell, mat)
        ci = pot.cup_ri(Z)
        co = ci + pot.cup_wall
        cup_wall = np.maximum.reduce([r - co, ci - r, -Z, Z - pot.cup_h])
        base = np.maximum.reduce([r - co, -Z, Z - pot.base])
        body = smin(smin(wall, base, 1.0), cup_wall, 1.0)
        # 45 deg chamfer on the bottom outer edge (elephant foot)
        return np.maximum(body, ((r - co) + (CHAMFER - Z)) / np.sqrt(2))

    return field
