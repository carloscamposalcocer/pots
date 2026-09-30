"""
The whole pot as one implicit field (negative = solid): patterned wall,
solid rim, solid base, and the plain drip cup fused to the base.
"""
import numpy as np

from .geometry import CHAMFER, PATTERN_GAP
from .patterns import PATTERNS, load_params
from .sdf import smin

BLEND = 1.5           # blend between a 3D pattern (weave) and the solid rim/base band
# The pattern is only evaluated where the plain shell is below this (mm):
# inside the wall or near it. Farther out the wall is at least NEAR outside
# either way, so its exact value changes neither the sign nor, through the
# 1 mm blends with the base and cup, any value near zero. Most of the
# grid is air or soil space, so this skips most of the pattern work.
NEAR = 2.5


def make_field(pot, pattern, params=None):
    """Return field(X, Y, Z) for `pot` with the wall pattern named `pattern`.
    `params`: the pattern settings (the `patterns` config node); None = the
    defaults in config/patterns.yaml."""
    kind, fn = PATTERNS[pattern].kind, PATTERNS[pattern].fn
    P = load_params(params)

    def patterned(shell, r, th, Z, ri):
        """The wall with its pattern, at points given as flat arrays."""
        s = np.clip((r - ri) / pot.wall, 0, 1)
        band_lo = Z - (pot.base + PATTERN_GAP)    # pattern starts just above the base
        band_hi = (pot.height - pot.rim) - Z
        band = np.minimum(band_lo, band_hi)       # >0 inside the pattern band
        if kind == "3d":
            mat = smin(fn(pot, P, th, Z, r, s), band, BLEND)   # solid outside the band
        else:
            hole = np.maximum(fn(pot, P, th, Z, r, s), -band)  # holes only inside the band
            mat = -hole
        if pot.skin > 0:
            # blind pockets above skin_z: the pattern shows outside, but a solid
            # soil-side skin keeps water from running out through the wall
            skin = np.minimum(ri + pot.skin - r, Z - pot.skin_z)   # >0 inside the skin
            mat = np.minimum(mat, -skin)
        return np.maximum(shell, mat)

    def field(X, Y, Z):
        X, Y, Z = np.broadcast_arrays(X, Y, Z)
        r = np.hypot(X, Y)
        ro, ri = pot.r_out(Z), pot.r_in(Z)
        shell = np.maximum.reduce([r - ro, ri - r, -Z, Z - pot.height])
        body = shell.copy()
        near = shell < NEAR
        th = np.arctan2(Y[near], X[near])
        body[near] = patterned(shell[near], r[near], th, Z[near], ri[near])
        # the base, cup and chamfer are all below cup_h; higher up they are
        # more than NEAR away and change no value near zero (see NEAR)
        low = Z < pot.cup_h + NEAR
        body[low] = bottom(body[low], r[low], Z[low])
        return body

    def bottom(wall, r, Z):
        """Fuse the base and cup to the wall, at points given as flat arrays."""
        ci = pot.cup_ri(Z)
        co = ci + pot.cup_wall
        cup_wall = np.maximum.reduce([r - co, ci - r, -Z, Z - pot.cup_h])
        base = np.maximum.reduce([r - co, -Z, Z - pot.base])
        body = smin(smin(wall, base, 1.0), cup_wall, 1.0)
        # the blends also round the z = 0 faces and bulge below the bed; cut
        # flat so the whole base touches the first layer
        body = np.maximum(body, -Z)
        # 45 deg chamfer on the bottom outer edge (elephant foot)
        return np.maximum(body, ((r - co) + (CHAMFER - Z)) / np.sqrt(2))

    return field
