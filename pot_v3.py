"""
Breathing plant pot v3: one single print, no supports.

- Lattice wall (graded gyroid) from the base up to a solid rim.
- Plain, smooth cup around the bottom, fused to the same solid base, with a
  moat between cup and lattice. Drips and drainage collect in the moat and
  the bottom of the pot; the soil wicks that water back up.

All dimensions in mm. Negative field = solid.
Usage:  python3 pot_v3.py [voxel_mm] [out.stl]
(Size: call set_height(h) first; make_pot.py --height does this.)
"""
import logging
import sys
import numpy as np
import pot_v2 as P2
from pot_v2 import WALL, RIM, T_IN, T_OUT, BLEND, r_out, r_in, smin, gyroid, build

BASE = 2.4            # shared solid floor
CUP_H_REF = 28.0      # cup height at P2.H_REF, measured from the build plate
CUP_GAP_REF = 5.0     # moat width at the bottom, at P2.H_REF
LIP = 1.0             # cup lip sticks out this far past the pot top
CUP_WALL = 2.4
CUP_H = CUP_GAP = CUP_FLARE = None   # set by set_height()


def set_height(h):
    """Scale pot and cup to height h (mm); see pot_v2.set_height."""
    global CUP_H, CUP_GAP, CUP_FLARE
    P2.set_height(h)
    CUP_H = CUP_H_REF * P2.K
    CUP_GAP = CUP_GAP_REF * P2.K
    # lip just wider than the pot top, so vertical drips land in the cup
    CUP_FLARE = (P2.R_TOP + LIP) - (P2.R_BOT + CUP_GAP)


set_height(P2.H_REF)


def cup_ri(z):
    return P2.R_BOT + CUP_GAP + CUP_FLARE * np.clip(z, 0, CUP_H) / CUP_H


def field(X, Y, Z):
    r = np.sqrt(X * X + Y * Y)
    th = np.arctan2(Y, X)
    ro, ri = r_out(Z), r_in(Z)

    # lattice pot wall, standing on the base, up to the solid rim
    shell = np.maximum.reduce([r - ro, ri - r, -Z, Z - P2.H])
    s = np.clip((r - ri) / WALL, 0, 1)
    dg = gyroid(th, Z, r, T_IN + (T_OUT - T_IN) * s)
    rim = (P2.H - RIM) - Z
    wall = np.maximum(shell, smin(dg, rim, BLEND))

    # plain cup wall + shared solid base
    ci = cup_ri(Z)
    co = ci + CUP_WALL
    cup_wall = np.maximum.reduce([r - co, ci - r, -Z, Z - CUP_H])
    base = np.maximum.reduce([r - co, -Z, Z - BASE])

    body = smin(smin(wall, base, 1.0), cup_wall, 1.0)
    # 45 deg chamfer on the bottom outer edge (elephant foot)
    body = np.maximum(body, ((r - co) + (0.6 - Z)) / np.sqrt(2))
    return body


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-5s  %(message)s", datefmt="%H:%M:%S")
    vox = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
    out = sys.argv[2] if len(sys.argv) > 2 else "pot_v3_raw.stl"
    m = build(field, P2.R_TOP + LIP + CUP_WALL + 1.0, P2.H, vox, out)
    logging.getLogger(__name__).info("faces=%s  volume=%.1f cm3", f"{len(m.faces):,}", m.volume / 1000)
