"""
Wall-pattern metrics, measured straight from the field.

Both faces of the wall are sampled in a window at mid-height at the same
arc length, theta = arc / r_out(z), so matching pixels lie on one radial ray.
"""
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

OUTER, INNER = 0.02, 0.98     # sample depth into the wall, from the outside


@dataclass
class FaceWindow:
    arc: np.ndarray       # arc length along the outer face (mm), 1D
    z: np.ndarray         # heights (mm), 1D
    px: float

    @property
    def extent(self):
        return [self.arc[0], self.arc[-1] + self.px, self.z[0], self.z[-1] + self.px]


def face_window(pot, width=70.0, height=50.0, px=0.1):
    """A window at mid-height, shrunk on small pots to stay inside the pattern band."""
    wz = min(height, pot.height - pot.base - pot.rim - 4.0)
    ww = min(width, np.pi * pot.r_bot)
    z0 = pot.height / 2 - wz / 2
    return FaceWindow(np.arange(0, ww, px), np.arange(z0, z0 + wz, px), px)


def sample_face(field, pot, frac, win):
    """Boolean image (z, arc) of where the wall is open, at depth `frac` of the wall."""
    A, Z = np.meshgrid(win.arc, win.z)
    TH = A / pot.r_out(Z)
    R = pot.r_out(Z) - pot.wall * frac
    return field((R * np.cos(TH)).astype(np.float32), (R * np.sin(TH)).astype(np.float32),
                 Z.astype(np.float32)) >= 0


def hole_diameter(open_, px):
    """Median hole diameter (mm): twice the largest inscribed radius of each hole."""
    labels, n = ndimage.label(open_)
    if n == 0:
        return 0.0
    edt = ndimage.distance_transform_edt(open_) * px
    return float(np.median(ndimage.maximum(edt, labels, np.arange(1, n + 1))) * 2)


@dataclass
class WallMetrics:
    outer_open: float     # fraction
    inner_open: float
    outer_hole: float     # median hole diameter (mm)
    inner_hole: float
    through: float        # fraction open on both faces along the same ray

    def __str__(self):
        return (f"outside {self.outer_open:.0%} open, holes ~{self.outer_hole:.1f} mm | "
                f"soil side {self.inner_open:.0%} open, holes ~{self.inner_hole:.1f} mm | "
                f"straight-through {self.through:.0%}")


def wall_metrics(field, pot, win=None):
    win = win or face_window(pot)
    out = sample_face(field, pot, OUTER, win)
    inn = sample_face(field, pot, INNER, win)
    return WallMetrics(out.mean(), inn.mean(), hole_diameter(out, win.px),
                       hole_diameter(inn, win.px), (out & inn).mean())
