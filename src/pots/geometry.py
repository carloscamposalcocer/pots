"""
Pot dimensions.

`Pot` holds every size of one pot. The height drives the shape: radii, cup
height, moat gap and the cell counts around the circumference all scale by
k = height / H_REF. The print-physics sizes (wall, rim, base, cup wall)
stay fixed in mm because they depend on the nozzle and the soil,
not on how big the pot is.

All dimensions in mm.
"""
from dataclasses import dataclass
from functools import cached_property

import numpy as np

H_REF = 130.0         # reference height; the shape below is defined at this size
R_BOT_REF = 56.0      # outer radius at the bottom, at H_REF
R_TOP_REF = 72.0      # outer radius at the top, at H_REF
CUP_H_REF = 28.0      # cup height from the build plate, at H_REF
CUP_GAP_REF = 5.0     # moat width at the bottom between pot and cup, at H_REF
R0_REF = 64.0         # radius of the unrolled (s, z) pattern coordinates, at H_REF
LIP = 1.0             # the cup lip sits this far outside the top of the pot
CHAMFER = 0.6         # 45 deg chamfer on the bottom outer edge (elephant foot)
PATTERN_GAP = 1.0     # the wall pattern starts this far above the base
MIN_PATTERN = 10.0    # minimum patterned band height


@dataclass(frozen=True)
class Pot:
    height: float = H_REF
    wall: float = 6.0          # lattice wall thickness (radial)
    rim: float = 5.0           # solid band at the top
    base: float = 2.4          # solid floor shared by pot and cup
    cup_wall: float = 2.4      # drip cup wall thickness
    skin: float = 0.0          # solid soil-side layer behind the pattern (0 = holes go through)
    skin_frac: float = 0.0     # share of the height, from the top, that gets the skin

    def __post_init__(self):
        if not 0 <= self.skin_frac <= 1:
            raise ValueError("skin_frac must be between 0 and 1")
        if not 0 <= self.skin < self.wall:
            raise ValueError("skin must be at least 0 and thinner than the wall")
        if self.height < self.min_height:
            raise ValueError(f"height must be at least {self.min_height:g} mm "
                             "(base + rim + some pattern)")

    @property
    def min_height(self):
        return self.base + self.rim + MIN_PATTERN

    @cached_property
    def k(self):
        """Scale factor relative to the reference pot."""
        return self.height / H_REF

    @cached_property
    def r_bot(self):
        """Outer radius at the bottom."""
        return R_BOT_REF * self.k

    @cached_property
    def r_top(self):
        """Outer radius at the top."""
        return R_TOP_REF * self.k

    @cached_property
    def cup_h(self):
        return CUP_H_REF * self.k

    @cached_property
    def cup_gap(self):
        return CUP_GAP_REF * self.k

    @cached_property
    def cup_flare(self):
        # lip just wider than the pot top, so vertical drips land in the cup
        return (self.r_top + LIP) - (self.r_bot + self.cup_gap)

    @cached_property
    def skin_z(self):
        """Height above which the soil side of the wall is closed by the skin."""
        return self.height * (1 - self.skin_frac)

    @cached_property
    def r0(self):
        """Radius at which the wall pattern is unrolled to (s, z) mm."""
        return R0_REF * self.k

    @cached_property
    def circ(self):
        return 2 * np.pi * self.r0

    @property
    def r_max(self):
        """Radius of a cylinder that contains the whole pot and cup."""
        return self.r_top + LIP + self.cup_wall + 1.0

    def count(self, ref):
        """Scale a cell count around the circumference, keeping it an integer
        (so the pattern stays seamless) and the cell size roughly constant."""
        return max(3, round(ref * self.k))

    def r_out(self, z):
        return self.r_bot + (self.r_top - self.r_bot) * np.clip(z, 0, self.height) / self.height

    def r_in(self, z):
        return self.r_out(z) - self.wall

    def cup_ri(self, z):
        """Inner radius of the cup."""
        return self.r_bot + self.cup_gap + self.cup_flare * np.clip(z, 0, self.cup_h) / self.cup_h
