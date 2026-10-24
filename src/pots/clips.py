"""
Optional wall mount (clips.enable): two pads on the back of the pot, each
with a T-slot, so the pot hangs on a pair of screws and lifts straight off
again.

Both pads end on one vertical plane (`x_wall`), a little outside everything
else, so the pot meets the wall on those two flat faces only. Both slots are
open at the bottom: hold the pot with the two screw heads at the mouths, push
it flat against the wall and slide it down `slide` mm.

  - The upper pad is at the top of the pot, its top face flush with the rim.
    Its slot ends in a flat seat as wide as the slot: the screw shank rests
    there and carries the pot.
  - The lower pad grows out of the cup wall, straight up from the build
    plate. Its slot runs right through, so it only holds the bottom of the
    pot against the wall (the pot hangs from the upper screw, which already
    presses the bottom in) and stops it swinging sideways. Nothing has to
    line up exactly: a few mm of drilling error only moves the lower screw
    along its slot.

Printability: the lower pad is a prism rising from the build plate, so it has
no overhang at all; the upper pad is a prism tangent to the wall whose
underside is a 45 degree wedge back into the pot wall. The only roofs in the
slots are the seat (a flat bridge as wide as the slot, ~4.6 mm) and a 45
degree ramp over the head recess.

Where a pad sits, the wall is filled solid from the soil face out (and the
lower pad keys KEY_CUP mm into the cup wall): a pad that only covered the
mouths of the pattern's holes would close them into sealed pockets, since the
soil-side skin closes them on the other side. The solid patch is also what
the mount hangs from.

Sizes are all fixed in mm, like the other print-physics sizes: screws don't
scale with the pot. What does depend on the pot is how much room the cup
gives the lower pad; `slide` is clamped to that (and a pot whose cup is too
short for any usable slide is refused).
"""
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from .geometry import CHAMFER, CUP_H_REF, H_REF, Pot

WEB = 1.6          # material left between the slot floor and the pot surface
ROOF = 2.0         # material above the top of the slot, under the rim
STANDOFF = 1.0     # the rest of the pot keeps at least this much air to the wall
KEY = 0.4          # overlap between a pad and the surface it is keyed into
KEY_CUP = 1.0      # how far the lower pad keys into the cup wall (< cup_wall)
HEAD_GAP = 0.3     # the seated screw head stops this far below the top of its recess
CAPTURE = 1.0      # material above and below the seated head in the lower pad
MIN_SLIDE = 2.5    # a shorter slide than this is not worth printing
ROOT2 = np.sqrt(2.0)


@dataclass(frozen=True)
class Clips:
    """The two wall-mount pads of one pot: all their sizes and positions."""
    pot: Pot
    screw: float = 4.0        # screw shank diameter
    head: float = 9.0         # screw head diameter
    head_h: float = 3.0       # screw head thickness
    clear: float = 0.6        # clearance around the screw
    lip: float = 2.0          # lips that trap the head, measured from the back face
    width: float = 16.0       # pad width, around the pot
    chamfer: float = 1.0      # 45 deg break on the pads' outside edges (0 = sharp)
    travel: float = 7.0       # how far the pot slides down to seat
    at: float = 180.0         # where around the pot the pads sit, degrees

    def __post_init__(self):
        if min(self.screw, self.head, self.head_h, self.clear, self.lip,
               self.width, self.travel) <= 0:
            raise ValueError("every clips size must be positive")
        if self.head <= self.screw:
            raise ValueError("clips.head must be wider than clips.screw")
        if self.slide < MIN_SLIDE:
            raise ValueError(
                f"clips need a taller pot: the cup is {self.pot.cup_h:.1f} mm tall, which "
                f"leaves a {max(self.cup_room, 0.0):.1f} mm slide; at least {MIN_SLIDE:g} mm "
                f"(height >= {self.min_height:.0f} mm, or a smaller clips.head) is needed")
        if self.width < self.head + 4 * self.clear:
            raise ValueError(f"clips.width must be at least {self.head + 4 * self.clear:g} mm "
                             "(the head recess plus some material beside it)")
        if self.width > self.pot.r_min:
            raise ValueError(f"clips.width must be at most {self.pot.r_min:.0f} mm on this pot "
                             "(the pads are flat, they would wrap too far around it)")
        if not 0 <= self.chamfer <= self.max_chamfer:
            raise ValueError(f"clips.chamfer must be between 0 and {self.max_chamfer:g} mm "
                             "(a bigger one would break into the slot)")

    # ---------------------------------------------------------------- the slot
    @cached_property
    def slot_w(self):
        """Width of the slot in the back face: the shank slides along it."""
        return self.screw + self.clear

    @cached_property
    def head_w(self):
        """Width of the head recess behind the lips."""
        return self.head + self.clear

    @cached_property
    def head_d(self):
        """Depth of the head recess."""
        return self.head_h + self.clear

    @cached_property
    def depth(self):
        """Slot depth from the back face: how far the heads stand off the wall."""
        return self.lip + self.head_d

    @cached_property
    def boss(self):
        """Pad thickness where the slot is."""
        return self.depth + WEB

    @cached_property
    def max_chamfer(self):
        """The biggest edge break that still misses the slot: it must stay
        beside the head recess and below the material over the slot."""
        return min((self.width - self.head_w) / 2, self.lip + ROOF)

    @property
    def stand_min(self):
        """Least the screw heads may stand off the wall (else they foul the lips)."""
        return self.lip + self.head_h

    # ----------------------------------------------------------- the lower pad
    @cached_property
    def cup_room(self):
        """The longest slide the lower pad has room for on the cup."""
        return self.pot.cup_h - 2 * CAPTURE - 0.5 - self.head_w / 2

    @cached_property
    def slide(self):
        """How far the pot slides down to seat: `travel`, or what the cup allows."""
        return min(self.travel, self.cup_room)

    @cached_property
    def min_height(self):
        """Shortest pot these screws fit on (the cup scales with the height)."""
        return (MIN_SLIDE + 2 * CAPTURE + 0.5 + self.head_w / 2) * H_REF / CUP_H_REF

    @cached_property
    def low_screw(self):
        """Height of the lower screw's axis once the pot is seated."""
        return self.slide + CAPTURE

    @cached_property
    def low_top(self):
        """Top of the lower pad."""
        return self.low_screw + self.head_w / 2 + CAPTURE

    # ----------------------------------------------------------- the upper pad
    @cached_property
    def top_z(self):
        """Top of the upper pad: flush with the rim."""
        return self.pot.height

    @cached_property
    def apex(self):
        """Top of the head recess, at the lip face (the highest point of the slot)."""
        return self.top_z - ROOF

    @cached_property
    def z_head(self):
        """Top of the head recess at its floor: the 45 degree ramp starts here."""
        return self.apex - self.head_d

    @cached_property
    def seat(self):
        """Top of the shank slot: the flat bridge the pot hangs from."""
        return self.z_head - self.head_w / 2 + self.screw / 2 - HEAD_GAP

    @cached_property
    def high_screw(self):
        """Height of the upper screw's axis once the pot is seated."""
        return self.seat - self.screw / 2

    @cached_property
    def mouth(self):
        """Bottom of the upper pad's back face: where its slot opens."""
        return self.seat - self.screw - self.slide

    @cached_property
    def slot_lo(self):
        """Lowest point of the upper slot (its floor meets the 45 degree underside)."""
        return self.mouth - self.depth

    # --------------------------------------------------------------- the frame
    @cached_property
    def spacing(self):
        """Distance between the two screws: what the user has to drill."""
        return self.high_screw - self.low_screw

    @cached_property
    def x_wall(self):
        """The wall plane: far enough out that both pads are `boss` thick where
        their slots are, and that nothing else on the pot touches the wall."""
        p = self.pot
        z = np.linspace(0, p.height, 601)
        env = max(float(p.r_out(z).max()), p.cup_lip + p.cup_wall) + STANDOFF
        high = np.linspace(self.slot_lo, self.top_z, 201)
        return max(env,
                   float(p.r_out(high).max()) + self.boss,
                   float(p.cup_ri(self.low_top)) + p.cup_wall + self.boss)

    @cached_property
    def r_max(self):
        """Radius of a cylinder that contains the pot and its pads."""
        return float(np.hypot(self.x_wall, self.width / 2)) + 1.0

    @cached_property
    def cup_key(self):
        """How far the lower pad reaches into the cup wall."""
        return min(KEY_CUP, max(self.pot.cup_wall - 0.8, 0.2))

    def local(self, X, Y):
        """(out, along): coordinates with the pads on the +out axis."""
        a = np.radians(self.at)
        c, s = np.cos(a), np.sin(a)
        return X * c + Y * s, Y * c - X * s

    # ------------------------------------------------------------- the field
    def carve(self, body, X, Y, Z):
        """Add the pads to a field (negative = solid) and cut their slots. Each
        slot is intersected with its own pad, so the clips can only remove
        material they added themselves."""
        u, v = self.local(X, Y)
        a, r = np.abs(v), np.hypot(X, Y)
        for pad, slot in (self._high(r, u, a, Z), self._low(r, u, a, Z)):
            body = np.maximum(np.minimum(body, pad), -np.maximum(slot, pad))
        return body

    def _edges(self, back, a, Z, top):
        """The 45 degree breaks on a pad's three outside edge families: where
        the back face meets the top, where it meets the sides, and where the
        top meets the sides. All convex, all cut by one plane each."""
        ch, half = self.chamfer, self.width / 2
        if ch <= 0:
            return []
        return [(back + Z - (top - ch)) / ROOT2,
                (back + a - (half - ch)) / ROOT2,
                (a + Z - (half + top - ch)) / ROOT2]

    def _slot(self, back, a, top=None, ramp=None):
        """The T-slot: a narrow slot in the back face up to `top`, and the head
        recess behind the lips, closed by a 45 degree `ramp` from `ramp` up."""
        narrow = [a - self.slot_w / 2, -self.lip - back]
        wide = [a - self.head_w / 2, -self.depth - back, back + self.lip]
        if top is not None:
            narrow.append(top)
            wide.append(ramp)
        return np.minimum(np.maximum.reduce(narrow), np.maximum.reduce(wide))

    def _high(self, r, u, a, Z):
        """The upper pad and its slot, as (pad, slot) fields."""
        back = u - self.x_wall
        pad = np.maximum.reduce([
            back,                                            # the wall plane
            a - self.width / 2,                              # sides
            Z - self.top_z,                                  # top, flush with the rim
            (back - (Z - self.mouth)) / ROOT2,               # 45 deg underside
            self.pot.r_in(Z) - KEY - r,                      # fills the wall, soil face out
            -u,                                              # this side of the pot only
            *self._edges(back, a, Z, self.top_z),            # broken edges
        ])
        slot = self._slot(back, a, Z - self.seat,
                          ((Z - self.z_head) - (back + self.depth)) / ROOT2)
        return pad, slot

    def _low(self, r, u, a, Z):
        """The lower pad (on the cup) and its slot, as (pad, slot) fields."""
        p = self.pot
        back = u - self.x_wall
        pad = np.maximum.reduce([
            back,                                            # the wall plane
            a - self.width / 2,                              # sides
            Z - self.low_top,                                # top
            -Z,                                              # the build plate
            (back + CHAMFER - Z) / ROOT2,                    # chamfer, bottom back edge
            p.cup_ri(Z) + p.cup_wall - self.cup_key - r,     # keyed into the cup wall
            -u,                                              # this side of the pot only
            *self._edges(back, a, Z, self.low_top),          # broken edges
        ])
        return pad, self._slot(back, a)                      # open top and bottom

    # ------------------------------------------------------------- the solid
    def solid(self, turn=None):
        """The two pads, slots cut, as one manifold3d Manifold (for solid.py).
        `turn`: the revolve offset build_solid is using (see solid.TURNS)."""
        from manifold3d import CrossSection, Manifold

        from .solid import TURNS, _ccw, _revolve

        def prism(sec, lo, hi, axis=1):
            """A section in the plane of the other two axes, extruded from `lo`
            to `hi` along `axis` (0 = out, 1 = across the pad, 2 = up), and
            turned to the pads' place on the pot. Intersecting prisms along
            different axes is how the edge chamfers are cut."""
            m = Manifold.extrude(CrossSection([_ccw(np.asarray(sec, float))]),
                                 hi - lo).translate([0, 0, lo])
            if axis == 0:                                  # section in (along, up)
                m = m.rotate([90, 0, 0]).rotate([0, 0, 90])
            elif axis == 1:                                # section in (out, up)
                m = m.rotate([90, 0, 0])
            return m.rotate([0, 0, self.at])

        def sides(top, zbot):
            """The two prisms that break a pad's edges: its plan with the back
            corners cut, and its back view with the top corners cut."""
            ch, half = self.chamfer, self.width / 2
            if ch <= 0:
                return []
            plan = [(far, -half), (x - ch, -half), (x, ch - half),
                    (x, half - ch), (x - ch, half), (far, half)]
            front = [(-half, zbot), (half, zbot), (half, top - ch),
                     (half - ch, top), (ch - half, top), (-half, top - ch)]
            return [prism(plan, zbot, top + 1, axis=2), prism(front, far, x + 1, axis=0)]

        def pad(sec, top, zbot):
            """One pad: its (out, up) section across the pad width, with the
            outside edges broken."""
            m = prism(sec, -self.width / 2, self.width / 2)
            for cut in sides(top, zbot):
                m = m ^ cut
            return m

        def core(radius, z0, z1, n=300):
            """Everything inside `radius(z)`, revolved: subtracted from a pad so
            it only fills outward from the pot's own surface."""
            z = np.linspace(z0, z1, n)
            pts = [(0.0, z[0]), *zip(np.maximum(radius(z), 0.0), z), (0.0, z[-1])]
            sec = CrossSection([_ccw(np.asarray(pts, float))]).simplify(0.005)
            return _revolve(sec, self.x_wall, TURNS[0] if turn is None else turn)

        p, x = self.pot, self.x_wall
        deep, lipx = x - self.depth, x - self.lip
        # ---- upper pad: back face on the wall plane, 45 degree underside
        far = float(p.r_in(np.linspace(0, p.height, 200)).min()) - 5.0   # inside every surface
        zlo = self.mouth - (x - far)
        ch, nar, wide = self.chamfer, self.slot_w / 2, self.head_w / 2
        top, zb = self.top_z, zlo - 1
        wedge = [(far, zlo), (x, self.mouth), (x, top - ch), (x - ch, top), (far, top)]
        high = pad(wedge, top, zb) - core(lambda z: p.r_in(z) - KEY, zb, top + 1) - (
            prism([(lipx, zb), (x + 1, zb), (x + 1, self.seat), (lipx, self.seat)], -nar, nar)
            + prism([(deep, zb), (lipx, zb), (lipx, self.apex), (deep, self.z_head)], -wide, wide))
        # ---- lower pad: a prism out of the cup wall, from the build plate up
        top, t = self.low_top, self.low_top + 1
        box = [(far, 0.0), (x - CHAMFER, 0.0), (x, CHAMFER), (x, top - ch), (x - ch, top), (far, top)]
        low = pad(box, top, -1) - core(
            lambda z: p.cup_ri(z) + p.cup_wall - self.cup_key, -1, t) - (
            prism([(lipx, -1), (x + 1, -1), (x + 1, t), (lipx, t)], -nar, nar)
            + prism([(deep, -1), (lipx, -1), (lipx, t), (deep, t)], -wide, wide))
        return high + low

    def __str__(self):
        return (f"two screws (shank {self.screw:g} mm, head up to {self.head:g} x "
                f"{self.head_h:g} mm) {self.spacing:.1f} mm apart vertically, standing "
                f"{self.stand_min:.1f}-{self.depth:.1f} mm off the wall; the pot slides "
                f"down {self.slide:.1f} mm onto them")
