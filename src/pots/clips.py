"""
Optional wall mount (clips.enable): a flat rail that is screwed to the wall
and stays there, plus two pads on the back of the pot that drop onto it.

The rail is a strip with two keys standing out of it, one for the cup and one
for the top of the pot. A key is a 45 degree dovetail: it leaves the strip
`key` mm wide and spreads to `key` + 2 `grip` mm at its face. The pads carry
the matching sockets, open at the bottom, so the pot goes on by holding it a
little high against the rail and sliding it down `slide` mm, and comes off by
lifting it straight up again.

  - The upper socket is closed at the top by a 45 degree chisel, and the
    upper key ends in the same plane: that face carries the pot, and because
    it slopes inward it also pulls the pot in against the rail.
  - The lower socket runs right through, so it only keeps the bottom of the
    pot in and stops it swinging sideways. Nothing has to line up there.

Two screws hold the rail, their heads sunk into counterbores so the pads can
slide right over them: one just below the upper key, one just above the lower
key, both hidden behind the pot. The lower hole is a slot, so only the first
hole has to be drilled accurately.

Printability. The rail prints flat, wall face down and keys up: every layer
is the same as the one below or narrower, except the dovetail flanks and the
chisel, which are 45 degrees, and the counterbores, which are pockets with no
roof. On the pot the sockets are vertical channels (no roof at all in the
lower one, a 45 degree chisel in the upper one), the lower pad is a prism off
the build plate, and the upper pad's underside is a 45 degree wedge back into
the wall. So the mount adds no overhang and no bridge anywhere.

Where a pad sits, the wall is filled solid from the soil face out (and the
lower pad keys KEY_CUP mm into the cup wall): a pad that only covered the
mouths of the pattern's holes would close them into sealed pockets, since the
soil-side skin closes them on the other side. The solid patch is also what
the mount hangs from.

Sizes are all fixed in mm, like the other print-physics sizes: screws don't
scale with the pot. What does depend on the pot is how much room the cup
gives the lower key; `slide` is clamped to that, and a pot whose cup is too
short for any usable slide is refused.
"""
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from .geometry import CHAMFER, CUP_H_REF, H_REF, Pot

WEB = 1.6          # material left between the socket floor and the pot surface
ROOF = 2.0         # material above the top of the upper socket, under the rim
GAP = 1.0          # air between the rest of the pot and the rail
KEY = 0.4          # overlap between a pad and the surface it is keyed into
KEY_CUP = 1.0      # how far the lower pad keys into the cup wall (< cup_wall)
FLOOR = 1.6        # rail left under a sunk screw head
SIDE = 1.8         # material beside a screw head, and beside a socket
CAPTURE = 1.0      # material above and below the lower key
GRAB = 10.0        # engagement: how long the upper key is
LOW_GRAB = 6.0     # ... and the lower one, which only holds the bottom in
MIN_GRAB = 4.0     # the shortest lower key worth printing
SLOP = 2.0         # the lower screw hole is a slot, this much each way
LEAD = 1.0         # lead-in on the end of the lower key
END = 1.0          # rail beyond the keys
MIN_SLIDE = 2.5    # a shorter slide than this is not worth printing
ROOT2 = np.sqrt(2.0)


@dataclass(frozen=True)
class Clips:
    """One pot's wall mount: the two pads on the pot and the rail on the wall."""
    pot: Pot
    screw: float = 4.0        # wall screw shank diameter
    head: float = 9.0         # wall screw head diameter
    head_h: float = 3.0       # wall screw head thickness
    clear: float = 0.6        # clearance: socket to key, hole to screw
    key: float = 5.0          # width of a key where it leaves the rail
    grip: float = 2.5         # how far a key stands out, and spreads each side
    width: float = 16.0       # pad width, around the pot
    chamfer: float = 1.0      # 45 deg break on the pads' outside edges (0 = sharp)
    travel: float = 7.0       # how far the pot slides down to seat
    at: float = 180.0         # where around the pot the pads sit, degrees

    def __post_init__(self):
        if min(self.screw, self.head, self.head_h, self.clear, self.key,
               self.grip, self.width, self.travel) <= 0:
            raise ValueError("every clips size must be positive")
        if self.head <= self.screw:
            raise ValueError("clips.head must be wider than clips.screw")
        if self.slide < MIN_SLIDE:
            raise ValueError(
                f"clips need a taller pot: the cup is {self.pot.cup_h:.1f} mm tall, which "
                f"leaves a {max(self.cup_room, 0.0):.1f} mm slide; at least {MIN_SLIDE:g} mm "
                f"(height >= {self.min_height:.0f} mm) is needed")
        if self.width < self.min_width:
            raise ValueError(f"clips.width must be at least {self.min_width:.1f} mm "
                             "(the socket and the rail, plus some material beside them)")
        if self.width > self.pot.r_min:
            raise ValueError(f"clips.width must be at most {self.pot.r_min:.0f} mm on this pot "
                             "(the pads are flat, they would wrap too far around it)")
        if not 0 <= self.chamfer <= self.max_chamfer:
            raise ValueError(f"clips.chamfer must be between 0 and {self.max_chamfer:g} mm "
                             "(a bigger one would break into the socket)")

    # -------------------------------------------------------------- the socket
    @cached_property
    def slot_w(self):
        """Width of the socket's mouth, in the pad's back face."""
        return self.key + self.clear

    @cached_property
    def depth(self):
        """Socket depth: the key plus clearance under its face."""
        return self.grip + self.clear

    @cached_property
    def floor_w(self):
        """Width of the socket at its floor: it spreads 45 degrees on the way in."""
        return self.slot_w + 2 * self.depth

    @cached_property
    def boss(self):
        """Pad thickness where the socket is."""
        return self.depth + WEB

    @cached_property
    def min_width(self):
        """Narrowest pad that keeps material beside the socket and hides the rail."""
        return max(self.floor_w + 2 * SIDE, self.strip)

    @cached_property
    def max_chamfer(self):
        """The biggest edge break that still misses the socket. Both the socket
        and the breaks run at 45 degrees, so one bound holds at every depth."""
        return min((self.width - self.slot_w) / 2, ROOF + self.depth)

    # ----------------------------------------------------------- the lower pad
    @cached_property
    def cup_room(self):
        """The longest slide the lower key has room for on the cup."""
        return self.pot.cup_h - 0.5 - MIN_GRAB - 2 * CAPTURE

    @cached_property
    def slide(self):
        """How far the pot slides down to seat: `travel`, or what the cup allows."""
        return min(self.travel, self.cup_room)

    @cached_property
    def min_height(self):
        """Shortest pot this mount fits on (the cup scales with the height)."""
        return (MIN_SLIDE + 0.5 + MIN_GRAB + 2 * CAPTURE) * H_REF / CUP_H_REF

    @cached_property
    def low_grab(self):
        """Length of the lower key: LOW_GRAB, or what is left of the cup."""
        return min(LOW_GRAB, self.pot.cup_h - 0.5 - self.slide - 2 * CAPTURE)

    @cached_property
    def low_top(self):
        """Top of the lower pad (its socket is open there)."""
        return self.slide + self.low_grab + 2 * CAPTURE

    @cached_property
    def low_key(self):
        """(bottom, top) of the lower key, with the pot seated."""
        return self.slide + CAPTURE, self.low_top - CAPTURE

    # ----------------------------------------------------------- the upper pad
    @cached_property
    def top_z(self):
        """Top of the upper pad: flush with the rim."""
        return self.pot.height

    @cached_property
    def seat(self):
        """Top of the socket at its mouth. The roof is a 45 degree chisel, so
        its highest point (at the floor) is ROOF below the rim."""
        return self.top_z - ROOF - self.depth

    @cached_property
    def high_key(self):
        """(bottom, top) of the upper key at the rail's face, pot seated."""
        return self.seat - GRAB, self.seat

    @cached_property
    def mouth(self):
        """Bottom of the upper pad's back face: where its socket opens."""
        return self.seat - GRAB - self.slide

    @cached_property
    def slot_lo(self):
        """Lowest point of the upper socket (its floor meets the 45 deg underside)."""
        return self.mouth - self.depth

    # --------------------------------------------------------------- the rail
    @cached_property
    def plate(self):
        """Rail thickness: a sunk screw head plus the material under it."""
        return FLOOR + self.head_h + self.clear

    @cached_property
    def strip(self):
        """Rail width: enough for a sunk head, and for a key's root."""
        return max(self.key + 2 * self.grip + 2 * 0.8, self.head + self.clear + 2 * SIDE)

    @cached_property
    def bore(self):
        """Screw hole through the rail."""
        return self.screw + self.clear

    @cached_property
    def sink(self):
        """Counterbore the head drops into."""
        return self.head + self.clear

    @cached_property
    def screw_hi(self):
        """Height of the upper screw: just below the upper key."""
        return self.high_key[0] - (self.head / 2 + 1.5)

    @cached_property
    def screw_lo(self):
        """Height of the lower screw's slot centre: just above the lower key."""
        return self.low_key[1] + self.head / 2 + 1.5 + SLOP

    @cached_property
    def spacing(self):
        """Distance between the two holes: what the user has to drill."""
        return self.screw_hi - self.screw_lo

    @cached_property
    def rail_lo(self):
        """Bottom end of the rail, in pot heights (hidden behind the cup)."""
        return self.low_key[0] - END

    @cached_property
    def rail_hi(self):
        """Top end of the rail (hidden behind the rim)."""
        return self.high_key[1] + self.grip + END

    @cached_property
    def rail_len(self):
        return self.rail_hi - self.rail_lo

    # --------------------------------------------------------------- the frame
    @cached_property
    def env(self):
        """Widest radius of the pot itself: only the rail may be outside it."""
        p = self.pot
        z = np.linspace(0, p.height, 601)
        return max(float(p.r_out(z).max()), p.cup_lip + p.cup_wall)

    @cached_property
    def x_wall(self):
        """The plane the pads' back faces lie on, which is also the rail's
        face: far enough out that both pads are `boss` thick where their
        sockets are, and that nothing else on the pot touches the rail."""
        p = self.pot
        high = np.linspace(self.slot_lo, self.top_z, 201)
        return max(self.env + GAP,
                   float(p.r_out(high).max()) + self.boss,
                   float(p.cup_ri(self.low_top)) + p.cup_wall + self.boss)

    @cached_property
    def standoff(self):
        """How far the widest part of the pot ends up from the wall."""
        return self.x_wall + self.plate - self.env

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

    def world(self, X, Y, Z):
        """(x, y, z) in the pot's frame of a point of the rail, given in the
        rail's own frame (across, up from its bottom end, out from the wall)."""
        u = self.x_wall + self.plate - np.asarray(Z, float)
        v = np.asarray(X, float)
        a = np.radians(self.at)
        c, s = np.cos(a), np.sin(a)
        return u * c - v * s, u * s + v * c, self.rail_lo + np.asarray(Y, float)

    # -------------------------------------------------------------- the field
    def carve(self, body, X, Y, Z):
        """Add the pads to a field (negative = solid) and cut their sockets.
        Each socket is intersected with its own pad, so the clips can only
        remove material they added themselves."""
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

    def _socket(self, back, a, top=None):
        """The dovetail channel: it spreads 45 degrees as it goes in, so it
        needs no lips. `top` closes it with the 45 degree chisel."""
        terms = [(a + back - self.slot_w / 2) / ROOT2,       # the flanks
                 -self.depth - back]                         # the floor
        if top is not None:
            terms.append(top)
        return np.maximum.reduce(terms)

    def _high(self, r, u, a, Z):
        """The upper pad and its socket, as (pad, slot) fields."""
        back = u - self.x_wall
        pad = np.maximum.reduce([
            back,                                            # the rail's face
            a - self.width / 2,                              # sides
            Z - self.top_z,                                  # top, flush with the rim
            (back - (Z - self.mouth)) / ROOT2,               # 45 deg underside
            self.pot.r_in(Z) - KEY - r,                      # fills the wall, soil face out
            -u,                                              # this side of the pot only
            *self._edges(back, a, Z, self.top_z),            # broken edges
        ])
        return pad, self._socket(back, a, (Z - self.seat + back) / ROOT2)

    def _low(self, r, u, a, Z):
        """The lower pad (on the cup) and its socket, as (pad, slot) fields."""
        p = self.pot
        back = u - self.x_wall
        pad = np.maximum.reduce([
            back,                                            # the rail's face
            a - self.width / 2,                              # sides
            Z - self.low_top,                                # top
            -Z,                                              # the build plate
            (back + CHAMFER - Z) / ROOT2,                    # chamfer, bottom back edge
            p.cup_ri(Z) + p.cup_wall - self.cup_key - r,     # keyed into the cup wall
            -u,                                              # this side of the pot only
            *self._edges(back, a, Z, self.low_top),          # broken edges
        ])
        return pad, self._socket(back, a)                    # open top and bottom

    # -------------------------------------------------------------- the solid
    def solid(self, turn=None):
        """The two pads, sockets cut, as one manifold3d Manifold (for
        solid.py). `turn`: the revolve offset build_solid is using."""
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

        def socket(z0, z1, chisel=None):
            """The dovetail channel as a prism along z: a trapezoid spreading
            45 degrees inward, closed by the 45 degree `chisel` if it has one."""
            e, nar, d = 0.5, self.slot_w / 2, self.depth
            trap = [(x + e, nar - e), (x - d, nar + d), (x - d, -(nar + d)), (x + e, -(nar - e))]
            m = prism(trap, z0, z1, axis=2)
            if chisel is not None:
                wide = nar + d + 1
                top = [(x - d - 1, z0), (x + e + 1, z0),
                       (x + e + 1, chisel - e - 1), (x - d - 1, chisel + d + 1)]
                m = m ^ prism(top, -wide, wide, axis=1)
            return m

        p, x = self.pot, self.x_wall
        # ---- upper pad: back face on the rail's plane, 45 degree underside
        far = float(p.r_in(np.linspace(0, p.height, 200)).min()) - 5.0   # inside every surface
        zlo = self.mouth - (x - far)
        ch, top, zb = self.chamfer, self.top_z, zlo - 1
        wedge = [(far, zlo), (x, self.mouth), (x, top - ch), (x - ch, top), (far, top)]
        high = (pad(wedge, top, zb) - core(lambda z: p.r_in(z) - KEY, zb, top + 1)
                - socket(zb, self.seat + self.depth + 1, self.seat))
        # ---- lower pad: a prism out of the cup wall, from the build plate up
        top = self.low_top
        box = [(far, 0.0), (x - CHAMFER, 0.0), (x, CHAMFER), (x, top - ch), (x - ch, top), (far, top)]
        low = (pad(box, top, -1)
               - core(lambda z: p.cup_ri(z) + p.cup_wall - self.cup_key, -1, top + 1)
               - socket(-1, top + 1))
        return high + low

    # --------------------------------------------------------------- the rail
    def rail(self):
        """The wall rail as a trimesh, in its print orientation: the face that
        goes against the wall flat on the bed (z = 0) and the keys pointing up,
        +y up the wall, so the long key is at the top."""
        import trimesh
        from manifold3d import CrossSection, Manifold

        from .solid import _ccw

        def prism(sec, lo, hi, axis=2):
            """A section in the plane of the other two axes, extruded from `lo`
            to `hi` along `axis`: 0 = across (section in (up, out)), 1 = up
            (section in (across, out)), 2 = out (section in (across, up))."""
            m = Manifold.extrude(CrossSection([_ccw(np.asarray(sec, float))]),
                                 hi - lo).translate([0, 0, lo])
            if axis == 0:
                return m.rotate([90, 0, 0]).rotate([0, 0, 90])
            if axis == 1:
                return m.rotate([90, 0, 0]).translate([0, lo + hi, 0])
            return m

        def bore(y, slop, dia, z0, h):
            """A screw hole, or a slot `slop` long each way, from z0 up h."""
            c = Manifold.cylinder(h, dia / 2, circular_segments=48).translate([0, 0, z0])
            if not slop:
                return c.translate([0, y, 0])
            return (c.translate([0, y - slop, 0]) + c.translate([0, y + slop, 0])
                    + Manifold.cube([dia, 2 * slop, h]).translate([-dia / 2, y - slop, z0]))

        g, plate = self.grip, self.plate
        half, side, y1 = self.key / 2, self.strip / 2, self.rail_len
        body = prism([(-side, 0.0), (side, 0.0), (side, y1), (-side, y1)], 0.0, plate)
        # the dovetail: its flanks run on down into the plate, so the two fuse
        # without a single face or edge of either lying in the plate's face
        root = 0.8
        sec = [(-half + root, plate - root), (half - root, plate - root),
               (half + g, plate + g), (-half - g, plate + g)]
        wide = half + g + 1
        for (lo, hi), slant in ((self.high_key, True), (self.low_key, False)):
            lo, hi = lo - self.rail_lo, hi - self.rail_lo
            key = prism(sec, lo - 1, hi + g + 1, axis=1)
            if slant:   # both ends 45 deg, parallel to the socket and its roof
                ends = [(lo - 1, plate - 1), (hi - 1, plate - 1),
                        (hi + g + 1, plate + g + 1), (lo + g + 1, plate + g + 1)]
            else:       # square, with a lead-in on the tip of the top end
                ends = [(lo, plate - 1), (hi, plate - 1), (hi, plate + g - LEAD),
                        (hi - LEAD - 1, plate + g + 1), (lo, plate + g + 1)]
            body = body + (key ^ prism(ends, -wide, wide, axis=0))
        for z, slop in ((self.screw_hi, 0.0), (self.screw_lo, SLOP)):
            y, sunk = z - self.rail_lo, self.head_h + self.clear
            body = (body - bore(y, slop, self.bore, -1.0, plate + 2)
                    - bore(y, slop, self.sink, plate - sunk, sunk + 1))
        m = body.to_mesh()
        return trimesh.Trimesh(m.vert_properties[:, :3], m.tri_verts, process=True)

    def __str__(self):
        return (f"a {self.strip:.0f} x {self.rail_len:.0f} x {self.plate:.1f} mm rail on the "
                f"wall, held by two {self.screw:g} mm screws {self.spacing:.0f} mm apart "
                f"(heads sunk in, the lower hole a +-{SLOP:g} mm slot); the pot slides "
                f"{self.slide:.1f} mm down onto its keys and stands {self.standoff:.1f} mm "
                "off the wall")
