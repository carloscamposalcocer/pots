"""
Wall holder (holder.enable): one separate printed part that holds an
unchanged pot against the wall.

A back plate is screwed to the wall. At its top a bridge reaches over the
rim, and a finger hangs `drop` mm down inside it; at its bottom a seat hugs
the back of the cup, with a ledge `ledge` mm under the base. The ledge
carries the pot. The pot's weight sits on its axis, in front of the ledge,
so the pot wants to tip forward: the finger stops that by holding the rim
in, and the seat takes the matching push of the cup back against the plate.

Everything that touches the pot follows its surface: the part is drawn as
a section in the plane through the pot's axis, extruded `width` mm across,
and then the pot itself, grown by `clear`, is cut out of it (revolved, so
the faces curve round the pot). Every size comes from the Pot, so the
holder fits any shape and height.

Taking the pot out: lift it (there is `lift` mm of air above the rim), swing
the bottom out about the finger's tip until it is past the ledge, slide it
down its own tilted axis off the finger and away. Putting it in is the same
backwards. The finger's outer face is vertical, inside the narrowest part
of the rim it passes, so the rim slides up and down past it whatever the
shape leans; the tests check this path for every shape (a swing of 4 to 6
degrees on the 100 to 130 mm pots, more on small ones).

Printability. The part prints lying on its side (the section flat on the
bed): every layer is the same section, so there is no overhang at all,
except the two screw holes, which run across the layers and get a
teardrop roof (45 degrees). The screw heads are sunk into counterbores in
the plate's front, behind the pot.
"""
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from .geometry import Pot

FLOOR = 1.6        # plate left behind a sunk screw head
EDGE = 1.0         # 45 deg break on the outside corners and the lead-ins
GAP = 0.2          # extra air between the cup lip and the plate (keeps them from touching)
SIDE = 2.0         # material between a screw head and the seat or the finger
INTO = 0.5         # the section reaches this far into the pot, so the cut makes the face
STEP = 0.25        # the bridge's top is `thick` - STEP above the rim
ROOT2 = np.sqrt(2.0)


@dataclass(frozen=True)
class Holder:
    """A wall holder for one pot."""
    pot: Pot
    screw: float = 4.0        # wall screw shank diameter
    head: float = 9.0         # wall screw head diameter
    head_h: float = 3.0       # wall screw head thickness
    clear: float = 0.5        # air between the holder and the pot, and around the screws
    width: float = 20.0       # how wide the holder is, across the pot
    thick: float = 4.0        # thickness of the bridge, the finger and the ledge
    drop: float = 6.0         # how far the finger reaches down inside the rim
    ledge: float = 6.0        # how far the ledge reaches in under the base
    lift: float = 1.5         # air above the rim: how far the pot lifts to come out
    groove: float = 1.5       # how deep the rim sits up in a groove under the bridge (<= lift to swing out rigidly)

    printed = "on its side"

    def __post_init__(self):
        if min(self.screw, self.head, self.head_h, self.width, self.thick, self.groove,
               self.drop, self.ledge) <= 0:
            raise ValueError("every holder size must be positive")
        if min(self.clear, self.lift) < 0:
            raise ValueError("holder.clear and holder.lift can't be negative (0: touching the pot)")
        if self.head <= self.screw:
            raise ValueError("holder.head must be wider than holder.screw")
        if self.screw_hi - self.screw_lo < self.sink + SIDE:
            raise ValueError(f"the pot is too short for the holder: there is no room for two "
                             f"screws between the cup and the finger (height >= "
                             f"{self.min_height:.0f} mm)")
        if self.width < self.min_width:
            raise ValueError(f"holder.width must be at least {self.min_width:.1f} mm "
                             "(the counterbore and its teardrop roof)")
        if self.width > self.pot.r_min:
            raise ValueError(f"holder.width must be at most {self.pot.r_min:.0f} mm on this pot "
                             "(the plate is flat, it would wrap too far around it)")
        if self.groove > self.drop:
            raise ValueError("holder.groove can't be deeper than holder.drop")
        if self.drop > self.pot.height - self.pot.cup_h - 2 * self.thick:
            raise ValueError("holder.drop reaches down to the cup on this pot")
        if self.ledge >= self.env(0.0) - self.thick:
            raise ValueError("holder.ledge reaches past the middle of the base")

    # ------------------------------------------------------------ the pot
    def env(self, z):
        """Outer radius of the pot at height z: the body, or the cup where it is wider."""
        p = self.pot
        r = p.r_out(z)
        cup = np.where(np.asarray(z) <= p.cup_h, p.cup_ri(z) + p.cup_wall, 0.0)
        return np.maximum(r, cup)

    @cached_property
    def env_max(self):
        z = np.linspace(0, self.pot.height, 2000)
        return float(self.env(z).max())

    # ------------------------------------------------------------ the part
    @property
    def bore(self):
        return self.screw + self.clear

    @property
    def sink(self):
        """Counterbore diameter."""
        return self.head + self.clear

    @property
    def min_width(self):
        return self.sink * ROOT2 + 2 * EDGE

    @cached_property
    def plate(self):
        """Back plate thickness: a sunk screw head plus the floor under it."""
        return FLOOR + self.head_h + self.clear

    @cached_property
    def face(self):
        """Distance from the pot axis to the plate's front face."""
        return self.env_max + self.clear + GAP

    @cached_property
    def x_wall(self):
        """Distance from the pot axis to the wall."""
        return self.face + self.plate

    @property
    def standoff(self):
        """Gap between the pot and the wall."""
        return self.x_wall - self.env_max

    @property
    def bottom(self):
        return -self.thick

    @property
    def plate_bottom(self):
        return self.bottom

    @property
    def under(self):
        """Underside of the bridge, beside the rim: `groove` below the top of
        the rim's slot, so the rim sits in a groove between the finger and
        the groove's outer side, which follows the outside of the rim. Over
        the rim the cut takes it up to `lift` above the rim (no face of the
        section lies in a face of the cut)."""
        return self.pot.height + self.lift - self.groove

    @property
    def top(self):
        return self.pot.height + self.lift - STEP + self.thick

    @property
    def tip(self):
        """Lowest point of the finger."""
        return self.pot.height - self.drop

    @cached_property
    def finger(self):
        """Radius of the finger's outer face: vertical, `clear` inside the
        narrowest part of the rim it sees as the pot lifts `lift` and drops
        off it, so the rim slides straight up and down past it whatever the
        shape leans."""
        z = np.linspace(self.tip - self.lift, self.pot.height, 200)
        return float(self.pot.r_in(z).min()) - self.clear

    @property
    def screw_lo(self):
        return self.pot.cup_h + self.sink / 2 + SIDE

    @property
    def screw_hi(self):
        return self.tip - self.sink / 2 - SIDE

    @property
    def screws(self):
        """Heights of the screws."""
        return (self.screw_lo, self.screw_hi)

    def sink_from(self, z):
        """Where the counterbore of the screw at height z starts (from the pot
        side): the plate's front."""
        return self.face - 3

    @property
    def min_height(self):
        """Shortest pot (same proportions) with room for both screws."""
        from .geometry import CUP_H_REF, H_REF
        need = self.drop + 2 * self.sink + 3 * SIDE
        return need / (1 - CUP_H_REF / H_REF)

    def section(self):
        """The part's outline in the (u, z) plane: u from the pot axis toward
        the wall, z up from the pot's bed. Where it overlaps the pot (+ clear)
        it is cut back by `cutter`."""
        from manifold3d import CrossSection

        from .solid import _ccw
        p, e, t = self.pot, EDGE, self.thick
        f, w, B, U, T = self.face, self.x_wall, self.bottom, self.under, self.top
        plate = [(f, B), (w - e, B), (w, B + e), (w, T - e), (w - e, T), (f, T)]
        # bridge + finger: the finger's outer side is drawn INTO mm into the
        # rim, the cut makes the real face (round the pot, `clear` off it),
        # with a lead-in on the tip
        r, tip = self.finger, self.tip
        hook = [(f, U), (r + INTO, U), (r + INTO, tip + e + INTO), (r - e, tip), (r - t + e, tip),
                (r - t, tip + e), (r - t, T - e), (r - t + e, T), (f, T)]
        # seat: a block under and behind the cup; the cut leaves the ledge
        # under the base and the face following the cup. Its front edge
        # crosses z = 0 one EDGE in, so the ledge gets a lead-in.
        lo = self.env(0.0) - self.ledge
        seat = [(lo, B + e), (lo + e, B), (f, B), (f, p.cup_h - GAP), (lo + 1.5 * e, p.cup_h - GAP),
                (lo + 1.5 * e, 0.5 * e), (lo, -0.5 * e)]
        return CrossSection([_ccw(s) for s in (plate, hook, seat)])

    def cutter(self):
        """The pot grown by `clear`, as a revolved (r, z) section: solid
        below the finger, and out from the finger's face up where it is."""
        from manifold3d import CrossSection

        from .solid import _ccw
        p, c = self.pot, self.clear
        z_mid = (p.cup_h + self.tip) / 2
        zs = np.linspace(0, p.height, int(np.ceil(p.height / 0.25)) + 1)
        zs = np.union1d(zs, [p.cup_h, np.nextafter(p.cup_h, np.inf)])
        top = p.height + self.lift
        pts = ([(0.0, 0.0)] + [(float(self.env(z)) + c, z) for z in zs]
               + [(float(self.env(p.height)) + c, top), (self.finger, top),
                  (self.finger, z_mid), (0.0, z_mid)])
        return CrossSection([_ccw(pts)])

    def manifold(self):
        """The holder in the pot's frame: the plate at +x, z up, centred on y = 0."""
        from manifold3d import Manifold

        from .solid import _segments
        W = self.width
        body = Manifold.extrude(self.section(), W)        # section in (x, y), across along z
        body = body - self._bores()
        body = body.rotate([90, 0, 0]).translate([0, W / 2, 0])   # (u, z, across) -> (x, y, z)
        n = _segments(self.x_wall)
        return body - Manifold.revolve(self.cutter(), n)

    def _bores(self):
        """The two screw holes and their counterbores, across the layers of the
        print frame (u, z, across), each with a teardrop roof."""
        from manifold3d import CrossSection, Manifold, OpType

        from .solid import _ccw
        W = self.width

        def teardrop(z, dia, u0, u1):
            r = dia / 2
            a = np.deg2rad(np.linspace(135, 405, 48))
            pts = [(0.0, r * ROOT2)] + list(zip(r * np.cos(a), r * np.sin(a)))
            # (z, across) of the hole -> section (a, b) = (-across, z) so the
            # extrusion, turned onto u, puts the point up (+across)
            sec = CrossSection([_ccw([(-(W / 2 + y), z + x) for x, y in pts])])
            return Manifold.extrude(sec, u1 - u0).rotate([0, 90, 0]).translate([u0, 0, 0])

        holes = []
        for z in self.screws:
            holes.append(teardrop(z, self.bore, self.face - 1, self.x_wall + 1))
            holes.append(teardrop(z, self.sink, self.sink_from(z), self.face + self.head_h + self.clear))
        return Manifold.batch_boolean(holes, OpType.Add)

    def mesh(self, frame="print"):
        """The holder as a trimesh. frame="print": lying on its side, the
        section flat on the bed; frame="pot": mounted, in the pot's frame."""
        import trimesh
        m = self.manifold()
        if frame == "print":
            m = m.translate([0, -self.width / 2, 0]).rotate([-90, 0, 0])
            lo = np.asarray(m.bounding_box()[:3])
            m = m.translate(-lo)
        elif frame != "pot":
            raise ValueError(f"unknown frame '{frame}'")
        g = m.to_mesh()
        return trimesh.Trimesh(g.vert_properties[:, :3], g.tri_verts, process=True)

    def __str__(self):
        return (f"a {self.width:.0f} mm wide holder, {self.top - self.bottom:.0f} mm tall: "
                f"two {self.screw:g} mm screws {self.screw_hi - self.screw_lo:.0f} mm apart, the "
                f"pot {self.standoff:.1f} mm off the wall, sitting on a {self.ledge:g} mm ledge "
                f"with a {self.drop:g} mm finger inside the rim")
