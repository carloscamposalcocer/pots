"""
Cup clip (clip.enable): one narrow printed part that holds an unchanged pot
against the wall by its cup alone, tight. Made for the standard tapered pot.

It is a strip at most `width` mm wide: a back plate on the wall, and a hook
whose inside has the shape of the back of the cup wall. The finger follows
the inside of the cup down `drop` mm (in the moat between the cup and the
pot wall), the bridge lies on the lip, the back follows the outside of the
cup all the way down, and a ledge reaches `ledge` mm in under the base. The
ledge carries the pot; the pot's weight is in front of it, so the pot tips
forward and the finger inside the cup stops it. Every face that touches the
cup reaches `squeeze` mm into it (the bridge just touches the lip, `lift`
above it), so the cup wall is held in a tight slot.

Because the slot follows the cup wall exactly, the pot can't swing in or
out without something giving: it is a press fit. Hanging the pot: tilt the
bottom out, push the back of the cup wall up into the slot, and swing the
bottom in over the ledge; the lip pushes the hook back a little as it goes
(about drop x the swing angle) and the clip springs back round it.

The screws go through the block behind the cup: their counterbores open in
the seat, so they are driven from the front before the pot goes in, and the
heads end up `clear` behind the cup. They have to stay below the finger's
tip (a counterbore higher up would cut the finger), so a 130 mm pot has room
for two, one above the other, and smaller ones for one; on pots under
~70 mm it cuts a notch through the middle of the ledge, and the clip
reaches a little below the ledge to keep FLOOR under it.

Printability is the holder's: it prints on its side, every layer the same
section, so there is no overhang; only the screw holes cross the layers,
with teardrop roofs.
"""
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from .holder import EDGE, FLOOR, SIDE, Holder

FIN = 1.2          # thinnest finger (the moat between cup and pot)
ABOVE = 0.5        # a counterbore stays this far below the finger's tip and above the ledge


@dataclass(frozen=True)
class Clip(Holder):
    """A cup clip for one pot (a Holder that hooks the cup instead of the rim)."""
    thick: float = 3.0        # thickness of the bridge and the ledge, and of the finger where the moat allows
    drop: float = 4.0         # how far the finger reaches down inside the cup
    ledge: float = 3.0        # how far the ledge reaches in under the base
    lift: float = 0.0         # air above the cup lip (0: the bridge lies on it)
    squeeze: float = 0.3      # how far the clip reaches into the cup wall, on each side: the grip

    unused = {"groove"}       # Holder settings the clip ignores (its slot is the cup wall's shape)

    def __post_init__(self):
        if min(self.screw, self.head, self.head_h, self.width, self.thick,
               self.drop, self.ledge) <= 0:
            raise ValueError("every clip size must be positive")
        if min(self.clear, self.lift) < 0:
            raise ValueError("clip.clear and clip.lift can't be negative")
        if not 0 <= self.squeeze <= 1:
            raise ValueError("clip.squeeze must be between 0 (no grip) and 1 mm")
        if self.head <= self.screw:
            raise ValueError("clip.head must be wider than clip.screw")
        if self.width < self.min_width:
            raise ValueError(f"clip.width must be at least {self.min_width:.1f} mm "
                             "(the counterbore and its teardrop roof)")
        if self.width > self.pot.r_min:
            raise ValueError(f"clip.width must be at most {self.pot.r_min:.0f} mm on this pot "
                             "(the plate is flat, it would wrap too far around it)")
        if self.tip < self.thick:
            raise ValueError("clip.drop reaches too far down the cup")
        if self.fin < FIN:
            raise ValueError(f"the finger doesn't fit inside the cup on this pot (it would be "
                             f"{self.fin:.1f} mm thick): make clip.drop or clip.clear smaller")
        if self.ledge >= self.env(0.0) - self.thick:
            raise ValueError("clip.ledge reaches past the middle of the base")

    # ------------------------------------------------------------ the part
    def ri(self, z):
        """Inner radius of the cup at height z (constant above the lip)."""
        return self.pot.cup_ri(np.clip(z, 0, self.pot.cup_h))

    @property
    def under(self):
        """Underside of the bridge: on the lip, or `lift` above it."""
        return self.pot.cup_h + self.lift

    @property
    def top(self):
        return self.under + self.thick

    @property
    def tip(self):
        """Lowest point of the finger."""
        return self.pot.cup_h - self.drop

    @property
    def finger(self):
        """Radius of the finger's outer face at its tip (it follows the cup up from there)."""
        return float(self.ri(self.tip)) + self.squeeze

    @cached_property
    def fin(self):
        """Finger thickness inside the cup wall: `thick`, or less where the
        moat between the cup and the pot wall is narrower (`clear` left on
        the pot side)."""
        z = np.linspace(self.tip, self.top, 200)
        room = float(np.min(self.ri(z) - self.pot.r_out(z))) - self.clear
        return min(self.thick, room)

    @property
    def screws(self):
        """Heights of the screws: two, one above the other, if they fit
        between the ledge and the finger's tip, else one right under the tip."""
        hi = self.tip - ABOVE - self.sink / 2
        lo = hi - self.sink - SIDE
        return (lo, hi) if lo - self.sink / 2 >= ABOVE else (hi,)

    def sink_from(self, z):
        """The counterbore starts inside the cup, so the head goes in from the front."""
        return float(self.env(z - self.sink / 2)) - self.squeeze - 1

    @property
    def plate_bottom(self):
        """The clip's bottom: under the ledge, or lower if a counterbore reaches down there."""
        return min(self.bottom, min(self.screws) - self.sink / 2 - FLOOR)

    def section(self):
        """The part's outline in the (u, z) plane: u from the pot axis toward
        the wall, z up from the pot's bed. A plate and one block from the
        ledge's tip to the plate; `cutter` carves the cup's shape out of it."""
        from manifold3d import CrossSection

        from .solid import _ccw
        e = EDGE
        f, w, B, T, P = self.face, self.x_wall, self.bottom, self.top, self.plate_bottom
        plate = [(f, P), (w - e, P), (w, P + e), (w, T - e), (w - e, T), (f, T)]
        # the ledge's front edge crosses z = 0 one EDGE in, so it gets a lead-in
        lo = float(self.env(0.0)) - self.ledge
        block = [(lo, B + e), (lo + e, B), (f, B), (f, T), (lo + 1.5 * e, T),
                 (lo + 1.5 * e, 0.5 * e), (lo, -0.5 * e)]
        return CrossSection([_ccw(q) for q in (plate, block)])

    def cutter(self):
        """The back of the cup wall as a revolved (r, z) section, `squeeze`
        thinner on each side: the whole cup below the finger (the seat and
        the ledge follow it), and above the finger's tip only the wall
        itself and what lies inside the finger, up past the bridge."""
        from manifold3d import CrossSection

        from .solid import _ccw
        p, s, t = self.pot, self.squeeze, self.fin
        k = min(EDGE, t / 3)                    # lead-ins on the finger's tip
        tip, top = self.tip, self.under
        zs = np.linspace(0, p.cup_h, int(np.ceil(p.cup_h / 0.25)) + 1)
        up = zs[zs > tip + k]
        ri = lambda z: float(self.ri(z))
        lip = float(self.env(p.cup_h)) - s
        pts = ([(0.0, 0.0)] + [(float(self.env(z)) - s, z) for z in zs]
               + ([(lip, top)] if top > p.cup_h else [])
               + [(ri(p.cup_h) + s, top)] + [(ri(z) + s, z) for z in up[::-1]]
               + [(ri(tip + k) + s, tip + k), (ri(tip) + s - k, tip),            # the finger's tip
                  (ri(tip) - t + k, tip), (ri(tip + k) - t, tip + k)]
               + [(ri(z) - t, z) for z in up]
               + [(ri(p.cup_h) - t, self.top + 1), (0.0, self.top + 1)])
        return CrossSection([_ccw(pts)])

    def __str__(self):
        return (f"{self.width:.0f} mm wide, {self.top - self.plate_bottom:.0f} mm tall: a {self.drop:g} mm "
                f"finger inside the cup, a {self.ledge:g} mm ledge, {self.squeeze:g} mm grip; "
                f"{['one', 'two'][len(self.screws) - 1]} {self.screw:g} mm screw{'s' * (len(self.screws) > 1)} "
                f"behind the cup, the lip {self.standoff:.1f} mm off the wall")
