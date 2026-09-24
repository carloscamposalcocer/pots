# Plant Pot Project

3D-printable plant pot, continued from an earlier Claude session. Read make_pot.py, patterns.py, pot_v3.py and pot_v2.py before changing anything. Use `.venv/Scripts/python.exe make_pot.py quality=draft` to confirm everything works (the system `python` lacks the deps; they are in requirements.txt / uv.lock).

## Settings (Hydra)
- User settings live in `conf/config.yaml` (+ `conf/quality/{full,draft}.yaml`): pattern, height, preview, out, quality (voxel, faces), design (wall, rim, base, cup_wall, strut_in, strut_out). Keep it to user-facing knobs; internal constants (cell counts, jitter, warps) stay in code.
- Override as `key=value`: `make_pot.py height=65 quality=draft design.wall=4`. `--show` prints the resolved config, `-v` = debug logs. Each run saves the resolved `config.yaml` into its `out` dir.
- make_pot.py uses Hydra's compose API, not `@hydra.main`: hydra-core 1.3.7 (latest stable) crashes in argparse on Python 3.14. `apply_config()` pushes the design values into the P2/P3/patterns module globals, then calls `patterns.set_height`.

## Goal
A single-piece, support-free FDM plant pot. The wall should let in as much air as possible without losing soil. A plain cup is fused to the base to catch drips and act as the water reserve. The wall pattern should be semi-organic and repeat seamlessly.

## Current design (decided)
- Size is driven by H (`make_pot.py height=65`; H_REF = 130 is the reference shape). `patterns.set_height(h)` (which chains to pot_v3/pot_v2 `set_height`) scales everything shape-related by k = H/130: radii, R0, cup height, moat gap, and the cell counts around the circumference (rounded to integers, so the pattern stays seamless and cells keep their mm size). Fixed in mm on purpose: wall, rim, base, cup wall, lip overhang, struts, VOR_H, hole size, blends, chamfer. Other modules must read sizes as `P2.H`, `P2.R_TOP`, ... at call time, never `from pot_v2 import H` (that copies a stale value).
- Reference size 151 x 130 mm. Pot body is tapered: outer radius 56 mm at the bottom, 72 mm at the top. Wall 6 mm thick, solid 5 mm rim at the top.
- Wall pattern: tapered Voronoi (`voronoi_taper` in patterns.py). The holes flare outward like funnels.
  - Struts are 2.2 mm on the soil side and thin to 1.1 mm outside (STRUT_IN, STRUT_OUT).
  - Result: outside face 58% open, typical hole 3.6 mm. Soil side 29% open, typical hole 2.5 mm.
  - Cells are a periodic jittered Voronoi (VOR_NC=76 cells around, VOR_H=5.2 mm row height, jitter ±0.30), with a slight sinusoidal warp.
- The pattern runs from just above the 2.4 mm solid base up to the rim.
- The cup is fused to the base (one piece). It is plain and smooth, 28 mm tall, 2.4 mm wall. It flares from a 5 mm gap at the bottom to a lip 73 mm from the centre, slightly wider than the top of the pot, so drips from the upper wall land inside it.
- There is no false floor: the soil sits directly on the solid base, so the bottom of the soil wicks water from the cup/moat. Earlier versions had a perforated 45-degree funnel floor with a wicking column; it might come back later.

## Printability rules (keep them)
- No supports anywhere.
- Every hole roof must be a flat bridge, no longer than the hole width (at most about 5 mm). The taper is done by shifting the pattern down by exactly the amount the edges recede, so holes only grow sideways and downward. Never let a hole ceiling rise toward the outside: that makes a sagging sloped ceiling.
- Minimum strut is about 1.1 mm (0.4 mm nozzle).
- 45-degree chamfer on the bottom outer edge (elephant foot).
- Every pattern must be periodic around the circumference: integer cell counts, and warp terms with integer frequency in theta.

## Code architecture
- The model is an implicit field (negative = solid), meshed with skimage marching_cubes in z-slabs.
  - Slab vertices stay in index space until after merge_vertices, then get scaled. That is what makes the slab seams watertight; don't change it.
- make_pot.py does build, then clean (fast_simplification decimation to about 900k faces, keep the largest component, pymeshfix, fix_normals, drop to z=0), then exports STL and 3MF plus a PNG preview (matplotlib render, cut-away, section, both wall faces with open %).
- patterns.py:
  - PATTERNS dict: gyroid, diamond (3D lattices, graded density through the wall) and voronoi, voronoi_taper, hex, coral (Gray-Scott reaction-diffusion; the texture is cached in coral_tex.npy), slots (2D radial perforations).
  - make_field(name) assembles lattice wall + base + cup.
  - Unrolled coordinates use R0 = 64 mm.
- pot_v3.py: cup and base parameters (CUP_H, CUP_GAP, CUP_FLARE, CUP_WALL, BASE, cup_ri()).
- pot_v2.py: shared helpers and global dimensions (H, R_BOT, R_TOP, WALL, RIM, CELL, gyroid(), smin(), build()). Its pot_field/cup_field are the older two-piece version and are unused now.

## Gotchas
- A full-res build (voxel 0.3) takes about 2 to 3 minutes and needs 2 to 3 GB RAM. Don't run several builds in parallel; it ran out of memory before.
- Tiny floating specks come out of marching cubes. clean() removes them; always check that the mesh is watertight and has 1 body.
- To check a pattern, sample the field on the outer face (frac 0.02) and the inner face (frac 0.98) at a fixed theta = arc / r_out(z), so the rays are radial. Report open %, median hole diameter (distance transform) and straight-through %.

## Possible next steps (not decided)
- Bring back the perforated 45-degree funnel floor so the soil sits above the water.
- Tune the funnel taper or the cell size; maybe mix patterns (e.g. a band of air-pruning slots near the base).
- Before calling any change done, verify: watertight, single body, overhang share, minimum strut, both-face metrics.
