# Plant Pot Project

3D-printable plant pot, continued from an earlier Claude session. It is an installable package in `src/pots/` with a `pots` console script. Read geometry.py, patterns.py, field.py and cli.py before changing anything. Run `uv run pytest` (~15 s) and `uv run pots quality=draft` to confirm everything works (the system `python` lacks the deps; they are in pyproject.toml / uv.lock).

## Settings (Hydra)
- User settings live in `src/pots/conf/config.yaml` (+ `quality/{full,draft}.yaml`), shipped inside the package: pattern, height, preview, out, quality (voxel, faces), design (wall, rim, base, cup_wall, strut_in, strut_out). Keep it to user-facing knobs; internal constants (cell counts, jitter, warps) stay in code.
- Override as `key=value`: `pots height=65 quality=draft design.wall=4`. `--show` prints the resolved config, `-v` = debug logs. Each run saves the resolved `config.yaml` into its `out` dir.
- cli.py uses Hydra's compose API (`initialize_config_dir` on the packaged conf/), not `@hydra.main`: hydra-core 1.3.7 (latest stable) crashes in argparse on Python 3.14. `pot_from_config()` turns `height` + `design.*` into a `Pot` (the design keys are exactly the `Pot` field names).

## Goal
A single-piece, support-free FDM plant pot. The wall should let in as much air as possible without losing soil. A plain cup is fused to the base to catch drips and act as the water reserve. The wall pattern should be semi-organic and repeat seamlessly.

## Current design (decided)
- Size is driven by H (`pots height=65`; H_REF = 130 is the reference shape). `Pot(height=h)` (geometry.py, a frozen dataclass) scales everything shape-related by k = H/130: radii, R0, cup height, moat gap, and the cell counts around the circumference (rounded to integers, so the pattern stays seamless and cells keep their mm size). Fixed in mm on purpose: wall, rim, base, cup wall, lip overhang, struts, VOR_H, hole size, blends, chamfer. There is no global size state: every function takes the `Pot` explicitly; cell counts are given at H_REF and scaled with `pot.count(ref)`.
- Reference size 151 x 130 mm. Pot body is tapered: outer radius 56 mm at the bottom, 72 mm at the top. Wall 6 mm thick, solid 5 mm rim at the top.
- Wall pattern: tapered Voronoi (`voronoi_taper` in patterns.py). The holes flare outward like funnels.
  - Struts are 2.2 mm on the soil side and thin to 1.1 mm outside (`pot.strut_in`, `pot.strut_out`).
  - Result: outside face 58% open, typical hole 3.6 mm. Soil side 29% open, typical hole 2.5 mm.
  - Cells are a periodic jittered Voronoi (VOR_NC_REF=76 cells around, VOR_H=5.2 mm row height, jitter ±0.30), with a slight sinusoidal warp.
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
- `src/pots/`:
  - geometry.py: `Pot` (all dimensions, derived ones as cached properties; r_out/r_in/cup_ri) and the reference constants (H_REF, radii, cup, LIP, CHAMFER).
  - patterns.py: `PATTERNS` registry of `Pattern(kind, fn, description)`; each pattern is `fn(pot, th, Z, r, s)`. Kinds: "3d" lattices (gyroid, diamond; graded density through the wall; field = material) and "2d" radial perforations (voronoi, voronoi_taper, hex, coral, slots; field = hole). Unrolled coordinates use `pot.r0` (64 mm at H_REF).
  - coral.py: Gray-Scott texture for `coral`, cached as `coral_{nx}x{nz}.npy` in `$POTS_CACHE_DIR` or `~/.cache/pots`.
  - field.py: `make_field(pot, pattern)` assembles lattice wall + rim + base + cup.
  - sdf.py: smin, cylindrical(), gyroid(), CELL, T_IN/T_OUT.
  - mesh.py: `build()` = skimage marching_cubes in z-slabs. Slab vertices stay in index space until after merge_vertices, then get scaled. That is what makes the slab seams watertight; don't change it. `clean()` = fast_simplification decimation, keep the largest component, pymeshfix, fix_normals, drop to z=0.
  - pipeline.py: `generate(pot, pattern, voxel, faces)` -> (mesh, field).
  - metrics.py: `wall_metrics(field, pot)`: open %, median hole diameter, straight-through % of both wall faces (logged on every build).
  - preview.py: matplotlib PNG (outside, cut-away, section, both wall faces with open %).
  - cli.py: the `pots` command; exports STL + 3MF + PNG + resolved config.yaml.
- tests/: pytest; geometry scaling, seamlessness at theta = ±pi for every pattern, solid base/rim/cup, voronoi_taper metrics, a small draft build (watertight, 1 body), CLI config handling.
- The old two-piece pot (pot_v2 pot_field/cup_field) and the standalone scripts were removed in the src/ refactor; they are in git history before that commit.

## Gotchas
- A full-res build (voxel 0.3) takes about 2 to 3 minutes and needs 2 to 3 GB RAM. Don't run several builds in parallel; it ran out of memory before.
- Tiny floating specks come out of marching cubes. clean() removes them; always check that the mesh is watertight and has 1 body (the build log reports both and warns otherwise).
- To check a pattern, use `pots.metrics.wall_metrics(field, pot)`: it samples the outer face (frac 0.02) and the inner face (frac 0.98) at theta = arc / r_out(z), so the rays are radial, and reports open %, median hole diameter (distance transform) and straight-through %.

## Possible next steps (not decided)
- Bring back the perforated 45-degree funnel floor so the soil sits above the water.
- Tune the funnel taper or the cell size; maybe mix patterns (e.g. a band of air-pruning slots near the base).
- Before calling any change done, verify: watertight, single body, overhang share, minimum strut, both-face metrics.
