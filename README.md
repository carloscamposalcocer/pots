# Breathing plant pot

A single-piece, support-free, 3D-printable (FDM) plant pot. The wall is
perforated to let in as much air as possible without losing soil, and a
plain cup fused to the base catches drips and holds a small water reserve
that the soil wicks back up.

The pot is generated from code (an implicit field meshed with marching
cubes) and exported as STL + 3MF, with a PNG preview.

## Setup

```sh
uv sync                      # or: pip install -r requirements.txt
```

Run everything with the project venv (`.venv/Scripts/python.exe` on Windows,
`.venv/bin/python` elsewhere, or `uv run python`).

## Usage

All settings live in [`conf/config.yaml`](conf/config.yaml) and have
defaults. Override only what you want to change, as `key=value`:

```sh
python make_pot.py                                  # build with the config defaults
python make_pot.py quality=draft                    # fast low-res check
python make_pot.py height=65                        # half-size pot, same shape
python make_pot.py pattern=coral design.wall=4      # another pattern, thinner wall
python make_pot.py height=65 --show                 # print the final settings, don't build
python make_pot.py -v                               # debug logging
```

Settings use Hydra/OmegaConf syntax, so nested keys use dots
(`design.strut_out=1.2`) and a misspelled key is an error.

### Outputs

Written to `out` (default `output/<pattern>_h<height>/`):

| File | What |
|---|---|
| `pot_<pattern>.stl` | print file |
| `pot_<pattern>.3mf` | same mesh, ~5x smaller |
| `pot_<pattern>.png` | preview: outside, cut-away, vertical section, both wall faces at true scale with their open % |
| `config.yaml` | the exact settings used for this pot |

## Parameters

### General

| Key | Default | Description |
|---|---|---|
| `pattern` | `voronoi_taper` | Wall pattern, see [Patterns](#patterns). |
| `height` | `130`* | Pot height in mm. **Drives the whole shape**, see [Scaling](#scaling-with-height). Minimum `base + rim + 10` (17.4 mm). |
| `preview` | `true` | Also render the PNG preview (~10 s). |
| `out` | `output/${pattern}_h${height}` | Output folder; existing files are overwritten. |
| `quality` | `full` | Mesh quality preset: `full` or `draft`. |

\* The reference shape is 130 mm; `conf/config.yaml` may currently hold a
different working value.

### Quality presets (`conf/quality/`)

| Key | `full` | `draft` | Description |
|---|---|---|---|
| `quality.voxel` | 0.3 | 0.6 | Mesh grid spacing in mm. Time and memory grow ~1/voxel³: full at 130 mm takes ~2–3 min and 2–3 GB RAM, draft ~1 min. Don't run several full builds in parallel. |
| `quality.faces` | 900 000 | 300 000 | Triangle budget; the mesh is decimated to this. |

You can also override a single value: `quality=draft quality.voxel=0.45`.

### Design sizes (`design.*`, mm)

Print-physics sizes: they depend on the nozzle and the soil, not on the
pot size, so they **do not scale** with `height`.

| Key | Default | Description |
|---|---|---|
| `design.wall` | 6.0 | Pot wall thickness (radial). The pattern goes through it. Thicker = stiffer, longer funnels; thinner = lighter. 4 mm suits small pots. |
| `design.rim` | 5.0 | Solid band at the top, for stiffness and a clean edge. |
| `design.base` | 2.4 | Solid floor shared by pot and cup. The soil sits on it and wicks water from the cup. The pattern starts 1 mm above it. |
| `design.cup_wall` | 2.4 | Wall thickness of the drip cup. |
| `design.strut_in` | 2.2 | `voronoi_taper` only. Strut width on the soil side. Wider = smaller inner holes = less soil loss, less air. |
| `design.strut_out` | 1.1 | `voronoi_taper` only. Strut width on the outside. Keep ≥ ~1.1 (0.4 mm nozzle) and ≤ `strut_in` so holes widen outward. |

With the defaults, `voronoi_taper` is 58% open outside (typical hole
3.6 mm) and 29% open on the soil side (typical hole 2.5 mm).

## Scaling with height

`height` sets a scale factor `k = height / 130` applied to everything that
defines the *shape*:

- outer radius: 56·k at the bottom, 72·k at the top
- cup height (28·k) and moat gap at the bottom (5·k); the cup lip always
  sits 1 mm outside the top of the pot so drips land in it
- number of pattern cells around the pot, rounded to an integer so the
  pattern stays seamless and each cell keeps its size in mm

What stays fixed in mm: the `design` sizes, hole size, row height of the
cells, the lip overhang and the bottom chamfer. So a small pot has fewer,
same-size holes and a proportionally thicker wall.

## Patterns

| Name | Type | Description |
|---|---|---|
| `voronoi_taper` | 2D, tapered | Organic cells that flare outward like funnels. Default. |
| `voronoi` | 2D | Same cells, straight-through holes, 1.8 mm struts. |
| `hex` | 2D | Warped honeycomb, pointy-top cells, 1.8 mm struts. |
| `coral` | 2D | Reaction-diffusion (Turing) labyrinth. Its texture is generated on first use at each height (slow) and cached as `coral_tex*.npy`. |
| `slots` | 2D | Narrow wavy vertical slots, air-pruning style. |
| `gyroid` | 3D lattice | Graded density through the wall, no straight line of sight. |
| `diamond` | 3D lattice | Schwarz diamond, straighter 45° channels. |

2D patterns are cut radially through the wall; 3D lattices are
tortuous channels. All wrap seamlessly around the pot.

## Printability rules

The generator follows these; keep them when changing the code:

- No supports anywhere.
- Every hole roof is a flat bridge no wider than the hole (≤ ~5 mm). The
  taper is done by shifting the pattern down as its edges recede, so holes
  only grow sideways and downward; a ceiling never rises toward the outside.
- Minimum strut ~1.1 mm (0.4 mm nozzle).
- 45° chamfer on the bottom outer edge (elephant foot).
- Every pattern is periodic around the circumference.

## Code layout

| File | Role |
|---|---|
| `make_pot.py` | Entry point: loads the config, builds, cleans (decimate, keep largest body, repair), exports and previews. |
| `conf/` | Hydra config: `config.yaml` and the `quality/` presets. |
| `patterns.py` | Wall patterns (`PATTERNS`), `make_field()` (wall + base + cup) and `set_height()`. |
| `pot_v3.py` | Cup and base parameters. |
| `pot_v2.py` | Global dimensions, shared helpers (`smin`, `gyroid`, `build` = slab-wise marching cubes). |

Hydra note: `make_pot.py` loads the config with Hydra's compose API rather
than `@hydra.main`, because hydra-core 1.3's own CLI crashes on Python 3.14.
The `key=value` syntax is the same; `--show` replaces `--cfg job`.
