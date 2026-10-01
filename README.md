# Topogame

A modular Python pipeline that creates random virtual terrain, a topographic map
with labeled heights and terrain colors, a contour-only map without camera bearing,
and a horizontal perspective image from a hill below the highest summit, facing
at least two visible hilltops. Uses diamond-square, NumPy, Matplotlib, and PyVista.

## Run

With [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev
uv run topogame --output outputs/demo --seed 42 --count 3
```

Or install in a Python 3.11+ virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m topogame --output outputs/demo --seed 42 --count 3
```

Omit `--seed` for a fresh random batch. `--count` is the number of successful
samples to save. Attempts use consecutive seeds starting at `--seed`; terrain
that cannot satisfy the camera rules is logged and skipped automatically. The
next seed is tried until exactly the requested count is saved, with consecutive
sample folder numbers and no gaps. Each sample's metadata records its actual
seed and settings. Choose a new output directory for another batch; existing
samples are never overwritten.

```text
outputs/demo/sample_0000/
├── topographic.png  # north-up map, biome legend, elevation contours, camera arrow
├── topographic_monochrome.png  # black contours, camera position only; no bearing
├── view.png         # perspective image from that arrow's position and bearing
├── terrain.npz      # heights, biome IDs, x/y axes, sea level
└── metadata.json    # seed, configuration, camera, palette, dependency versions
```

Useful options:

```bash
uv run topogame --output outputs/large --seed 7 --size 513 \
  --extent 4000 --min-height -150 --max-height 900 --sea-level 0 \
  --roughness 0.6 --contour-interval 50 --eye-height 15 \
  --field-of-view 65 --min-visible-hills 2 --width 1600 --height 900
uv run topogame --help
```

Grid size must be `2**n + 1`, at least 9: 129, 257, and 513 are practical choices.
Higher roughness retains more small-scale height variation. By default, shoreline
falloff makes an island; `--no-island` leaves the fractal terrain unmasked.
`--eye-height` is meters above the selected ground point (default 12 for visibility;
use around 1.7 for human eye level). The camera stays level with Z up and zero
pitch, giving a forward perspective with space above, below, left, and right.
`--pitch` accepts only 0. Field of view is vertical; image aspect ratio determines
the horizontal field of view. `--width` and `--height` affect the camera image.

## Camera composition rules

- Stand on a detected local hilltop inside the map's 10% boundary margin,
  strictly below the terrain's highest elevation.
- Include at least `--min-visible-hills` distinct **other** hilltops (default 2,
  minimum 2), separated by at least 5 degrees in the view.
- Reject hilltops hidden by intervening terrain. Sightline checks cover grid
  edges and triangle diagonals, conservatively allowing either cell triangulation.
- Aim at a suitable bearing, not just N/E/S/W, while keeping the camera horizontal.
- Keep the selected summits inside a 10% margin on all four image edges, so their
  tops are visible with headroom. The margin applies to the selected hilltops;
  other terrain can extend outside the frame.

Hill detection ignores small surface bumps: a summit must dominate a neighborhood
about 5% of the map wide, rise at least 2.5% of the land's elevation range above
its neighborhood median, and stand at least 15% of that range above sea level.
Detected hills are at least 6% of the map width apart.

Candidate camera hills are visited in seeded random order. At the first location
with a suitable view, the bearing favors more visible summits, then centered
framing. If a terrain cannot satisfy the rules, `generate_sample()` raises
`NoSuitableViewError` and creates no directory for that sample. The command-line
runner catches this specific error and tries the next seed until `--count`
successful samples have been saved. Invalid configuration, file errors, and
rendering failures still stop the run; earlier completed samples remain saved.
Small grids and single-hill maps can be unsuitable. Retries have no fixed limit;
settings that prevent any valid view need to be adjusted (field of view, image
shape, eye height, or required hill count).

## Modules

| Module | Responsibility |
| --- | --- |
| `config.py` | Parameters and validation |
| `terrain.py` | Diamond-square height field and optional island falloff |
| `biomes.py` | Altitude-based terrain types and shared colors |
| `viewpoint.py` | Hill detection, terrain visibility checks, level camera composition |
| `topographic.py` | Colored map, labeled contours, camera marker |
| `monochrome.py` | Black-on-white labeled contours and camera location, without camera direction |
| `rendering.py` | Off-screen PyVista mesh, sea surface, perspective camera |
| `pipeline.py` | Reproducible stage orchestration and artifact persistence |
| `cli.py` | Single-sample and batch command-line interface |

The additional `topographic_monochrome.png` has no colored fills or biome legend.
It keeps elevation labels, north-up map orientation, and a black camera location
dot labeled "Camera", with no camera arrow or facing-direction label. Camera
direction remains available in the original colored map and metadata.

The map and rendered scene use the same heights and biome palette. Biomes are
synthetic altitude zones: water at/below sea level; dry land is beach below 4%,
grassland below 30%, forest below 58%, mountain below 85%, and snow above, relative
to the highest point above sea level. Forest is a color zone, not individual trees.
The renderer adds an opaque horizontal sea surface; underwater heights remain in
the raw data and map contours. Mesh lighting and interpolation affect image colors.

Coordinates use meters: +X east, +Y north, +Z up. Array access is
`heights[row_y, column_x]`, with row zero at the southern boundary. Both the map
and camera use this convention. The colored map arrow matches the exact camera
bearing; its compass label is rounded to N/NE/E/SE/S/SW/W/NW. Metadata schema 2
records the exact bearing in `coordinates.bearing_degrees_clockwise_from_north`
and selected summit `[row, column]` indices in `camera.visible_hills`.

## Python API

```python
from topogame import PipelineConfig, generate_sample

sample = generate_sample("outputs/custom", PipelineConfig(seed=123, size=257))
print(sample / "view.png")
```

Stages can also be imported independently. To read a generated terrain:

```python
import numpy as np

with np.load("outputs/custom/terrain.npz") as data:
    heights = data["heights"]
    biomes = data["biomes"]
    print(heights.shape, heights.min(), heights.max())
```

Separate NumPy random streams drive terrain and camera selection. The same seed,
settings, and dependency versions reproduce terrain and camera metadata. PNG
pixels may vary across graphics drivers and rendering-library versions.

## Rendering and tests

PyVista renders off screen, without opening an interactive window. VTK still needs
a working OpenGL runtime. On headless Linux, install EGL/Mesa runtime libraries
if absent (Debian/Ubuntu: `libegl1 libgl1 libopengl0`). See the official
[PyVista installation guide](https://docs.pyvista.org/getting-started/installation.html)
and [screenshot example](https://docs.pyvista.org/examples/02-plot/screenshot).

```bash
uv run pytest                    # includes real off-screen rendering
uv run pytest -m 'not render'    # geometry, camera, validation, failure cleanup
```

Tests verify seeded generation, coastlines, biome thresholds, level hilltop views,
summit framing, blocked sightlines, unsuitable terrain, output preservation,
failure cleanup, and complete paired artifacts.
