# Topogame

A modular Python pipeline that creates random virtual terrain, a topographic map
with labeled heights and terrain colors, and a perspective image taken at a random
land point facing **N, E, S, or W**. Uses diamond-square, NumPy, Matplotlib, and PyVista.

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

Omit `--seed` for a fresh random batch. Each sample records its seed and settings;
sample `i` uses the base seed plus `i`. Choose a new output directory for another
batch; existing samples are never overwritten.

```text
outputs/demo/sample_0000/
├── topographic.png  # north-up map, biome legend, elevation contours, camera arrow
├── view.png         # perspective image from that arrow's position and bearing
├── terrain.npz      # heights, biome IDs, x/y axes, sea level
└── metadata.json    # seed, configuration, camera, palette, dependency versions
```

Useful options:

```bash
uv run topogame --output outputs/large --seed 7 --size 513 \
  --extent 4000 --min-height -150 --max-height 900 --sea-level 0 \
  --roughness 0.6 --contour-interval 50 --eye-height 15 \
  --pitch -5 --width 1600 --height 900
uv run topogame --help
```

Grid size must be `2**n + 1`, at least 9: 129, 257, and 513 are practical choices.
Higher roughness retains more small-scale height variation. By default, shoreline
falloff makes an island; `--no-island` leaves the fractal terrain unmasked.
`--eye-height` is meters above the selected ground point (default 12 for visibility;
use around 1.7 for human eye level). Pitch defaults to horizontal; negative looks
down. Field of view is vertical. Distant terrain can be hidden by nearby ridges,
as in a real ground-level camera; visibility is not guaranteed.

## Modules

| Module | Responsibility |
| --- | --- |
| `config.py` | Parameters and validation |
| `terrain.py` | Diamond-square height field and optional island falloff |
| `biomes.py` | Altitude-based terrain types and shared colors |
| `viewpoint.py` | Random dry interior location and independent random cardinal bearing |
| `topographic.py` | Colored map, labeled contours, camera marker |
| `rendering.py` | Off-screen PyVista mesh, sea surface, perspective camera |
| `pipeline.py` | Reproducible stage orchestration and artifact persistence |
| `cli.py` | Single-sample and batch command-line interface |

The map and rendered scene use the same heights and biome palette. Biomes are
synthetic altitude zones: water at/below sea level; dry land is beach below 4%,
grassland below 30%, forest below 58%, mountain below 85%, and snow above, relative
to the highest point above sea level. Forest is a color zone, not individual trees.
The renderer adds an opaque horizontal sea surface; underwater heights remain in
the raw data and map contours. Mesh lighting and interpolation affect image colors.

Coordinates use meters: +X east, +Y north, +Z up. Array access is
`heights[row_y, column_x]`, with row zero at the southern boundary. Both the map
and camera use this convention. The camera is sampled uniformly among dry grid
vertices inside a 10% boundary margin. Bearings are sampled uniformly and
independently, without choosing a preferred scenic direction.

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

Tests verify seeded generation, coastlines, biome thresholds, all four camera
bearings, output preservation, failure cleanup, and complete paired artifacts.
