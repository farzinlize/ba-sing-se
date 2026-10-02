# Topogame

A modular Python pipeline that creates random virtual terrain, a topographic map
with labeled heights and terrain colors, a contour-only map without camera bearing,
and a horizontal perspective image from a hill below the highest summit, facing
at least two visible hilltops. Uses NumPy, Matplotlib, and PyVista, with rounded
hills/ridges by default and an optional diamond-square terrain mode.

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
├── topographic.png  # hillshaded map, contours, large red camera marker, compass guide
├── topographic_monochrome.png  # black contours, red camera point, compass guide; no bearing
├── view.png         # perspective image from that arrow's position and bearing
├── terrain.npz      # final heights, biome IDs, shared surface_rgb, x/y axes, sea level
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
falloff makes an island; `--no-island` leaves the terrain unmasked.
`--eye-height` is meters above the selected ground point (default 12 for visibility;
use around 1.7 for human eye level). The camera stays level with Z up and zero
pitch, giving a forward perspective with space above, below, left, and right.
`--pitch` accepts only 0. Field of view is vertical; image aspect ratio determines
the horizontal field of view. `--width` and `--height` affect the camera image.

## Rounded terrain and lighting

The default `--terrain-mode rounded` draws a seeded count of freely placed broad
hills (4–15 by default), with varied scale, elongation, orientation, asymmetric
shoulders, occasional dominant mountains, broad valleys, and a partial network
of curved ridges. Some hills form clusters or chains while others remain isolated,
so seeds change the large-scale terrain structure rather than only surface noise.
Set the essential count range with `--rounded-hills-min` and
`--rounded-hills-max`. `--terrain-smoothing 12` applies a light Gaussian filter
with sigma **12 meters**, independent of grid resolution; 0 disables it. Filtering
happens before colors, contours, and camera checks, and the final heights are
normalized to `--min-height` / `--max-height`. For the original terrain shape use
`--terrain-mode fractal --terrain-smoothing 0`.

For continuous dry land, disable island falloff and put sea level below the
minimum terrain height:

```bash
uv run topogame --output outputs/rounded --seed 42 --count 3 \
  --terrain-mode rounded --no-island --min-height 50 --max-height 450 --sea-level 0 \
  --rounded-hills-min 4 --rounded-hills-max 15 --terrain-smoothing 12 \
  --sun-azimuth 315 --sun-elevation 35 --contour-interval 25
```

Sun azimuth is clockwise from north, toward the sun: 315° is northwest. Default
elevation is 35°. The camera uses matte materials, directional sunlight, moderated
cast shadows, and `--ambient-light 0.3`. Use `--no-shadows` to disable cast shadows.
The colored map uses the same sun direction for subtle local slope hillshading;
`--hillshade-strength 0.3` controls its strength (0 disables it). Contours are drawn
above shading with light halos around labels. The monochrome map stays contour-only.

These are synthetic landscapes, not an erosion/geology simulation. The requested
hill count describes source landforms; overlapping hills and ridges can merge into
fewer detected summits after smoothing. More varied terrain also means some seeds
cannot meet the strict camera composition rules and are skipped by batch runs.
Map hillshade does not include cast shadows, and lighting makes map/view pixel
colors differ.
VTK shadow quality depends on the graphics backend and image size; see the
[PyVista lighting notes](https://docs.pyvista.org/api/plotting/lights.html).

## Stylized low-poly camera images

`--render-style natural` preserves the existing appearance (default).
`--render-style stylized` adds flat-shaded terrain with subtle face-color variation,
irregular rocks, clustered pine/oak/birch trees, near-camera shrubs and grass,
warm sunlight with cool sky fill, small low-poly clouds, and gentle distance haze.
The terrain mesh is **not simplified or displaced**: heights, camera position,
horizontal framing, and the matching contours stay unchanged. The colored map
shares the richer ground palette; the monochrome puzzle map stays undecorated.

```bash
uv run topogame --output outputs/stylized --render-style stylized --seed 42 --count 3 \
  --no-island --min-height 50 --max-height 450 --sea-level 0 \
  --tree-density 12 --rock-density 3 --sky-fill 0.45 --contour-interval 25
# Island with optional snow and shallow/deep water colors:
uv run topogame --output outputs/stylized-island --render-style stylized --seed 42 --snow-line 450
```

Essential stylized controls (also fields on `PipelineConfig`):

| Option | Default / meaning |
| --- | --- |
| `--tree-density`, `--rock-density` | 12 trees / 3 rocks per hectare, before suitability and spacing filters; 0 disables |
| `--tree-species` | `pine oak birch`; pass one or more distinct species |
| `--tree-height-min`, `--tree-height-max` | 10–24 meters |
| `--ground-cover-density` | 1; shrub/grass density multiplier, 0 disables |
| `--vegetation-seed`, `--decoration-seed` | Independent seeds derived from the sample seed unless overridden |
| `--sky-fill`, `--facet-variation` | 0.45 sky illumination / 0.035 face-color variation |
| `--no-clouds`, `--haze-strength` | Clouds enabled / 0.12 distance color haze; 0 disables haze |
| `--snow-line` | Optional absolute elevation in meters; snow disabled by default |

Placement follows slope/elevation and seeded clusters, with spaced tree crowns
and clearings. Rocks favor exposed slopes. Models are batched by species, variant,
and distance detail level; distant trees use fewer triangles and grass is limited
to the foreground. Conservative placement protects summit sightlines, followed
by ray intersections against the actual final decoration triangles. Any blocking
object is removed in full. Unsuitable terrain still triggers the existing retry
logic until the requested sample count is saved; the camera is never tilted.

Metadata records the style, controls, resolved scenery seeds, actual object counts,
batch/triangle counts, removed obstructions, and a reproducible placement hash.
Density is a target, not an exact count: spacing, terrain resolution, snow, and
visibility filtering can reduce it. Higher densities cost memory and render time;
batching avoids thousands of actors but still expands repeated mesh geometry.
Haze is a lightweight distance-based color blend, not volumetric fog. Water colors
follow depth within the map bounds; waves/refraction are not simulated. Models and
clouds are procedural, and existing VTK shadow/backend limitations still apply.

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
the median of a wider neighborhood (about 12%), and stand at least 15% of that
range above the higher of sea level and the minimum terrain height. This recognizes
broad summits even on fully dry land. Detected hills are at least 6% of the map
width apart. All camera checks use the final smoothed heightmap.

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
| `terrain.py` | Rounded hills/ridges or diamond-square terrain, island falloff, smoothing |
| `biomes.py` | Shared elevation/slope colors, map hillshade, categorical biome IDs |
| `viewpoint.py` | Hill detection, terrain visibility checks, level camera composition |
| `topographic.py` | Colored map, labeled contours, camera marker |
| `monochrome.py` | Black-on-white labeled contours and camera location, without camera direction |
| `rendering.py` | Off-screen PyVista mesh, sea surface, perspective camera |
| `decoration.py` | Seeded low-poly models, clustered placement, batching, final-scene sightlines |
| `pipeline.py` | Reproducible stage orchestration and artifact persistence |
| `cli.py` | Single-sample and batch command-line interface |

The colored map uses a prominent red camera marker and a subtle eight-direction
compass guide in its bottom-right corner. The additional
`topographic_monochrome.png` has no colored fills or biome legend.
It keeps elevation labels, north-up map orientation, and a red camera location
dot labeled "Camera", plus the same subtle eight-direction compass guide. It has
no camera arrow or facing-direction label. Camera
direction remains available in the original colored map and metadata.

The map and scene share the same final heights and muted grass/earth/rock colors,
blended gradually by elevation and slope; steeper slopes expose more rock.
Unlit colors are saved as `surface_rgb` in `terrain.npz`. The existing categorical
biome IDs remain available as raw data, but no longer determine rendered colors.
Where water is present, the renderer adds a matte sea surface; fully dry terrain
has no sea plane or coastline. Underwater heights remain in the data and contours.

Coordinates use meters: +X east, +Y north, +Z up. Array access is
`heights[row_y, column_x]`, with row zero at the southern boundary. Both the map
and camera use this convention. The colored map arrow matches the exact camera
bearing; its compass label is rounded to N/NE/E/SE/S/SW/W/NW. Metadata schema 5
records the exact bearing in `coordinates.bearing_degrees_clockwise_from_north`,
selected summit `[row, column]` indices in `camera.visible_hills`, all settings,
and realized rounded-generation details such as hill centers, shape parameters,
ridge connections, valleys, and connectivity.

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

Separate NumPy random streams drive terrain, camera selection, vegetation, and
decoration. The same seed, settings, and dependency versions reproduce terrain
and camera metadata. PNG pixels may vary across graphics drivers and
rendering-library versions.

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
failure cleanup, rounded-terrain smoothness, dry land, slope colors, sun direction,
lighting controls, stylized scenery/spacing, final-mesh obstruction removal,
unchanged terrain/cameras across styles, and complete paired artifacts.
