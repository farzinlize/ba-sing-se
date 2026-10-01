"""Shared categorical land-cover palette for the map and 3D renderer."""

from enum import IntEnum
import numpy as np
from numpy.typing import NDArray

from .terrain import Terrain
from .config import PipelineConfig


class Biome(IntEnum):
    WATER = 0
    BEACH = 1
    GRASSLAND = 2
    FOREST = 3
    MOUNTAIN = 4
    SNOW = 5


LABELS = ("Water", "Beach", "Grassland", "Forest", "Mountain", "Snow")
COLORS = np.array([
    [46, 123, 170], [222, 205, 148], [133, 172, 86],
    [54, 112, 68], [137, 128, 118], [238, 241, 240],
], dtype=np.uint8)

# Continuous surface colors are separate from the legacy categorical biome IDs.
SURFACE_LABELS = ("Grass", "Earth", "Rock", "Water")
SURFACE_COLORS = np.array([[125, 143, 99], [163, 148, 120], [157, 155, 147],
                           [91, 133, 145]], dtype=np.uint8)
STYLIZED_COLORS = np.array([[133, 176, 63], [174, 150, 102], [147, 149, 142],
                           [25, 127, 187]], dtype=np.uint8)
SAND_COLOR = np.array([225, 205, 145])
SNOW_COLOR = np.array([237, 242, 244])


def surface_palette(config: PipelineConfig) -> np.ndarray:
    return STYLIZED_COLORS if config.render_style == "stylized" else SURFACE_COLORS


def terrain_normals(terrain: Terrain) -> np.ndarray:
    """World-space normals for +X east, +Y north, +Z up, in physical meters."""
    dy, dx = np.gradient(terrain.heights, terrain.y, terrain.x)
    normals = np.stack((-dx, -dy, np.ones_like(dx)), axis=-1)
    return normals / np.linalg.norm(normals, axis=-1, keepdims=True)


def _smoothstep(low: float, high: float, values: np.ndarray) -> np.ndarray:
    fraction = np.clip((values - low) / (high - low), 0, 1)
    return fraction * fraction * (3 - 2 * fraction)


def water_colors(depth: np.ndarray, extent: float) -> NDArray[np.uint8]:
    """Shared shallow/deep/shore colors for the map and stylized sea surface."""
    fraction = _smoothstep(0, max(extent * 0.025, 1), depth)
    shallow, deep = np.array([65, 190, 205]), np.array([22, 108, 175])
    rgb = shallow * (1 - fraction[..., None]) + deep * fraction[..., None]
    shore = np.clip(1 - depth / 1.5, 0, 1)
    rgb = rgb * (1 - shore[..., None]) + np.array([188, 226, 215]) * shore[..., None]
    return np.rint(rgb).astype(np.uint8)


def terrain_colors(terrain: Terrain, config: PipelineConfig | None = None) -> NDArray[np.uint8]:
    """Muted, smoothly blended grass/earth/rock using final elevation and slope."""
    base = max(float(terrain.heights.min()), terrain.sea_level)
    relief = max(float(terrain.heights.max() - base), np.finfo(float).eps)
    relative = np.clip((terrain.heights - base) / relief, 0, 1)
    slope = np.degrees(np.arccos(np.clip(terrain_normals(terrain)[..., 2], 0, 1)))
    earth = _smoothstep(0.2, 0.75, relative)[..., None]
    stylized = config is not None and config.render_style == "stylized"
    palette = STYLIZED_COLORS if stylized else SURFACE_COLORS
    if stylized:
        earth *= 0.45
    rgb = palette[0] * (1 - earth) + palette[1] * earth
    rock = np.maximum(_smoothstep(0.8 if stylized else 0.65, 1.0, relative),
                      _smoothstep(35 if stylized else 20, 62 if stylized else 50, slope))[..., None]
    rgb = rgb * (1 - rock) + palette[2] * rock
    if stylized:
        # Only create beach bands if this terrain actually intersects water.
        if terrain.heights.min() <= terrain.sea_level:
            beach = (1 - _smoothstep(0, config.extent * 0.007, terrain.heights - terrain.sea_level))
            beach *= 1 - _smoothstep(20, 45, slope)
            rgb = rgb * (1 - beach[..., None]) + SAND_COLOR * beach[..., None]
        if config.snow_line is not None:
            snow = _smoothstep(config.snow_line, config.snow_line + 30, terrain.heights)
            snow *= 1 - _smoothstep(45, 75, slope)
            rgb = rgb * (1 - snow[..., None]) + SNOW_COLOR * snow[..., None]
    wet = terrain.heights <= terrain.sea_level
    rgb[wet] = water_colors(terrain.sea_level - terrain.heights[wet], config.extent) if stylized else palette[3]
    return np.rint(rgb).astype(np.uint8)


def hillshade_colors(terrain: Terrain, rgb: np.ndarray, config: PipelineConfig) -> np.ndarray:
    """Subtle local slope shading; a flat surface keeps its original color."""
    illumination = np.clip(terrain_normals(terrain) @ np.asarray(config.sun_direction), 0, 1)
    factor = 1 + config.hillshade_strength * (illumination - config.sun_direction[2])
    factor[terrain.heights <= terrain.sea_level] = 1
    return np.clip(rgb.astype(float) / 255 * factor[..., None], 0, 1)


def classify_biomes(terrain: Terrain) -> NDArray[np.uint8]:
    """Synthetic altitude zones; this is a visual model, not an ecology simulation.

    Dry elevations as fractions of peak height above sea: beach < .04,
    grassland < .30, forest < .58, mountain < .85, snow above.
    """
    relief = float(terrain.heights.max() - terrain.sea_level)
    if relief <= 0:
        return np.zeros_like(terrain.heights, dtype=np.uint8)
    relative = (terrain.heights - terrain.sea_level) / relief
    types = (np.digitize(relative, [0.04, 0.30, 0.58, 0.85]) + 1).astype(np.uint8)
    types[terrain.heights <= terrain.sea_level] = Biome.WATER
    return types
