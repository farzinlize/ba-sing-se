"""Shared categorical land-cover palette for the map and 3D renderer."""

from enum import IntEnum
import numpy as np
from numpy.typing import NDArray

from .terrain import Terrain


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
