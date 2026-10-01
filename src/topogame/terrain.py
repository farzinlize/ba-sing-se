"""Diamond-square height generation, independent of plotting and rendering."""

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

from .config import PipelineConfig


@dataclass(frozen=True)
class Terrain:
    # Rows run south to north; columns run west to east.
    heights: NDArray[np.float64]
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    sea_level: float


def diamond_square(size: int, roughness: float, rng: np.random.Generator) -> NDArray[np.float64]:
    """Return a nonperiodic fractal grid, normalized to [0, 1]."""
    if size < 3 or (size - 1) & (size - 2):
        raise ValueError("size must be 2**n + 1")
    if not 0 < roughness < 1:
        raise ValueError("roughness must be between 0 and 1")
    grid = np.zeros((size, size), dtype=np.float64)
    grid[::size - 1, ::size - 1] = rng.uniform(-1, 1, (2, 2))
    step, amplitude = size - 1, 1.0
    while step > 1:
        half = step // 2
        for row in range(half, size - 1, step):
            for col in range(half, size - 1, step):
                grid[row, col] = (
                    grid[row - half, col - half] + grid[row - half, col + half]
                    + grid[row + half, col - half] + grid[row + half, col + half]
                ) / 4 + rng.uniform(-amplitude, amplitude)
        for row in range(0, size, half):
            for col in range((row + half) % step, size, step):
                neighbors = [grid[r, c] for r, c in (
                    (row - half, col), (row + half, col),
                    (row, col - half), (row, col + half),
                ) if 0 <= r < size and 0 <= c < size]
                grid[row, col] = np.mean(neighbors) + rng.uniform(-amplitude, amplitude)
        step = half
        amplitude *= roughness
    return (grid - grid.min()) / np.ptp(grid)


def generate_terrain(config: PipelineConfig, rng: np.random.Generator) -> Terrain:
    normalized = diamond_square(config.size, config.roughness, rng)
    if config.island:
        axis = np.linspace(-1, 1, config.size)
        xx, yy = np.meshgrid(axis, axis)
        # Radial shoreline falloff avoids a square coast and submerges the boundary.
        falloff = np.minimum(np.hypot(xx, yy), 1.0) ** 3
        normalized = (normalized + 0.25) * (1 - falloff)
        normalized /= normalized.max()
    heights = config.min_height + normalized * (config.max_height - config.min_height)
    axis = np.linspace(0, config.extent, config.size)
    return Terrain(heights, axis, axis.copy(), config.sea_level)
