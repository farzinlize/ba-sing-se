"""Rounded hills/ridges and diamond-square terrain, independent of rendering."""

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


def smooth_heightmap(heights: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian smoothing; sigma is in cells, with reflected boundaries."""
    if sigma <= 0:
        return heights.copy()
    radius = max(1, int(np.ceil(3 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    result = heights.copy()
    for axis in (0, 1):
        padding = [(0, 0), (0, 0)]
        padding[axis] = (radius, radius)
        padded = np.pad(result, padding, mode="reflect")
        result = np.apply_along_axis(lambda line: np.convolve(line, kernel, mode="valid"), axis, padded)
    return result


def rounded_heightmap(size: int, roughness: float, rng: np.random.Generator) -> np.ndarray:
    """Broad elliptical hills linked by a spanning tree of lower, rounded ridges."""
    axis = np.linspace(0, 1, size)
    xx, yy = np.meshgrid(axis, axis)
    centers = np.array([(x, y) for y in (0.2, 0.5, 0.8) for x in (0.2, 0.5, 0.8)])
    centers += rng.uniform(-0.055, 0.055, centers.shape)
    amplitudes = rng.uniform(0.65, 1.15, len(centers))
    field = np.zeros_like(xx)
    for (cx, cy), amplitude in zip(centers, amplitudes):
        angle = rng.uniform(0, np.pi)
        major, minor = rng.uniform(0.085, 0.12), rng.uniform(0.065, 0.09)
        u = (xx - cx) * np.cos(angle) + (yy - cy) * np.sin(angle)
        v = -(xx - cx) * np.sin(angle) + (yy - cy) * np.cos(angle)
        field += amplitude * np.exp(-0.5 * ((u / major) ** 2 + (v / minor) ** 2))
    connected = {0}
    while len(connected) < len(centers):
        _, i, j = min((np.sum((centers[i] - centers[j]) ** 2), i, j)
                      for i in sorted(connected) for j in range(len(centers)) if j not in connected)
        start, delta = centers[i], centers[j] - centers[i]
        t = np.clip(((xx - start[0]) * delta[0] + (yy - start[1]) * delta[1]) / np.dot(delta, delta), 0, 1)
        distance2 = (xx - start[0] - t * delta[0]) ** 2 + (yy - start[1] - t * delta[1]) ** 2
        field += 0.3 * min(amplitudes[i], amplitudes[j]) * np.exp(-0.5 * distance2 / 0.055 ** 2)
        connected.add(j)
    # Reuse the existing generator for subdued natural irregularity, not jagged peaks.
    field += 0.035 * diamond_square(size, roughness, rng)
    return (field - field.min()) / np.ptp(field)


def generate_terrain(config: PipelineConfig, rng: np.random.Generator) -> Terrain:
    generator = rounded_heightmap if config.terrain_mode == "rounded" else diamond_square
    normalized = generator(config.size, config.roughness, rng)
    if config.island:
        axis = np.linspace(-1, 1, config.size)
        xx, yy = np.meshgrid(axis, axis)
        # Radial shoreline falloff avoids a square coast and submerges the boundary.
        falloff = np.minimum(np.hypot(xx, yy), 1.0) ** 3
        normalized = (normalized + 0.25) * (1 - falloff)
    cell_size = config.extent / (config.size - 1)
    normalized = smooth_heightmap(normalized, config.terrain_smoothing / cell_size)
    if config.island:
        # Keep the submerged boundary after filtering without modifying interior hills.
        normalized[[0, -1], :] = 0
        normalized[:, [0, -1]] = 0
    normalized = (normalized - normalized.min()) / np.ptp(normalized)
    heights = config.min_height + normalized * (config.max_height - config.min_height)
    axis = np.linspace(0, config.extent, config.size)
    return Terrain(heights, axis, axis.copy(), config.sea_level)
