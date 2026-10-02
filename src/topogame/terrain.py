"""Rounded hills/ridges and diamond-square terrain, independent of rendering."""

from dataclasses import dataclass
from typing import Any
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
    generation_details: dict[str, Any] | None = None


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


def _irregular_centers(count: int, rng: np.random.Generator,
                       cluster_tendency: float) -> np.ndarray:
    """Place spaced hills freely, mixing local groups with isolated centers."""
    margin, minimum_spacing = 0.07, 0.055
    centers: list[np.ndarray] = []
    for _ in range(count):
        accepted = None
        for _attempt in range(160):
            if centers and rng.random() < cluster_tendency:
                anchor = centers[int(rng.integers(len(centers)))]
                distance = rng.uniform(0.07, 0.24)
                angle = rng.uniform(0, 2 * np.pi)
                candidate = anchor + distance * np.array([np.cos(angle), np.sin(angle)])
            else:
                candidate = rng.uniform(margin, 1 - margin, 2)
            if (np.all((margin <= candidate) & (candidate <= 1 - margin))
                    and all(np.linalg.norm(candidate - other) >= minimum_spacing for other in centers)):
                accepted = candidate
                break
        if accepted is None:
            # Deterministic maximin fallback avoids silently lowering the requested count.
            choices = rng.uniform(margin, 1 - margin, (512, 2))
            distances = np.min(np.linalg.norm(choices[:, None] - np.asarray(centers)[None], axis=2), axis=1)
            accepted = choices[int(np.argmax(distances))]
        centers.append(accepted)
    return np.asarray(centers)


def _ridge_edges(centers: np.ndarray, rng: np.random.Generator,
                 connectivity: float) -> list[tuple[int, int]]:
    """Create a partial, variable network with branches, chains and isolated hills."""
    count = len(centers)
    distance = np.linalg.norm(centers[:, None] - centers[None, :], axis=2)
    candidates = sorted((distance[i, j] * rng.uniform(0.8, 1.25), i, j)
                        for i in range(count) for j in range(i + 1, count)
                        if distance[i, j] < 0.48)
    target = int(round(connectivity * max(count - 1, 1)))
    target += int(rng.integers(0, max(2, count // 4 + 1)))
    target = min(target, len(candidates), max(0, count + 2))
    degree = np.zeros(count, dtype=int)
    edges: list[tuple[int, int]] = []
    for _, i, j in candidates:
        # Degree limits retain open groups and prevent identical all-connected webs.
        if len(edges) >= target:
            break
        if degree[i] >= 3 or degree[j] >= 3:
            continue
        if rng.random() > 0.82 and edges:
            continue
        edges.append((i, j))
        degree[[i, j]] += 1
    return edges


def rounded_heightmap(size: int, roughness: float, rng: np.random.Generator,
                      hill_count_range: tuple[int, int] = (4, 15),
                      *, return_details: bool = False) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Build seeded broad hills, asymmetric peaks, valleys and varied curved ridges."""
    axis = np.linspace(0, 1, size)
    xx, yy = np.meshgrid(axis, axis)
    minimum, maximum = hill_count_range
    hill_count = int(rng.integers(minimum, maximum + 1))
    cluster_tendency = float(rng.beta(1.5, 1.5) * 0.75)
    ridge_connectivity = float(rng.uniform(0.18, 0.92))
    directional_alignment = float(rng.uniform(0, 0.8))
    common_angle = float(rng.uniform(0, np.pi))
    centers = _irregular_centers(hill_count, rng, cluster_tendency)
    amplitudes = rng.lognormal(mean=-0.12, sigma=0.36, size=hill_count)
    dominant = None
    if hill_count >= 6 and rng.random() < 0.58:
        dominant = int(rng.integers(hill_count))
        amplitudes[dominant] *= rng.uniform(1.35, 2.05)
    major_axes = rng.uniform(0.055, 0.18, hill_count)
    minor_axes = major_axes * rng.uniform(0.38, 0.95, hill_count)
    random_angles = rng.uniform(0, np.pi, hill_count)
    angles = ((1 - directional_alignment) * random_angles
              + directional_alignment * (common_angle + rng.normal(0, 0.22, hill_count)))
    asymmetry = rng.uniform(-0.38, 0.38, hill_count)
    field = np.zeros_like(xx)
    for (cx, cy), amplitude, major, minor, angle, skew in zip(
            centers, amplitudes, major_axes, minor_axes, angles, asymmetry):
        u = (xx - cx) * np.cos(angle) + (yy - cy) * np.sin(angle)
        v = -(xx - cx) * np.sin(angle) + (yy - cy) * np.cos(angle)
        exponent = rng.uniform(1.75, 2.45)
        radius = ((np.abs(u) / major) ** exponent + (np.abs(v) / minor) ** exponent)
        hill = amplitude * np.exp(-0.5 * radius) * (1 + skew * np.tanh(u / major))
        field += hill
        # An offset shoulder makes broad hills and dominant mountains less symmetric.
        if rng.random() < 0.72:
            offset_angle = angle + rng.uniform(-1.2, 1.2)
            offset = rng.uniform(0.35, 0.8) * major
            sx, sy = cx + np.cos(offset_angle) * offset, cy + np.sin(offset_angle) * offset
            shoulder = np.exp(-0.5 * (((xx - sx) / (major * rng.uniform(0.38, 0.7))) ** 2
                                      + ((yy - sy) / (minor * rng.uniform(0.55, 0.9))) ** 2))
            field += amplitude * rng.uniform(0.16, 0.38) * shoulder
    edges = _ridge_edges(centers, rng, ridge_connectivity)
    ridge_parameters = []
    for i, j in edges:
        start, end = centers[i], centers[j]
        delta = end - start
        length = float(np.linalg.norm(delta))
        perpendicular = np.array([-delta[1], delta[0]]) / length
        curvature = float(rng.uniform(-0.2, 0.2) * length)
        control = (start + end) / 2 + perpendicular * curvature
        width = float(rng.uniform(0.025, 0.07))
        height = float(rng.uniform(0.12, 0.42) * min(amplitudes[i], amplitudes[j]))
        nearest = np.full_like(field, np.inf)
        along = np.zeros_like(field)
        samples = 18
        for sample in range(samples):
            t = sample / (samples - 1)
            point = (1 - t) ** 2 * start + 2 * (1 - t) * t * control + t ** 2 * end
            distance2 = (xx - point[0]) ** 2 + (yy - point[1]) ** 2
            update = distance2 < nearest
            nearest[update] = distance2[update]
            along[update] = t
        taper = np.sin(np.pi * along) ** 0.6
        field += height * np.exp(-0.5 * nearest / width ** 2) * (0.35 + 0.65 * taper)
        ridge_parameters.append({"hills": [i, j], "width": width, "height": height,
                                 "curvature": curvature})
    valley_count = int(rng.integers(0, 3))
    for _ in range(valley_count):
        cx, cy = rng.uniform(0.08, 0.92, 2)
        major, minor = rng.uniform(0.16, 0.34), rng.uniform(0.08, 0.2)
        angle = rng.uniform(0, np.pi)
        u = (xx - cx) * np.cos(angle) + (yy - cy) * np.sin(angle)
        v = -(xx - cx) * np.sin(angle) + (yy - cy) * np.cos(angle)
        field -= rng.uniform(0.06, 0.2) * amplitudes.mean() * np.exp(
            -0.5 * ((u / major) ** 2 + (v / minor) ** 2))
    # Reuse the existing generator for subdued natural irregularity, not jagged peaks.
    detail_strength = float(rng.uniform(0.018, 0.05))
    field += detail_strength * diamond_square(size, roughness, rng)
    normalized = (field - field.min()) / np.ptp(field)
    pair_distances = np.linalg.norm(centers[:, None] - centers[None, :], axis=2)
    details = {
        "mode": "rounded", "hill_count": hill_count,
        "hill_centers_normalized": centers.round(6).tolist(),
        "hill_amplitudes": amplitudes.round(6).tolist(),
        "hill_major_axes_normalized": major_axes.round(6).tolist(),
        "hill_minor_axes_normalized": minor_axes.round(6).tolist(),
        "hill_angles_degrees": np.degrees(angles % np.pi).round(3).tolist(),
        "minimum_center_spacing_normalized": float(pair_distances[pair_distances > 0].min()),
        "cluster_tendency": cluster_tendency, "directional_alignment": directional_alignment,
        "dominant_hill": dominant, "ridge_connectivity": ridge_connectivity,
        "ridge_count": len(edges), "ridges": ridge_parameters,
        "isolated_hill_count": int(hill_count - len({node for edge in edges for node in edge})),
        "valley_count": valley_count, "detail_strength": detail_strength,
    }
    return (normalized, details) if return_details else normalized


def generate_terrain(config: PipelineConfig, rng: np.random.Generator) -> Terrain:
    if config.terrain_mode == "rounded":
        normalized, details = rounded_heightmap(
            config.size, config.roughness, rng,
            (config.rounded_hills_min, config.rounded_hills_max), return_details=True)
    else:
        normalized = diamond_square(config.size, config.roughness, rng)
        details = {"mode": "fractal"}
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
    details.update({"smoothing_meters": config.terrain_smoothing,
                    "island_falloff": config.island,
                    "realized_min_height": float(heights.min()),
                    "realized_max_height": float(heights.max())})
    return Terrain(heights, axis, axis.copy(), config.sea_level, details)
