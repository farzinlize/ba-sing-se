"""Choose a level hilltop view with multiple unobstructed, fully framed summits."""

from dataclasses import dataclass
import math
import numpy as np

from .config import PipelineConfig
from .terrain import Terrain

DIRECTIONS = {"N": (0, 1), "E": (1, 0), "S": (0, -1), "W": (-1, 0)}
# Reserve 10% of each image edge so selected summits are not cropped.
FRAME_LIMIT = 0.8
MIN_HILL_SEPARATION_DEGREES = 5.0


class NoSuitableViewError(ValueError):
    """This terrain cannot provide a camera view satisfying the composition rules."""


@dataclass(frozen=True)
class Viewpoint:
    row: int
    column: int
    direction: str
    position: tuple[float, float, float]
    focal_point: tuple[float, float, float]
    ground_height: float
    pitch_degrees: float
    field_of_view_degrees: float
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    visible_hills: tuple[tuple[int, int], ...] = ()

    @property
    def bearing_degrees(self) -> float:
        return math.degrees(math.atan2(self.focal_point[0] - self.position[0],
                                       self.focal_point[1] - self.position[1])) % 360


def find_hills(terrain: Terrain) -> np.ndarray:
    """Find separated local summits with relief, ignoring tiny surface bumps.

    A summit dominates a neighborhood about 5% of the map wide, rises at
    least 2.5% of the land's relief above the median of a wider (12%) neighborhood,
    and stands at least 15% of that relief above the land base (sea or minimum
    elevation, whichever is higher). Retain one summit per 6% of
    the map width so adjacent vertices cannot count as different hills.
    """
    heights = terrain.heights
    base = max(float(heights.min()), terrain.sea_level)
    relief = float(heights.max() - base)
    if relief <= 0:
        return np.empty((0, 2), dtype=int)
    radius = max(1, int(round(min(heights.shape) * 0.025)))
    relief_radius = max(radius, int(round(min(heights.shape) * 0.06)))
    local_max = heights.copy()
    for axis in (0, 1):
        padding = [(0, 0), (0, 0)]
        padding[axis] = (radius, radius)
        windows = np.lib.stride_tricks.sliding_window_view(
            np.pad(local_max, padding, mode="edge"), 2 * radius + 1, axis=axis)
        local_max = windows.max(axis=-1)
    candidates = np.argwhere((heights == local_max)
                             & (heights > base + relief * 0.15))
    candidates = sorted(candidates, key=lambda rc: (-heights[tuple(rc)], *rc))
    separation = 0.06 * min(np.ptp(terrain.x), np.ptp(terrain.y))
    peaks = []
    for row, col in candidates:
        # Boundary maxima need not be hills; there is no terrain beyond them.
        if not (0 < row < heights.shape[0] - 1 and 0 < col < heights.shape[1] - 1):
            continue
        # Broad, rounded summits need a wider neighborhood to measure relief.
        patch = heights[max(0, row - relief_radius):row + relief_radius + 1,
                        max(0, col - relief_radius):col + relief_radius + 1]
        if heights[row, col] - np.median(patch) < relief * 0.025:
            continue
        if all(math.hypot(terrain.x[col] - terrain.x[c], terrain.y[row] - terrain.y[r])
               >= separation for r, c in peaks):
            peaks.append((int(row), int(col)))
    return np.asarray(peaks, dtype=int).reshape(-1, 2)


def hill_is_visible(terrain: Terrain, row: int, column: int, eye_z: float,
                    target_row: int, target_column: int) -> bool:
    """Check the whole sightline against the terrain, including cell diagonals.

    Heights are linear within each rendered triangle. Check every grid and
    diagonal crossing, conservatively using the higher of the two possible
    cell triangulations so the result does not depend on VTK's diagonal choice.
    """
    dr, dc = target_row - row, target_column - column
    crossings = [np.array([0.0, 1.0])]
    for start, delta in ((row, dr), (column, dc), (row + column, dr + dc),
                         (row - column, dr - dc)):
        if delta:
            lines = np.arange(min(start, start + delta) + 1, max(start, start + delta))
            crossings.append((lines - start) / delta)
    t = np.unique(np.concatenate(crossings))
    rr, cc = row + dr * t, column + dc * t
    r = np.clip(np.floor(rr).astype(int), 0, terrain.heights.shape[0] - 2)
    c = np.clip(np.floor(cc).astype(int), 0, terrain.heights.shape[1] - 2)
    u, v = cc - c, rr - r
    h00, h10 = terrain.heights[r, c], terrain.heights[r, c + 1]
    h01, h11 = terrain.heights[r + 1, c], terrain.heights[r + 1, c + 1]
    diagonal_a = np.where(u >= v, h00 + (h10 - h00) * u + (h11 - h10) * v,
                          h00 + (h11 - h01) * u + (h01 - h00) * v)
    diagonal_b = np.where(u + v <= 1, h00 + (h10 - h00) * u + (h01 - h00) * v,
                          h11 + (h01 - h11) * (1 - u) + (h10 - h11) * (1 - v))
    surface = np.maximum(diagonal_a, diagonal_b)
    sightline = eye_z + (terrain.heights[target_row, target_column] - eye_z) * t
    # The endpoint is the summit itself; an earlier contact blocks the view.
    return bool(np.all(surface[:-1] < sightline[:-1] - 1e-8))


def select_viewpoint(terrain: Terrain, config: PipelineConfig,
                     rng: np.random.Generator) -> Viewpoint:
    heights = terrain.heights
    margin = max(1, config.size // 10)
    if not np.any(heights[margin:-margin, margin:-margin] > terrain.sea_level):
        raise NoSuitableViewError("No dry interior camera location; lower sea_level or try another seed")
    hills = find_hills(terrain)
    candidates = [(int(r), int(c)) for r, c in hills
                  if margin <= r < heights.shape[0] - margin
                  and margin <= c < heights.shape[1] - margin
                  and heights[r, c] < heights.max()]
    tan_v = math.tan(math.radians(config.field_of_view / 2))
    tan_h = tan_v * config.width / config.height
    for index in rng.permutation(len(candidates)):
        row, column = candidates[index]
        ground = float(heights[row, column])
        position = (float(terrain.x[column]), float(terrain.y[row]), ground + config.eye_height)
        visible = []
        for r, c in hills:
            if (r, c) == (row, column):
                continue
            dx, dy = terrain.x[c] - position[0], terrain.y[r] - position[1]
            distance = math.hypot(dx, dy)
            dz = heights[r, c] - position[2]
            if abs(dz) > distance * tan_v * FRAME_LIMIT:
                continue
            if hill_is_visible(terrain, row, column, position[2], int(r), int(c)):
                visible.append((int(r), int(c), math.atan2(dx, dy), distance, float(dz)))
        if len(visible) < config.min_visible_hills:
            continue
        angles = np.array([hill[2] for hill in visible])
        distances = np.array([hill[3] for hill in visible])
        dz = np.array([hill[4] for hill in visible])
        # Center each possible group between its outer summits, including N wraparound.
        bearings = list(angles)
        for i in range(len(angles)):
            for j in range(i + 1, len(angles)):
                gap = (angles[j] - angles[i] + math.pi) % (2 * math.pi) - math.pi
                bearings.append(angles[i] + gap / 2)
        best = None
        for bearing in bearings:
            offsets = (angles - bearing + math.pi) % (2 * math.pi) - math.pi
            depth = distances * np.cos(offsets)
            in_front = depth > 0.1
            safe_depth = np.where(in_front, depth, 1.0)
            horizontal = distances * np.sin(offsets) / (safe_depth * tan_h)
            vertical = dz / (safe_depth * tan_v)
            framed = np.flatnonzero(in_front & (np.abs(horizontal) <= FRAME_LIMIT)
                                     & (np.abs(vertical) <= FRAME_LIMIT))
            # Distinct geography must also be visibly separated in the image.
            selected = []
            for i in sorted(framed, key=lambda i: offsets[i]):
                if not selected or offsets[i] - offsets[selected[-1]] >= math.radians(MIN_HILL_SEPARATION_DEGREES):
                    selected.append(int(i))
            if len(selected) < config.min_visible_hills:
                continue
            score = (len(selected), -float(np.max(np.abs(horizontal[selected]))))
            if best is None or score > best[0]:
                best = (score, bearing, selected)
        if best is not None:
            _, bearing, selected = best
            distance = config.extent * 0.2
            focal_point = (position[0] + math.sin(bearing) * distance,
                           position[1] + math.cos(bearing) * distance, position[2])
            bearing_degrees = math.degrees(bearing) % 360
            direction = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")[int((bearing_degrees + 22.5) // 45) % 8]
            return Viewpoint(row, column, direction, position, focal_point, ground,
                             0.0, config.field_of_view,
                             visible_hills=tuple((visible[i][0], visible[i][1]) for i in selected))
    raise NoSuitableViewError(
        f"No horizontal hilltop view contains {config.min_visible_hills} distinct, unobstructed "
        "hilltops inside the image margins, from a hill below the highest summit. "
        "Try another seed, a wider field of view/image, or a different eye height.")
