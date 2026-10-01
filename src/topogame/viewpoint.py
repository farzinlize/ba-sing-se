"""Select a reproducible land position and a cardinal camera bearing."""

from dataclasses import dataclass
import math
import numpy as np

from .config import PipelineConfig
from .terrain import Terrain

DIRECTIONS = {"N": (0, 1), "E": (1, 0), "S": (0, -1), "W": (-1, 0)}


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


def select_viewpoint(terrain: Terrain, config: PipelineConfig,
                     rng: np.random.Generator) -> Viewpoint:
    candidates = terrain.heights > terrain.sea_level
    # Leave a margin so the camera isn't standing at the edge of the mesh.
    margin = max(1, config.size // 10)
    interior = np.zeros_like(candidates)
    interior[margin:-margin, margin:-margin] = True
    rows, columns = np.where(candidates & interior)
    if not len(rows):
        raise ValueError("No dry interior camera location; lower sea_level or try another seed")
    index = int(rng.integers(len(rows)))
    row, column = int(rows[index]), int(columns[index])
    direction = str(rng.choice(list(DIRECTIONS)))
    dx, dy = DIRECTIONS[direction]
    ground = float(terrain.heights[row, column])
    position = (float(terrain.x[column]), float(terrain.y[row]), ground + config.eye_height)
    distance = config.extent * 0.2
    focal_point = (position[0] + dx * distance, position[1] + dy * distance,
                   position[2] + distance * math.tan(math.radians(config.pitch)))
    return Viewpoint(row, column, direction, position, focal_point, ground,
                     config.pitch, config.field_of_view)
