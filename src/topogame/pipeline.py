"""Orchestrate independent stages and persist a complete sample together."""

from dataclasses import asdict
from importlib.metadata import version
import json
from pathlib import Path
import tempfile
import numpy as np

from .biomes import COLORS, LABELS, classify_biomes
from .config import PipelineConfig
from .monochrome import save_monochrome_map
from .rendering import render_view
from .terrain import generate_terrain
from .topographic import save_topographic_map
from .viewpoint import select_viewpoint


def generate_sample(output: str | Path, config: PipelineConfig | None = None) -> Path:
    """Create a new directory containing three images, raw terrain, and metadata.

    Existing paths are never overwritten. Failed runs discard their staging files.
    """
    config = config or PipelineConfig()
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    terrain_seed, camera_seed = np.random.SeedSequence(config.seed).spawn(2)
    terrain = generate_terrain(config, np.random.default_rng(terrain_seed))
    biomes = classify_biomes(terrain)
    camera = select_viewpoint(terrain, config, np.random.default_rng(camera_seed))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".topogame-", dir=output.parent) as temporary:
        stage = Path(temporary)
        save_topographic_map(terrain, biomes, camera, config, stage / "topographic.png")
        save_monochrome_map(terrain, camera, config, stage / "topographic_monochrome.png")
        render_view(terrain, biomes, camera, config, stage / "view.png")
        np.savez_compressed(stage / "terrain.npz", heights=terrain.heights, biomes=biomes,
                            x=terrain.x, y=terrain.y, sea_level=terrain.sea_level)
        metadata = {
            "schema_version": 1, "config": asdict(config), "camera": asdict(camera),
            "coordinates": {"units": "meters", "x": "east", "y": "north", "z": "up",
                            "array_indexing": "heights[row_y, column_x]; row 0 is south",
                            "bearing_degrees_clockwise_from_north": {"N": 0, "E": 90, "S": 180, "W": 270}[camera.direction]},
            "biomes": [{"id": i, "name": label, "rgb": color.tolist()}
                       for i, (label, color) in enumerate(zip(LABELS, COLORS))],
            "versions": {name: version(name) for name in ("topogame", "numpy", "matplotlib", "pyvista", "vtk")},
            "files": {"map": "topographic.png", "monochrome_map": "topographic_monochrome.png",
                      "view": "view.png", "terrain": "terrain.npz"},
        }
        (stage / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        # mkdir is exclusive even if another process chose the same destination.
        output.mkdir()
        for artifact in stage.iterdir():
            artifact.replace(output / artifact.name)
    return output
