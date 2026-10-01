"""Off-screen perspective rendering of the same terrain and biome colors."""

from pathlib import Path
import numpy as np

from .biomes import COLORS, Biome
from .config import PipelineConfig
from .terrain import Terrain
from .viewpoint import Viewpoint


def render_view(terrain: Terrain, biomes: np.ndarray, camera: Viewpoint,
                config: PipelineConfig, path: Path) -> None:
    import pyvista as pv

    xx, yy = np.meshgrid(terrain.x, terrain.y)
    grid = pv.StructuredGrid(xx, yy, terrain.heights)
    grid.point_data["biome_rgb"] = COLORS[biomes.ravel(order="F")]
    plotter = pv.Plotter(off_screen=True, window_size=(config.width, config.height))
    try:
        plotter.set_background("#badcef", top="#659ec8")
        surface = grid.extract_surface(algorithm="dataset_surface").triangulate()
        plotter.add_mesh(surface, scalars="biome_rgb", rgb=True,
                         show_scalar_bar=False, smooth_shading=True,
                         ambient=0.35, diffuse=0.65, specular=0.05)
        water = pv.Plane(center=(config.extent / 2, config.extent / 2, terrain.sea_level),
                         direction=(0, 0, 1), i_size=config.extent, j_size=config.extent)
        plotter.add_mesh(water, color=COLORS[Biome.WATER] / 255, lighting=False)
        plotter.camera_position = [camera.position, camera.focal_point, camera.up]
        plotter.camera.view_angle = camera.field_of_view_degrees
        plotter.camera.clipping_range = (0.1, config.extent * 5 + config.max_height - config.min_height)
        plotter.show(screenshot=str(path), auto_close=False)
    finally:
        plotter.close()
