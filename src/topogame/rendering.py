"""Off-screen matte terrain rendering with shared natural colors and sunlight."""

from pathlib import Path
import numpy as np

from .biomes import SURFACE_COLORS, terrain_colors
from .config import PipelineConfig
from .terrain import Terrain
from .viewpoint import Viewpoint


def render_view(terrain: Terrain, biomes: np.ndarray, camera: Viewpoint,
                config: PipelineConfig, path: Path, *, surface_rgb: np.ndarray | None = None) -> None:
    import pyvista as pv

    xx, yy = np.meshgrid(terrain.x, terrain.y)
    grid = pv.StructuredGrid(xx, yy, terrain.heights)
    rgb = terrain_colors(terrain) if surface_rgb is None else surface_rgb
    grid.point_data["surface_rgb"] = rgb.reshape(-1, 3, order="F")
    plotter = pv.Plotter(off_screen=True, window_size=(config.width, config.height), lighting="none")
    try:
        plotter.set_background("#dce4e5", top="#a4bac9")
        surface = grid.extract_surface(algorithm="dataset_surface").triangulate()
        plotter.add_mesh(surface, scalars="surface_rgb", rgb=True,
                         show_scalar_bar=False, smooth_shading=True,
                         ambient=config.ambient_light, diffuse=1 - config.ambient_light, specular=0)
        if np.any(terrain.heights <= terrain.sea_level):
            water = pv.Plane(center=(config.extent / 2, config.extent / 2, terrain.sea_level),
                             direction=(0, 0, 1), i_size=config.extent, j_size=config.extent)
            plotter.add_mesh(water, color=SURFACE_COLORS[3] / 255,
                             ambient=config.ambient_light, diffuse=1 - config.ambient_light, specular=0)
        center = np.array([config.extent / 2, config.extent / 2, float(terrain.heights.mean())])
        sun = pv.Light(position=center + np.asarray(config.sun_direction) * config.extent * 3,
                       focal_point=center, light_type="scene light", positional=False, intensity=1.0,
                       shadow_attenuation=0.7)
        plotter.add_light(sun)
        if config.shadows:
            plotter.enable_shadows()
        plotter.camera_position = [camera.position, camera.focal_point, camera.up]
        plotter.camera.parallel_projection = False
        plotter.camera.view_angle = camera.field_of_view_degrees
        plotter.camera.clipping_range = (0.1, config.extent * 5 + config.max_height - config.min_height)
        plotter.show(screenshot=str(path), auto_close=False)
    finally:
        plotter.close()
