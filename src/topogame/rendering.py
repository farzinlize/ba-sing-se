"""Natural and stylized PyVista views of the exact same terrain and camera."""
from pathlib import Path
import numpy as np

from .biomes import SURFACE_COLORS, terrain_colors, water_colors
from .config import PipelineConfig
from .decoration import Scene, build_scene, scenery_seeds
from .terrain import Terrain
from .viewpoint import Viewpoint


def stylized_water(surface, terrain: Terrain, config: PipelineConfig):
    """Clip a sea-level mesh at the actual shoreline; color by seabed depth."""
    if terrain.heights.min() >= terrain.sea_level:
        return None
    water = surface.copy(deep=True)
    depth = terrain.sea_level - water.points[:, 2]
    water.point_data['depth'] = depth
    water.points[:, 2] = terrain.sea_level
    water = water.clip_scalar(scalars='depth', value=0, invert=False)
    water.point_data['rgb'] = water_colors(water['depth'], config.extent)
    return water


def _haze(rgb, positions, camera: Viewpoint, config: PipelineConfig):
    distance = np.linalg.norm(positions - camera.position, axis=1)
    amount = np.minimum(0.35, 1 - np.exp(-config.haze_strength * distance / config.extent))
    return np.rint(np.clip(rgb * (1 - amount[:, None]) + np.array([172, 208, 232]) * amount[:, None], 0, 255)).astype(np.uint8)


def render_view(terrain: Terrain, biomes: np.ndarray, camera: Viewpoint,
                config: PipelineConfig, path: Path, *, surface_rgb: np.ndarray | None = None,
                scene: Scene | None = None) -> None:
    import pyvista as pv

    stylized = config.render_style == 'stylized'
    if stylized and scene is None:
        scene = build_scene(terrain, camera, config)
    xx, yy = np.meshgrid(terrain.x, terrain.y)
    grid = pv.StructuredGrid(xx, yy, terrain.heights)
    rgb = terrain_colors(terrain, config) if surface_rgb is None else surface_rgb
    grid.point_data['surface_rgb'] = rgb.reshape(-1, 3, order='F')
    plotter = pv.Plotter(off_screen=True, window_size=(config.width, config.height), lighting='none')
    try:
        plotter.set_background('#b9ddf3' if stylized else '#dce4e5', top='#479bdd' if stylized else '#a4bac9')
        surface = grid.extract_surface(algorithm='dataset_surface').triangulate()
        if stylized:
            # No decimation or displacement: all geometry stays identical to camera checks.
            faceted = surface.point_data_to_cell_data(pass_point_data=True)
            seed = scenery_seeds(config)['decoration']
            rng = np.random.default_rng(np.random.SeedSequence([seed, 991]))
            factor = rng.uniform(1 - config.facet_variation, 1 + config.facet_variation, (faceted.n_cells, 1))
            faceted.cell_data['face_rgb'] = _haze(faceted.cell_data['surface_rgb'] * factor,
                                                 faceted.cell_centers().points, camera, config)
            plotter.add_mesh(faceted, scalars='face_rgb', preference='cell', rgb=True,
                             show_scalar_bar=False, smooth_shading=False,
                             ambient=config.ambient_light, diffuse=1 - config.ambient_light, specular=0)
        else:
            plotter.add_mesh(surface, scalars='surface_rgb', rgb=True,
                             show_scalar_bar=False, smooth_shading=True,
                             ambient=config.ambient_light, diffuse=1 - config.ambient_light, specular=0)
        if stylized:
            water = stylized_water(surface, terrain, config)
            if water is not None and water.n_cells:
                water['rgb'] = _haze(water['rgb'], water.points, camera, config)
                plotter.add_mesh(water, scalars='rgb', rgb=True, show_scalar_bar=False,
                                 ambient=config.ambient_light, diffuse=1 - config.ambient_light, specular=0)
            for batch in scene.batches:
                mesh = batch.copy(deep=False)
                cloud = str(mesh.field_data['kind'][0]) == 'cloud'
                mesh.point_data['render_rgb'] = _haze(mesh.point_data['rgb'], mesh.points, camera, config)
                plotter.add_mesh(mesh, scalars='render_rgb', rgb=True, show_scalar_bar=False,
                                 smooth_shading=False, ambient=0.8 if cloud else config.ambient_light,
                                 diffuse=0.2 if cloud else 1 - config.ambient_light, specular=0)
        elif np.any(terrain.heights <= terrain.sea_level):
            water = pv.Plane(center=(config.extent / 2, config.extent / 2, terrain.sea_level),
                             direction=(0, 0, 1), i_size=config.extent, j_size=config.extent)
            plotter.add_mesh(water, color=SURFACE_COLORS[3] / 255,
                             ambient=config.ambient_light, diffuse=1 - config.ambient_light, specular=0)
        center = np.array([config.extent / 2, config.extent / 2, float(terrain.heights.mean())])
        sun = pv.Light(position=center + np.asarray(config.sun_direction) * config.extent * 3,
                       focal_point=center, light_type='scene light', positional=False,
                       color='#fff4df' if stylized else 'white', intensity=1.1 if stylized else 1.0,
                       shadow_attenuation=0.7)
        plotter.add_light(sun)
        if stylized and config.sky_fill:
            # Unattenuated fill keeps sun-shadowed faces readable without a second shadow.
            fill = pv.Light(position=center + [0, 0, config.extent * 3], focal_point=center,
                            light_type='scene light', positional=False, color='#d7e9ff',
                            intensity=config.sky_fill, shadow_attenuation=0)
            plotter.add_light(fill)
        if config.shadows:
            plotter.enable_shadows()
        plotter.camera_position = [camera.position, camera.focal_point, camera.up]
        plotter.camera.parallel_projection = False
        plotter.camera.view_angle = camera.field_of_view_degrees
        plotter.camera.clipping_range = (0.1, config.extent * 5 + config.max_height - config.min_height)
        plotter.show(screenshot=str(path), auto_close=False)
    finally:
        plotter.close()
