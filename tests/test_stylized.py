"""Stylized scenery must remain reproducible, grounded and puzzle-safe."""
from dataclasses import asdict, replace
import json

import numpy as np
from PIL import Image
import pyvista as pv
import pytest

from topogame import PipelineConfig, generate_sample
from topogame.biomes import classify_biomes, terrain_colors, water_colors
from topogame.cli import main
from topogame.decoration import (Instance, Scene, batch_instances, build_scene,
                                 clear_summit_sightlines, generate_instances, prototype, scenery_seeds)
from topogame.rendering import render_view, stylized_water
from topogame.terrain import Terrain, generate_terrain
from topogame.viewpoint import NoSuitableViewError, select_viewpoint


@pytest.fixture
def landscape():
    config = PipelineConfig(seed=3, size=65, render_style='stylized', island=False, min_height=50,
                            max_height=450, tree_density=1.5, ground_cover_density=0.1,
                            rock_density=0.2, width=400, height=240)
    a, b = np.random.SeedSequence(config.seed).spawn(2)
    terrain = generate_terrain(config, np.random.default_rng(a))
    camera = select_viewpoint(terrain, config, np.random.default_rng(b))
    return terrain, camera, config


@pytest.mark.parametrize('changes', [
    {'render_style': 'invalid'}, {'tree_density': -1}, {'tree_density': 101},
    {'ground_cover_density': 6}, {'rock_density': -1}, {'tree_height_min': 0},
    {'tree_height_min': 30, 'tree_height_max': 20}, {'tree_height_max': 101},
    {'tree_species': ()}, {'tree_species': ('pine', 'pine')}, {'tree_species': ('invalid',)},
    {'tree_species': 'pine'}, {'vegetation_seed': -1}, {'decoration_seed': True},
    {'sky_fill': 1.1}, {'facet_variation': 0.21}, {'clouds': 1}, {'haze_strength': -1},
    {'snow_line': float('nan')}, {'tree_density': float('inf')},
])
def test_stylized_configuration_validation(changes):
    with pytest.raises(ValueError):
        PipelineConfig(**changes)


def test_config_replays_from_json():
    config = PipelineConfig(render_style='stylized', tree_species=('oak', 'pine'), vegetation_seed=9)
    assert PipelineConfig(**json.loads(json.dumps(asdict(config)))) == config


def test_scenery_reproducibility_and_independent_streams(landscape):
    terrain, camera, config = landscape
    first = generate_instances(terrain, camera, config)
    assert first == generate_instances(terrain, camera, config)
    changed = generate_instances(terrain, camera, replace(config, vegetation_seed=98))
    vegetation = {'pine', 'oak', 'birch', 'shrub', 'grass'}
    assert [i for i in first if i.kind in vegetation] != [i for i in changed if i.kind in vegetation]
    # Decorative rocks can be rejected near trees, but cloud generation is independent.
    assert [i for i in first if i.kind == 'cloud'] == [i for i in changed if i.kind == 'cloud']
    changed = generate_instances(terrain, camera, replace(config, decoration_seed=98))
    assert [i for i in first if i.kind in vegetation] == [i for i in changed if i.kind in vegetation]
    assert scenery_seeds(config)['vegetation'] != scenery_seeds(config)['decoration']


def test_placements_are_grounded_spaced_and_respect_snow(landscape):
    terrain, camera, config = landscape
    config = replace(config, snow_line=320)
    items = generate_instances(terrain, camera, config)
    trees = [i for i in items if i.kind in config.tree_species]
    assert {i.kind for i in trees} == set(config.tree_species)
    for item in items:
        if item.kind == 'cloud':
            continue
        row = int(np.argmin(abs(terrain.y - item.position[1])))
        col = int(np.argmin(abs(terrain.x - item.position[0])))
        assert item.position[2] == terrain.heights[row, col]
        assert item.position[2] > terrain.sea_level
        if item.kind != 'rock':
            assert item.position[2] < config.snow_line
    for index, tree in enumerate(trees):
        for other in trees[index + 1:]:
            assert np.linalg.norm(np.subtract(tree.position[:2], other.position[:2])) >= .43 * (tree.size + other.size) - 1e-8


def test_lod_reduces_tree_geometry_and_models_are_distinct():
    point_counts = []
    for species in ('pine', 'oak', 'birch'):
        close, distant = prototype(species, 0, 0), prototype(species, 0, 1)
        assert distant.n_cells < close.n_cells
        assert close.bounds.z_min < 0 < close.bounds.z_max
        point_counts.append(close.n_points)
    assert len(set(point_counts)) >= 2


def test_all_decorations_can_be_disabled(landscape):
    terrain, camera, config = landscape
    scene = build_scene(terrain, camera, replace(config, tree_density=0, ground_cover_density=0, rock_density=0, clouds=False))
    assert scene.instances == [] and scene.batches == []


def test_final_mesh_checks_remove_whole_obstructing_objects(landscape):
    terrain, camera, _ = landscape
    row, col = camera.visible_hills[0]
    target = np.array([terrain.x[col], terrain.y[row], terrain.heights[row, col]])
    midpoint = (np.asarray(camera.position) + target) / 2
    # Center a rock on the sightline, including its partially buried local origin.
    obstruction = Instance('rock', tuple(midpoint + [0, 0, 1.8]), 10, 0, 1, 0, 0)
    safe = replace(obstruction, position=(0, 0, -100))
    batch = batch_instances([obstruction, safe], [0, 1])
    before = prototype('rock', 0, 0).points.copy()
    cleaned, removed = clear_summit_sightlines([batch], terrain, camera)
    assert removed == {0}
    assert set(cleaned[0].cell_data['instance_id']) == {1}
    for mesh in cleaned:
        hits, _ = mesh.ray_trace(camera.position, target)
        assert not len(hits)
    np.testing.assert_array_equal(prototype('rock', 0, 0).points, before)


def test_final_scene_preserves_all_summits_and_records_reproducible_layout(landscape):
    terrain, camera, config = landscape
    scene = build_scene(terrain, camera, config)
    assert scene.metadata() == build_scene(terrain, camera, config).metadata()
    assert len(scene.batches) < len(scene.instances) / 5
    assert scene.metadata()['enabled'] is True
    for row, col in camera.visible_hills:
        target = (terrain.x[col], terrain.y[row], terrain.heights[row, col])
        for mesh in scene.batches:
            hits, _ = mesh.ray_trace(camera.position, target)
            assert not len(hits)
    with pytest.raises(NoSuitableViewError):
        build_scene(terrain, replace(camera, ground_height=float(terrain.heights.max())), config)


def test_stylized_water_only_covers_submerged_terrain():
    axis = np.linspace(0, 100, 9)
    xx, yy = np.meshgrid(axis, axis)
    heights = xx - 50
    terrain = Terrain(heights, axis, axis, 0)
    surface = pv.StructuredGrid(xx, yy, heights).extract_surface(algorithm='dataset_surface').triangulate()
    config = PipelineConfig(size=9, extent=100, terrain_smoothing=0)
    water = stylized_water(surface, terrain, config)
    assert water.n_cells > 0
    assert np.all(water.points[:, 0] <= 50 + 1e-6)
    assert np.all(water.points[:, 2] == 0)
    assert np.ptp(water['rgb'].astype(float), axis=0).max() > 20
    styled_config = replace(config, render_style='stylized')
    wet = heights <= 0
    np.testing.assert_array_equal(terrain_colors(terrain, styled_config)[wet], water_colors(-heights[wet], config.extent))
    assert stylized_water(surface, replace(terrain, sea_level=-100), config) is None


def test_dry_palette_has_no_beach_and_snow_is_optional(landscape):
    terrain, _, config = landscape
    np.testing.assert_array_equal(terrain_colors(terrain, config), terrain_colors(replace(terrain, sea_level=-1000), config))
    snow = terrain_colors(terrain, replace(config, snow_line=300))
    clear = terrain_colors(terrain, config)
    summit = np.unravel_index(terrain.heights.argmax(), terrain.heights.shape)
    assert snow[summit].mean() > clear[summit].mean() + 30


@pytest.mark.render
def test_style_comparison_keeps_terrain_camera_and_monochrome_map(landscape, tmp_path):
    _, _, config = landscape
    natural = generate_sample(tmp_path / 'natural', replace(config, render_style='natural'))
    stylized = generate_sample(tmp_path / 'stylized', config)
    one = json.loads((natural / 'metadata.json').read_text())
    two = json.loads((stylized / 'metadata.json').read_text())
    assert one['camera'] == two['camera']
    assert two['scenery']['enabled'] is True and two['scenery']['counts']['rock'] > 0
    with np.load(natural / 'terrain.npz') as a, np.load(stylized / 'terrain.npz') as b:
        np.testing.assert_array_equal(a['heights'], b['heights'])
        assert not np.array_equal(a['surface_rgb'], b['surface_rgb'])
    with Image.open(natural / 'topographic_monochrome.png') as a, Image.open(stylized / 'topographic_monochrome.png') as b:
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert (stylized / 'view.png').is_file()


@pytest.mark.render
def test_added_geometry_sun_shadows_and_sky_fill_affect_pixels(landscape, tmp_path):
    terrain, camera, config = landscape
    config = replace(config, clouds=False, haze_strength=0)
    scene = build_scene(terrain, camera, config)
    images = {}
    for name, changes, objects in [('baseline', {}, scene), ('bare', {}, Scene([], [], scenery_seeds(config))),
                                    ('unshadowed', {'shadows': False}, scene),
                                    ('opposite_sun', {'sun_azimuth': 135}, scene), ('no_fill', {'sky_fill': 0}, scene)]:
        path = tmp_path / f'{name}.png'
        render_view(terrain, classify_biomes(terrain), camera, replace(config, **changes), path, scene=objects)
        with Image.open(path) as image:
            images[name] = np.array(image.convert('RGB'), dtype=float)
    for name in ('bare', 'unshadowed', 'opposite_sun', 'no_fill'):
        assert np.count_nonzero(abs(images['baseline'] - images[name]).max(axis=-1) > 2) > 100
    assert images['baseline'][120:].mean() > images['no_fill'][120:].mean()


@pytest.mark.render
def test_stylized_batch_retries_unsuitable_terrain_and_saves_exact_count(tmp_path, capsys):
    assert main(['--output', str(tmp_path), '--seed', '42', '--count', '2', '--size', '17',
                 '--terrain-mode', 'fractal', '--terrain-smoothing', '0', '--render-style', 'stylized',
                 '--tree-density', '0.2', '--rock-density', '0.1', '--ground-cover-density', '0',
                 '--no-clouds', '--width', '320', '--height', '180']) == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == ['sample_0000', 'sample_0001']
    assert 'Skipping seed 42' in capsys.readouterr().out
