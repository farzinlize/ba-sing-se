"""Rounded geometry, physical coordinates, and shared map/view appearance."""

from dataclasses import replace
import json

import numpy as np
from PIL import Image
import pytest

from topogame import PipelineConfig, generate_sample
from topogame.biomes import (SURFACE_COLORS, classify_biomes, hillshade_colors,
                            terrain_colors, terrain_normals)
from topogame.cli import main
from topogame.rendering import render_view
from topogame.terrain import (Terrain, diamond_square, generate_terrain,
                             rounded_heightmap, smooth_heightmap)
from topogame.viewpoint import find_hills, hill_is_visible, select_viewpoint


@pytest.mark.parametrize("changes", [
    {"terrain_mode": "unknown"}, {"terrain_smoothing": -1}, {"terrain_smoothing": 501},
    {"terrain_smoothing": float("nan")}, {"sun_azimuth": -1}, {"sun_azimuth": 360},
    {"sun_elevation": 0}, {"sun_elevation": 91}, {"sun_elevation": float("inf")},
    {"ambient_light": -0.1}, {"ambient_light": 1.1}, {"hillshade_strength": -0.1},
    {"hillshade_strength": 1.1}, {"shadows": "false"}, {"min_height": 600},
])
def test_invalid_terrain_and_lighting_settings(changes):
    with pytest.raises(ValueError):
        PipelineConfig(**changes)


def curvature(heights):
    return sum(np.mean(np.diff(heights, n=2, axis=axis) ** 2) for axis in (0, 1))


def test_gaussian_smoothing_reduces_sharpness_without_border_artifacts():
    heights = np.random.default_rng(3).uniform(size=(65, 65))
    smoothed = smooth_heightmap(heights, 1.5)
    assert smoothed.shape == heights.shape
    assert np.isfinite(smoothed).all()
    assert curvature(smoothed) < curvature(heights) * 0.05
    np.testing.assert_allclose(smooth_heightmap(np.full((9, 9), 7.0), 2), 7)
    np.testing.assert_array_equal(smooth_heightmap(heights, 0), heights)


def test_rounded_terrain_is_reproducible_dry_and_less_sharp_than_fractal():
    config = PipelineConfig(island=False, min_height=50, max_height=450, sea_level=-20)
    rounded = generate_terrain(config, np.random.default_rng(42))
    same = generate_terrain(config, np.random.default_rng(42))
    other = generate_terrain(config, np.random.default_rng(43))
    fractal = generate_terrain(replace(config, terrain_mode="fractal"), np.random.default_rng(42))
    np.testing.assert_array_equal(rounded.heights, same.heights)
    assert not np.array_equal(rounded.heights, other.heights)
    assert rounded.heights.min() == pytest.approx(50)
    assert rounded.heights.max() == pytest.approx(450)
    assert np.all(rounded.heights > rounded.sea_level)
    assert curvature(rounded.heights) < curvature(fractal.heights) * 0.3
    assert len(find_hills(rounded)) >= 3
    # No island mask forcing all boundary points to the minimum elevation.
    assert np.ptp(rounded.heights[0]) > 1


def test_rounded_hill_count_and_free_placement_are_seeded_and_constrained():
    first, details = rounded_heightmap(
        65, 0.55, np.random.default_rng(7), (6, 6), return_details=True)
    same, same_details = rounded_heightmap(
        65, 0.55, np.random.default_rng(7), (6, 6), return_details=True)
    np.testing.assert_array_equal(first, same)
    assert details == same_details
    assert details["hill_count"] == 6
    centers = np.asarray(details["hill_centers_normalized"])
    assert np.all((0.07 <= centers) & (centers <= 0.93))
    assert details["minimum_center_spacing_normalized"] >= 0.055
    # The former layout kept every coordinate close to the fixed 0.2/0.5/0.8 grid.
    distance_from_old_grid = np.min(
        np.abs(centers[..., None] - np.array([0.2, 0.5, 0.8])), axis=2)
    assert distance_from_old_grid.max() > 0.1


def test_rounded_seeds_vary_large_scale_landscape_structure():
    hill_counts, ridge_counts, isolated_counts = set(), set(), set()
    highland_centers, open_plain_fractions, dominant_quadrants = [], [], set()
    config = PipelineConfig(size=65, island=False)
    for seed in range(20):
        terrain = generate_terrain(config, np.random.default_rng(seed))
        details = terrain.generation_details
        hill_counts.add(details["hill_count"])
        ridge_counts.add(details["ridge_count"])
        isolated_counts.add(details["isolated_hill_count"])
        normalized = (terrain.heights - terrain.heights.min()) / np.ptp(terrain.heights)
        rows, columns = np.where(normalized >= np.quantile(normalized, 0.85))
        highland_centers.append((columns.mean() / 64, rows.mean() / 64))
        open_plain_fractions.append(float(np.mean(normalized < 0.22)))
        quadrant_means = [normalized[:32, :32].mean(), normalized[:32, 33:].mean(),
                          normalized[33:, :32].mean(), normalized[33:, 33:].mean()]
        dominant_quadrants.add(int(np.argmax(quadrant_means)))
    highland_centers = np.asarray(highland_centers)
    # These geometry metrics distinguish shifted clusters, broad plains, and
    # different connectivity, instead of merely checking unequal height arrays.
    assert len(hill_counts) >= 6
    assert len(ridge_counts) >= 6
    assert len(isolated_counts) >= 5
    assert np.ptp(highland_centers[:, 0]) > 0.35
    assert np.ptp(highland_centers[:, 1]) > 0.35
    assert np.ptp(open_plain_fractions) > 0.3
    assert dominant_quadrants == {0, 1, 2, 3}


def test_original_fractal_shape_is_available_without_smoothing():
    config = PipelineConfig(size=33, island=False, terrain_mode="fractal", terrain_smoothing=0)
    terrain = generate_terrain(config, np.random.default_rng(12))
    original = diamond_square(33, config.roughness, np.random.default_rng(12))
    np.testing.assert_allclose(terrain.heights, config.min_height + original * (config.max_height - config.min_height))


def test_lowering_sea_below_dry_land_does_not_change_summits_or_surface_colors():
    config = PipelineConfig(size=65, island=False, min_height=50, max_height=450)
    terrain = generate_terrain(config, np.random.default_rng(42))
    lowered = replace(terrain, sea_level=-10000)
    np.testing.assert_array_equal(find_hills(terrain), find_hills(lowered))
    np.testing.assert_array_equal(terrain_colors(terrain), terrain_colors(lowered))


def test_surface_colors_blend_smoothly_and_steep_slopes_expose_rock():
    x = np.linspace(0, 2000, 257)
    heights = np.tile(np.linspace(50, 150, 257), (9, 1))
    gentle = Terrain(heights, x, np.linspace(0, 100, 9), 0)
    steep = replace(gentle, x=x / 40)
    rgb, steep_rgb = terrain_colors(gentle), terrain_colors(steep)
    assert rgb.dtype == np.uint8
    assert np.max(np.abs(np.diff(rgb.astype(float), axis=1))) <= 2
    rock = SURFACE_COLORS[2].astype(float)
    assert np.linalg.norm(steep_rgb[4, 64] - rock) < np.linalg.norm(rgb[4, 64] - rock)
    wet = replace(gentle, sea_level=75)
    assert np.all(terrain_colors(wet)[heights <= 75] == SURFACE_COLORS[3])


def test_hillshade_uses_world_north_and_is_optional():
    axis = np.linspace(0, 100, 9)
    xx, yy = np.meshgrid(axis, axis)
    terrain = Terrain(200 - yy * 0.5, axis, axis, 0)  # North-facing slope.
    rgb = np.full((9, 9, 3), 128, dtype=np.uint8)
    north = PipelineConfig(sun_azimuth=0)
    south = replace(north, sun_azimuth=180)
    assert hillshade_colors(terrain, rgb, north).mean() > hillshade_colors(terrain, rgb, south).mean()
    np.testing.assert_allclose(hillshade_colors(terrain, rgb, replace(north, hillshade_strength=0)), rgb / 255)
    np.testing.assert_allclose(terrain_normals(terrain)[4, 4], np.array([0, 0.5, 1]) / np.sqrt(1.25))
    default = PipelineConfig()
    assert default.sun_direction[0] < 0 < default.sun_direction[1]
    assert np.linalg.norm(default.sun_direction) == pytest.approx(1)
    assert default.sun_direction[2] == pytest.approx(np.sin(np.radians(35)))


def test_camera_rules_hold_on_final_smoothed_dry_terrain():
    config = PipelineConfig(size=65, seed=3, island=False, min_height=50,
                            max_height=450, terrain_smoothing=25)
    terrain_seed, camera_seed = np.random.SeedSequence(config.seed).spawn(2)
    terrain = generate_terrain(config, np.random.default_rng(terrain_seed))
    camera = select_viewpoint(terrain, config, np.random.default_rng(camera_seed))
    assert camera.ground_height == terrain.heights[camera.row, camera.column]
    assert camera.ground_height < terrain.heights.max()
    assert camera.position[2] == camera.focal_point[2]
    assert len(camera.visible_hills) >= config.min_visible_hills
    forward = np.subtract(camera.focal_point, camera.position)
    forward /= np.linalg.norm(forward)
    for row, col in camera.visible_hills:
        assert hill_is_visible(terrain, camera.row, camera.column, camera.position[2], row, col)
        vector = np.array([terrain.x[col], terrain.y[row], terrain.heights[row, col]]) - camera.position
        half_height = (vector @ forward) * np.tan(np.radians(config.field_of_view / 2))
        assert abs(vector[2]) <= 0.8 * half_height
        assert abs(vector @ np.cross(forward, camera.up)) <= 0.8 * half_height * config.width / config.height


@pytest.mark.render
def test_dry_pair_metadata_and_lighting_controls(tmp_path):
    config = PipelineConfig(size=65, seed=3, island=False, min_height=50, max_height=450,
                            width=400, height=240, contour_interval=25)
    sample = generate_sample(tmp_path / "pair", config)
    metadata = json.loads((sample / "metadata.json").read_text())
    assert metadata["config"]["terrain_mode"] == "rounded"
    assert metadata["config"]["sun_azimuth"] == 315
    assert metadata["config"]["sun_elevation"] == 35
    assert metadata["config"]["terrain_smoothing"] == 12
    assert metadata["config"]["rounded_hills_min"] == 4
    assert metadata["config"]["rounded_hills_max"] == 15
    assert metadata["terrain_generation"]["hill_count"] >= 4
    assert metadata["config"]["shadows"] is True
    np.testing.assert_allclose(metadata["appearance"]["sun_direction"], config.sun_direction)
    with np.load(sample / "terrain.npz") as data:
        terrain = Terrain(data["heights"], data["x"], data["y"], float(data["sea_level"]))
        np.testing.assert_array_equal(data["surface_rgb"], terrain_colors(terrain))
    _, camera_seed = np.random.SeedSequence(config.seed).spawn(2)
    camera = select_viewpoint(terrain, config, np.random.default_rng(camera_seed))
    for name, changes in [("no_shadows", {"shadows": False}), ("opposite_sun", {"sun_azimuth": 135})]:
        path = tmp_path / f"{name}.png"
        render_view(terrain, classify_biomes(terrain), camera, replace(config, **changes), path)
        with Image.open(sample / "view.png") as original, Image.open(path) as changed:
            difference = np.abs(np.asarray(original.convert("RGB"), dtype=float) - np.asarray(changed.convert("RGB"), dtype=float))
            # A substantial set of pixels changes, not just a saved metadata value.
            assert np.count_nonzero(difference.max(axis=-1) > 2) > 100
    for filename in ("view.png", "topographic.png", "topographic_monochrome.png"):
        with Image.open(sample / filename) as image:
            assert image.width > 0 and image.height > 0


@pytest.mark.render
def test_cli_saves_requested_count_of_rounded_dry_pairs(tmp_path):
    assert main(["--output", str(tmp_path), "--seed", "42", "--count", "2", "--size", "65",
                 "--terrain-mode", "rounded", "--no-island", "--min-height", "50", "--max-height", "450",
                 "--sea-level", "-10", "--rounded-hills-min", "5", "--rounded-hills-max", "9",
                 "--terrain-smoothing", "20", "--sun-azimuth", "300",
                 "--sun-elevation", "40", "--ambient-light", "0.4", "--hillshade-strength", "0.2",
                 "--no-shadows", "--width", "320", "--height", "180"]) == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == ["sample_0000", "sample_0001"]
    for path in sorted(tmp_path.iterdir()):
        metadata = json.loads((path / "metadata.json").read_text())
        assert metadata["config"]["sun_azimuth"] == 300
        assert metadata["config"]["shadows"] is False
        assert metadata["config"]["terrain_smoothing"] == 20
        assert metadata["config"]["rounded_hills_min"] == 5
        assert metadata["config"]["rounded_hills_max"] == 9
        assert 5 <= metadata["terrain_generation"]["hill_count"] <= 9
        with np.load(path / "terrain.npz") as data:
            assert np.all(data["heights"] > data["sea_level"])
