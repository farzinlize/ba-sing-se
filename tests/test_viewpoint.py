"""Camera composition tests using known hills, ridges and image shapes."""

from dataclasses import replace

import numpy as np
import pytest

from topogame import PipelineConfig, generate_sample
from topogame.terrain import Terrain, generate_terrain
from topogame.viewpoint import find_hills, hill_is_visible, select_viewpoint


def three_hills():
    axis = np.linspace(0, 1600, 65)
    xx, yy = np.meshgrid(axis, axis)
    heights = np.full(xx.shape, 10.0)
    for row, col, peak in [(32, 12, 80), (20, 44, 160), (44, 44, 220)]:
        distance = np.hypot(xx - axis[col], yy - axis[row])
        heights = np.maximum(heights, 10 + (peak - 10) * np.maximum(1 - distance / 180, 0))
    return Terrain(heights, axis, axis.copy(), 0)


def test_selects_lower_hill_and_frames_two_separate_summits():
    terrain = three_hills()
    config = PipelineConfig(size=65, extent=1600)
    camera = select_viewpoint(terrain, config, np.random.default_rng(42))
    assert (camera.row, camera.column) == (32, 12)
    assert set(camera.visible_hills) == {(20, 44), (44, 44)}
    assert camera.bearing_degrees == pytest.approx(90)
    assert camera.position[2] == camera.focal_point[2]
    assert camera.position[2] == 80 + config.eye_height
    for row, col in camera.visible_hills:
        offset = np.array([terrain.x[col], terrain.y[row], terrain.heights[row, col]]) - camera.position
        forward = np.subtract(camera.focal_point, camera.position)
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, camera.up)
        depth = offset @ forward
        half_height = depth * np.tan(np.radians(config.field_of_view / 2))
        half_width = half_height * config.width / config.height
        # Independently project into pixel space: 10% margin on all sides.
        pixel_x = config.width * (1 + (offset @ right) / half_width) / 2
        pixel_y = config.height * (1 - offset[2] / half_height) / 2
        assert 0.1 * config.width <= pixel_x <= 0.9 * config.width
        assert 0.1 * config.height <= pixel_y <= 0.9 * config.height


@pytest.mark.parametrize("changes", [
    {"width": 360, "height": 720}, {"field_of_view": 10},
    {"eye_height": 2000}, {"min_visible_hills": 3},
])
def test_rejects_views_that_cannot_fit_the_required_hills(changes):
    config = replace(PipelineConfig(size=65, extent=1600), **changes)
    with pytest.raises(ValueError, match="No horizontal hilltop view"):
        select_viewpoint(three_hills(), config, np.random.default_rng(42))


def test_camera_can_face_across_north_without_angle_wraparound_error():
    original = three_hills()
    terrain = replace(original, heights=original.heights.T.copy())
    camera = select_viewpoint(terrain, PipelineConfig(size=65, extent=1600), np.random.default_rng(42))
    assert min(camera.bearing_degrees, 360 - camera.bearing_degrees) < 1e-8
    assert set(camera.visible_hills) == {(44, 20), (44, 44)}


def test_intervening_ridge_blocks_a_summit():
    heights = np.zeros((9, 9))
    heights[7, 4] = 20
    terrain = Terrain(heights, np.arange(9.0), np.arange(9.0), 0)
    assert hill_is_visible(terrain, 1, 4, 10, 7, 4)
    heights[4, :] = 30
    assert not hill_is_visible(terrain, 1, 4, 10, 7, 4)
    # A higher eye can clear the ridge; visibility is not merely peak detection.
    assert hill_is_visible(terrain, 1, 4, 50, 7, 4)


def test_diagonal_cell_obstruction_is_not_missed():
    heights = np.zeros((9, 9))
    heights[7, 7] = 10
    heights[3, 4] = heights[4, 3] = 30
    terrain = Terrain(heights, np.arange(9.0), np.arange(9.0), 0)
    # Grid vertices on the diagonal are clear, but the intervening triangle isn't.
    assert not hill_is_visible(terrain, 1, 1, 12, 7, 7)


@pytest.mark.parametrize("shape", ["flat", "one_hill", "plateau"])
def test_rejects_land_without_multiple_distinct_hills(shape):
    axis = np.linspace(0, 1600, 65)
    xx, yy = np.meshgrid(axis, axis)
    heights = np.full((65, 65), 10.0)
    if shape == "one_hill":
        heights += 100 * np.maximum(1 - np.hypot(xx - 800, yy - 800) / 300, 0)
    elif shape == "plateau":
        heights[20:45, 20:45] = 100
    terrain = Terrain(heights, axis, axis.copy(), 0)
    with pytest.raises(ValueError, match="No horizontal hilltop view"):
        select_viewpoint(terrain, PipelineConfig(size=65, extent=1600), np.random.default_rng(0))


def test_unsuitable_terrain_leaves_no_sample(tmp_path):
    with pytest.raises(ValueError, match="No horizontal hilltop view"):
        generate_sample(tmp_path / "sample", PipelineConfig(size=17, seed=42,
                                                          terrain_mode="fractal", terrain_smoothing=0))
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("seed", [7, 14, 21, 42])
def test_generated_terrain_has_valid_camera_and_visible_hills(seed):
    config = PipelineConfig(seed=seed)
    terrain_seed, camera_seed = np.random.SeedSequence(seed).spawn(2)
    terrain = generate_terrain(config, np.random.default_rng(terrain_seed))
    camera = select_viewpoint(terrain, config, np.random.default_rng(camera_seed))
    assert (camera.row, camera.column) in map(tuple, find_hills(terrain))
    assert camera.ground_height < terrain.heights.max()
    assert len(camera.visible_hills) >= 2
    for row, col in camera.visible_hills:
        assert hill_is_visible(terrain, camera.row, camera.column, camera.position[2], row, col)
