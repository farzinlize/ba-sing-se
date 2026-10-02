from dataclasses import replace
import json

import numpy as np
from matplotlib.figure import Figure
from PIL import Image
import pytest

from topogame import PipelineConfig, generate_sample
from topogame.biomes import Biome, classify_biomes
from topogame.cli import main
from topogame.monochrome import save_monochrome_map
from topogame.terrain import Terrain, diamond_square, generate_terrain
from topogame.topographic import CAMERA_MARKER_SIZE, COMPASS_DIRECTIONS, add_compass_guide
from topogame.viewpoint import DIRECTIONS, select_viewpoint


@pytest.mark.parametrize("changes", [
    {"size": 128}, {"size": 1}, {"size": 17.0}, {"seed": -1},
    {"roughness": 0}, {"roughness": 1}, {"sea_level": 700},
    {"extent": 0}, {"pitch": 90}, {"eye_height": -1},
    {"width": 0}, {"contour_interval": 0}, {"max_height": float("nan")},
    {"pitch": -10}, {"pitch": 1}, {"min_visible_hills": 1},
    {"min_visible_hills": True}, {"min_visible_hills": 2.5},
    {"rounded_hills_min": 2}, {"rounded_hills_min": 16, "rounded_hills_max": 15},
    {"rounded_hills_max": 41}, {"rounded_hills_min": 4.0},
])
def test_invalid_settings(changes):
    with pytest.raises(ValueError):
        PipelineConfig(**changes)


def test_diamond_square_reproducible_and_nontrivial():
    first = diamond_square(33, 0.55, np.random.default_rng(12))
    same = diamond_square(33, 0.55, np.random.default_rng(12))
    other = diamond_square(33, 0.55, np.random.default_rng(13))
    np.testing.assert_array_equal(first, same)
    assert not np.array_equal(first, other)
    assert first.shape == (33, 33)
    assert np.isfinite(first).all()
    assert first.min() == 0 and first.max() == 1
    assert np.unique(first).size > 1000


def test_island_bounds_and_coast():
    config = PipelineConfig(size=33)
    terrain = generate_terrain(config, np.random.default_rng(42))
    assert terrain.heights.min() == pytest.approx(config.min_height)
    assert terrain.heights.max() == pytest.approx(config.max_height)
    assert (terrain.heights[[0, -1], :] < terrain.sea_level).all()
    assert (terrain.heights[:, [0, -1]] < terrain.sea_level).all()
    assert terrain.x[-1] == config.extent


def test_altitude_zones_and_sea_level_boundary():
    terrain = Terrain(np.array([[-1, 0, 1, 10, 40, 70, 100.0]]),
                      np.arange(7), np.array([0]), 0)
    assert classify_biomes(terrain).tolist() == [[0, 0, 1, 2, 3, 4, 5]]


def test_camera_is_level_reproducible_and_consistent_with_grid():
    config = PipelineConfig(size=33)
    terrain = generate_terrain(config, np.random.default_rng(12))
    for seed in range(8):
        camera = select_viewpoint(terrain, config, np.random.default_rng(seed))
        same = select_viewpoint(terrain, config, np.random.default_rng(seed))
        assert camera == same
        assert camera.ground_height > terrain.sea_level
        assert camera.position[0] == terrain.x[camera.column]
        assert camera.position[1] == terrain.y[camera.row]
        assert camera.position[2] == camera.ground_height + config.eye_height
        delta = np.subtract(camera.focal_point, camera.position)
        assert np.linalg.norm(delta[:2]) == pytest.approx(config.extent * 0.2)
        assert delta[2] == 0
        assert camera.up == (0, 0, 1)
        assert camera.ground_height < terrain.heights.max()
        assert len(camera.visible_hills) >= 2


def test_camera_reports_no_land():
    config = PipelineConfig(size=9)
    terrain = Terrain(np.full((9, 9), -1.0), np.arange(9), np.arange(9), 0)
    with pytest.raises(ValueError, match="No dry interior"):
        select_viewpoint(terrain, config, np.random.default_rng(0))


def test_existing_output_is_preserved(tmp_path):
    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("keep")
    with pytest.raises(FileExistsError):
        generate_sample(tmp_path)
    assert sentinel.read_text() == "keep"


def test_failed_render_leaves_no_partial_sample(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("renderer unavailable")

    monkeypatch.setattr("topogame.pipeline.render_view", fail)
    with pytest.raises(RuntimeError, match="renderer unavailable"):
        generate_sample(tmp_path / "sample", PipelineConfig(size=17, seed=20))
    assert list(tmp_path.iterdir()) == []


def test_cli_rejects_invalid_input_before_creating_output(tmp_path):
    output = tmp_path / "samples"
    with pytest.raises(SystemExit) as error:
        main(["--output", str(output), "--size", "128"])
    assert error.value.code == 2
    assert not output.exists()


@pytest.mark.render
def test_complete_sample_and_reproduction(tmp_path):
    config = PipelineConfig(size=33, width=320, height=180, seed=21)
    first = generate_sample(tmp_path / "first", config)
    second = generate_sample(tmp_path / "second", replace(config))
    assert {p.name for p in first.iterdir()} == {
        "topographic.png", "topographic_monochrome.png", "view.png", "terrain.npz", "metadata.json",
    }
    metadata = json.loads((first / "metadata.json").read_text())
    assert metadata["schema_version"] == 5
    assert metadata["terrain_generation"]["mode"] == "rounded"
    assert 4 <= metadata["terrain_generation"]["hill_count"] <= 15
    assert len(metadata["camera"]["visible_hills"]) >= 2
    camera_data = metadata["camera"]
    delta = np.subtract(camera_data["focal_point"], camera_data["position"])
    bearing = np.degrees(np.arctan2(delta[0], delta[1])) % 360
    assert metadata["coordinates"]["bearing_degrees_clockwise_from_north"] == pytest.approx(bearing)
    assert metadata == json.loads((second / "metadata.json").read_text())
    assert (first / metadata["files"]["monochrome_map"]).is_file()
    with np.load(first / "terrain.npz") as data, np.load(second / "terrain.npz") as copy:
        np.testing.assert_array_equal(data["heights"], copy["heights"])
        np.testing.assert_array_equal(data["biomes"], copy["biomes"])
        camera = metadata["camera"]
        assert data["heights"][camera["row"], camera["column"]] == camera["ground_height"]
        assert data["biomes"][camera["row"], camera["column"]] != Biome.WATER
    with Image.open(first / "view.png") as view:
        assert view.size == (320, 180)
        # A scene must contain meaningful detail, beyond a blank background.
        pixels = np.array(view.convert("RGB"))
        assert np.unique(pixels.reshape(-1, 3), axis=0).shape[0] > 100
    with Image.open(first / "topographic.png") as topo:
        assert topo.width >= 1000 and topo.height >= 1000


def test_monochrome_map_only_colors_camera_position_and_hides_orientation(tmp_path):
    config = PipelineConfig(size=17)
    terrain = generate_terrain(config, np.random.default_rng(42))
    camera = select_viewpoint(terrain, config, np.random.default_rng(12))
    baseline = None
    for direction, (dx, dy) in DIRECTIONS.items():
        x, y, z = camera.position
        rotated = replace(camera, direction=direction, focal_point=(x + dx * 100, y + dy * 100, z))
        path = tmp_path / f"map_{direction}.png"
        save_monochrome_map(terrain, rotated, config, path)
        with Image.open(path) as image:
            pixels = np.array(image.convert("RGB"))
        chromatic = np.ptp(pixels, axis=2) > 5
        red = ((pixels[:, :, 0] > 180) & (pixels[:, :, 1] < 140)
               & (pixels[:, :, 2] < 140))
        assert red.sum() > 100
        assert chromatic.sum() < 2000
        assert pixels.min() == 0 and pixels.max() == 255
        if baseline is not None:
            np.testing.assert_array_equal(pixels, baseline)
        baseline = pixels


def test_colored_map_uses_large_camera_marker_and_subtle_bottom_right_compass():
    figure = Figure()
    ax = figure.subplots()
    add_compass_guide(ax)
    labels = {text.get_text(): text for text in ax.texts}
    assert CAMERA_MARKER_SIZE == 120
    assert set(labels) == set(COMPASS_DIRECTIONS)
    assert all(text.get_alpha() < 1 for text in labels.values())
    assert all(text.get_position()[0] > 0.8 and text.get_position()[1] < 0.2
               for text in labels.values())
    assert ax.patches[0].get_alpha() < 0.5
