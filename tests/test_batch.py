"""Only completed, valid views count toward a requested batch size."""

import json

import pytest

from topogame.cli import main
from topogame.viewpoint import NoSuitableViewError


def test_batch_skips_unsuitable_seeds_and_keeps_contiguous_sample_numbers(tmp_path, monkeypatch, capsys):
    attempts = []

    def generate(path, config):
        attempts.append((path.name, config.seed))
        if config.seed in (42, 44):
            raise NoSuitableViewError("No suitable hilltop view")
        path.mkdir()
        (path / "seed.txt").write_text(str(config.seed))

    monkeypatch.setattr("topogame.cli.generate_sample", generate)
    assert main(["--output", str(tmp_path), "--seed", "42", "--count", "3"]) == 0
    assert attempts == [
        ("sample_0000", 42), ("sample_0000", 43),
        ("sample_0001", 44), ("sample_0001", 45), ("sample_0002", 46),
    ]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["sample_0000", "sample_0001", "sample_0002"]
    assert [(tmp_path / f"sample_{i:04d}" / "seed.txt").read_text() for i in range(3)] == ["43", "45", "46"]
    output = capsys.readouterr().out
    assert "Skipping seed 42" in output and "Skipping seed 44" in output
    assert "Completed 3 samples; skipped 2" in output


@pytest.mark.parametrize("error", [ValueError("invalid render setting"), OSError("write failed")])
def test_batch_does_not_retry_unrelated_errors(tmp_path, monkeypatch, error):
    attempts = []

    def fail(path, config):
        attempts.append(config.seed)
        raise error

    monkeypatch.setattr("topogame.cli.generate_sample", fail)
    with pytest.raises(SystemExit) as caught:
        main(["--output", str(tmp_path), "--seed", "42", "--count", "3"])
    assert caught.value.code == 2
    assert attempts == [42]


def test_batch_checks_all_existing_destinations_before_attempting_generation(tmp_path, monkeypatch):
    existing = tmp_path / "sample_0001"
    existing.mkdir()
    sentinel = existing / "keep.txt"
    sentinel.write_text("keep")

    def unexpected(*args):
        pytest.fail("Generation must not start when a destination already exists")

    monkeypatch.setattr("topogame.cli.generate_sample", unexpected)
    with pytest.raises(SystemExit):
        main(["--output", str(tmp_path), "--count", "2"])
    assert sentinel.read_text() == "keep"
    assert not (tmp_path / "sample_0000").exists()


@pytest.mark.render
def test_real_batch_retries_and_saves_exact_count_with_actual_seeds(tmp_path, capsys):
    # This terrain at seed 42 has only one hill and must be skipped.
    assert main(["--output", str(tmp_path), "--seed", "42", "--size", "17",
                 "--terrain-mode", "fractal", "--terrain-smoothing", "0",
                 "--count", "2", "--width", "320", "--height", "180"]) == 0
    samples = sorted(tmp_path.iterdir())
    assert [p.name for p in samples] == ["sample_0000", "sample_0001"]
    seeds = []
    for sample in samples:
        metadata = json.loads((sample / "metadata.json").read_text())
        seeds.append(metadata["config"]["seed"])
        assert len(metadata["camera"]["visible_hills"]) >= 2
        assert {p.name for p in sample.iterdir()} == {
            "topographic.png", "topographic_monochrome.png", "view.png", "terrain.npz", "metadata.json",
        }
    assert 42 < seeds[0] < seeds[1]
    assert "Skipping seed 42" in capsys.readouterr().out
