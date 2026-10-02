"""Command-line batch generation."""

import argparse
from dataclasses import replace
from pathlib import Path
import secrets

from .config import PipelineConfig
from .pipeline import generate_sample
from .viewpoint import NoSuitableViewError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate paired topographic maps and horizontal hilltop camera images.")
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--count", type=int, default=1, help="Number of successful samples; unsuitable terrain is skipped")
    parser.add_argument("--seed", type=int, help="Starting seed; each attempt uses the next seed. Random if omitted.")
    parser.add_argument("--size", type=int, default=257, help="2**n + 1 grid points per side")
    parser.add_argument("--extent", type=float, default=2000, help="Map side length in meters")
    parser.add_argument("--min-height", type=float, default=-100)
    parser.add_argument("--max-height", type=float, default=600)
    parser.add_argument("--sea-level", type=float, default=0)
    parser.add_argument("--roughness", type=float, default=0.55)
    parser.add_argument("--no-island", dest="island", action="store_false", help="Disable shoreline falloff")
    parser.add_argument("--terrain-mode", choices=("rounded", "fractal"), default="rounded",
                        help="Broad hills and connected ridges (default), or diamond-square terrain")
    parser.add_argument("--terrain-smoothing", type=float, default=12, help="Gaussian smoothing radius (sigma) in meters; 0 disables")
    parser.add_argument("--rounded-hills-min", type=int, default=4, help="Minimum broad hills in rounded mode")
    parser.add_argument("--rounded-hills-max", type=int, default=15, help="Maximum broad hills in rounded mode")
    parser.add_argument("--sun-azimuth", type=float, default=315, help="Sun bearing clockwise from north; default NW (315)")
    parser.add_argument("--sun-elevation", type=float, default=35, help="Sun elevation above the horizon in degrees")
    parser.add_argument("--no-shadows", dest="shadows", action="store_false", help="Disable camera-image cast shadows")
    parser.add_argument("--ambient-light", type=float, default=0.3, help="Ambient surface lighting from 0 to 1")
    parser.add_argument("--hillshade-strength", type=float, default=0.3, help="Map hillshade strength from 0 to 1; 0 disables")
    parser.add_argument("--render-style", choices=("natural", "stylized"), default="natural")
    parser.add_argument("--tree-density", type=float, default=12, help="Stylized trees per hectare before spacing/visibility filtering")
    parser.add_argument("--tree-height-min", type=float, default=10)
    parser.add_argument("--tree-height-max", type=float, default=24)
    parser.add_argument("--tree-species", nargs="+", choices=("pine", "oak", "birch"), default=("pine", "oak", "birch"))
    parser.add_argument("--ground-cover-density", type=float, default=1, help="Shrub/grass density multiplier; 0 disables")
    parser.add_argument("--rock-density", type=float, default=3, help="Stylized rocks per hectare before filtering")
    parser.add_argument("--vegetation-seed", type=int, help="Override the independent vegetation seed")
    parser.add_argument("--decoration-seed", type=int, help="Override the independent rock/cloud/facet seed")
    parser.add_argument("--sky-fill", type=float, default=0.45, help="Stylized sky illumination, 0 to 1")
    parser.add_argument("--facet-variation", type=float, default=0.035, help="Stylized face color variation, 0 to 0.2")
    parser.add_argument("--no-clouds", dest="clouds", action="store_false")
    parser.add_argument("--haze-strength", type=float, default=0.12, help="Stylized distance color haze, 0 to 1")
    parser.add_argument("--snow-line", type=float, help="Optional stylized snow elevation in meters")
    parser.add_argument("--eye-height", type=float, default=12, help="Camera meters above local ground")
    parser.add_argument("--pitch", type=float, default=0, help="Must be 0: hilltop views keep the camera horizontal")
    parser.add_argument("--field-of-view", type=float, default=65, help="Vertical field of view in degrees")
    parser.add_argument("--min-visible-hills", type=int, default=2, help="Minimum distinct visible hilltops (at least 2)")
    parser.add_argument("--contour-interval", type=float, default=50, help="Contour spacing in meters")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = vars(parser.parse_args(argv))
    args["tree_species"] = tuple(args["tree_species"])
    output, count = args.pop("output"), args.pop("count")
    if count < 1:
        parser.error("count must be positive")
    if args["seed"] is None:
        args["seed"] = secrets.randbits(32)
    try:
        config = PipelineConfig(**args)
        destinations = [output / f"sample_{i:04d}" for i in range(count)]
        for path in destinations:
            if path.exists():
                raise FileExistsError(f"Output already exists: {path}; choose another --output directory")
        next_seed, skipped = config.seed, 0
        for i, path in enumerate(destinations):
            while True:
                sample_config = replace(config, seed=next_seed)
                next_seed += 1
                print(f"Generating {path} ({i + 1}/{count}, seed={sample_config.seed})", flush=True)
                try:
                    generate_sample(path, sample_config)
                except NoSuitableViewError as error:
                    skipped += 1
                    print(f"Skipping seed {sample_config.seed}: {error}", flush=True)
                    continue
                break
            print(f"Saved {path / 'topographic.png'}, "
                  f"{path / 'topographic_monochrome.png'}, and {path / 'view.png'}", flush=True)
        print(f"Completed {count} samples; skipped {skipped} unsuitable terrains.", flush=True)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    return 0
