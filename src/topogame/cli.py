"""Command-line batch generation."""

import argparse
from pathlib import Path
import secrets

from .config import PipelineConfig
from .pipeline import generate_sample


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate paired topographic maps and cardinal camera images.")
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--seed", type=int, help="Base seed; sample i uses seed + i. Random if omitted.")
    parser.add_argument("--size", type=int, default=257, help="2**n + 1 grid points per side")
    parser.add_argument("--extent", type=float, default=2000, help="Map side length in meters")
    parser.add_argument("--min-height", type=float, default=-100)
    parser.add_argument("--max-height", type=float, default=600)
    parser.add_argument("--sea-level", type=float, default=0)
    parser.add_argument("--roughness", type=float, default=0.55)
    parser.add_argument("--no-island", dest="island", action="store_false", help="Disable shoreline falloff")
    parser.add_argument("--eye-height", type=float, default=12, help="Camera meters above local ground")
    parser.add_argument("--pitch", type=float, default=0, help="Camera pitch in degrees; positive looks upward")
    parser.add_argument("--field-of-view", type=float, default=65, help="Vertical field of view in degrees")
    parser.add_argument("--contour-interval", type=float, default=50, help="Contour spacing in meters")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = vars(parser.parse_args(argv))
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
        for i, path in enumerate(destinations):
            sample_config = PipelineConfig(**{**args, "seed": config.seed + i})
            print(f"Generating {path} (seed={sample_config.seed})", flush=True)
            generate_sample(path, sample_config)
            print(f"Saved {path / 'topographic.png'}, "
                  f"{path / 'topographic_monochrome.png'}, and {path / 'view.png'}", flush=True)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    return 0
