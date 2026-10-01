"""Shared, validated pipeline settings; all distances are in meters."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PipelineConfig:
    seed: int = 42
    size: int = 257
    extent: float = 2000.0
    min_height: float = -100.0
    max_height: float = 600.0
    sea_level: float = 0.0
    roughness: float = 0.55
    island: bool = True
    eye_height: float = 12.0
    pitch: float = 0.0
    field_of_view: float = 65.0
    min_visible_hills: int = 2
    contour_interval: float = 50.0
    width: int = 1280
    height: int = 720

    def __post_init__(self) -> None:
        for name in ("seed", "size", "width", "height", "min_visible_hills"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        if self.size < 9 or (self.size - 1) & (self.size - 2):
            raise ValueError("size must be 2**n + 1 and at least 9 (e.g. 129, 257, 513)")
        for name in ("extent", "min_height", "max_height", "sea_level", "roughness",
                     "eye_height", "pitch", "field_of_view", "contour_interval"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if not self.min_height < self.sea_level < self.max_height:
            raise ValueError("min_height < sea_level < max_height is required")
        if not 0 < self.roughness < 1:
            raise ValueError("roughness must be between 0 and 1")
        if min(self.extent, self.eye_height, self.contour_interval) <= 0:
            raise ValueError("extent, eye_height, and contour_interval must be positive")
        if self.pitch != 0:
            raise ValueError("pitch must be 0 for a horizontal, human-style camera view")
        if self.min_visible_hills < 2:
            raise ValueError("min_visible_hills must be at least 2")
        if not 10 <= self.field_of_view <= 120:
            raise ValueError("field_of_view must be between 10 and 120 degrees")
        if min(self.width, self.height) < 64:
            raise ValueError("image dimensions must be at least 64 pixels")
