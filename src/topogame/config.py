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
    terrain_mode: str = "rounded"
    terrain_smoothing: float = 12.0
    rounded_hills_min: int = 4
    rounded_hills_max: int = 15
    sun_azimuth: float = 315.0
    sun_elevation: float = 35.0
    shadows: bool = True
    ambient_light: float = 0.3
    hillshade_strength: float = 0.3
    render_style: str = "natural"
    tree_density: float = 12.0
    tree_height_min: float = 10.0
    tree_height_max: float = 24.0
    tree_species: tuple[str, ...] = ("pine", "oak", "birch")
    ground_cover_density: float = 1.0
    rock_density: float = 3.0
    vegetation_seed: int | None = None
    decoration_seed: int | None = None
    sky_fill: float = 0.45
    facet_variation: float = 0.035
    clouds: bool = True
    haze_strength: float = 0.12
    snow_line: float | None = None
    eye_height: float = 12.0
    pitch: float = 0.0
    field_of_view: float = 65.0
    min_visible_hills: int = 2
    contour_interval: float = 50.0
    width: int = 1280
    height: int = 720

    def __post_init__(self) -> None:
        # Accept a saved JSON configuration as well as the Python tuple form.
        if isinstance(self.tree_species, list):
            object.__setattr__(self, "tree_species", tuple(self.tree_species))
        for name in ("seed", "size", "width", "height", "min_visible_hills",
                     "rounded_hills_min", "rounded_hills_max"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        for name in ("vegetation_seed", "decoration_seed"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError(f"{name} must be a nonnegative integer or None")
        if self.size < 9 or (self.size - 1) & (self.size - 2):
            raise ValueError("size must be 2**n + 1 and at least 9 (e.g. 129, 257, 513)")
        for name in ("extent", "min_height", "max_height", "sea_level", "roughness",
                     "eye_height", "pitch", "field_of_view", "contour_interval",
                     "terrain_smoothing", "sun_azimuth", "sun_elevation", "ambient_light",
                     "hillshade_strength", "tree_density", "tree_height_min", "tree_height_max",
                     "ground_cover_density", "rock_density", "sky_fill", "facet_variation", "haze_strength"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if not self.min_height < self.max_height or self.sea_level >= self.max_height:
            raise ValueError("min_height < max_height and sea_level < max_height are required")
        if self.terrain_mode not in ("rounded", "fractal"):
            raise ValueError("terrain_mode must be rounded or fractal")
        if not 3 <= self.rounded_hills_min <= self.rounded_hills_max <= 40:
            raise ValueError("rounded hill counts must satisfy 3 <= min <= max <= 40")
        if self.render_style not in ("natural", "stylized"):
            raise ValueError("render_style must be natural or stylized")
        if not 0 <= self.tree_density <= 100 or not 0 <= self.rock_density <= 100:
            raise ValueError("tree_density and rock_density must be between 0 and 100 per hectare")
        if not 0 < self.tree_height_min <= self.tree_height_max <= 100:
            raise ValueError("tree heights must satisfy 0 < min <= max <= 100 meters")
        if not 0 <= self.ground_cover_density <= 5:
            raise ValueError("ground_cover_density must be between 0 and 5")
        if not isinstance(self.tree_species, tuple) or not self.tree_species or any(
                species not in ("pine", "oak", "birch") for species in self.tree_species):
            raise ValueError("tree_species must be a nonempty tuple of pine, oak, birch")
        if len(set(self.tree_species)) != len(self.tree_species):
            raise ValueError("tree_species must not contain duplicates")
        if not 0 <= self.sky_fill <= 1 or not 0 <= self.haze_strength <= 1 or not 0 <= self.facet_variation <= 0.2:
            raise ValueError("sky_fill/haze_strength must be in [0, 1]; facet_variation in [0, 0.2]")
        if self.snow_line is not None and not math.isfinite(self.snow_line):
            raise ValueError("snow_line must be finite or None")
        if not 0 <= self.terrain_smoothing <= self.extent / 4:
            raise ValueError("terrain_smoothing must be between 0 and extent / 4 meters")
        if not 0 <= self.sun_azimuth < 360 or not 0 < self.sun_elevation <= 90:
            raise ValueError("sun_azimuth must be in [0, 360) and sun_elevation in (0, 90]")
        if not 0 <= self.ambient_light <= 1 or not 0 <= self.hillshade_strength <= 1:
            raise ValueError("ambient_light and hillshade_strength must be between 0 and 1")
        if any(not isinstance(getattr(self, name), bool) for name in ("shadows", "island", "clouds")):
            raise ValueError("shadows, island and clouds must be booleans")
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

    @property
    def sun_direction(self) -> tuple[float, float, float]:
        """Unit vector toward the sun; azimuth is clockwise from north (+Y)."""
        azimuth, elevation = math.radians(self.sun_azimuth), math.radians(self.sun_elevation)
        return (math.sin(azimuth) * math.cos(elevation),
                math.cos(azimuth) * math.cos(elevation), math.sin(elevation))
