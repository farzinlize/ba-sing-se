"""Draw a north-up hillshaded map with readable elevation contours."""

from pathlib import Path
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Patch
from matplotlib import patheffects

from .biomes import SURFACE_LABELS, SNOW_COLOR, hillshade_colors, surface_palette, terrain_colors
from .config import PipelineConfig
from .terrain import Terrain
from .viewpoint import Viewpoint


CAMERA_MARKER_SIZE = 120
COMPASS_DIRECTIONS = {
    "N": (0.0, 1.0), "NE": (2 ** -0.5, 2 ** -0.5),
    "E": (1.0, 0.0), "SE": (2 ** -0.5, -(2 ** -0.5)),
    "S": (0.0, -1.0), "SW": (-(2 ** -0.5), -(2 ** -0.5)),
    "W": (-1.0, 0.0), "NW": (-(2 ** -0.5), 2 ** -0.5),
}


def add_compass_guide(ax) -> None:
    """Add a subtle eight-direction guide inside the map's bottom-right corner."""
    center_x, center_y = 0.91, 0.09
    ray_radius, label_radius = 0.034, 0.064
    transform = ax.transAxes
    ax.add_patch(Circle((center_x, center_y), 0.073, transform=transform,
                        facecolor="white", edgecolor="#38443c", linewidth=0.65,
                        alpha=0.34, zorder=6, clip_on=False))
    for label, (dx, dy) in COMPASS_DIRECTIONS.items():
        ax.plot([center_x, center_x + dx * ray_radius],
                [center_y, center_y + dy * ray_radius],
                transform=transform, color="#303932", linewidth=0.65,
                alpha=0.48, zorder=7, solid_capstyle="round")
        ax.text(center_x + dx * label_radius, center_y + dy * label_radius, label,
                transform=transform, ha="center", va="center", fontsize=6.5,
                color="#263129", alpha=0.66, zorder=8)
    ax.scatter([center_x], [center_y], transform=transform, s=5,
               c="#303932", alpha=0.5, zorder=8)


def save_topographic_map(terrain: Terrain, biomes: np.ndarray, camera: Viewpoint,
                         config: PipelineConfig, path: Path, *, surface_rgb: np.ndarray | None = None) -> None:
    figure = Figure(figsize=(9, 9), layout="constrained")
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    half_cell = (terrain.x[1] - terrain.x[0]) / 2
    rgb = terrain_colors(terrain, config) if surface_rgb is None else surface_rgb
    ax.imshow(hillshade_colors(terrain, rgb, config), origin="lower",
              extent=(-half_cell, config.extent + half_cell, -half_cell, config.extent + half_cell),
              interpolation="bilinear")
    start = np.ceil(terrain.heights.min() / config.contour_interval) * config.contour_interval
    levels = np.arange(start, terrain.heights.max(), config.contour_interval)
    if len(levels):
        contours = ax.contour(terrain.x, terrain.y, terrain.heights, levels=levels,
                              colors="#30392d", linewidths=0.6, alpha=0.85)
        labels = ax.clabel(contours, inline=True, fontsize=7, fmt="%g m")
        for label in labels:
            label.set_path_effects([patheffects.withStroke(linewidth=2, foreground="#f6f5ef")])
    if terrain.heights.min() < terrain.sea_level < terrain.heights.max():
        ax.contour(terrain.x, terrain.y, terrain.heights, levels=[terrain.sea_level],
                   colors="#315867", linewidths=1.0)
    x, y, _ = camera.position
    delta = np.subtract(camera.focal_point[:2], camera.position[:2])
    dx, dy = delta / np.linalg.norm(delta)
    length = config.extent * 0.075
    ax.scatter([x], [y], c="#f54242", edgecolors="white",
               linewidths=1.0, s=CAMERA_MARKER_SIZE, zorder=5)
    ax.annotate("", xy=(x + dx * length, y + dy * length), xytext=(x, y),
                arrowprops={"arrowstyle": "-|>", "color": "#ee3030", "lw": 2.5})
    ax.annotate(f"Camera · {camera.direction}", (x, y), xytext=(8, 8),
                textcoords="offset points", fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
    ax.set(xlim=(0, config.extent), ylim=(0, config.extent), aspect="equal",
           xlabel="East (m)", ylabel="North (m)",
           title=f"Terrain {config.seed} · North ↑ · contours {config.contour_interval:g} m")
    add_compass_guide(ax)
    legend = [Patch(facecolor=color / 255, label=label)
              for color, label in zip(surface_palette(config), SURFACE_LABELS)
              if label != "Water" or np.any(terrain.heights <= terrain.sea_level)]
    if config.render_style == "stylized" and config.snow_line is not None and terrain.heights.max() > config.snow_line:
        legend.append(Patch(facecolor=SNOW_COLOR / 255, label="Snow"))
    ax.legend(handles=legend,
              loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=4, frameon=False,
              title="Surface colors · blended by elevation and slope")
    figure.savefig(path, dpi=160)
    figure.clear()
