"""Draw a north-up hillshaded map with readable elevation contours."""

from pathlib import Path
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib import patheffects

from .biomes import SURFACE_LABELS, SNOW_COLOR, hillshade_colors, surface_palette, terrain_colors
from .config import PipelineConfig
from .terrain import Terrain
from .viewpoint import Viewpoint


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
    ax.scatter([x], [y], c="#f54242", edgecolors="white", s=60, zorder=5)
    ax.annotate("", xy=(x + dx * length, y + dy * length), xytext=(x, y),
                arrowprops={"arrowstyle": "-|>", "color": "#ee3030", "lw": 2.5})
    ax.annotate(f"Camera · {camera.direction}", (x, y), xytext=(8, 8),
                textcoords="offset points", fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
    ax.set(xlim=(0, config.extent), ylim=(0, config.extent), aspect="equal",
           xlabel="East (m)", ylabel="North (m)",
           title=f"Terrain {config.seed} · North ↑ · contours {config.contour_interval:g} m")
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
