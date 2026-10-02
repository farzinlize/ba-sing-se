"""Contour-only maps that reveal camera location but never its orientation."""

from pathlib import Path
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .config import PipelineConfig
from .terrain import Terrain
from .topographic import CAMERA_MARKER_SIZE, add_compass_guide
from .viewpoint import Viewpoint


def save_monochrome_map(terrain: Terrain, camera: Viewpoint,
                        config: PipelineConfig, path: Path) -> None:
    """Save black contours on white, with a red nondirectional camera position dot."""
    figure = Figure(figsize=(9, 9), layout="constrained", facecolor="white")
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    ax.set_facecolor("white")
    start = np.ceil(terrain.heights.min() / config.contour_interval) * config.contour_interval
    levels = np.arange(start, terrain.heights.max(), config.contour_interval)
    if len(levels):
        contours = ax.contour(terrain.x, terrain.y, terrain.heights, levels=levels,
                              colors="black", linewidths=0.45, alpha=0.7)
        ax.clabel(contours, inline=True, fontsize=7, fmt="%g m", colors="black")
    if terrain.heights.min() < terrain.sea_level < terrain.heights.max():
        ax.contour(terrain.x, terrain.y, terrain.heights, levels=[terrain.sea_level],
                   colors="black", linewidths=1.0)
    x, y, _ = camera.position
    ax.scatter([x], [y], c="#f54242", edgecolors="white",
               linewidths=1.0, s=CAMERA_MARKER_SIZE, zorder=5)
    ax.annotate("Camera", (x, y), xytext=(8, 8), textcoords="offset points",
                fontsize=9, color="black",
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
    ax.set(xlim=(0, config.extent), ylim=(0, config.extent), aspect="equal",
           xlabel="East (m)", ylabel="North (m)",
           title=f"Terrain {config.seed} · North ↑ · contours {config.contour_interval:g} m")
    add_compass_guide(ax)
    ax.tick_params(colors="black")
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")
    ax.title.set_color("black")
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
    figure.savefig(path, dpi=160, facecolor="white", transparent=False)
    figure.clear()
