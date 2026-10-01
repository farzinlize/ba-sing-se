"""Generate paired topographic maps and first-person terrain images."""

from .config import PipelineConfig
from .pipeline import generate_sample

__all__ = ["PipelineConfig", "generate_sample"]
