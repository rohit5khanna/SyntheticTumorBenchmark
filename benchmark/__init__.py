"""Synthetic tumor benchmark package."""

from .config import load_config
from .generator import generate_benchmark_dataset

__all__ = ["load_config", "generate_benchmark_dataset"]

