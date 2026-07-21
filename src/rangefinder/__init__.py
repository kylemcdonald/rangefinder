"""Rangefinder: automated ranging, species identification, and composition for
time-of-flight mass spectra.

Goes from a raw event list (``.pos``/``.epos`` or any array of m/z values) to
ranged peaks, identified species, and isotope-resolved composition, with no
user-supplied ranges, element list, or interactive tuning.
"""
from __future__ import annotations

from importlib import resources
from pathlib import Path

from rangefinder.analysis.custom_pipeline import run_custom_pipeline
from rangefinder.analysis.naive_pipeline import run_naive_pipeline
from rangefinder.utils.config import load_config

__all__ = [
    "run_custom_pipeline",
    "run_naive_pipeline",
    "default_config_path",
    "load_default_config",
]

__version__ = "0.1.0"


def default_config_path() -> Path:
    """Path to the bundled default pipeline configuration."""
    with resources.as_file(resources.files("rangefinder").joinpath("config/defaults.yaml")) as path:
        return Path(path)


def load_default_config() -> dict:
    """Load the bundled default pipeline configuration."""
    return load_config(default_config_path())
