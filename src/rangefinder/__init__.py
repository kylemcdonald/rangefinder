"""Rangefinder: automated ranging, species identification, and composition for
atom probe tomography mass spectra.

The end-to-end pipeline is specific to positive-ion APT event data and its
field-evaporation chemistry. Some low-level detection utilities can be adapted to
other one-dimensional TOF mass spectra, but non-APT assignment and composition are
outside the package's validated scope.
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

__version__ = "0.3.0"


def default_config_path() -> Path:
    """Path to the bundled default pipeline configuration."""
    with resources.as_file(resources.files("rangefinder").joinpath("config/defaults.yaml")) as path:
        return Path(path)


def load_default_config() -> dict:
    """Load the bundled default pipeline configuration."""
    return load_config(default_config_path())
