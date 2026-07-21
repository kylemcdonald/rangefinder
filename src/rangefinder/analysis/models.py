from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class MethodArtifacts:
    method_name: str
    method_label: str
    output_dir: Path
    peaks: pd.DataFrame
    assignments: pd.DataFrame
    elemental_composition: pd.DataFrame
    species_composition: pd.DataFrame
    isotope_composition: pd.DataFrame
    isotope_audit: pd.DataFrame
    isotope_anomalies: pd.DataFrame
    ambiguous_peaks: pd.DataFrame
    diagnostics: dict[str, object] = field(default_factory=dict)


@dataclass
class SampleArtifacts:
    sample_name: str
    sample_slug: str
    output_dir: Path
    method_results: list[MethodArtifacts]
    comparison: pd.DataFrame | None
    summary: dict[str, object]
