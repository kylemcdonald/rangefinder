from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from rangefinder.analysis.assignment import assign_peaks, composition_tables
from rangefinder.analysis.isotope_audit import audit_isotope_families, reconcile_isotope_anomalies
from rangefinder.analysis.models import MethodArtifacts
from rangefinder.analysis.plotting import plot_spectrum, plot_zoom_regions
from rangefinder.analysis.common import dataframe_to_csv_ready
from rangefinder.analysis.quality import mass_scale_diagnostics, spatial_composition_stability


def write_method_outputs(
    *,
    method_name: str,
    method_label: str,
    output_dir: Path,
    peaks: pd.DataFrame,
    assignments: pd.DataFrame,
    elemental_composition: pd.DataFrame,
    species_composition: pd.DataFrame,
    isotope_composition: pd.DataFrame,
    isotope_audit: pd.DataFrame,
    isotope_anomalies: pd.DataFrame,
    ambiguous_peaks: pd.DataFrame,
    diagnostics: dict[str, object],
) -> MethodArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataframe_to_csv_ready(peaks).to_csv(output_dir / "peaks.csv", index=False)
    dataframe_to_csv_ready(assignments).to_csv(output_dir / "assignments.csv", index=False)
    dataframe_to_csv_ready(elemental_composition).to_csv(
        output_dir / "elemental_composition.csv", index=False
    )
    dataframe_to_csv_ready(species_composition).to_csv(
        output_dir / "species_composition.csv", index=False
    )
    dataframe_to_csv_ready(isotope_composition).to_csv(
        output_dir / "isotope_composition.csv", index=False
    )
    dataframe_to_csv_ready(isotope_audit).to_csv(
        output_dir / "isotope_audit.csv", index=False
    )
    dataframe_to_csv_ready(isotope_anomalies).to_csv(
        output_dir / "isotope_anomalies.csv", index=False
    )
    dataframe_to_csv_ready(ambiguous_peaks).to_csv(output_dir / "ambiguous_peaks.csv", index=False)
    (output_dir / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, default=str),
        encoding="utf-8",
    )
    return MethodArtifacts(
        method_name=method_name,
        method_label=method_label,
        output_dir=output_dir,
        peaks=peaks,
        assignments=assignments,
        elemental_composition=elemental_composition,
        species_composition=species_composition,
        isotope_composition=isotope_composition,
        isotope_audit=isotope_audit,
        isotope_anomalies=isotope_anomalies,
        ambiguous_peaks=ambiguous_peaks,
        diagnostics=diagnostics,
    )


def load_method_outputs(*, method_name: str, method_label: str, output_dir: Path) -> MethodArtifacts:
    def read_csv_or_empty(path: Path) -> pd.DataFrame:
        try:
            return pd.read_csv(path)
        except EmptyDataError:
            return pd.DataFrame()

    diagnostics_path = output_dir / "diagnostics.json"
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8")) if diagnostics_path.exists() else {}
    return MethodArtifacts(
        method_name=method_name,
        method_label=method_label,
        output_dir=output_dir,
        peaks=read_csv_or_empty(output_dir / "peaks.csv"),
        assignments=read_csv_or_empty(output_dir / "assignments.csv"),
        elemental_composition=read_csv_or_empty(output_dir / "elemental_composition.csv"),
        species_composition=read_csv_or_empty(output_dir / "species_composition.csv"),
        isotope_composition=read_csv_or_empty(output_dir / "isotope_composition.csv"),
        isotope_audit=read_csv_or_empty(output_dir / "isotope_audit.csv"),
        isotope_anomalies=read_csv_or_empty(output_dir / "isotope_anomalies.csv"),
        ambiguous_peaks=read_csv_or_empty(output_dir / "ambiguous_peaks.csv"),
        diagnostics=diagnostics,
    )


def finalize_method_analysis(
    *,
    method_name: str,
    method_label: str,
    output_dir: Path,
    centers,
    counts,
    peaks: pd.DataFrame,
    raw_mz,
    sample_name: str,
    config: dict,
    diagnostics: dict[str, object],
    positions: tuple | None = None,
) -> MethodArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)
    assignments, peak_summary = assign_peaks(peaks, config=config)
    species, elements, isotopes, anomalies, ambiguous = composition_tables(
        peak_summary,
        assignments,
        config=config,
    )
    overlap_diagnostics = species.attrs.get("overlap_deconvolution", {})
    isotope_audit = audit_isotope_families(
        mz_values=raw_mz,
        peaks=peak_summary,
        assignments=assignments,
        isotope_composition=isotopes,
        config=config,
    )
    if not anomalies.empty:
        anomalies = reconcile_isotope_anomalies(anomalies, isotope_audit, config=config)
    diagnostics = diagnostics | {
        "mass_scale": mass_scale_diagnostics(assignments),
        "overlap_deconvolution": overlap_diagnostics,
    }
    if positions is not None:
        x_nm, y_nm, z_nm = positions
        spatial_frame, spatial_summary = spatial_composition_stability(
            x_nm=x_nm,
            y_nm=y_nm,
            z_nm=z_nm,
            m_over_z_da=raw_mz,
            peaks=peak_summary,
            assignments=assignments,
            config=config,
        )
        if not spatial_frame.empty:
            spatial_frame.to_csv(output_dir / "spatial_composition.csv", index=False)
        diagnostics = diagnostics | {"spatial_stability": spatial_summary}
    peak_summary.to_csv(output_dir / "peaks_summary.csv", index=False)
    plot_spectrum(
        centers,
        counts,
        peak_summary,
        output_dir / "full_spectrum.png",
        title=f"{sample_name} | {method_label}",
        max_mz=float(config["analysis"]["max_mz"]),
    )
    plot_spectrum(
        centers,
        counts,
        peak_summary,
        output_dir / "spectrum_zoom_0_120.png",
        title=f"{sample_name} | {method_label} | 0-120 Da",
        max_mz=float(config["analysis"]["zoom_max_mz"]),
    )
    zoom_paths = plot_zoom_regions(
        centers,
        counts,
        peak_summary,
        assignments,
        output_dir / "zoom_regions",
        window_da=1.0,
        max_regions=int(config["analysis"]["max_zoom_regions_per_method"]),
    )
    diagnostics = diagnostics | {
        "zoom_region_count": len(zoom_paths),
        "peak_count": int(peak_summary.shape[0]),
    }
    return write_method_outputs(
        method_name=method_name,
        method_label=method_label,
        output_dir=output_dir,
        peaks=peak_summary,
        assignments=assignments,
        elemental_composition=elements,
        species_composition=species,
        isotope_composition=isotopes,
        isotope_audit=isotope_audit,
        isotope_anomalies=anomalies,
        ambiguous_peaks=ambiguous,
        diagnostics=diagnostics,
    )
