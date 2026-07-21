#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


from rangefinder.analysis.apyt_support import (
    apyt_species_tables_from_counts,
    build_apyt_species_dictionary,
    select_apyt_species_candidates,
)
from rangefinder.analysis.assignment import assign_peaks, composition_tables
from rangefinder.analysis.isotope_audit import audit_isotope_families, reconcile_isotope_anomalies
from rangefinder.analysis.quality import mass_scale_diagnostics
from rangefinder.analysis.common import (
    build_histogram,
    characterize_centroid_peaks,
    default_prominence,
    detect_peaks_from_histogram,
    detection_bin_width,
    integrate_peak_areas,
    neutral_formula_from_element_counter,
)
from rangefinder.analysis.pipeline_utils import write_method_outputs
from rangefinder.analysis.plotting import plot_spectrum, plot_zoom_regions
from rangefinder.utils.config import load_config


def load_pos_mass(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=">f4")
    return raw.reshape(-1, 4)[:, 3].astype(np.float32)


def finalize_external_method(
    *,
    method_name: str,
    method_label: str,
    output_dir: Path,
    sample_name: str,
    config: dict,
    centers: np.ndarray,
    counts: np.ndarray,
    peaks: pd.DataFrame,
    raw_mz: np.ndarray,
    species_override: pd.DataFrame | None = None,
    elements_override: pd.DataFrame | None = None,
    diagnostics: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    assignments, peak_summary = assign_peaks(peaks, config=config)
    species, elements, isotopes, anomalies, ambiguous = composition_tables(
        peak_summary, assignments, config=config
    )
    isotope_audit = audit_isotope_families(
        mz_values=raw_mz,
        peaks=peak_summary,
        assignments=assignments,
        isotope_composition=isotopes,
        config=config,
    )
    if not anomalies.empty:
        anomalies = reconcile_isotope_anomalies(anomalies, isotope_audit, config=config)
    diagnostics = diagnostics | {"mass_scale": mass_scale_diagnostics(assignments)}
    if species_override is not None and not species_override.empty:
        species = species_override
    if elements_override is not None and not elements_override.empty:
        elements = elements_override
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
    write_method_outputs(
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


def build_apav_ranges(peak_summary: pd.DataFrame, assignments: pd.DataFrame):
    import apav
    from apav.core.range import Range as ApavRange
    from apav.core.range import RangeCollection

    top = assignments[assignments["rank"] == 1].copy()
    merged = peak_summary.merge(
        top[["peak_id", "species_label", "charge", "element_counts"]],
        on="peak_id",
        how="left",
    )
    merged = merged[merged["species_label"] != "unassigned"].sort_values("peak_mz_da").reset_index(drop=True)
    ranges = RangeCollection()
    eps = 1e-4
    for i, row in merged.iterrows():
        element_counts = row["element_counts"]
        if isinstance(element_counts, str):
            element_counts = ast.literal_eval(element_counts)
        formula = neutral_formula_from_element_counter(Counter(element_counts))
        charge = int(row["charge"])
        left = float(row["integration_left_da"])
        right = float(row["integration_right_da"])
        if i > 0:
            prev_mz = float(merged.iloc[i - 1]["peak_mz_da"])
            left = max(left, (prev_mz + float(row["peak_mz_da"])) / 2.0 + eps)
        if i < len(merged) - 1:
            next_mz = float(merged.iloc[i + 1]["peak_mz_da"])
            right = min(right, (next_mz + float(row["peak_mz_da"])) / 2.0 - eps)
        if right <= left:
            continue
        ranges.add(ApavRange(apav.Ion(formula, charge), (left, right)))
    return ranges


def apav_quant_tables(
    roi,
    peak_summary: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    bin_width: float,
    upper: float,
    fallback_species: pd.DataFrame,
    fallback_elements: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from apav.analysis.massspectrum import NoiseCorrectedMassSpectrum

    ranges = build_apav_ranges(peak_summary, assignments)
    if len(ranges) == 0:
        return fallback_species, fallback_elements
    species_quant = NoiseCorrectedMassSpectrum(
        roi, ranges, percent=True, decompose=False, bin_width=bin_width, upper=upper
    )
    element_quant = NoiseCorrectedMassSpectrum(
        roi, ranges, percent=True, decompose=True, bin_width=bin_width, upper=upper
    )
    species_rows = [
        {"species_label": key, "weighted_counts": value[1], "atomic_percent_weighted": value[0]}
        for key, value in species_quant.quant_dict.items()
    ]
    element_rows = [
        {"element": key, "weighted_counts": value[1], "atomic_percent_weighted": value[0]}
        for key, value in element_quant.quant_dict.items()
    ]
    species = pd.DataFrame(species_rows)
    elements = pd.DataFrame(element_rows)
    if not fallback_species.empty:
        robust = fallback_species.groupby("species_label", as_index=False)[
            ["robust_counts", "atomic_percent_robust"]
        ].sum(numeric_only=True)
        species = species.merge(
            robust,
            on="species_label",
            how="left",
        ).fillna({"robust_counts": 0.0, "atomic_percent_robust": 0.0})
        if "category" not in species.columns:
            species["category"] = "apav_range"
        if "charge" not in species.columns:
            species["charge"] = 0
    if not fallback_elements.empty:
        robust = fallback_elements[["element", "robust_counts", "atomic_percent_robust"]]
        elements = elements.merge(
            robust,
            on="element",
            how="left",
        ).fillna({"robust_counts": 0.0, "atomic_percent_robust": 0.0})
    return species, elements


def run_apav(sample_path: Path, output_dir: Path, config: dict) -> None:
    import ctypes
    import sys

    libomp_path = Path("/opt/homebrew/opt/libomp/lib/libomp.dylib")
    if libomp_path.exists():
        ctypes.CDLL(str(libomp_path), mode=ctypes.RTLD_GLOBAL)
    import apav
    import apav.core.roi
    import apav.utils.helpers

    analysis_cfg = config["analysis"]
    sys_byteorder = "<" if sys.byteorder == "little" else ">"

    def _array2native_byteorder_compat(array: np.ndarray) -> np.ndarray:
        if array.dtype.byteorder in ("=", "|"):
            return array
        native_dtype = array.dtype.newbyteorder(sys_byteorder)
        return array.byteswap().view(native_dtype)

    apav.utils.helpers.array2native_byteorder = _array2native_byteorder_compat
    apav.core.roi.array2native_byteorder = _array2native_byteorder_compat
    roi = apav.load_pos(str(sample_path))
    mass = np.asarray(roi.mass, dtype=np.float32)
    bin_width = detection_bin_width(
        mass,
        list(analysis_cfg["bin_width_candidates_da"]),
        min_mz=float(analysis_cfg["min_mz"]),
        max_mz=float(analysis_cfg["zoom_max_mz"]),
        max_detection_bin_width_da=float(analysis_cfg["max_detection_bin_width_da"]),
    )
    centers, counts = roi.mass_histogram(
        bin_width=bin_width,
        lower=float(analysis_cfg["min_mz"]),
        upper=float(analysis_cfg["max_mz"]),
        norm=False,
    )
    prominence = default_prominence(
        counts,
        quantile=float(analysis_cfg["pyccapt_prominence_quantile"]),
        minimum=float(analysis_cfg["pyccapt_min_prominence"]),
    )
    peaks = detect_peaks_from_histogram(
        centers,
        counts,
        prominence=prominence,
        distance_bins=max(1, int(round(0.15 / bin_width))),
        rel_height=0.5,
        smooth_window_bins=0,
        smooth_polyorder=2,
        method_label="apav",
    )
    peaks = integrate_peak_areas(
        peaks,
        centers,
        counts,
        integration_half_width_da=float(analysis_cfg["peak_integration_half_width_da"]),
        baseline_half_width_da=float(analysis_cfg["local_baseline_half_width_da"]),
    )
    assignments, peak_summary = assign_peaks(peaks, config=config)
    species_fallback, elements_fallback, isotopes, anomalies, ambiguous = composition_tables(
        peak_summary, assignments, config=config
    )
    range_species, range_elements = apav_quant_tables(
        roi,
        peak_summary,
        assignments,
        bin_width=bin_width,
        upper=float(analysis_cfg["max_mz"]),
        fallback_species=species_fallback,
        fallback_elements=elements_fallback,
    )
    isotope_audit = audit_isotope_families(
        mz_values=mass,
        peaks=peak_summary,
        assignments=assignments,
        isotope_composition=isotopes,
        config=config,
    )
    if not anomalies.empty:
        anomalies = reconcile_isotope_anomalies(anomalies, isotope_audit, config=config)
    peak_summary.to_csv(output_dir / "peaks_summary.csv", index=False)
    plot_spectrum(
        centers,
        counts,
        peak_summary,
        output_dir / "full_spectrum.png",
        title=f"{sample_path.stem} | APAV",
        max_mz=float(analysis_cfg["max_mz"]),
    )
    plot_spectrum(
        centers,
        counts,
        peak_summary,
        output_dir / "spectrum_zoom_0_120.png",
        title=f"{sample_path.stem} | APAV | 0-120 Da",
        max_mz=float(analysis_cfg["zoom_max_mz"]),
    )
    zoom_paths = plot_zoom_regions(
        centers,
        counts,
        peak_summary,
        assignments,
        output_dir / "zoom_regions",
        window_da=1.0,
        max_regions=int(analysis_cfg["max_zoom_regions_per_method"]),
    )
    write_method_outputs(
        method_name="apav",
        method_label="APAV",
        output_dir=output_dir,
        peaks=peak_summary,
        assignments=assignments,
        elemental_composition=elements_fallback,
        species_composition=species_fallback,
        isotope_composition=isotopes,
        isotope_audit=isotope_audit,
        isotope_anomalies=anomalies,
        ambiguous_peaks=ambiguous,
        diagnostics={
            "bin_width_da": bin_width,
            "prominence": prominence,
            "mass_scale": mass_scale_diagnostics(assignments),
            "library": "apav",
            "apav_quantification": "NoiseCorrectedMassSpectrum",
            "range_count": len(build_apav_ranges(peak_summary, assignments)),
            "zoom_region_count": len(zoom_paths),
            "peak_count": int(peak_summary.shape[0]),
        },
    )


def run_ms_deisotope(sample_path: Path, output_dir: Path, config: dict) -> None:
    import ms_deisotope
    from ms_peak_picker import pick_peaks

    analysis_cfg = config["analysis"]
    mass = load_pos_mass(sample_path)
    bin_width = detection_bin_width(
        mass,
        list(analysis_cfg["bin_width_candidates_da"]),
        min_mz=float(analysis_cfg["min_mz"]),
        max_mz=float(analysis_cfg["zoom_max_mz"]),
        max_detection_bin_width_da=float(analysis_cfg["max_detection_bin_width_da"]),
    )
    centers, counts, _ = build_histogram(
        mass,
        min_mz=float(analysis_cfg["min_mz"]),
        max_mz=float(analysis_cfg["max_mz"]),
        bin_width_da=bin_width,
    )
    prominence = default_prominence(
        counts,
        quantile=float(analysis_cfg["pyccapt_prominence_quantile"]),
        minimum=float(analysis_cfg["pyccapt_min_prominence"]),
    )
    picked = pick_peaks(
        centers.astype(float),
        counts.astype(float),
        fit_type="quadratic",
        signal_to_noise_threshold=0.0,
    )
    result = ms_deisotope.deconvolute_peaks(
        picked,
        charge_range=(1, int(analysis_cfg.get("ms_deisotope_max_charge", analysis_cfg["max_charge"]))),
        error_tolerance=float(analysis_cfg["ppm_tolerance"]) * 1e-6,
        use_quick_charge=True,
    )
    mz = []
    intensity = []
    charge = []
    neutral_mass = []
    score = []
    for peak in result.peak_set:
        if peak.intensity < prominence:
            continue
        mz.append(float(peak.mz))
        intensity.append(float(peak.intensity))
        charge.append(int(abs(peak.charge)))
        neutral_mass.append(float(peak.neutral_mass))
        score.append(float(getattr(peak, "score", peak.intensity)))
    peaks = characterize_centroid_peaks(
        np.asarray(mz, dtype=float),
        np.asarray(intensity, dtype=float),
        centers,
        counts,
        method_label="ms_deisotope",
        integration_half_width_da=float(analysis_cfg["peak_integration_half_width_da"]),
        baseline_half_width_da=float(analysis_cfg["local_baseline_half_width_da"]),
    )
    if not peaks.empty:
        peaks["charge_hint"] = charge
        peaks["neutral_mass_hint"] = neutral_mass
        peaks["deisotope_score"] = score
    finalize_external_method(
        method_name="ms_deisotope",
        method_label="ms_deisotope",
        output_dir=output_dir,
        sample_name=sample_path.stem,
        config=config,
        centers=centers,
        counts=counts,
        peaks=peaks,
        raw_mz=mass,
        diagnostics={
            "bin_width_da": bin_width,
            "prominence_filter": prominence,
            "library": "ms_deisotope",
            "deconvoluted_peak_count": len(mz),
            "charge_range": [1, int(analysis_cfg.get("ms_deisotope_max_charge", analysis_cfg["max_charge"]))],
        },
    )


def run_apyt(sample_path: Path, output_dir: Path, config: dict) -> None:
    import apyt.spectrum.fit as apyt_fit

    analysis_cfg = config["analysis"]
    mass = load_pos_mass(sample_path)
    bin_width = detection_bin_width(
        mass,
        list(analysis_cfg["bin_width_candidates_da"]),
        min_mz=float(analysis_cfg["min_mz"]),
        max_mz=float(analysis_cfg["zoom_max_mz"]),
        max_detection_bin_width_da=float(analysis_cfg["max_detection_bin_width_da"]),
    )
    centers, counts, _ = build_histogram(
        mass,
        min_mz=float(analysis_cfg["min_mz"]),
        max_mz=float(analysis_cfg["max_mz"]),
        bin_width_da=bin_width,
    )
    prominence = default_prominence(
        counts,
        quantile=float(analysis_cfg["pyccapt_prominence_quantile"]),
        minimum=float(analysis_cfg["pyccapt_min_prominence"]),
    )
    peaks = detect_peaks_from_histogram(
        centers,
        counts,
        prominence=prominence,
        distance_bins=max(1, int(round(float(analysis_cfg["pyccapt_distance_bins"])))),
        rel_height=0.5,
        smooth_window_bins=0,
        smooth_polyorder=2,
        method_label="apyt",
    )
    peaks = integrate_peak_areas(
        peaks,
        centers,
        counts,
        integration_half_width_da=float(analysis_cfg["peak_integration_half_width_da"]),
        baseline_half_width_da=float(analysis_cfg["local_baseline_half_width_da"]),
    )
    selection = select_apyt_species_candidates(peaks, config=config)
    species_dict = build_apyt_species_dictionary(
        selection,
        atomic_volume_nm3=float(analysis_cfg.get("apyt_atomic_volume_nm3", 0.015)),
    )
    if not species_dict:
        raise RuntimeError("APyT species selection returned no candidate formulas.")
    fit_min_mz = float(analysis_cfg.get("apyt_min_fit_mz", min(bin_width, 0.01)))
    spectrum = np.column_stack((centers, counts))
    spectrum = spectrum[spectrum[:, 0] >= fit_min_mz]
    if spectrum.size == 0:
        raise RuntimeError("APyT fit spectrum is empty after enforcing a positive m/z range.")
    mass_decimals = max(0, int(np.ceil(-np.log10(max(bin_width, 1.0e-6)))))
    peaks_list, _ = apyt_fit.peaks_list(species_dict, mass_decimals=mass_decimals, verbose=False)
    result = apyt_fit.fit(
        spectrum,
        peaks_list,
        str(analysis_cfg.get("apyt_function", "error-expDecay")),
        scale_width=bool(analysis_cfg.get("apyt_scale_width", False)),
        verbose=False,
    )
    counts_list, total_counts, background = apyt_fit.counts(
        peaks_list,
        str(analysis_cfg.get("apyt_function", "error-expDecay")),
        result.best_values,
        (float(spectrum[0, 0]), float(spectrum[-1, 0])),
        float(bin_width),
        ignore_list=[],
        verbose=False,
    )
    species_override, elements_override = apyt_species_tables_from_counts(counts_list)
    finalize_external_method(
        method_name="apyt",
        method_label="APyT",
        output_dir=output_dir,
        sample_name=sample_path.stem,
        config=config,
        centers=centers,
        counts=counts,
        peaks=peaks,
        raw_mz=mass,
        species_override=species_override,
        elements_override=elements_override,
        diagnostics={
            "bin_width_da": bin_width,
            "prominence_filter": prominence,
            "library": "apyt",
            "fit_function": str(analysis_cfg.get("apyt_function", "error-expDecay")),
            "scale_width": bool(analysis_cfg.get("apyt_scale_width", False)),
            "fit_success": bool(getattr(result, "success", False)),
            "chi_square": float(getattr(result, "chisqr", np.nan)),
            "reduced_chi_square": float(getattr(result, "redchi", np.nan)),
            "nfev": int(getattr(result, "nfev", 0)),
            "selected_formula_count": int(len(species_dict)),
            "selected_atomic_formula_count": int(
                selection.loc[selection["category"] == "atomic", "formula"].nunique()
            ),
            "selected_molecular_formula_count": int(
                selection.loc[selection["category"] == "molecular", "formula"].nunique()
            ),
            "selected_formulas": [
                {
                    "formula": str(row["formula"]),
                    "charge": int(row["charge"]),
                    "category": str(row["category"]),
                    "support_score": float(row["support_score"]),
                    "natural_probability": float(row["natural_probability"]),
                }
                for row in selection.to_dict(orient="records")
            ],
            "apyt_total_counts": float(total_counts),
            "apyt_background_counts": float(background),
            "peak_model_count": int(len(peaks_list)),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["apav", "ms_deisotope", "apyt"])
    parser.add_argument("--sample", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.method == "apav":
        run_apav(args.sample, args.output_dir, config)
    elif args.method == "ms_deisotope":
        run_ms_deisotope(args.sample, args.output_dir, config)
    else:
        run_apyt(args.sample, args.output_dir, config)


if __name__ == "__main__":
    main()
