"""Detection-stage ablation harness.

Every detector below produces a list of peak-center bin indices on one shared
integration histogram; the harness then integrates areas and runs the shared
assignment stage identically for all of them. Composition and detection scores
therefore reflect *only* the peak-detection algorithm, holding integration and
species assignment constant. This is how we isolate "which detector" from
"which assigner" — the detector-only comparators in the main benchmark
(pyOpenMS, ms_deisotope, PyCCAPT, naive) make the same shared-assigner
assumption, and these baselines round out the algorithm space with the standard
1D mass-spectrum detectors (CWT ridge lines, persistent homology, derivative
zero-crossings) that APT ranging never adopted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema, find_peaks, find_peaks_cwt, peak_widths

from rangefinder.analysis.common import (
    build_histogram,
    default_prominence,
    detection_bin_width,
    integrate_peak_areas,
    smooth_counts,
)
from rangefinder.analysis.custom_pipeline import detect_custom_peaks
from rangefinder.analysis.pipeline_utils import finalize_method_analysis


def build_detection_histogram(mz: np.ndarray, config: dict) -> tuple[np.ndarray, np.ndarray, float]:
    """Reproduce the integration histogram Rangefinder builds, so every
    detector sees the same binned spectrum."""
    analysis_cfg = config["analysis"]
    base_bin_width = detection_bin_width(
        mz,
        list(analysis_cfg["bin_width_candidates_da"]),
        min_mz=float(analysis_cfg["min_mz"]),
        max_mz=float(analysis_cfg["zoom_max_mz"]),
        max_detection_bin_width_da=float(analysis_cfg["max_detection_bin_width_da"]),
    )
    detection_bin_widths = sorted(
        {
            float(candidate)
            for candidate in analysis_cfg["bin_width_candidates_da"]
            if float(candidate) <= float(analysis_cfg["max_detection_bin_width_da"])
        }
        | {float(base_bin_width)}
    )
    integration_bin_width = float(min(detection_bin_widths))
    centers, counts, _edges = build_histogram(
        mz,
        min_mz=float(analysis_cfg["min_mz"]),
        max_mz=float(analysis_cfg["max_mz"]),
        bin_width_da=integration_bin_width,
    )
    return centers, counts, integration_bin_width


# --- individual detectors: (centers, counts, config) -> peak bin indices ---


def _prominence_floor(counts: np.ndarray, config: dict) -> float:
    analysis_cfg = config["analysis"]
    return default_prominence(
        counts,
        quantile=float(analysis_cfg["naive_prominence_quantile"]),
        minimum=float(analysis_cfg["naive_min_prominence"]),
    )


def detect_naive(centers, counts, config) -> np.ndarray:
    """scipy.signal.find_peaks on the raw histogram, prominence-filtered."""
    prominence = _prominence_floor(counts, config)
    idx, _ = find_peaks(counts, prominence=prominence, distance=int(config["analysis"]["naive_distance_bins"]), height=0)
    return idx


def detect_smoothed(centers, counts, config) -> np.ndarray:
    """find_peaks on Savitzky-Golay-smoothed counts (classic denoise-then-peak)."""
    prominence = _prominence_floor(counts, config)
    window = int(config["analysis"]["custom_savgol_window_bins"])
    poly = int(config["analysis"]["custom_savgol_polyorder"])
    working = smooth_counts(counts, window, poly) if window > 2 else counts
    idx, _ = find_peaks(working, prominence=prominence, distance=int(config["analysis"]["naive_distance_bins"]), height=0)
    return idx


def detect_cwt(centers, counts, config) -> np.ndarray:
    """Continuous-wavelet-transform ridge-line detection (Ricker/Mexican-hat),
    the MassSpecWavelet family. Scale-robust; carries no baseline model, so we
    apply the shared prominence floor as a post-filter."""
    widths = np.arange(1.0, 16.0)
    idx = find_peaks_cwt(counts.astype(float), widths, min_snr=1.0, noise_perc=10.0)
    idx = np.asarray(idx, dtype=int)
    if idx.size == 0:
        return idx
    floor = _prominence_floor(counts, config)
    # CWT reports ridge maxima on the smoothed transform; snap each to the local
    # count maximum and keep those clearing the shared prominence floor.
    snapped = []
    for i in idx:
        lo, hi = max(0, i - 3), min(counts.size, i + 4)
        j = lo + int(np.argmax(counts[lo:hi]))
        snapped.append(j)
    snapped = np.unique(np.asarray(snapped, dtype=int))
    keep = [j for j in snapped if counts[j] >= floor]
    return np.asarray(keep, dtype=int)


def detect_persistent_homology(centers, counts, config) -> np.ndarray:
    """0-th persistent-homology peak detection (Huber 2021): sweep from high to
    low counts with union-find; a component's persistence is its birth height
    minus the level at which it merges into a taller neighbour. Persistence is
    a topological analogue of prominence and is stable to noise."""
    seq = counts.astype(float)
    n = seq.size
    if n == 0:
        return np.asarray([], dtype=int)
    idx_to_peak = np.full(n, -1, dtype=int)
    born: list[int] = []
    died: list[float | None] = []
    for idx in sorted(range(n), key=lambda i: seq[i], reverse=True):
        left_done = idx > 0 and idx_to_peak[idx - 1] != -1
        right_done = idx < n - 1 and idx_to_peak[idx + 1] != -1
        il = idx_to_peak[idx - 1] if left_done else -1
        ir = idx_to_peak[idx + 1] if right_done else -1
        if not left_done and not right_done:
            born.append(idx)
            died.append(None)
            idx_to_peak[idx] = len(born) - 1
        elif left_done and not right_done:
            idx_to_peak[idx] = il
        elif right_done and not left_done:
            idx_to_peak[idx] = ir
        else:
            # merge: the taller (earlier-born) component survives.
            if seq[born[il]] >= seq[born[ir]]:
                survivor, dier = il, ir
            else:
                survivor, dier = ir, il
            if died[dier] is None:
                died[dier] = seq[idx]
            idx_to_peak[idx] = survivor
    floor = _prominence_floor(counts, config)
    global_min = float(seq.min())
    keep = []
    for peak_id, birth_idx in enumerate(born):
        death_level = died[peak_id] if died[peak_id] is not None else global_min
        persistence = seq[birth_idx] - float(death_level)
        if persistence >= floor:
            keep.append(birth_idx)
    return np.asarray(sorted(keep), dtype=int)


def detect_derivative(centers, counts, config) -> np.ndarray:
    """Classic analytical-chemistry detector: Savitzky-Golay smooth, then flag
    local maxima (first-derivative sign change), prominence-filtered."""
    window = int(config["analysis"]["custom_savgol_window_bins"])
    poly = int(config["analysis"]["custom_savgol_polyorder"])
    working = smooth_counts(counts, window, poly) if window > 2 else counts
    maxima = argrelextrema(working, np.greater, order=int(config["analysis"]["naive_distance_bins"]))[0]
    floor = _prominence_floor(counts, config)
    keep = [i for i in maxima if counts[i] >= floor]
    return np.asarray(keep, dtype=int)


DETECTORS = {
    "naive": detect_naive,
    "smoothed": detect_smoothed,
    "cwt": detect_cwt,
    "persistent_homology": detect_persistent_homology,
    "derivative": detect_derivative,
}

DETECTOR_LABELS = {
    "rangefinder": "Rangefinder (multi-resolution)",
    "naive": "Naive prominence",
    "smoothed": "Savitzky--Golay prominence",
    "cwt": "CWT ridge lines",
    "persistent_homology": "Persistent homology",
    "derivative": "Derivative zero-crossing",
}


def _peaks_from_indices(centers, counts, indices, method_label) -> pd.DataFrame:
    indices = np.asarray(sorted(set(int(i) for i in indices)), dtype=int)
    if indices.size == 0:
        return pd.DataFrame()
    widths = peak_widths(counts, indices, rel_height=0.5)
    grid = np.arange(centers.size)
    rows = []
    for k, i in enumerate(indices):
        left = float(np.interp(widths[2][k], grid, centers))
        right = float(np.interp(widths[3][k], grid, centers))
        rows.append(
            {
                "peak_id": f"{method_label}-peak-{k + 1:04d}",
                "method": method_label,
                "peak_index": int(i),
                "peak_mz_da": float(centers[i]),
                "peak_height": float(counts[i]),
                "smoothed_height": float(counts[i]),
                "prominence": float(widths[1][k]),
                "left_fwhm_mz_da": left,
                "right_fwhm_mz_da": right,
                "fwhm_da": max(right - left, float(centers[i]) / 800.0),
            }
        )
    return pd.DataFrame(rows).sort_values("peak_mz_da", ignore_index=True)


def run_detection_ablation(sample_data, output_dir, config, detector_name: str):
    """Run one detector through the shared integration + assignment stages."""
    analysis_cfg = config["analysis"]
    mz = sample_data.m_over_z_da
    if detector_name == "rangefinder":
        centers, counts, peaks, _diag = detect_custom_peaks(sample_data, config)
        # Re-integrate the detected centres with the shared integrator so the
        # only difference from the other detectors is the peak list itself.
        peaks = integrate_peak_areas(
            peaks[["peak_id", "method", "peak_index", "peak_mz_da", "peak_height", "smoothed_height", "fwhm_da"]].copy()
            if not peaks.empty else peaks,
            centers,
            counts,
            integration_half_width_da=float(analysis_cfg["peak_integration_half_width_da"]),
            baseline_half_width_da=float(analysis_cfg["local_baseline_half_width_da"]),
            max_integration_half_width_da=float(analysis_cfg["custom_max_integration_half_width_da"]),
        )
    else:
        centers, counts, _bw = build_detection_histogram(mz, config)
        indices = DETECTORS[detector_name](centers, counts, config)
        peaks = _peaks_from_indices(centers, counts, indices, detector_name)
        if not peaks.empty:
            peaks = integrate_peak_areas(
                peaks,
                centers,
                counts,
                integration_half_width_da=float(analysis_cfg["peak_integration_half_width_da"]),
                baseline_half_width_da=float(analysis_cfg["local_baseline_half_width_da"]),
                max_integration_half_width_da=float(analysis_cfg["custom_max_integration_half_width_da"]),
            )
    return finalize_method_analysis(
        method_name=f"det_{detector_name}",
        method_label=DETECTOR_LABELS.get(detector_name, detector_name),
        output_dir=output_dir,
        centers=centers,
        counts=counts,
        peaks=peaks,
        raw_mz=mz,
        sample_name=sample_data.metadata.sample_name,
        config=config,
        diagnostics={"detector": detector_name, "peak_detection": f"detection ablation: {detector_name}"},
    )
