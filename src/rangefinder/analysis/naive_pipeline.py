from __future__ import annotations

import pandas as pd

from rangefinder.analysis.common import (
    build_histogram,
    default_prominence,
    detect_peaks_from_histogram,
    detection_bin_width,
)
from rangefinder.analysis.pipeline_utils import finalize_method_analysis


def _apply_sideband_subtraction(
    peaks: pd.DataFrame,
    centers,
    counts,
    *,
    bin_width_da: float,
    core_half_width_bins: int,
    sideband_gap_bins: int,
    sideband_width_bins: int,
) -> pd.DataFrame:
    if peaks.empty:
        return peaks.copy()
    augmented = peaks.copy()
    integration_left = []
    integration_right = []
    integrated_area = []
    sideband_level = []
    for peak_idx in augmented["peak_index"].astype(int):
        core_left = max(0, peak_idx - core_half_width_bins)
        core_right = min(counts.size, peak_idx + core_half_width_bins + 1)
        left_start = max(0, core_left - sideband_gap_bins - sideband_width_bins)
        left_end = max(0, core_left - sideband_gap_bins)
        right_start = min(counts.size, core_right + sideband_gap_bins)
        right_end = min(counts.size, core_right + sideband_gap_bins + sideband_width_bins)

        core = counts[core_left:core_right]
        left_side = counts[left_start:left_end]
        right_side = counts[right_start:right_end]
        sidebands = []
        if left_side.size:
            sidebands.append(left_side)
        if right_side.size:
            sidebands.append(right_side)
        baseline = float(sum(side.sum() for side in sidebands) / sum(side.size for side in sidebands)) if sidebands else 0.0
        corrected = max(float(core.sum()) - baseline * float(core.size), 0.0)

        integration_left.append(float(centers[core_left] - bin_width_da / 2.0))
        integration_right.append(float(centers[core_right - 1] + bin_width_da / 2.0))
        integrated_area.append(corrected)
        sideband_level.append(baseline)

    augmented["left_fwhm_mz_da"] = augmented["peak_mz_da"] - bin_width_da / 2.0
    augmented["right_fwhm_mz_da"] = augmented["peak_mz_da"] + bin_width_da / 2.0
    augmented["fwhm_da"] = float(bin_width_da)
    augmented["integration_left_da"] = integration_left
    augmented["integration_right_da"] = integration_right
    augmented["integrated_area"] = integrated_area
    augmented["sideband_level"] = sideband_level
    return augmented


def run_naive_pipeline(sample_data, output_dir, config):
    analysis_cfg = config["analysis"]
    bin_width = detection_bin_width(
        sample_data.m_over_z_da,
        list(analysis_cfg["bin_width_candidates_da"]),
        min_mz=float(analysis_cfg["min_mz"]),
        max_mz=float(analysis_cfg["zoom_max_mz"]),
        max_detection_bin_width_da=float(analysis_cfg["max_detection_bin_width_da"]),
    )
    centers, counts, _edges = build_histogram(
        sample_data.m_over_z_da,
        min_mz=float(analysis_cfg["min_mz"]),
        max_mz=float(analysis_cfg["max_mz"]),
        bin_width_da=bin_width,
    )
    prominence = default_prominence(
        counts,
        quantile=float(analysis_cfg["naive_prominence_quantile"]),
        minimum=float(analysis_cfg["naive_min_prominence"]),
    )
    peaks = detect_peaks_from_histogram(
        centers,
        counts,
        prominence=prominence,
        distance_bins=int(analysis_cfg["naive_distance_bins"]),
        rel_height=0.5,
        smooth_window_bins=0,
        smooth_polyorder=1,
        method_label="naive",
        max_fwhm_da=bin_width,
    )
    peaks = _apply_sideband_subtraction(
        peaks,
        centers,
        counts,
        bin_width_da=float(bin_width),
        core_half_width_bins=int(analysis_cfg["naive_core_half_width_bins"]),
        sideband_gap_bins=int(analysis_cfg["naive_sideband_gap_bins"]),
        sideband_width_bins=int(analysis_cfg["naive_sideband_width_bins"]),
    )
    return finalize_method_analysis(
        method_name="naive",
        method_label="Naive sideband",
        output_dir=output_dir,
        centers=centers,
        counts=counts,
        peaks=peaks,
        raw_mz=sample_data.m_over_z_da,
        sample_name=sample_data.metadata.sample_name,
        config=config,
        diagnostics={
            "bin_width_da": bin_width,
            "prominence": prominence,
            "peak_detection": "raw histogram local maxima",
            "quantification": {
                "core_half_width_bins": int(analysis_cfg["naive_core_half_width_bins"]),
                "sideband_gap_bins": int(analysis_cfg["naive_sideband_gap_bins"]),
                "sideband_width_bins": int(analysis_cfg["naive_sideband_width_bins"]),
                "rule": "center-window counts minus mean sideband level",
            },
        },
    )
