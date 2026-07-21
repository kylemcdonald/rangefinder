from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pyccapt.calibration.core.mc_plot import AptHistPlotter
from pyccapt.calibration.core.share_variables import Variables

from rangefinder.analysis.common import (
    build_histogram,
    detection_bin_width,
    default_prominence,
    integrate_peak_areas,
)
from rangefinder.analysis.pipeline_utils import finalize_method_analysis


def run_pyccapt_pipeline(sample_data, output_dir, config):
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
        quantile=float(analysis_cfg["pyccapt_prominence_quantile"]),
        minimum=float(analysis_cfg["pyccapt_min_prominence"]),
    )
    variables = Variables()
    variables.mc = sample_data.m_over_z_da
    plotter = AptHistPlotter(sample_data.m_over_z_da, variables)
    y_values, x_values = plotter.plot_histogram(
        bin_width=bin_width,
        normalize=False,
        label="mc",
        log=True,
        grid=False,
        steps="stepfilled",
        plot_show=False,
    )
    peaks_idx, _properties, widths, prominences = plotter.find_peaks_and_widths(
        prominence=prominence,
        distance=int(analysis_cfg["pyccapt_distance_bins"]),
        percent=50,
    )
    rows = []
    if peaks_idx is not None:
        for i, peak_idx in enumerate(peaks_idx):
            rows.append(
                {
                    "peak_id": f"pyccapt-peak-{i + 1:04d}",
                    "method": "pyccapt",
                    "peak_index": int(peak_idx),
                    # AptHistPlotter returns histogram bin edges; the peak sits at
                    # the bin center, half a bin above the left edge.
                    "peak_mz_da": float(x_values[peak_idx]) + bin_width / 2.0,
                    "peak_height": float(y_values[peak_idx]),
                    "smoothed_height": float(y_values[peak_idx]),
                    "prominence": float(prominences[0][i]),
                    "left_fwhm_mz_da": float(x_values[round(widths[2][i])]) + bin_width / 2.0,
                    "right_fwhm_mz_da": float(x_values[round(widths[3][i])]) + bin_width / 2.0,
                    "fwhm_da": float(x_values[round(widths[3][i])] - x_values[round(widths[2][i])]),
                }
            )
    import pandas as pd

    peaks = pd.DataFrame(rows)
    peaks = integrate_peak_areas(
        peaks,
        centers,
        counts,
        integration_half_width_da=float(analysis_cfg["peak_integration_half_width_da"]),
        baseline_half_width_da=float(analysis_cfg["local_baseline_half_width_da"]),
    )
    return finalize_method_analysis(
        method_name="pyccapt",
        method_label="PyCCAPT",
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
            "library": "pyccapt",
            "peak_detection": "AptHistPlotter.find_peaks_and_widths",
        },
    )
