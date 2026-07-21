from __future__ import annotations

import numpy as np
import pyopenms

from rangefinder.analysis.common import (
    build_histogram,
    characterize_centroid_peaks,
    detection_bin_width,
    default_prominence,
)
from rangefinder.analysis.pipeline_utils import finalize_method_analysis


def run_pyopenms_pipeline(sample_data, output_dir, config):
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
    spectrum = pyopenms.MSSpectrum()
    spectrum.set_peaks((centers.astype(float), counts.astype(float)))
    picked = pyopenms.MSSpectrum()
    pyopenms.PeakPickerHiRes().pick(spectrum, picked)
    if bool(analysis_cfg["pyopenms_use_deisotoper"]):
        pyopenms.Deisotoper.deisotopeAndSingleCharge(
            picked,
            float(max(bin_width, 0.03)),
            False,
            int(analysis_cfg["pyopenms_deisotope_min_charge"]),
            int(analysis_cfg["pyopenms_deisotope_max_charge"]),
            False,
            2,
            6,
            False,
            True,
            True,
            False,
            1,
            False,
            False,
        )
    mz, intensity = picked.get_peaks()
    mz = np.asarray(mz)
    intensity = np.asarray(intensity)
    keep = intensity >= prominence
    peaks = characterize_centroid_peaks(
        mz[keep],
        intensity[keep],
        centers,
        counts,
        method_label="pyopenms",
        integration_half_width_da=float(analysis_cfg["peak_integration_half_width_da"]),
        baseline_half_width_da=float(analysis_cfg["local_baseline_half_width_da"]),
    )
    return finalize_method_analysis(
        method_name="pyopenms",
        method_label="pyOpenMS",
        output_dir=output_dir,
        centers=centers,
        counts=counts,
        peaks=peaks,
        raw_mz=sample_data.m_over_z_da,
        sample_name=sample_data.metadata.sample_name,
        config=config,
        diagnostics={
            "bin_width_da": bin_width,
            "prominence_filter": prominence,
            "library": "pyopenms",
            "peak_picker": "PeakPickerHiRes",
            "deisotoper": (
                "Deisotoper.deisotopeAndSingleCharge"
                if bool(analysis_cfg["pyopenms_use_deisotoper"])
                else "disabled"
            ),
        },
    )
