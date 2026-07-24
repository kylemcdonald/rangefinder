from __future__ import annotations

import numpy as np
import pandas as pd

from rangefinder.analysis.assignment import (
    _ambiguous_template_columns,
    _apportion_integer_counts,
    robust_family_observation,
)
from rangefinder.analysis.custom_pipeline import _peak_has_significance_rescue


def test_integer_apportionment_preserves_expected_total() -> None:
    frame = pd.DataFrame({"weighted_counts": [8288.301088, 13675.885855, 0.49]})
    result = _apportion_integer_counts(frame)
    assert result["integer_count"].tolist() == [8288, 13676, 1]
    assert int(result["integer_count"].sum()) == int(
        np.floor(float(result["weighted_counts"].sum()) + 0.5)
    )


def test_positive_isotope_overlap_is_capped_without_hiding_missing_members() -> None:
    expected = np.asarray([0.0825, 0.0744, 0.7372, 0.0541, 0.0518])
    observed = expected * 26750.0
    observed[-1] *= 22.0
    _, fractions, interference, _ = robust_family_observation(observed, expected)
    assert interference.tolist() == [False, False, False, False, True]
    np.testing.assert_allclose(fractions, expected, atol=1.0e-12)

    missing = observed.copy()
    missing[1] = 0.0
    _, missing_fractions, missing_interference, _ = robust_family_observation(
        missing, expected
    )
    assert not bool(missing_interference[1])
    assert missing_fractions[1] == 0.0


def test_degenerate_local_templates_do_not_create_family_evidence() -> None:
    design = np.asarray(
        [
            [1.0, 1.0, 0.0],
            [0.2, 0.2, 1.0],
            [0.0, 0.0, 0.4],
        ]
    )
    ambiguous = _ambiguous_template_columns(
        design,
        max_correlation=0.9995,
    )
    assert ambiguous.tolist() == [True, True, False]


def test_clean_multiresolution_range_rescues_crowded_trace_peak() -> None:
    config = {
        "custom_peak_significance_rescue_enabled": True,
        "custom_peak_significance_rescue_min_detection_sources": 2,
        "custom_peak_significance_rescue_min_eer_purity": 0.8,
        "custom_peak_significance_rescue_min_eer_snr": 10.0,
    }
    peak = pd.Series(
        {
            "detection_source_count": 2,
            "eer_integrated_area": 725.0,
            "eer_counting_std": 30.0,
            "eer_purity": 0.87,
        }
    )
    assert _peak_has_significance_rescue(peak, analysis_cfg=config)
    impure = peak.copy()
    impure["eer_purity"] = 0.79
    assert not _peak_has_significance_rescue(
        impure,
        analysis_cfg=config,
    )
