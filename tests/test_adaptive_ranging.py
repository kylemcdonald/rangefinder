from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from rangefinder.analysis.adaptive_ranging import apply_adaptive_ranging, equal_error_ranges


class EqualErrorRangingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.centers = np.arange(1.005, 80.0, 0.01, dtype=np.float64)
        background = 35.0 / np.sqrt(self.centers)
        first = 900.0 * np.exp(-0.5 * ((self.centers - 40.0) / 0.035) ** 2)
        first += np.where(
            self.centers > 40.0,
            160.0 * np.exp(-(self.centers - 40.0) / 0.09),
            0.0,
        )
        second = 500.0 * np.exp(-0.5 * ((self.centers - 40.42) / 0.045) ** 2)
        self.counts = background + first + second
        self.peaks = pd.DataFrame(
            [
                {
                    "peak_id": "p1",
                    "peak_mz_da": 40.0,
                    "fwhm_da": 0.08,
                    "integration_left_da": 39.82,
                    "integration_right_da": 40.18,
                    "integrated_area": 1000.0,
                },
                {
                    "peak_id": "p2",
                    "peak_mz_da": 40.42,
                    "fwhm_da": 0.10,
                    "integration_left_da": 40.24,
                    "integration_right_da": 40.60,
                    "integrated_area": 800.0,
                },
            ]
        )

    def test_ranges_are_non_overlapping_and_report_quality(self) -> None:
        ranged, diagnostics = equal_error_ranges(self.peaks, self.centers, self.counts)
        self.assertLessEqual(
            float(ranged.iloc[0]["eer_integration_right_da"]),
            float(ranged.iloc[1]["eer_integration_left_da"]),
        )
        self.assertTrue(((ranged["eer_purity"] >= 0.0) & (ranged["eer_purity"] <= 1.0)).all())
        self.assertTrue(((ranged["eer_recovery"] >= 0.0) & (ranged["eer_recovery"] <= 1.0)).all())
        self.assertTrue((ranged["eer_integrated_area"] > 0.0).all())
        self.assertGreater(float(diagnostics["background_coefficient"]), 0.0)

    def test_shadow_is_non_mutating_and_eer_mode_promotes_bounds(self) -> None:
        shadow_config = {"analysis": {"adaptive_ranging_mode": "shadow"}}
        shadow, diagnostics = apply_adaptive_ranging(
            self.peaks, self.centers, self.counts, config=shadow_config
        )
        np.testing.assert_allclose(shadow["integrated_area"], self.peaks["integrated_area"])
        self.assertEqual(int(diagnostics["selected_peak_count"]), 0)
        self.assertIn("eer_integrated_area", shadow.columns)

        eer_config = {"analysis": {"adaptive_ranging_mode": "eer"}}
        promoted, diagnostics = apply_adaptive_ranging(
            self.peaks, self.centers, self.counts, config=eer_config
        )
        self.assertEqual(int(diagnostics["selected_peak_count"]), 2)
        self.assertTrue((promoted["integration_method"] == "equal_error").all())
        np.testing.assert_allclose(promoted["integrated_area"], promoted["eer_integrated_area"])


if __name__ == "__main__":
    unittest.main()
