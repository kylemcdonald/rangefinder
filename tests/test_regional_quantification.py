from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from rangefinder.analysis.regions import partition_composition_by_region


def _config() -> dict:
    return {
        "analysis": {
            "composition_probability_floor": 0.0,
            "overlap_deconvolution_enabled": False,
            "min_isotope_counts_for_anomaly": 100,
            "slight_anomaly_min_abs_fraction_diff": 0.05,
            "strong_anomaly_min_abs_fraction_diff": 0.10,
            "slight_anomaly_min_z": 3.0,
            "strong_anomaly_min_z": 5.0,
        }
    }


def _assignment(
    peak_id: str,
    species: str,
    element: str,
    isotope: str,
) -> dict[str, object]:
    return {
        "peak_id": peak_id,
        "rank": 1,
        "species_label": species,
        "isotopologue_label": isotope,
        "category": "atomic",
        "charge": 1,
        "probability": 1.0,
        "confidence": "high",
        "candidate_count": 1,
        "mass_error_da": 0.0,
        "element_counts": {element: 1},
        "isotope_counts": {isotope: 1},
    }


class RegionalQuantificationTests(unittest.TestCase):
    def test_fixed_global_identities_partition_and_recombine_exactly(self) -> None:
        peaks = pd.DataFrame(
            [
                {
                    "peak_id": "oxygen",
                    "peak_mz_da": 15.995,
                    "integration_left_da": 15.90,
                    "integration_right_da": 16.10,
                    "integrated_area": 1000.0,
                },
                {
                    "peak_id": "sulfur",
                    "peak_mz_da": 31.972,
                    "integration_left_da": 31.90,
                    "integration_right_da": 32.10,
                    "integrated_area": 100.0,
                },
            ]
        )
        assignments = pd.DataFrame(
            [
                _assignment("oxygen", "O+", "O", "16O"),
                _assignment("sulfur", "S+", "S", "32S"),
            ]
        )
        # Region 1 contains most of the 16 Da peak but none of the actual
        # sulfur peak. A regional rerange could mistake 16 Da for 32S2+;
        # fixed global ranging must retain O+ and report zero regional S.
        mz = np.concatenate(
            [
                np.full(100, 15.995),
                np.full(900, 15.995),
                np.full(100, 31.972),
            ]
        )
        labels = np.concatenate(
            [
                np.zeros(100, dtype=np.int64),
                np.ones(900, dtype=np.int64),
                np.zeros(100, dtype=np.int64),
            ]
        )
        result = partition_composition_by_region(
            mz_values=mz,
            event_region=labels,
            peaks=peaks,
            assignments=assignments,
            config=_config(),
        )

        self.assertEqual(result["quantification_mode"], "fixed_global_ranging")
        self.assertTrue(result["recombination"]["passed"])
        self.assertLessEqual(result["recombination"]["max_absolute_error"], 1.0e-9)
        region_one = result["regions"][1]
        top = region_one["assignments"].set_index("peak_id")
        self.assertEqual(top.loc["oxygen", "species_label"], "O+")
        elements = region_one["elements"].set_index("element")
        self.assertAlmostEqual(float(elements.loc["O", "weighted_counts"]), 900.0)
        self.assertAlmostEqual(float(elements.loc["S", "weighted_counts"]), 0.0)
        self.assertAlmostEqual(float(elements.loc["S", "atomic_percent_weighted"]), 0.0)

        recombined = (
            pd.concat([region["elements"] for region in result["regions"]])
            .groupby("element")["weighted_counts"]
            .sum()
        )
        self.assertAlmostEqual(float(recombined["O"]), 1000.0)
        self.assertAlmostEqual(float(recombined["S"]), 100.0)


if __name__ == "__main__":
    unittest.main()
