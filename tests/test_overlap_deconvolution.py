from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from rangefinder.analysis.overlap_deconvolution import (
    deconvolve_peak_overlaps,
    isotopologue_envelope,
)


def _config() -> dict:
    return {
        "analysis": {
            "ppm_tolerance": 2500.0,
            "min_absolute_tolerance_da": 0.03,
            "max_absolute_tolerance_da": 0.15,
            "overlap_deconvolution_enabled": True,
            "overlap_deconvolution_max_rank": 4,
            "overlap_deconvolution_min_probability": 0.01,
            "overlap_deconvolution_min_vote_area": 0.0,
            "overlap_deconvolution_max_species": 8,
            "overlap_deconvolution_anchor_min_probability": 0.60,
            "overlap_deconvolution_isotopologue_probability_floor": 1.0e-5,
            "overlap_deconvolution_match_tolerance_scale": 1.0,
            "overlap_deconvolution_min_envelope_coverage": 0.50,
            "overlap_deconvolution_max_condition_number": 1.0e6,
            "overlap_deconvolution_max_relative_rmse": 0.05,
        }
    }


def _fe_ni_problem() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    envelopes = {
        "Fe+": isotopologue_envelope({"Fe": 1}, 1, min_probability=1.0e-5),
        "Ni+": isotopologue_envelope({"Ni": 1}, 1, min_probability=1.0e-5),
    }
    grouped: list[list[tuple[str, float, float]]] = []
    for label, envelope in envelopes.items():
        for mass, probability in envelope:
            for group in grouped:
                group_mass = float(np.average([item[1] for item in group], weights=[item[2] for item in group]))
                if abs(group_mass - mass) <= 0.01:
                    group.append((label, mass, probability))
                    break
            else:
                grouped.append([(label, mass, probability)])
    grouped.sort(key=lambda group: min(item[1] for item in group))
    abundance = {"Fe+": 120_000.0, "Ni+": 80_000.0}
    peak_rows = []
    assignment_rows = []
    expected_overlap_fraction = None
    for index, group in enumerate(grouped, start=1):
        peak_id = f"p{index}"
        mass = float(np.average([item[1] for item in group], weights=[item[2] for item in group]))
        contributions = {label: abundance[label] * probability for label, _, probability in group}
        area = float(sum(contributions.values()))
        peak_rows.append(
            {
                "peak_id": peak_id,
                "peak_mz_da": mass,
                "fwhm_da": 0.05,
                "integrated_area": area,
            }
        )
        ordered = sorted(contributions, key=contributions.get, reverse=True)
        for rank, label in enumerate(ordered, start=1):
            element = label.removesuffix("+")
            assignment_rows.append(
                {
                    "peak_id": peak_id,
                    "rank": rank,
                    "species_label": label,
                    "element_counts": {element: 1},
                    "charge": 1,
                    "probability": 1.0 / len(ordered),
                }
            )
        if len(contributions) == 2:
            expected_overlap_fraction = {
                "peak_id": peak_id,
                "Fe+": contributions["Fe+"] / area,
                "Ni+": contributions["Ni+"] / area,
            }
    assert expected_overlap_fraction is not None
    return pd.DataFrame(peak_rows), pd.DataFrame(assignment_rows), expected_overlap_fraction


class OverlapDeconvolutionTests(unittest.TestCase):
    def test_solvable_fe_ni_overlap_recovers_peak_allocation(self) -> None:
        peaks, assignments, expected = _fe_ni_problem()
        result, diagnostics = deconvolve_peak_overlaps(peaks, assignments, config=_config())
        overlap = result[result["peak_id"] == expected["peak_id"]].set_index("species_label")
        self.assertEqual(int(diagnostics["accepted_component_count"]), 1)
        self.assertEqual(set(overlap["quantification_method"]), {"isotope_envelope_poisson"})
        self.assertAlmostEqual(
            float(overlap.loc["Fe+", "quantification_probability"]), expected["Fe+"], places=5
        )
        self.assertAlmostEqual(
            float(overlap.loc["Ni+", "quantification_probability"]), expected["Ni+"], places=5
        )
        self.assertAlmostEqual(float(overlap["quantification_probability"].sum()), 1.0, places=12)

    def test_rank_deficient_component_falls_back_without_mutation(self) -> None:
        peaks = pd.DataFrame(
            [
                {"peak_id": "p1", "peak_mz_da": 55.9349, "integrated_area": 1000.0},
                {"peak_id": "p2", "peak_mz_da": 57.9333, "integrated_area": 100.0},
            ]
        )
        assignments = pd.DataFrame(
            [
                {"peak_id": "p1", "rank": 1, "species_label": "Fe+",
                 "element_counts": {"Fe": 1}, "charge": 1, "probability": 0.7},
                {"peak_id": "p1", "rank": 2, "species_label": "Fe_alias+",
                 "element_counts": {"Fe": 1}, "charge": 1, "probability": 0.3},
                {"peak_id": "p2", "rank": 1, "species_label": "Fe_alias+",
                 "element_counts": {"Fe": 1}, "charge": 1, "probability": 0.7},
                {"peak_id": "p2", "rank": 2, "species_label": "Fe+",
                 "element_counts": {"Fe": 1}, "charge": 1, "probability": 0.3},
            ]
        )
        result, diagnostics = deconvolve_peak_overlaps(peaks, assignments, config=_config())
        np.testing.assert_allclose(result["quantification_probability"], assignments["probability"])
        self.assertEqual(int(diagnostics["accepted_component_count"]), 0)
        self.assertGreaterEqual(int(diagnostics["rejected_component_count"]), 1)
        reasons = {component.get("reason") for component in diagnostics["components"]}
        self.assertIn("underdetermined envelope matrix", reasons)


if __name__ == "__main__":
    unittest.main()
