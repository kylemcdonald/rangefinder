"""Regression tests for the assignment parsimony guards.

These lock in the failure modes fixed in July 2026: phantom trace elements on
single-element spectra (atomic support penalty + element pruning) and
minor-isotopologue molecular capture (envelope veto).
"""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from rangefinder import default_config_path
from rangefinder.analysis.assignment import _anchored_elements, assign_peaks
from rangefinder.analysis.custom_pipeline import run_custom_pipeline
from rangefinder.io.pos_loader import PosSampleData, PosSampleMetadata
from rangefinder.utils.config import load_config
from rangefinder.validation.synthetic import (
    SyntheticMaterial,
    SyntheticSpecies,
    synthesize_material_events,
)

ROOT = Path(__file__).resolve().parent.parent


def _run(material: SyntheticMaterial, tmp_name: str):
    config = load_config(default_config_path())
    rng = np.random.default_rng(7)
    x, y, z, mz = synthesize_material_events(material, rng=rng)
    metadata = PosSampleMetadata(
        path=Path(f"/tmp/{tmp_name}"), sample_name=tmp_name, sample_slug=tmp_name,
        event_count=int(mz.size), file_size_bytes=int(mz.size) * 16,
        verified_big_endian_float32=True,
        verified_columns=("x_nm", "y_nm", "z_nm", "m_over_z_da"),
    )
    sample = PosSampleData(metadata=metadata, x_nm=x, y_nm=y, z_nm=z, m_over_z_da=mz)
    out_dir = ROOT / "outputs" / "_test_parsimony" / tmp_name
    artifacts = run_custom_pipeline(sample, out_dir, config, enable_segmentation=False)
    elements = artifacts.elemental_composition
    return {
        str(row["element"]): float(row["atomic_percent_weighted"])
        for _, row in elements.iterrows()
    }


class ParsimonyGuardTests(unittest.TestCase):
    def test_single_ambiguous_atomic_match_cannot_self_anchor(self):
        config = load_config(default_config_path())
        peaks = pd.DataFrame(
            {
                "peak_id": ["s-only", "c-one", "c-two", "o-strong"],
                "peak_mz_da": [31.975, 6.0, 12.0, 16.0],
                "integrated_area": [100_000.0, 10_000.0, 20_000.0, 200_000.0],
            }
        )
        assignments = pd.DataFrame(
            [
                {
                    "peak_id": "s-only", "rank": 1, "species_label": "S+",
                    "category": "atomic", "probability": 0.55,
                    "confidence": "ambiguous", "element_counts": {"S": 1},
                },
                {
                    "peak_id": "c-one", "rank": 1, "species_label": "C2+",
                    "category": "atomic", "probability": 0.52,
                    "confidence": "ambiguous", "element_counts": {"C": 1},
                },
                {
                    "peak_id": "c-two", "rank": 1, "species_label": "C+",
                    "category": "atomic", "probability": 0.61,
                    "confidence": "ambiguous", "element_counts": {"C": 1},
                },
                {
                    "peak_id": "o-strong", "rank": 1, "species_label": "O+",
                    "category": "atomic", "probability": 0.90,
                    "confidence": "high", "element_counts": {"O": 1},
                },
            ]
        )
        anchored = _anchored_elements(peaks, assignments, config=config)
        self.assertNotIn("S", anchored)
        self.assertIn("C", anchored)
        self.assertIn("O", anchored)

    def test_32_da_oxide_peak_rejects_uncorroborated_s_and_zn(self):
        """A calibration-shifted O2+ peak lies closer to 32S+ and 64Zn2+.
        A local mass advantage must not bootstrap either element without
        independent peaks elsewhere in the spectrum or a material prior."""
        config = load_config(default_config_path())
        rows = [
            ("h", 1.005000, 107_907.25),
            ("c2", 6.005000, 12_059.13),
            ("c", 11.995000, 34_730.00),
            ("si28-2", 13.988471, 577_077.27),
            ("si29-2", 14.488247, 40_598.05),
            ("si30-2", 14.986878, 23_446.99),
            ("o", 15.994934, 600_041.68),
            ("oh", 16.995000, 13_298.96),
            ("o18", 17.999141, 2_842.67),
            ("si28", 27.976783, 71_312.93),
            ("si29", 28.976495, 4_459.45),
            ("si30", 29.983380, 5_182.12),
            ("target", 31.975000, 99_561.70),
        ]
        peaks = pd.DataFrame(rows, columns=["peak_id", "peak_mz_da", "integrated_area"])
        peaks["peak_height"] = peaks["integrated_area"] / 5.0
        peaks["fwhm_da"] = 0.10
        assignments, _summary = assign_peaks(peaks, config=config)
        target = assignments[
            (assignments["peak_id"] == "target") & (assignments["rank"] == 1)
        ].iloc[0]
        self.assertEqual("O2+", target["species_label"])
        self.assertEqual({"O": 2}, target["element_counts"])
        self.assertGreater(float(target["probability"]), 0.95)
        pruned = str(target.get("pruned_elements", ""))
        self.assertIn("S", pruned)
        self.assertIn("Zn", pruned)

    def test_pure_tungsten_has_no_phantom_elements(self):
        """W evaporates as 3+/4+; no N/O/B should be fabricated from the 4+
        charge-state quartet."""
        material = SyntheticMaterial(
            name="test_pure_w",
            species=[
                SyntheticSpecies({"W": 1}, 3, 0.70),
                SyntheticSpecies({"W": 1}, 4, 0.30),
            ],
            total_events=400_000,
            background_fraction=0.01,
        )
        composition = _run(material, "test_pure_w")
        self.assertGreater(composition.get("W", 0.0), 99.0)
        fabricated = sum(pct for element, pct in composition.items() if element != "W")
        self.assertLess(fabricated, 1.0)

    def test_oxide_molecular_ions_apportioned_to_real_species(self):
        """O2+ at 32 Da must not leak to minor isotopologues of hydrides
        (the H2Si+ failure mode)."""
        material = SyntheticMaterial(
            name="test_si_oxide",
            species=[
                SyntheticSpecies({"Si": 1}, 2, 0.35),
                SyntheticSpecies({"Si": 1}, 1, 0.25),
                SyntheticSpecies({"Si": 1, "O": 1}, 1, 0.15),
                SyntheticSpecies({"O": 1}, 1, 0.10),
                SyntheticSpecies({"O": 2}, 1, 0.08),
                SyntheticSpecies({"H": 1}, 1, 0.07),
            ],
            total_events=600_000,
            background_fraction=0.01,
        )
        composition = _run(material, "test_si_oxide")
        truth = material.true_atomic_fractions()
        for element in ("Si", "O", "H"):
            self.assertLess(
                abs(composition.get(element, 0.0) - 100.0 * truth[element]),
                3.0,
                msg=f"{element}: {composition.get(element)} vs {100.0 * truth[element]}",
            )
        fabricated = sum(
            pct for element, pct in composition.items() if element not in truth
        )
        self.assertLess(fabricated, 0.5)


if __name__ == "__main__":
    unittest.main()
