"""Regression tests for the assignment parsimony guards.

These lock in the failure modes fixed in July 2026: phantom trace elements on
single-element spectra (atomic support penalty + element pruning) and
minor-isotopologue molecular capture (envelope veto).
"""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from rangefinder import default_config_path
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
