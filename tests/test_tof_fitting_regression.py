from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from rangefinder.analysis.custom_pipeline import _reduce_fitted_components
from rangefinder.analysis.tof_fitting import fit_local_tof_peak_mixture
from rangefinder.io.pos_loader import list_pos_samples, load_pos_sample

REPO_ROOT = Path(__file__).resolve().parents[1]

SAMPLE_SLUG_2166 = "2166bb75-7ff6-4c85-bf2d-564431f0b089"
SAMPLE_SLUG_E14 = "e14f067b-4c6f-4c34-8e25-ac1e4a217c57"
DATA_DIR = REPO_ROOT / "data"


def _fit_reduced_components(raw_mz: np.ndarray, *, seed_mz_da: np.ndarray, seed_fwhm_da: np.ndarray):
    fit = fit_local_tof_peak_mixture(
        np.asarray(raw_mz, dtype=np.float64),
        seed_mz_da=np.asarray(seed_mz_da, dtype=np.float64),
        seed_fwhm_da=np.asarray(seed_fwhm_da, dtype=np.float64),
        window_margin_da=0.2,
        center_tolerance_da=0.05,
        sigma_floor_da=0.015,
        sigma_ceiling_da=0.1,
        bins_per_sigma=5.0,
        offset_penalty=0.15,
        sigma_penalty=0.08,
    )
    if fit is None:
        raise AssertionError("fit_local_tof_peak_mixture returned None")
    reduced_counts, _, _ = _reduce_fitted_components(
        np.asarray(fit["fitted_counts"], dtype=np.float64),
        np.asarray(fit["fitted_mz_da"], dtype=np.float64),
        np.asarray(fit["fitted_fwhm_da"], dtype=np.float64),
        merge_tolerance_da=0.045,
        min_relative_amplitude=0.012,
        min_absolute_amplitude=25.0,
    )
    return fit, reduced_counts


@unittest.skipUnless(DATA_DIR.exists(), "Local APT POS data is not available in this workspace.")
class TofFittingRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        samples = {metadata.sample_slug: metadata for metadata in list_pos_samples(DATA_DIR)}
        cls.raw_2166 = np.asarray(load_pos_sample(samples[SAMPLE_SLUG_2166]).m_over_z_da, dtype=np.float64)
        cls.raw_e14 = np.asarray(load_pos_sample(samples[SAMPLE_SLUG_E14]).m_over_z_da, dtype=np.float64)

    def test_2166_crowded_silicon_window_stays_near_natural(self) -> None:
        # These are the exact pre-fit custom seeds in the 14-16 Da crowded window.
        fit, reduced_counts = _fit_reduced_components(
            self.raw_2166,
            seed_mz_da=np.array([13.995, 14.485, 14.985, 15.475, 15.995], dtype=np.float64),
            seed_fwhm_da=np.array([0.106761, 0.122842, 0.111765, 0.25, 0.095603], dtype=np.float64),
        )
        self.assertEqual(reduced_counts.size, 3)
        fractions = reduced_counts[:3] / max(float(np.sum(reduced_counts[:3])), 1.0)
        self.assertGreater(float(fractions[0]), 0.88)
        self.assertLess(float(fractions[1]), 0.08)
        self.assertLess(float(fractions[2]), 0.06)
        self.assertLess(float(fit["sigma_t"]), 0.0023)

    def test_e14_reference_silicon_window_remains_stable(self) -> None:
        fit, reduced_counts = _fit_reduced_components(
            self.raw_e14,
            seed_mz_da=np.array([13.985, 14.485, 14.985], dtype=np.float64),
            seed_fwhm_da=np.array([0.107144, 0.106862, 0.10952], dtype=np.float64),
        )
        self.assertEqual(reduced_counts.size, 3)
        fractions = reduced_counts[:3] / max(float(np.sum(reduced_counts[:3])), 1.0)
        self.assertGreater(float(fractions[0]), 0.88)
        self.assertLess(float(fractions[1]), 0.08)
        self.assertLess(float(fractions[2]), 0.06)
        self.assertLess(float(fit["sigma_t"]), 0.0023)


if __name__ == "__main__":
    unittest.main()
