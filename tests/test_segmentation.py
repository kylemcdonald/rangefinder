from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from rangefinder.analysis.segmentation import segment_sample_regions

AL_MZ = 26.9815
ZN_MZ = 63.9291


def _config() -> dict:
    return {
        "analysis": {
            "segmentation_enabled": True,
            "segmentation_min_total_events": 10000,
            "segmentation_target_events_per_voxel": 800,
            "segmentation_min_events_per_voxel": 200,
            "segmentation_max_regions": 4,
            "segmentation_min_region_fraction": 0.05,
            "segmentation_min_composition_separation_at_pct": 10.0,
            "segmentation_min_silhouette": 0.35,
            "segmentation_min_neighbor_agreement": 0.60,
            "segmentation_max_feature_elements": 6,
        }
    }


def _peaks_and_assignments() -> tuple[pd.DataFrame, pd.DataFrame]:
    peaks = pd.DataFrame(
        [
            {"peak_id": "p1", "peak_mz_da": AL_MZ, "integration_left_da": AL_MZ - 0.1, "integration_right_da": AL_MZ + 0.1, "fwhm_da": 0.05},
            {"peak_id": "p2", "peak_mz_da": ZN_MZ, "integration_left_da": ZN_MZ - 0.1, "integration_right_da": ZN_MZ + 0.1, "fwhm_da": 0.05},
        ]
    )
    assignments = pd.DataFrame(
        [
            {"peak_id": "p1", "rank": 1, "species_label": "Al+", "confidence": "high", "element_counts": {"Al": 1}},
            {"peak_id": "p2", "rank": 1, "species_label": "Zn+", "confidence": "high", "element_counts": {"Zn": 1}},
        ]
    )
    return peaks, assignments


def _events(n: int, zn_fraction_by_z, rng: np.random.Generator):
    x = rng.uniform(-20, 20, n)
    y = rng.uniform(-20, 20, n)
    z = rng.uniform(0, 100, n)
    zn_fraction = zn_fraction_by_z(z)
    is_zn = rng.random(n) < zn_fraction
    mz = np.where(is_zn, ZN_MZ, AL_MZ) + rng.normal(0, 0.01, n)
    return x, y, z, mz


class SegmentationTest(unittest.TestCase):
    def test_homogeneous_sample_reports_one_region(self) -> None:
        rng = np.random.default_rng(11)
        peaks, assignments = _peaks_and_assignments()
        x, y, z, mz = _events(120_000, lambda z: np.full_like(z, 0.5), rng)
        result = segment_sample_regions(
            x_nm=x, y_nm=y, z_nm=z, m_over_z_da=mz,
            peaks=peaks, assignments=assignments, config=_config(),
        )
        self.assertIsNotNone(result)
        self.assertTrue(result["homogeneous"])
        self.assertEqual(result["n_regions"], 1)

    def test_layered_sample_reports_two_regions(self) -> None:
        rng = np.random.default_rng(12)
        peaks, assignments = _peaks_and_assignments()
        x, y, z, mz = _events(120_000, lambda z: np.where(z < 50.0, 0.95, 0.05), rng)
        result = segment_sample_regions(
            x_nm=x, y_nm=y, z_nm=z, m_over_z_da=mz,
            peaks=peaks, assignments=assignments, config=_config(),
        )
        self.assertIsNotNone(result)
        self.assertFalse(result["homogeneous"])
        self.assertEqual(result["n_regions"], 2)
        labels = result["event_region"]
        self.assertEqual(labels.shape[0], mz.shape[0])
        # Regions must be depth-organized: each side of the interface is
        # dominated by one label.
        top = labels[z < 45]
        bottom = labels[z > 55]
        top_mode = np.bincount(top).argmax()
        bottom_mode = np.bincount(bottom).argmax()
        self.assertNotEqual(top_mode, bottom_mode)
        self.assertGreater(float((top == top_mode).mean()), 0.9)
        self.assertGreater(float((bottom == bottom_mode).mean()), 0.9)

    def test_voxel_table_ijk_and_centers_match_data_bounds(self) -> None:
        # Regression: the voxel-key packing used 2^12-wide fields for +/-4096
        # offsets, so iy/iz carried into neighboring fields and the voxel table
        # reported indices like iy=-4093 and centers micrometers outside the
        # specimen.
        rng = np.random.default_rng(13)
        peaks, assignments = _peaks_and_assignments()
        x, y, z, mz = _events(120_000, lambda z: np.where(z < 50.0, 0.95, 0.05), rng)
        result = segment_sample_regions(
            x_nm=x, y_nm=y, z_nm=z, m_over_z_da=mz,
            peaks=peaks, assignments=assignments, config=_config(),
        )
        voxels = result["voxel_table"]
        edge = float(result["voxel_edge_nm"])
        n_x = int(np.ceil((x.max() - x.min()) / edge)) + 1
        n_y = int(np.ceil((y.max() - y.min()) / edge)) + 1
        n_z = int(np.ceil((z.max() - z.min()) / edge)) + 1
        for axis, n_axis in (("ix", n_x), ("iy", n_y), ("iz", n_z)):
            self.assertTrue((voxels[axis] >= -1).all(), f"{axis} below grid")
            self.assertTrue((voxels[axis] <= n_axis).all(), f"{axis} beyond grid")
        for axis, lo, hi in (("x_nm", x.min(), x.max()), ("y_nm", y.min(), y.max()), ("z_nm", z.min(), z.max())):
            self.assertTrue((voxels[axis] > lo - edge).all(), f"{axis} below data")
            self.assertTrue((voxels[axis] < hi + edge).all(), f"{axis} above data")


if __name__ == "__main__":
    unittest.main()
