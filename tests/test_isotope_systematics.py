from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from rangefinder.analysis.isotope_audit import reconcile_isotope_anomalies
from rangefinder.analysis.tof_fitting import (
    fit_isotope_family_tof,
    fit_isotope_family_tof_tail,
)

MG_MASSES = np.asarray([23.985041, 24.985836, 25.982593])
MG_FRACTIONS = np.asarray([0.7899, 0.1000, 0.1101])


def _synthetic_family_events(
    *,
    n_ions: int,
    tail_fraction: float,
    tail_tau_sigma_scale: float,
    seed: int = 7,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(n_ions, MG_FRACTIONS / MG_FRACTIONS.sum())
    chunks = []
    for mass, count in zip(MG_MASSES, counts, strict=True):
        t0 = np.sqrt(mass)
        sigma_t = t0 / (2.0 * 800.0) / 2.354820045
        t = t0 + rng.normal(0.0, sigma_t, size=count)
        tail_mask = rng.random(count) < tail_fraction
        t[tail_mask] += rng.exponential(tail_tau_sigma_scale * sigma_t, size=int(tail_mask.sum()))
        chunks.append(np.square(t))
    return np.concatenate(chunks)


class TailAwareFamilyFitTest(unittest.TestCase):
    def test_emg_fit_recovers_natural_fractions_with_tails(self) -> None:
        events = _synthetic_family_events(
            n_ions=200_000, tail_fraction=0.5, tail_tau_sigma_scale=3.0
        )
        kwargs = dict(
            target_mz_da=MG_MASSES,
            sigma_da_init=0.02,
            window_margin_da=0.25,
            shift_tolerance_da=0.04,
            sigma_floor_da=0.005,
            sigma_ceiling_da=0.08,
        )
        gauss = fit_isotope_family_tof(events, **kwargs)
        tail = fit_isotope_family_tof_tail(events, **kwargs)
        self.assertIsNotNone(gauss)
        self.assertIsNotNone(tail)
        tail_fraction = np.asarray(tail["fitted_fraction"], dtype=float)
        expected = MG_FRACTIONS / MG_FRACTIONS.sum()
        # The EMG model must recover the true fractions closely even with a
        # strong thermal tail present.
        self.assertLess(float(np.abs(tail_fraction - expected).max()), 0.01)
        # And it should be at least as accurate as the Gaussian-only model.
        gauss_error = float(np.abs(np.asarray(gauss["fitted_fraction"]) - expected).max())
        tail_error = float(np.abs(tail_fraction - expected).max())
        self.assertLessEqual(tail_error, gauss_error + 1e-9)


class AnomalyReconciliationTest(unittest.TestCase):
    def _anomaly_row(self, observed: float, expected: float, total: int) -> pd.DataFrame:
        variance = expected * (1.0 - expected) / total
        z = (observed - expected) / np.sqrt(variance)
        return pd.DataFrame(
            [
                {
                    "element": "Mg",
                    "isotope_label": "24Mg",
                    "observed_fraction": observed,
                    "expected_fraction": expected,
                    "fraction_difference": observed - expected,
                    "observed_to_natural_ratio": observed / expected,
                    "z_score": z,
                    "status": "strong",
                }
            ]
        )

    def _config(self) -> dict:
        return {
            "analysis": {
                "isotope_anomaly_systematic_fraction_floor": 0.005,
                "isotope_anomaly_audit_support_ratio": 0.4,
                "slight_anomaly_min_abs_fraction_diff": 0.01,
                "strong_anomaly_min_abs_fraction_diff": 0.03,
                "slight_anomaly_min_z": 3.0,
                "strong_anomaly_min_z": 5.0,
            }
        }

    def test_unaudited_strong_anomaly_is_downgraded(self) -> None:
        anomalies = self._anomaly_row(0.84, 0.79, 100_000)
        result = reconcile_isotope_anomalies(anomalies, pd.DataFrame(), config=self._config())
        self.assertEqual(result.iloc[0]["status"], "slight")
        self.assertIn("no independent audit", result.iloc[0]["gating_notes"])

    def test_audit_confirmed_anomaly_stays_strong(self) -> None:
        anomalies = self._anomaly_row(0.84, 0.79, 100_000)
        audit = pd.DataFrame(
            [
                {
                    "element": "Mg",
                    "isotope_label": "24Mg",
                    "audit_fraction": 0.842,
                    "audit_shape_delta": 0.002,
                    "overlap_risk": False,
                }
            ]
        )
        result = reconcile_isotope_anomalies(anomalies, audit, config=self._config())
        self.assertEqual(result.iloc[0]["status"], "strong")

    def test_overlap_risk_downgrades(self) -> None:
        anomalies = self._anomaly_row(0.84, 0.79, 100_000)
        audit = pd.DataFrame(
            [
                {
                    "element": "Mg",
                    "isotope_label": "24Mg",
                    "audit_fraction": 0.845,
                    "audit_shape_delta": 0.002,
                    "overlap_risk": True,
                }
            ]
        )
        result = reconcile_isotope_anomalies(anomalies, audit, config=self._config())
        self.assertEqual(result.iloc[0]["status"], "slight")

    def test_small_bias_with_huge_counts_not_strong(self) -> None:
        # 1.2% deviation at 2e6 counts: counting z is astronomical, but the
        # systematic floor keeps this from being labeled strong.
        anomalies = self._anomaly_row(0.802, 0.79, 2_000_000)
        audit = pd.DataFrame(
            [
                {
                    "element": "Mg",
                    "isotope_label": "24Mg",
                    "audit_fraction": 0.801,
                    "audit_shape_delta": 0.01,
                    "overlap_risk": False,
                }
            ]
        )
        result = reconcile_isotope_anomalies(anomalies, audit, config=self._config())
        self.assertNotEqual(result.iloc[0]["status"], "strong")


if __name__ == "__main__":
    unittest.main()
