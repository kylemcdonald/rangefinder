"""Benchmark metrics: composition recovery, peak detection, mass accuracy.

Composition truths are dicts of atomic percent. Peak truths come in two
flavors: exact isotopologue lines (synthetic data, where every emitted line
position and intensity is known) and reference range intervals (public
datasets shipped with an expert range file).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def composition_metrics(
    recovered_at_pct: dict[str, float],
    truth_at_pct: dict[str, float],
    *,
    fabrication_floor_at_pct: float = 0.01,
    allowed_extra_elements: set[str] | None = None,
) -> dict[str, object]:
    """allowed_extra_elements: elements known present (e.g. from the expert
    range file) but absent from an idealized nominal truth; they do not
    count as fabricated. H is never counted as fabricated: residual-gas
    hydrogen is present in every real APT measurement."""
    allowed = set(allowed_extra_elements or ()) | {"H"}
    elements = sorted(set(recovered_at_pct) | set(truth_at_pct))
    l1 = float(
        sum(
            abs(recovered_at_pct.get(element, 0.0) - truth_at_pct.get(element, 0.0))
            for element in elements
        )
    )
    fabricated = float(
        sum(
            pct
            for element, pct in recovered_at_pct.items()
            if truth_at_pct.get(element, 0.0) < fabrication_floor_at_pct
            and element not in allowed
        )
    )
    major_truth = {e: p for e, p in truth_at_pct.items() if p >= 1.0}
    max_major_error = max(
        (abs(recovered_at_pct.get(e, 0.0) - p) for e, p in major_truth.items()),
        default=float("nan"),
    )
    missed_major = sorted(
        e for e, p in major_truth.items() if recovered_at_pct.get(e, 0.0) < 0.2 * p
    )
    top_truth = max(truth_at_pct, key=truth_at_pct.get) if truth_at_pct else None
    top_recovered = max(recovered_at_pct, key=recovered_at_pct.get) if recovered_at_pct else None
    return {
        "elemental_l1_at_pct": l1,
        "fabricated_at_pct": fabricated,
        "max_major_element_error_at_pct": float(max_major_error),
        "missed_major_elements": missed_major,
        "top_element_true": top_truth,
        "top_element_recovered": top_recovered,
        "top_element_correct": bool(top_truth == top_recovered),
    }


def peak_metrics_vs_lines(
    detected_mz: np.ndarray,
    truth_lines: pd.DataFrame,
    *,
    mass_resolving_power: float,
    min_line_weight: float = 1.0e-4,
    tolerance_floor_da: float = 0.03,
) -> dict[str, object]:
    """Match detected peak centers against known emitted lines.

    truth_lines columns: mz_da, weight (fraction of all signal ions).
    A line is 'detectable' if its weight is above min_line_weight. The match
    tolerance scales with the peak width implied by the mass resolving power.
    """
    detected = np.sort(np.asarray(detected_mz, dtype=np.float64))
    lines = truth_lines[truth_lines["weight"] >= float(min_line_weight)].copy()
    if lines.empty:
        return {}
    line_mz = lines["mz_da"].to_numpy(dtype=np.float64)
    line_weight = lines["weight"].to_numpy(dtype=np.float64)
    tolerance = np.maximum(float(tolerance_floor_da), line_mz / float(mass_resolving_power))
    matched = np.zeros(line_mz.size, dtype=bool)
    ppm_errors: list[float] = []
    used_detected: set[int] = set()
    for idx in np.argsort(-line_weight):
        if detected.size == 0:
            break
        deltas = np.abs(detected - line_mz[idx])
        order = np.argsort(deltas)
        for det_idx in order[:8]:
            if deltas[det_idx] > tolerance[idx]:
                break
            if int(det_idx) in used_detected:
                continue
            matched[idx] = True
            used_detected.add(int(det_idx))
            ppm_errors.append(1.0e6 * (detected[det_idx] - line_mz[idx]) / line_mz[idx])
            break
    # Detected peaks not near any truth line (within the generous per-line
    # tolerance) count as false peaks; near-duplicates of matched lines do not.
    false_mask = np.ones(detected.size, dtype=bool)
    for idx in range(line_mz.size):
        near = np.abs(detected - line_mz[idx]) <= max(2.0 * tolerance[idx], 0.08)
        false_mask &= ~near
    n_false = int(false_mask.sum())
    recall = float(matched.mean()) if matched.size else float("nan")
    weighted_recall = float(
        np.sum(line_weight[matched]) / max(np.sum(line_weight), 1.0e-12)
    )
    precision = float(len(used_detected) / max(detected.size, 1))
    return {
        "n_truth_lines": int(line_mz.size),
        "n_detected_peaks": int(detected.size),
        "line_recall": recall,
        "line_recall_weighted": weighted_recall,
        "peak_precision": precision,
        "n_false_peaks": n_false,
        "mass_accuracy_median_abs_ppm": float(np.median(np.abs(ppm_errors))) if ppm_errors else float("nan"),
        "mass_accuracy_median_ppm": float(np.median(ppm_errors)) if ppm_errors else float("nan"),
    }


def peak_metrics_vs_ranges(
    detected_mz: np.ndarray,
    ranges: pd.DataFrame,
    mz_values: np.ndarray,
    *,
    min_range_ions: int = 200,
) -> dict[str, object]:
    """Score detected peaks against an expert range file.

    Recall: fraction of reference ranges (holding at least min_range_ions raw
    ions) containing at least one detected peak. In-reference fraction: share
    of detected peaks inside any reference range — real spectra contain
    unranged minor peaks, so this is a coverage indicator, not a strict
    precision.
    """
    detected = np.sort(np.asarray(detected_mz, dtype=np.float64))
    mz_sorted = np.sort(np.asarray(mz_values, dtype=np.float64))
    if ranges.empty:
        return {}
    lo = ranges["mz_lo_da"].to_numpy(dtype=np.float64)
    hi = ranges["mz_hi_da"].to_numpy(dtype=np.float64)
    ions = (
        np.searchsorted(mz_sorted, hi, side="right")
        - np.searchsorted(mz_sorted, lo, side="left")
    )
    populated = ions >= int(min_range_ions)
    if not populated.any():
        return {}
    hit = np.zeros(lo.size, dtype=bool)
    for idx in range(lo.size):
        left = np.searchsorted(detected, lo[idx], side="left")
        right = np.searchsorted(detected, hi[idx], side="right")
        hit[idx] = right > left
    in_any = np.zeros(detected.size, dtype=bool)
    for idx in range(lo.size):
        in_any |= (detected >= lo[idx]) & (detected <= hi[idx])
    ion_weights = ions.astype(np.float64)
    populated_hit = hit[populated]
    return {
        "n_reference_ranges": int(populated.sum()),
        "n_detected_peaks": int(detected.size),
        "range_recall": float(populated_hit.mean()),
        "range_recall_ion_weighted": float(
            np.sum(ion_weights[populated & hit]) / max(np.sum(ion_weights[populated]), 1.0)
        ),
        "in_reference_fraction": float(in_any.mean()) if detected.size else float("nan"),
    }
