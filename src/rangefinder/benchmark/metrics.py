"""Benchmark metrics: composition recovery, peak detection, mass accuracy.

Composition truths are dicts of atomic percent. Peak truths come in two
flavors: exact isotopologue lines (synthetic data, where every emitted line
position and intensity is known) and reference range intervals (public
datasets shipped with an expert range file).
"""
from __future__ import annotations

import ast
import json

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


def _formula_dict(value: object) -> dict[str, int]:
    if isinstance(value, dict):
        parsed = value
    elif isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                return {}
    else:
        return {}
    if not isinstance(parsed, dict):
        return {}
    formula: dict[str, int] = {}
    for element, count in parsed.items():
        try:
            integer_count = int(count)
        except (TypeError, ValueError):
            continue
        if integer_count > 0:
            formula[str(element)] = integer_count
    return formula


def assignment_metrics_vs_ranges(
    peaks: pd.DataFrame,
    assignments: pd.DataFrame,
    ranges: pd.DataFrame,
    mz_values: np.ndarray,
    *,
    min_range_ions: int = 200,
) -> dict[str, object]:
    """Score top species formulae against populated expert ranges.

    Detection and identification are kept separate: coverage reports how many
    expert ranges contain a detected peak, while formula accuracy is evaluated
    among covered ranges. The ion-weighted score prevents dozens of trace
    ranges from obscuring correctness on the material's dominant signal.
    """
    if peaks.empty or assignments.empty or ranges.empty:
        return {}
    mz_sorted = np.sort(np.asarray(mz_values, dtype=np.float64))
    top = assignments.loc[assignments["rank"] == 1].copy()
    if top.empty:
        return {}
    top["predicted_formula"] = top["element_counts"].map(_formula_dict)
    peak_frame = peaks[["peak_id", "peak_mz_da"]].copy()
    if "integrated_area" in peaks.columns:
        peak_frame["integrated_area"] = peaks["integrated_area"].fillna(0.0)
    else:
        peak_frame["integrated_area"] = 0.0
    peak_frame = peak_frame.merge(
        top[["peak_id", "predicted_formula"]],
        on="peak_id",
        how="left",
    )
    rows: list[dict[str, object]] = []
    for _, reference in ranges.iterrows():
        lo = float(reference["mz_lo_da"])
        hi = float(reference["mz_hi_da"])
        ion_count = int(
            np.searchsorted(mz_sorted, hi, side="right")
            - np.searchsorted(mz_sorted, lo, side="left")
        )
        if ion_count < int(min_range_ions):
            continue
        candidates = peak_frame[
            (peak_frame["peak_mz_da"] >= lo) & (peak_frame["peak_mz_da"] <= hi)
        ]
        if candidates.empty:
            rows.append({"ions": ion_count, "covered": False, "exact": False, "elements": False})
            continue
        selected = candidates.sort_values(
            ["integrated_area", "peak_mz_da"], ascending=[False, True]
        ).iloc[0]
        expected = _formula_dict(reference["formula"])
        predicted = _formula_dict(selected["predicted_formula"])
        rows.append(
            {
                "ions": ion_count,
                "covered": True,
                "exact": predicted == expected,
                "elements": bool(set(predicted) & set(expected)),
            }
        )
    if not rows:
        return {}
    covered = [row for row in rows if bool(row["covered"])]
    total_ions = max(sum(int(row["ions"]) for row in rows), 1)
    return {
        "n_reference_assignments": len(rows),
        "n_reference_assignments_covered": len(covered),
        "assignment_range_coverage": len(covered) / len(rows),
        "assignment_formula_accuracy": (
            sum(bool(row["exact"]) for row in covered) / len(covered)
            if covered
            else float("nan")
        ),
        "assignment_formula_recall": (
            sum(bool(row["exact"]) for row in rows) / len(rows)
        ),
        "assignment_formula_accuracy_ion_weighted": (
            sum(int(row["ions"]) for row in rows if bool(row["exact"])) / total_ions
        ),
        "assignment_element_hit_rate": (
            sum(bool(row["elements"]) for row in covered) / len(covered)
            if covered
            else float("nan")
        ),
        "assignment_element_recall": (
            sum(bool(row["elements"]) for row in rows) / len(rows)
        ),
        "assignment_element_hit_rate_ion_weighted": (
            sum(int(row["ions"]) for row in rows if bool(row["elements"])) / total_ions
        ),
    }
