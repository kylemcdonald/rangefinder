"""Adaptive peak boundaries and range-quality diagnostics.

The equal-error criterion implemented here is a clean-room implementation of
the general decision rule described by Felfer's Atom-Probe-Toolbox: move a
boundary away from a peak until the estimated signal left outside the range is
balanced by the background admitted inside it.  No source code from that GPL
project is included here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class _SideBoundary:
    mass_da: float
    included_signal: float
    total_signal: float
    missed_signal: float


def _inverse_sqrt_background(
    centers: np.ndarray,
    counts: np.ndarray,
    peak_mz: np.ndarray,
    *,
    exclusion_half_width_da: float,
    lower_fraction: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Fit the constant-TOF continuum ``A/sqrt(m)`` through inter-peak bins.

    The fit is deliberately conservative: peak neighborhoods are excluded and
    only the lower part of the transformed ``counts * sqrt(m)`` distribution
    contributes.  The model is capped below 1 Da where the asymptotic change of
    variables is not physically valid.
    """
    safe_mass = np.maximum(np.asarray(centers, dtype=np.float64), 1.0)
    raw = np.asarray(counts, dtype=np.float64)
    eligible = np.isfinite(raw) & np.isfinite(safe_mass) & (centers > 1.0)
    for mz in np.asarray(peak_mz, dtype=np.float64):
        eligible &= np.abs(centers - mz) > float(exclusion_half_width_da)

    if int(eligible.sum()) < 10:
        eligible = np.isfinite(raw) & (centers > 1.0)
    transformed = raw[eligible] * np.sqrt(safe_mass[eligible])
    if transformed.size == 0:
        background = np.zeros_like(raw)
        return background, {
            "coefficient": 0.0,
            "relative_mad": float("nan"),
            "poisson_deviance_per_bin": float("inf"),
            "fit_bin_count": 0,
        }

    quantile = float(np.clip(lower_fraction, 0.05, 1.0))
    cutoff = float(np.quantile(transformed, quantile))
    fit_mask = eligible.copy()
    fit_mask[eligible] = transformed <= cutoff
    if int(fit_mask.sum()) < 3:
        fit_mask = eligible

    predictor = 1.0 / np.sqrt(safe_mass[fit_mask])
    denominator = float(np.dot(predictor, predictor))
    coefficient = (
        max(0.0, float(np.dot(raw[fit_mask], predictor)) / denominator)
        if denominator > 0.0
        else 0.0
    )
    background = coefficient / np.sqrt(safe_mass)

    transformed_fit = raw[fit_mask] * np.sqrt(safe_mass[fit_mask])
    median = float(np.median(transformed_fit)) if transformed_fit.size else 0.0
    mad = (
        float(np.median(np.abs(transformed_fit - median)))
        if transformed_fit.size
        else float("nan")
    )
    relative_mad = mad / max(abs(median), 1.0)
    expected = np.clip(background[eligible], 1.0e-12, None)
    observed = np.clip(raw[eligible], 0.0, None)
    positive = observed > 0.0
    deviance = np.empty_like(observed)
    deviance[positive] = 2.0 * (
        observed[positive] * np.log(observed[positive] / expected[positive])
        - (observed[positive] - expected[positive])
    )
    deviance[~positive] = 2.0 * expected[~positive]
    poisson_deviance_per_bin = float(np.mean(np.clip(deviance, 0.0, None)))
    return background, {
        "coefficient": coefficient,
        "relative_mad": relative_mad,
        "poisson_deviance_per_bin": poisson_deviance_per_bin,
        "fit_bin_count": int(fit_mask.sum()),
    }


def _side_boundary(
    centers: np.ndarray,
    counts: np.ndarray,
    background: np.ndarray,
    indices: np.ndarray,
    *,
    purity_factor: float,
) -> _SideBoundary:
    """Find one equal-error crossing while walking outward from an apex."""
    if indices.size == 0:
        return _SideBoundary(float("nan"), 0.0, 0.0, 0.0)

    signal = np.clip(counts[indices] - background[indices], 0.0, None)
    # A single peak's contribution cannot increase away from its apex.  The
    # cumulative minimum removes noise bumps and prevents a neighboring peak
    # from pulling the boundary through the intervening valley.
    monotone_signal = np.minimum.accumulate(signal)
    total_signal = float(monotone_signal.sum())
    if total_signal <= 0.0:
        return _SideBoundary(float(centers[indices[0]]), 0.0, 0.0, 0.0)

    cumulative_signal = np.cumsum(monotone_signal)
    cumulative_background = np.cumsum(np.clip(background[indices], 0.0, None))
    missed = total_signal - cumulative_signal
    balance = missed - max(float(purity_factor), 0.0) * cumulative_background
    crossings = np.flatnonzero(balance <= 0.0)
    if crossings.size:
        crossing = int(crossings[0])
    else:
        nonzero = np.flatnonzero(monotone_signal > 0.0)
        crossing = int(nonzero[-1]) if nonzero.size else 0

    boundary_mass = float(centers[indices[crossing]])
    included_signal = float(cumulative_signal[crossing])
    missed_signal = float(missed[crossing])
    if crossing > 0 and balance[crossing - 1] > 0.0 and balance[crossing] <= 0.0:
        previous = float(balance[crossing - 1])
        current = float(balance[crossing])
        fraction = previous / max(previous - current, np.finfo(float).eps)
        previous_mass = float(centers[indices[crossing - 1]])
        boundary_mass = previous_mass + fraction * (boundary_mass - previous_mass)
        included_signal = float(
            cumulative_signal[crossing - 1]
            + fraction * monotone_signal[crossing]
        )
        missed_signal = max(0.0, total_signal - included_signal)

    return _SideBoundary(
        boundary_mass,
        included_signal,
        total_signal,
        missed_signal,
    )


def equal_error_ranges(
    peaks: pd.DataFrame,
    centers: np.ndarray,
    counts: np.ndarray,
    *,
    purity_factor: float = 1.0,
    background_exclusion_half_width_da: float = 0.30,
    background_lower_fraction: float = 0.50,
    snap_half_width_da: float = 0.05,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return one non-overlapping adaptive range and diagnostics per peak."""
    if peaks.empty:
        return peaks.copy(), {
            "background_coefficient": 0.0,
            "background_relative_mad": float("nan"),
            "range_count": 0,
        }
    centers = np.asarray(centers, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    if centers.ndim != 1 or counts.shape != centers.shape or centers.size < 2:
        raise ValueError("centers and counts must be aligned one-dimensional arrays")

    ordered = peaks.sort_values("peak_mz_da").copy()
    peak_mz = ordered["peak_mz_da"].to_numpy(dtype=np.float64)
    background, background_diagnostics = _inverse_sqrt_background(
        centers,
        counts,
        peak_mz,
        exclusion_half_width_da=background_exclusion_half_width_da,
        lower_fraction=background_lower_fraction,
    )

    bin_width = float(np.median(np.diff(centers)))
    snap_bins = max(1, int(round(float(snap_half_width_da) / max(bin_width, 1.0e-12))))
    apex_indices: list[int] = []
    for mz in peak_mz:
        nominal = int(np.clip(np.searchsorted(centers, mz), 0, centers.size - 1))
        left = max(0, nominal - snap_bins)
        right = min(centers.size, nominal + snap_bins + 1)
        apex_indices.append(left + int(np.argmax(counts[left:right])))
    apex = np.asarray(apex_indices, dtype=int)

    # If two reported peaks snap to the same bin, preserve their mass order for
    # partitioning.  This is rare, but avoiding a reversed interval keeps the
    # diagnostic path total and deterministic.
    apex = np.maximum.accumulate(apex)
    left_limit = np.zeros(apex.size, dtype=int)
    right_limit = np.full(apex.size, centers.size - 1, dtype=int)
    for idx in range(apex.size - 1):
        first = int(apex[idx])
        second = int(apex[idx + 1])
        if second <= first:
            valley = first
        else:
            segment = counts[first : second + 1]
            valley = first + int(np.argmin(segment))
        right_limit[idx] = valley
        left_limit[idx + 1] = valley

    rows: list[dict[str, object]] = []
    for position, (row_index, row) in enumerate(ordered.iterrows()):
        peak_index = int(apex[position])
        left_indices = np.arange(peak_index, int(left_limit[position]) - 1, -1, dtype=int)
        right_indices = np.arange(peak_index, int(right_limit[position]) + 1, dtype=int)
        left = _side_boundary(
            centers,
            counts,
            background,
            left_indices,
            purity_factor=purity_factor,
        )
        right = _side_boundary(
            centers,
            counts,
            background,
            right_indices,
            purity_factor=purity_factor,
        )
        range_left = min(left.mass_da, right.mass_da)
        range_right = max(left.mass_da, right.mass_da)
        in_range = (centers >= range_left) & (centers <= range_right)
        observed_counts = float(counts[in_range].sum())
        background_counts = float(background[in_range].sum())
        signal_counts = max(0.0, observed_counts - background_counts)

        apex_signal = max(0.0, float(counts[peak_index] - background[peak_index]))
        total_monotone_signal = max(
            0.0,
            left.total_signal + right.total_signal - apex_signal,
        )
        included_monotone_signal = max(
            0.0,
            left.included_signal + right.included_signal - apex_signal,
        )
        missed_signal = max(0.0, total_monotone_signal - included_monotone_signal)
        purity = signal_counts / observed_counts if observed_counts > 0.0 else 0.0
        recovery = (
            min(1.0, included_monotone_signal / total_monotone_signal)
            if total_monotone_signal > 0.0
            else 0.0
        )
        rows.append(
            {
                "_row_index": row_index,
                "eer_integration_left_da": float(range_left),
                "eer_integration_right_da": float(range_right),
                "eer_observed_counts": observed_counts,
                "eer_background_counts": background_counts,
                "eer_integrated_area": signal_counts,
                "eer_missed_signal_counts": missed_signal,
                "eer_purity": float(np.clip(purity, 0.0, 1.0)),
                "eer_recovery": float(np.clip(recovery, 0.0, 1.0)),
                "eer_counting_std": float(np.sqrt(max(observed_counts + background_counts, 0.0))),
                "eer_apex_mz_da": float(centers[peak_index]),
            }
        )

    diagnostics_frame = pd.DataFrame(rows).set_index("_row_index")
    result = peaks.copy().join(diagnostics_frame, how="left")
    diagnostics: dict[str, object] = {
        "background_model": "constant TOF continuum transformed as A/sqrt(max(m, 1 Da))",
        "background_coefficient": float(background_diagnostics["coefficient"]),
        "background_relative_mad": float(background_diagnostics["relative_mad"]),
        "background_poisson_deviance_per_bin": float(
            background_diagnostics["poisson_deviance_per_bin"]
        ),
        "background_fit_bin_count": int(background_diagnostics["fit_bin_count"]),
        "purity_factor": float(purity_factor),
        "range_count": int(result.shape[0]),
        "median_purity": float(result["eer_purity"].median()),
        "median_recovery": float(result["eer_recovery"].median()),
    }
    return result, diagnostics


def apply_adaptive_ranging(
    peaks: pd.DataFrame,
    centers: np.ndarray,
    counts: np.ndarray,
    *,
    config: dict,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Annotate legacy ranges with EER estimates and select the configured area.

    Modes are ``off`` (legacy output only), ``shadow`` (diagnostics only),
    ``eer`` (use every valid EER estimate), and ``hybrid`` (quality-gated EER).
    """
    analysis_cfg = config["analysis"]
    mode = str(analysis_cfg.get("adaptive_ranging_mode", "off")).lower()
    if peaks.empty or mode == "off":
        result = peaks.copy()
        if not result.empty:
            result["integration_method"] = np.where(
                result.get("fit_peak_area", pd.Series(np.nan, index=result.index)).notna(),
                "local_mixture",
                "fixed_sideband",
            )
        return result, {"mode": mode, "selected_peak_count": 0, "range_count": int(result.shape[0])}

    result = peaks.copy()
    result["legacy_integration_left_da"] = result["integration_left_da"]
    result["legacy_integration_right_da"] = result["integration_right_da"]
    result["legacy_integrated_area"] = result["integrated_area"]
    result, diagnostics = equal_error_ranges(
        result,
        centers,
        counts,
        purity_factor=float(analysis_cfg.get("eer_purity_factor", 1.0)),
        background_exclusion_half_width_da=float(
            analysis_cfg.get("eer_background_exclusion_half_width_da", 0.30)
        ),
        background_lower_fraction=float(analysis_cfg.get("eer_background_lower_fraction", 0.50)),
        snap_half_width_da=float(analysis_cfg.get("eer_snap_half_width_da", 0.05)),
    )
    valid = (
        np.isfinite(result["eer_integrated_area"])
        & (result["eer_integrated_area"] > 0.0)
        & (result["eer_integration_right_da"] > result["eer_integration_left_da"])
    )
    if mode == "eer":
        selected = valid
    elif mode == "hybrid":
        legacy = result["legacy_integrated_area"].fillna(0.0).to_numpy(dtype=np.float64)
        adaptive = result["eer_integrated_area"].fillna(0.0).to_numpy(dtype=np.float64)
        ratio = adaptive / np.maximum(legacy, 1.0)
        selected = (
            valid
            & (result["eer_purity"] >= float(analysis_cfg.get("eer_hybrid_min_purity", 0.50)))
            & (result["eer_recovery"] >= float(analysis_cfg.get("eer_hybrid_min_recovery", 0.50)))
            & (ratio >= float(analysis_cfg.get("eer_hybrid_min_area_ratio", 0.25)))
            & (ratio <= float(analysis_cfg.get("eer_hybrid_max_area_ratio", 4.0)))
        )
        if bool(analysis_cfg.get("eer_hybrid_preserve_local_mixture", True)):
            selected &= result.get("fit_peak_area", pd.Series(np.nan, index=result.index)).isna()
        max_background_mad = float(analysis_cfg.get("eer_hybrid_max_background_relative_mad", np.inf))
        if float(diagnostics.get("background_relative_mad", np.inf)) > max_background_mad:
            selected &= False
    else:
        selected = pd.Series(False, index=result.index)

    selected = pd.Series(np.asarray(selected, dtype=bool), index=result.index)
    result["range_placement_systematic_counts"] = (
        result["legacy_integrated_area"] - result["eer_integrated_area"]
    ).abs()
    result["range_placement_systematic_fraction"] = (
        result["range_placement_systematic_counts"]
        / result["legacy_integrated_area"].abs().clip(lower=1.0)
    )
    result["adaptive_range_selected"] = selected
    result["integration_method"] = np.where(
        selected,
        "equal_error",
        np.where(
            result.get("fit_peak_area", pd.Series(np.nan, index=result.index)).notna(),
            "local_mixture",
            "fixed_sideband",
        ),
    )
    result.loc[selected, "integration_left_da"] = result.loc[
        selected, "eer_integration_left_da"
    ]
    result.loc[selected, "integration_right_da"] = result.loc[
        selected, "eer_integration_right_da"
    ]
    result.loc[selected, "integrated_area"] = result.loc[selected, "eer_integrated_area"]
    diagnostics |= {
        "mode": mode,
        "selected_peak_count": int(selected.sum()),
        "selected_area_fraction": float(
            result.loc[selected, "integrated_area"].sum()
            / max(float(result["integrated_area"].sum()), 1.0)
        ),
        "median_range_placement_systematic_fraction": float(
            result["range_placement_systematic_fraction"].median()
        ),
    }
    return result, diagnostics
