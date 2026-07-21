from __future__ import annotations

import numpy as np
import pandas as pd


def mass_scale_diagnostics(assignments: pd.DataFrame) -> dict[str, object]:
    """Summarize signed mass residuals of confident top assignments.

    A nonzero median indicates a residual calibration offset in the instrument's
    exported m/z scale; a wide spread indicates the assignment tolerances are
    doing more work than they should. POS files carry no voltage or detector
    information, so this cross-check against reference isotope masses is the
    only calibration audit available at this stage.
    """
    if assignments.empty:
        return {"n_high_confidence": 0}
    top = assignments[
        (assignments["rank"] == 1)
        & (assignments["confidence"] == "high")
        & (assignments["species_label"] != "unassigned")
    ].copy()
    top = top[top["mass_error_da"].notna() & top["candidate_mz_da"].notna()]
    if top.empty:
        return {"n_high_confidence": 0}
    error_da = top["mass_error_da"].to_numpy(dtype=np.float64)
    candidate_mz = np.maximum(top["candidate_mz_da"].to_numpy(dtype=np.float64), 1.0e-9)
    ppm = 1.0e6 * error_da / candidate_mz
    return {
        "n_high_confidence": int(top.shape[0]),
        "median_error_da": float(np.median(error_da)),
        "median_abs_error_da": float(np.median(np.abs(error_da))),
        "p95_abs_error_da": float(np.percentile(np.abs(error_da), 95.0)),
        "median_ppm": float(np.median(ppm)),
        "mad_ppm": float(np.median(np.abs(ppm - np.median(ppm)))),
    }


def spatial_composition_stability(
    *,
    x_nm: np.ndarray,
    y_nm: np.ndarray,
    z_nm: np.ndarray,
    m_over_z_da: np.ndarray,
    peaks: pd.DataFrame,
    assignments: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Check whether the reported composition is stable through the analyzed volume.

    The global mass spectrum silently averages over any phase or layer structure
    in the specimen. This diagnostic splits events into equal-count depth slices,
    re-counts each confidently assigned peak window per slice, and reports the
    per-element composition drift. Large drift means single-number compositions
    and isotope ratios should be treated as mixtures, not bulk values.
    """
    analysis_cfg = config["analysis"]
    n_slices = int(analysis_cfg.get("spatial_stability_slices", 5))
    min_events = int(analysis_cfg.get("spatial_stability_min_events_per_slice", 20000))
    drift_threshold = float(analysis_cfg.get("spatial_stability_max_drift_at_pct", 3.0))
    empty = pd.DataFrame(), {"enabled": False}
    if not bool(analysis_cfg.get("spatial_stability_enabled", True)):
        return empty
    if peaks.empty or assignments.empty:
        return empty
    z = np.asarray(z_nm, dtype=np.float64)
    mz = np.asarray(m_over_z_da, dtype=np.float64)
    finite = np.isfinite(z) & np.isfinite(mz)
    z = z[finite]
    mz = mz[finite]
    if z.size < n_slices * min_events:
        n_slices = max(2, int(z.size // max(min_events, 1)))
    if z.size < 2 * max(min_events, 1) or n_slices < 2:
        return pd.DataFrame(), {"enabled": True, "skipped": "too few events for slicing"}
    top = assignments[
        (assignments["rank"] == 1)
        & (assignments["species_label"] != "unassigned")
        & (assignments["confidence"].isin(["high", "ambiguous"]))
    ].copy()
    if top.empty:
        return pd.DataFrame(), {"enabled": True, "skipped": "no assigned peaks"}
    peaks_indexed = peaks.set_index("peak_id")
    windows: list[tuple[float, float, dict[str, float]]] = []
    for _, row in top.iterrows():
        peak_id = str(row["peak_id"])
        if peak_id not in peaks_indexed.index:
            continue
        peak = peaks_indexed.loc[peak_id]
        left = float(peak.get("integration_left_da", np.nan))
        right = float(peak.get("integration_right_da", np.nan))
        if not (np.isfinite(left) and np.isfinite(right)) or right <= left:
            center = float(peak["peak_mz_da"])
            half = max(float(peak.get("fwhm_da", 0.05)), 0.05)
            left, right = center - half, center + half
        element_counts = row.get("element_counts")
        if not isinstance(element_counts, dict) or not element_counts:
            continue
        windows.append((left, right, {str(k): float(v) for k, v in element_counts.items()}))
    if not windows:
        return pd.DataFrame(), {"enabled": True, "skipped": "no usable windows"}
    slice_edges = np.quantile(z, np.linspace(0.0, 1.0, n_slices + 1))
    slice_edges[0] -= 1.0e-6
    slice_edges[-1] += 1.0e-6
    slice_index = np.clip(np.searchsorted(slice_edges, z, side="right") - 1, 0, n_slices - 1)
    order = np.argsort(mz)
    mz_sorted = mz[order]
    slice_sorted = slice_index[order]
    rows: list[dict[str, object]] = []
    element_totals_per_slice: list[dict[str, float]] = [dict() for _ in range(n_slices)]
    for left, right, element_counts in windows:
        lo = int(np.searchsorted(mz_sorted, left, side="left"))
        hi = int(np.searchsorted(mz_sorted, right, side="right"))
        if hi <= lo:
            continue
        slice_counts = np.bincount(slice_sorted[lo:hi], minlength=n_slices).astype(np.float64)
        for element, atom_count in element_counts.items():
            for slice_id in range(n_slices):
                element_totals_per_slice[slice_id][element] = (
                    element_totals_per_slice[slice_id].get(element, 0.0)
                    + slice_counts[slice_id] * atom_count
                )
    for slice_id in range(n_slices):
        totals = element_totals_per_slice[slice_id]
        denom = max(sum(totals.values()), 1.0)
        z_lo = float(slice_edges[slice_id])
        z_hi = float(slice_edges[slice_id + 1])
        for element, count in sorted(totals.items()):
            rows.append(
                {
                    "slice": slice_id,
                    "z_min_nm": z_lo,
                    "z_max_nm": z_hi,
                    "element": element,
                    "atom_counts": float(count),
                    "atomic_percent": 100.0 * float(count) / denom,
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame, {"enabled": True, "skipped": "no events in windows"}
    drift_rows = []
    for element, group in frame.groupby("element"):
        values = group["atomic_percent"].to_numpy(dtype=np.float64)
        median_pct = float(np.median(values))
        if median_pct < 0.5:
            continue
        drift_rows.append(
            {
                "element": str(element),
                "median_at_pct": median_pct,
                "drift_at_pct": float(values.max() - values.min()),
            }
        )
    summary: dict[str, object] = {"enabled": True, "n_slices": int(n_slices)}
    if drift_rows:
        drift_frame = pd.DataFrame(drift_rows).sort_values("drift_at_pct", ascending=False)
        worst = drift_frame.iloc[0]
        summary |= {
            "max_drift_element": str(worst["element"]),
            "max_drift_at_pct": float(worst["drift_at_pct"]),
            "spatially_uniform": bool(float(worst["drift_at_pct"]) <= drift_threshold),
            "drift_threshold_at_pct": drift_threshold,
        }
    return frame, summary
