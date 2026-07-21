from __future__ import annotations

import json
from collections import Counter
from itertools import combinations_with_replacement

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import find_peaks, peak_widths, savgol_filter
from scipy.special import softmax


def choose_bin_width(
    m_over_z_da: np.ndarray,
    candidates: list[float],
    *,
    min_mz: float,
    max_mz: float,
) -> float:
    window = np.asarray(m_over_z_da, dtype=np.float64)
    window = window[np.isfinite(window)]
    window = window[(window >= min_mz) & (window <= max_mz)]
    if window.size < 3:
        return min(candidates)
    q25, q75 = np.quantile(window, [0.25, 0.75])
    iqr = max(q75 - q25, 1e-6)
    fd_width = 2.0 * iqr / np.cbrt(window.size)
    return min(candidates, key=lambda candidate: abs(candidate - fd_width))


def detection_bin_width(
    m_over_z_da: np.ndarray,
    candidates: list[float],
    *,
    min_mz: float,
    max_mz: float,
    max_detection_bin_width_da: float,
) -> float:
    chosen = choose_bin_width(
        m_over_z_da,
        candidates,
        min_mz=min_mz,
        max_mz=max_mz,
    )
    return float(min(chosen, max_detection_bin_width_da))


def build_histogram(
    m_over_z_da: np.ndarray,
    *,
    min_mz: float,
    max_mz: float,
    bin_width_da: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.arange(min_mz, max_mz + bin_width_da, bin_width_da, dtype=np.float64)
    counts, edges = np.histogram(m_over_z_da, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return centers, counts.astype(np.float64), edges


def default_prominence(counts: np.ndarray, quantile: float, minimum: float) -> float:
    nonzero = counts[counts > 0]
    if nonzero.size == 0:
        return minimum
    return float(max(np.quantile(nonzero, quantile), minimum))


def smooth_counts(counts: np.ndarray, window_length: int, polyorder: int) -> np.ndarray:
    if window_length < 3 or counts.size < window_length:
        return counts.copy()
    if window_length % 2 == 0:
        window_length += 1
    return savgol_filter(counts, window_length=window_length, polyorder=polyorder, mode="interp")


def detect_peaks_from_histogram(
    centers: np.ndarray,
    counts: np.ndarray,
    *,
    prominence: float,
    distance_bins: int,
    rel_height: float = 0.5,
    smooth_window_bins: int = 0,
    smooth_polyorder: int = 2,
    method_label: str,
    max_fwhm_da: float | None = None,
) -> pd.DataFrame:
    working = counts
    if smooth_window_bins and smooth_window_bins > 2:
        working = smooth_counts(counts, smooth_window_bins, smooth_polyorder)
    peak_indices, properties = find_peaks(
        working,
        prominence=prominence,
        distance=distance_bins,
        height=0,
    )
    widths = peak_widths(working, peak_indices, rel_height=rel_height)
    rows: list[dict[str, float | int | str]] = []
    for idx, peak_idx in enumerate(peak_indices):
        fwhm_da = float(
            np.interp(widths[3][idx], np.arange(centers.size), centers)
            - np.interp(widths[2][idx], np.arange(centers.size), centers)
        )
        if max_fwhm_da is not None:
            fwhm_da = min(fwhm_da, float(max_fwhm_da))
        rows.append(
            {
                "peak_id": f"{method_label}-peak-{idx + 1:04d}",
                "method": method_label,
                "peak_index": int(peak_idx),
                "peak_mz_da": float(centers[peak_idx]),
                "peak_height": float(counts[peak_idx]),
                "smoothed_height": float(working[peak_idx]),
                "prominence": float(properties["prominences"][idx]),
                "left_fwhm_mz_da": float(np.interp(widths[2][idx], np.arange(centers.size), centers)),
                "right_fwhm_mz_da": float(np.interp(widths[3][idx], np.arange(centers.size), centers)),
                "fwhm_da": fwhm_da,
            }
        )
    return pd.DataFrame(rows).sort_values("peak_mz_da", ignore_index=True)


def _log_peakiness_candidate_table(
    centers: np.ndarray,
    counts: np.ndarray,
    *,
    method_label: str,
    smooth_sigma_bins: float,
    baseline_window_bins: int,
    min_excess: float,
    min_prominence: float,
    min_count: float,
    min_distance_bins: int,
    min_detection_percentile: float,
    max_detection_percentile: float,
    tail_strength_center: float,
    tail_strength_scale: float,
    max_fwhm_da: float | None = None,
) -> tuple[pd.DataFrame, float]:
    if centers.size == 0 or counts.size == 0:
        return pd.DataFrame(), float("inf")
    window_bins = max(int(baseline_window_bins), 3)
    if window_bins % 2 == 0:
        window_bins += 1
    log_counts = np.log10(np.maximum(counts.astype(np.float64), 0.0) + 1.0)
    smoothed = gaussian_filter1d(log_counts, sigma=float(smooth_sigma_bins), mode="nearest")
    baseline = median_filter(smoothed, size=window_bins, mode="nearest")
    excess = smoothed - baseline
    median_excess = median_filter(excess, size=window_bins, mode="nearest")
    residual_abs = np.abs(excess - median_excess)
    local_mad = median_filter(residual_abs, size=window_bins, mode="nearest")
    local_sigma = np.maximum(1.4826 * local_mad, 1.0e-4)
    peak_indices, properties = find_peaks(
        excess,
        height=0.0,
        prominence=0.0,
        distance=max(int(min_distance_bins), 1),
    )
    candidate_rows: list[dict[str, float | int | str]] = []
    refined_indices: list[int] = []
    score_values: list[float] = []
    seen_indices: set[int] = set()
    for peak_idx, prominence in zip(peak_indices, properties.get("prominences", []), strict=False):
        lo = max(0, int(peak_idx) - 2)
        hi = min(len(counts), int(peak_idx) + 3)
        refined_idx = lo + int(np.argmax(counts[lo:hi]))
        if refined_idx in seen_indices:
            continue
        seen_indices.add(refined_idx)
        peak_count = float(counts[refined_idx])
        if peak_count < float(min_count):
            continue
        excess_value = float(excess[int(peak_idx)])
        prominence_value = float(prominence)
        if excess_value < float(min_excess) or prominence_value < float(min_prominence):
            continue
        sigma_value = float(local_sigma[int(peak_idx)])
        peakiness = prominence_value / sigma_value if sigma_value > 0 else float("inf")
        candidate_rows.append(
            {
                "peak_id": "",
                "method": method_label,
                "peak_index": int(refined_idx),
                "peak_mz_da": float(centers[refined_idx]),
                "peak_height": peak_count,
                "smoothed_height": float(counts[refined_idx]),
                "prominence": prominence_value,
                "peakiness": peakiness,
                "excess": excess_value,
                "noise_sigma": sigma_value,
            }
        )
        refined_indices.append(int(peak_idx))
        score_values.append(float(peakiness))
    if not candidate_rows:
        return pd.DataFrame(), float("inf")
    peakiness_values = np.asarray(score_values, dtype=np.float64)
    p90 = float(np.percentile(peakiness_values, 90.0))
    tail_quality = 1.0 / (1.0 + np.exp(-(p90 - float(tail_strength_center)) / float(tail_strength_scale)))
    percentile = float(max_detection_percentile) - (
        float(max_detection_percentile) - float(min_detection_percentile)
    ) * tail_quality
    cutoff = float(np.percentile(peakiness_values, percentile))
    widths = peak_widths(excess, np.asarray(refined_indices, dtype=int), rel_height=0.5)
    rows: list[dict[str, float | int | str]] = []
    for output_idx, candidate in enumerate(candidate_rows, start=1):
        candidate = dict(candidate)
        width_pos = output_idx - 1
        fwhm_da = float(
            np.interp(widths[3][width_pos], np.arange(centers.size), centers)
            - np.interp(widths[2][width_pos], np.arange(centers.size), centers)
        )
        if max_fwhm_da is not None:
            fwhm_da = min(fwhm_da, float(max_fwhm_da))
        candidate["peak_id"] = f"{method_label}-peak-{output_idx:04d}"
        candidate["left_fwhm_mz_da"] = float(np.interp(widths[2][width_pos], np.arange(centers.size), centers))
        candidate["right_fwhm_mz_da"] = float(np.interp(widths[3][width_pos], np.arange(centers.size), centers))
        candidate["fwhm_da"] = fwhm_da
        rows.append(candidate)
    return pd.DataFrame(rows).sort_values("peak_mz_da", ignore_index=True), cutoff


def detect_peaks_log_peakiness_candidates(
    centers: np.ndarray,
    counts: np.ndarray,
    *,
    method_label: str,
    smooth_sigma_bins: float,
    baseline_window_bins: int,
    min_excess: float,
    min_prominence: float,
    min_count: float,
    min_distance_bins: int,
    min_detection_percentile: float,
    max_detection_percentile: float,
    tail_strength_center: float,
    tail_strength_scale: float,
    max_fwhm_da: float | None = None,
) -> pd.DataFrame:
    candidates, _cutoff = _log_peakiness_candidate_table(
        centers,
        counts,
        method_label=method_label,
        smooth_sigma_bins=smooth_sigma_bins,
        baseline_window_bins=baseline_window_bins,
        min_excess=min_excess,
        min_prominence=min_prominence,
        min_count=min_count,
        min_distance_bins=min_distance_bins,
        min_detection_percentile=min_detection_percentile,
        max_detection_percentile=max_detection_percentile,
        tail_strength_center=tail_strength_center,
        tail_strength_scale=tail_strength_scale,
        max_fwhm_da=max_fwhm_da,
    )
    return candidates


def detect_peaks_log_peakiness(
    centers: np.ndarray,
    counts: np.ndarray,
    *,
    method_label: str,
    smooth_sigma_bins: float,
    baseline_window_bins: int,
    min_excess: float,
    min_prominence: float,
    min_count: float,
    min_distance_bins: int,
    min_detection_percentile: float,
    max_detection_percentile: float,
    tail_strength_center: float,
    tail_strength_scale: float,
    max_fwhm_da: float | None = None,
) -> pd.DataFrame:
    candidates, cutoff = _log_peakiness_candidate_table(
        centers,
        counts,
        method_label=method_label,
        smooth_sigma_bins=smooth_sigma_bins,
        baseline_window_bins=baseline_window_bins,
        min_excess=min_excess,
        min_prominence=min_prominence,
        min_count=min_count,
        min_distance_bins=min_distance_bins,
        min_detection_percentile=min_detection_percentile,
        max_detection_percentile=max_detection_percentile,
        tail_strength_center=tail_strength_center,
        tail_strength_scale=tail_strength_scale,
        max_fwhm_da=max_fwhm_da,
    )
    if candidates.empty:
        return candidates
    return candidates.loc[candidates["peakiness"].fillna(0.0) >= float(cutoff)].reset_index(drop=True)


def refine_peak_centers_parabolic(
    peaks: pd.DataFrame,
    centers: np.ndarray,
    counts: np.ndarray,
) -> pd.DataFrame:
    if peaks.empty:
        return peaks.copy()
    refined = peaks.copy()
    refined_mz = []
    refined_height = []
    refined_index = []
    for peak_idx in refined["peak_index"].astype(int):
        if peak_idx <= 0 or peak_idx >= counts.size - 1:
            refined_index.append(float(peak_idx))
            refined_mz.append(float(centers[peak_idx]))
            refined_height.append(float(counts[peak_idx]))
            continue
        y_left = float(counts[peak_idx - 1])
        y_center = float(counts[peak_idx])
        y_right = float(counts[peak_idx + 1])
        denom = y_left - 2.0 * y_center + y_right
        delta = 0.0
        if abs(denom) > 1e-12:
            delta = 0.5 * (y_left - y_right) / denom
            delta = float(np.clip(delta, -1.0, 1.0))
        refined_pos = float(peak_idx) + delta
        refined_index.append(refined_pos)
        refined_mz.append(float(np.interp(refined_pos, np.arange(centers.size), centers)))
        refined_height.append(float(y_center - 0.25 * (y_left - y_right) * delta))
    refined["peak_index_refined"] = refined_index
    refined["peak_mz_da"] = refined_mz
    refined["smoothed_height"] = refined_height
    return refined.sort_values("peak_mz_da", ignore_index=True)


def refine_peak_centers_raw_local_mode(
    peaks: pd.DataFrame,
    raw_mz: np.ndarray,
    *,
    search_half_width_da: float,
    local_bin_width_da: float,
    smooth_sigma_da: float,
    min_events: int,
    max_shift_da: float,
) -> pd.DataFrame:
    if peaks.empty:
        return peaks.copy()
    raw = np.asarray(raw_mz, dtype=np.float64)
    raw = raw[np.isfinite(raw)]
    if raw.size == 0:
        return peaks.copy()
    raw.sort()
    refined = peaks.copy()
    refined_center = []
    refined_height = []
    refined_shift = []
    refined_event_count = []
    refined_method = []
    bin_width = float(max(local_bin_width_da, 1.0e-4))
    sigma_bins = max(float(smooth_sigma_da) / bin_width, 0.75)
    for _, peak in refined.iterrows():
        center_da = float(peak["peak_mz_da"])
        peak_height = float(peak.get("smoothed_height", peak.get("peak_height", 0.0)))
        half_width = max(float(search_half_width_da), 0.75 * float(peak.get("fwhm_da", 0.0)), 3.0 * bin_width)
        left = center_da - half_width
        right = center_da + half_width
        lo = int(np.searchsorted(raw, left, side="left"))
        hi = int(np.searchsorted(raw, right, side="right"))
        local = raw[lo:hi]
        if local.size < int(min_events):
            refined_center.append(center_da)
            refined_height.append(peak_height)
            refined_shift.append(0.0)
            refined_event_count.append(int(local.size))
            refined_method.append("parabolic-fallback")
            continue
        edges = np.arange(left, right + bin_width, bin_width, dtype=np.float64)
        if edges.size < 8:
            edges = np.linspace(left, right, 9, dtype=np.float64)
        local_counts, edges = np.histogram(local, bins=edges)
        local_centers = 0.5 * (edges[:-1] + edges[1:])
        smoothed = gaussian_filter1d(local_counts.astype(np.float64), sigma=sigma_bins, mode="nearest")
        side_count = max(2, min(8, smoothed.size // 6))
        left_level = float(np.median(smoothed[:side_count]))
        right_level = float(np.median(smoothed[-side_count:]))
        baseline = np.linspace(left_level, right_level, smoothed.size)
        signal = np.clip(smoothed - baseline, 0.0, None)
        if float(signal.sum()) <= 0.0:
            signal = smoothed
        mode_index = int(np.argmax(signal))
        if mode_index <= 0 or mode_index >= signal.size - 1:
            mode_center = float(local_centers[mode_index])
            mode_height = float(signal[mode_index])
        else:
            y_left = float(np.log1p(signal[mode_index - 1]))
            y_center = float(np.log1p(signal[mode_index]))
            y_right = float(np.log1p(signal[mode_index + 1]))
            denom = y_left - 2.0 * y_center + y_right
            delta_bins = 0.0
            if abs(denom) > 1.0e-12:
                delta_bins = float(np.clip(0.5 * (y_left - y_right) / denom, -1.0, 1.0))
            mode_center = float(local_centers[mode_index] + delta_bins * (local_centers[1] - local_centers[0]))
            mode_height = float(np.exp(y_center - 0.25 * (y_left - y_right) * delta_bins) - 1.0)
        shift_da = float(np.clip(mode_center - center_da, -float(max_shift_da), float(max_shift_da)))
        refined_center.append(center_da + shift_da)
        refined_height.append(max(mode_height, peak_height))
        refined_shift.append(shift_da)
        refined_event_count.append(int(local.size))
        refined_method.append("raw-local-mode")
    refined["peak_mz_da"] = refined_center
    refined["smoothed_height"] = refined_height
    refined["raw_refine_shift_da"] = refined_shift
    refined["raw_refine_event_count"] = refined_event_count
    refined["raw_refine_method"] = refined_method
    return refined.sort_values("peak_mz_da", ignore_index=True)


def merge_peak_tables(
    peak_tables: list[pd.DataFrame],
    *,
    merge_tolerance_da: float,
    method_label: str,
) -> pd.DataFrame:
    nonempty = [table.copy() for table in peak_tables if table is not None and not table.empty]
    if not nonempty:
        return pd.DataFrame()
    combined = pd.concat(nonempty, ignore_index=True, sort=False)
    combined = combined.sort_values(["peak_mz_da", "prominence"], ascending=[True, False]).reset_index(drop=True)
    clusters: list[list[dict[str, object]]] = []
    current: list[dict[str, object]] = []
    for row in combined.to_dict("records"):
        if not current:
            current = [row]
            continue
        current_center = float(np.average(
            [float(item["peak_mz_da"]) for item in current],
            weights=[max(float(item.get("prominence", 0.0)), 1.0) for item in current],
        ))
        if abs(float(row["peak_mz_da"]) - current_center) <= merge_tolerance_da:
            current.append(row)
        else:
            clusters.append(current)
            current = [row]
    if current:
        clusters.append(current)
    rows: list[dict[str, object]] = []
    for idx, cluster in enumerate(clusters, start=1):
        weights = np.asarray([max(float(item.get("prominence", 0.0)), 1.0) for item in cluster], dtype=np.float64)
        mz_values = np.asarray([float(item["peak_mz_da"]) for item in cluster], dtype=np.float64)
        peak_idx_values = np.asarray([float(item.get("peak_index_refined", item["peak_index"])) for item in cluster], dtype=np.float64)
        representative = max(
            cluster,
            key=lambda item: (
                float(item.get("prominence", 0.0)),
                float(item.get("smoothed_height", item.get("peak_height", 0.0))),
            ),
        )
        row = dict(representative)
        row["peak_id"] = f"{method_label}-peak-{idx:04d}"
        row["method"] = method_label
        row["peak_mz_da"] = float(np.average(mz_values, weights=weights))
        row["peak_index_refined"] = float(np.average(peak_idx_values, weights=weights))
        row["prominence"] = float(max(float(item.get("prominence", 0.0)) for item in cluster))
        row["detection_source_count"] = int(len(cluster))
        row["detection_sources"] = sorted({str(item.get("detection_source", "unknown")) for item in cluster})
        rows.append(row)
    return pd.DataFrame(rows).sort_values("peak_mz_da", ignore_index=True)


def integrate_peak_areas(
    peaks: pd.DataFrame,
    centers: np.ndarray,
    counts: np.ndarray,
    *,
    integration_half_width_da: float,
    baseline_half_width_da: float,
    max_integration_half_width_da: float | None = None,
) -> pd.DataFrame:
    if peaks.empty:
        return peaks.copy()
    augmented = peaks.copy()
    peak_areas = []
    left_edges = []
    right_edges = []
    for peak_mz, fwhm_da in zip(augmented["peak_mz_da"], augmented["fwhm_da"]):
        half_width = max(integration_half_width_da, float(fwhm_da) * 1.5)
        if max_integration_half_width_da is not None:
            half_width = min(half_width, float(max_integration_half_width_da))
        left = peak_mz - half_width
        right = peak_mz + half_width
        left_mask = (centers >= left - baseline_half_width_da) & (centers < left)
        right_mask = (centers > right) & (centers <= right + baseline_half_width_da)
        core_mask = (centers >= left) & (centers <= right)
        if not core_mask.any():
            peak_areas.append(0.0)
            left_edges.append(left)
            right_edges.append(right)
            continue
        left_level = float(np.median(counts[left_mask])) if left_mask.any() else float(counts[core_mask][0])
        right_level = float(np.median(counts[right_mask])) if right_mask.any() else float(counts[core_mask][-1])
        baseline = np.linspace(left_level, right_level, core_mask.sum())
        signal = np.clip(counts[core_mask] - baseline, 0.0, None)
        area = float(signal.sum())
        peak_areas.append(area)
        left_edges.append(float(left))
        right_edges.append(float(right))
    augmented["integration_left_da"] = left_edges
    augmented["integration_right_da"] = right_edges
    augmented["integrated_area"] = peak_areas
    return augmented


def characterize_centroid_peaks(
    centroid_mz: np.ndarray,
    centroid_intensity: np.ndarray,
    centers: np.ndarray,
    counts: np.ndarray,
    *,
    method_label: str,
    integration_half_width_da: float,
    baseline_half_width_da: float,
) -> pd.DataFrame:
    if centroid_mz.size == 0:
        return pd.DataFrame(
            columns=[
                "peak_id",
                "method",
                "peak_index",
                "peak_mz_da",
                "peak_height",
                "smoothed_height",
                "prominence",
                "left_fwhm_mz_da",
                "right_fwhm_mz_da",
                "fwhm_da",
                "integration_left_da",
                "integration_right_da",
                "integrated_area",
            ]
        )
    nearest_indices = np.searchsorted(centers, centroid_mz, side="left").clip(0, centers.size - 1)
    rows = []
    for idx, (mz, intensity, center_idx) in enumerate(zip(centroid_mz, centroid_intensity, nearest_indices, strict=True)):
        rows.append(
            {
                "peak_id": f"{method_label}-peak-{idx + 1:04d}",
                "method": method_label,
                "peak_index": int(center_idx),
                "peak_mz_da": float(mz),
                "peak_height": float(counts[center_idx]),
                "smoothed_height": float(intensity),
                "prominence": float(intensity),
                "left_fwhm_mz_da": float(mz - integration_half_width_da / 2.0),
                "right_fwhm_mz_da": float(mz + integration_half_width_da / 2.0),
                "fwhm_da": float(integration_half_width_da),
            }
        )
    peaks = pd.DataFrame(rows)
    return integrate_peak_areas(
        peaks,
        centers,
        counts,
        integration_half_width_da=integration_half_width_da,
        baseline_half_width_da=baseline_half_width_da,
    )


def neutral_formula_from_element_counter(counter: Counter[str]) -> str:
    parts = []
    for symbol in sorted(counter):
        count = counter[symbol]
        parts.append(symbol if count == 1 else f"{symbol}{count}")
    return "".join(parts)


def formula_from_element_counter(counter: Counter[str], charge: int) -> str:
    parts = [neutral_formula_from_element_counter(counter)]
    charge_str = "+" if charge == 1 else f"{charge}+"
    return f"{''.join(parts)}{charge_str}"


def isotopologue_label(counter: Counter[str], charge: int) -> str:
    parts = []
    for label in sorted(counter):
        count = counter[label]
        parts.append(label if count == 1 else f"{label}{count}")
    charge_str = "+" if charge == 1 else f"{charge}+"
    return f"{'-'.join(parts)}{charge_str}"


def normalize_series(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    denom = float(series.max())
    if denom <= 0:
        return pd.Series(np.zeros(series.shape[0]), index=series.index)
    return series / denom


def top_assignment_table(assignments: pd.DataFrame, peaks: pd.DataFrame) -> pd.DataFrame:
    if assignments.empty:
        merged = peaks.copy()
        merged["top_species"] = "unassigned"
        merged["top_isotopologue"] = "unassigned"
        merged["top_probability"] = 0.0
        merged["confidence"] = "unassigned"
        return merged
    top = assignments.sort_values(
        ["peak_id", "rank"], ignore_index=True
    ).groupby("peak_id", as_index=False).first()
    return peaks.merge(
        top[
            [
                "peak_id",
                "species_label",
                "isotopologue_label",
                "probability",
                "confidence",
                "candidate_count",
            ]
        ],
        on="peak_id",
        how="left",
    ).rename(
        columns={
            "species_label": "top_species",
            "isotopologue_label": "top_isotopologue",
            "probability": "top_probability",
        }
    )


def dataframe_to_csv_ready(frame: pd.DataFrame) -> pd.DataFrame:
    converted = frame.copy()
    for column in converted.columns:
        if converted[column].map(lambda value: isinstance(value, (dict, list))).any():
            converted[column] = converted[column].map(json.dumps)
    return converted


def assignment_softmax(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    return softmax(scores / 0.35)


def pairwise_peak_matches(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    tolerance_da: float,
    left_label: str,
    right_label: str,
) -> pd.DataFrame:
    matches = []
    right_peaks = right["peak_mz_da"].to_numpy()
    for _, row in left.iterrows():
        mz = float(row["peak_mz_da"])
        if right_peaks.size == 0:
            matches.append(
                {
                    f"{left_label}_peak_id": row["peak_id"],
                    f"{left_label}_mz_da": mz,
                    f"{right_label}_peak_id": None,
                    f"{right_label}_mz_da": np.nan,
                    "delta_da": np.nan,
                    "matched": False,
                }
            )
            continue
        idx = int(np.argmin(np.abs(right_peaks - mz)))
        delta = float(abs(right_peaks[idx] - mz))
        right_row = right.iloc[idx]
        matches.append(
            {
                f"{left_label}_peak_id": row["peak_id"],
                f"{left_label}_mz_da": mz,
                f"{right_label}_peak_id": right_row["peak_id"] if delta <= tolerance_da else None,
                f"{right_label}_mz_da": float(right_row["peak_mz_da"]) if delta <= tolerance_da else np.nan,
                "delta_da": delta,
                "matched": delta <= tolerance_da,
            }
        )
    return pd.DataFrame(matches)


def species_envelope_score(
    species_candidates: pd.DataFrame,
    element_support: dict[str, float],
) -> pd.Series:
    if species_candidates.empty:
        return pd.Series(dtype=float)
    # Hot path: the species library can hold hundreds of thousands of
    # candidates and this runs once per assignment iteration, so we avoid
    # per-row Series construction (iterrows) and loop over raw object arrays.
    symbols_arr = species_candidates["symbols"].to_numpy()
    is_molecular = (species_candidates["category"].to_numpy() == "molecular")
    get = element_support.get
    support = np.empty(symbols_arr.shape[0], dtype=float)
    for i in range(symbols_arr.shape[0]):
        symbols = symbols_arr[i]
        if not symbols:
            support[i] = 0.0
        elif len(symbols) == 1:
            support[i] = get(symbols[0], 0.0)
        elif is_molecular[i]:
            # A molecular ion requires every constituent element to exist in the
            # specimen, so its support is bounded by its weakest element; a mean
            # would let one abundant element carry implausible partners.
            support[i] = min(get(symbol, 0.0) for symbol in symbols)
        else:
            support[i] = sum(get(symbol, 0.0) for symbol in symbols) / len(symbols)
    abundance = (np.log10(np.clip(species_candidates["natural_probability"].to_numpy(dtype=float), 1e-12, None)) + 12.0) / 12.0
    complexity = species_candidates["complexity_penalty"].to_numpy(dtype=float)
    return pd.Series(0.8 * support + 0.4 * abundance - complexity, index=species_candidates.index)


def build_species_library(
    isotope_table: pd.DataFrame,
    element_support: dict[str, float],
    *,
    always_include: list[str],
    max_candidate_elements: int,
    max_charge: int,
    max_molecular_charge: int,
    max_molecular_atoms: int,
) -> pd.DataFrame:
    ranked_elements = sorted(
        element_support.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    chosen = [symbol for symbol, _score in ranked_elements[:max_candidate_elements]]
    for symbol in always_include:
        if symbol in isotope_table["symbol"].values and symbol not in chosen:
            chosen.append(symbol)
    chosen = chosen[: max_candidate_elements + len(always_include)]
    selected = isotope_table[isotope_table["symbol"].isin(chosen)].reset_index(drop=True)
    rows: list[dict[str, object]] = []
    candidate_id = 1
    for _, isotope in selected.iterrows():
        for charge in range(1, max_charge + 1):
            # An ion cannot lose more electrons than its atomic number
            # (H^2+ does not exist; He^3+ does not exist).
            if charge > int(isotope.get("atomic_number", 127)):
                break
            rows.append(
                {
                    "candidate_id": f"cand-{candidate_id:07d}",
                    "category": "atomic",
                    "charge": charge,
                    "neutral_mass_da": float(isotope["mass_da"]),
                    "m_over_z_da": float(isotope["mass_da"] / charge),
                    "species_label": f"{isotope['symbol']}{'+' if charge == 1 else f'{charge}+'}",
                    "isotopologue_label": f"{isotope['isotope_label']}{'+' if charge == 1 else f'{charge}+'}",
                    "element_counts": {str(isotope["symbol"]): 1},
                    "isotope_counts": {str(isotope["isotope_label"]): 1},
                    "symbols": [str(isotope["symbol"])],
                    "natural_probability": float(isotope["natural_abundance"]),
                    "complexity_penalty": 0.02 * (charge - 1),
                }
            )
            candidate_id += 1
    isotope_records = selected.to_dict("records")
    for atom_count in range(2, max_molecular_atoms + 1):
        for combo in combinations_with_replacement(isotope_records, atom_count):
            neutral_mass = float(sum(item["mass_da"] for item in combo))
            element_counter = Counter(str(item["symbol"]) for item in combo)
            isotope_counter = Counter(str(item["isotope_label"]) for item in combo)
            natural_probability = float(np.prod([item["natural_abundance"] for item in combo]))
            total_protons = int(sum(int(item.get("atomic_number", 127)) for item in combo))
            pure_hydrogen = all(str(item["symbol"]) == "H" for item in combo)
            for charge in range(1, max_molecular_charge + 1):
                if charge > total_protons:
                    break
                # H2^2+ and H3^(2+/3+) have no bound states; a pure-hydrogen
                # molecular ion above charge 1 is mass-degenerate with H+ and
                # would silently multiply the hydrogen count.
                if pure_hydrogen and charge >= 2:
                    continue
                rows.append(
                    {
                        "candidate_id": f"cand-{candidate_id:07d}",
                        "category": "molecular",
                        "charge": charge,
                        "neutral_mass_da": neutral_mass,
                        "m_over_z_da": neutral_mass / charge,
                        "species_label": formula_from_element_counter(element_counter, charge),
                        "isotopologue_label": isotopologue_label(isotope_counter, charge),
                        "element_counts": dict(element_counter),
                        "isotope_counts": dict(isotope_counter),
                        "symbols": sorted(element_counter),
                        "natural_probability": natural_probability,
                        "complexity_penalty": 0.12 + 0.08 * (atom_count - 1) + 0.03 * (charge - 1),
                    }
                )
                candidate_id += 1
    library = pd.DataFrame(rows)
    library["support_score"] = species_envelope_score(library, element_support)
    return library.sort_values("m_over_z_da", ignore_index=True)
