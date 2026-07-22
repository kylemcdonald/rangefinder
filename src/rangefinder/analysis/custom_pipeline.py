from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from rangefinder.analysis.common import (
    build_histogram,
    detection_bin_width,
    default_prominence,
    detect_peaks_log_peakiness_candidates,
    detect_peaks_log_peakiness,
    detect_peaks_from_histogram,
    integrate_peak_areas,
    merge_peak_tables,
    refine_peak_centers_raw_local_mode,
    refine_peak_centers_parabolic,
    smooth_counts,
)
from rangefinder.analysis.adaptive_ranging import apply_adaptive_ranging
from rangefinder.analysis.assignment import infer_element_support
from rangefinder.analysis.pipeline_utils import finalize_method_analysis
from rangefinder.analysis.segmentation import segment_sample_regions
from rangefinder.analysis.tof_fitting import fit_isotope_family_tof, fit_local_tof_peak_mixture
from rangefinder.references.isotopes import isotope_table


def _rescue_log_only_peaks(
    log_candidates: pd.DataFrame,
    *,
    regular_peak_mz: np.ndarray,
    regular_support_mz: np.ndarray,
    merge_tolerance_da: float,
    config: dict,
) -> pd.DataFrame:
    if log_candidates.empty:
        return log_candidates.copy()
    analysis_cfg = config["analysis"]
    rescue = log_candidates.copy()
    if regular_peak_mz.size > 0:
        rescue = rescue[
            rescue["peak_mz_da"].map(
                lambda value: bool(np.min(np.abs(regular_peak_mz - float(value))) > float(merge_tolerance_da))
            )
        ].reset_index(drop=True)
    if rescue.empty:
        return rescue
    strong_mask = (
        (rescue["peakiness"].fillna(0.0) >= float(analysis_cfg.get("custom_log_rescue_min_peakiness", 10.0)))
        & (rescue["peak_height"].fillna(0.0) >= float(analysis_cfg.get("custom_log_rescue_min_count", 40.0)))
        & (rescue["prominence"].fillna(0.0) >= float(analysis_cfg.get("custom_log_rescue_min_prominence", 0.30)))
    )
    relaxed_mask = (
        (rescue["peakiness"].fillna(0.0) >= float(analysis_cfg.get("custom_log_rescue_relaxed_min_peakiness", 4.0)))
        & (rescue["peak_height"].fillna(0.0) >= float(analysis_cfg.get("custom_log_rescue_relaxed_min_count", 60.0)))
        & (rescue["prominence"].fillna(0.0) >= float(analysis_cfg.get("custom_log_rescue_relaxed_min_prominence", 0.40)))
    )
    support_radius_da = float(analysis_cfg.get("custom_log_rescue_support_radius_da", 1.05))
    isolated_support_radius_da = float(analysis_cfg.get("custom_log_rescue_isolated_support_radius_da", 0.80))
    candidate_positions = rescue["peak_mz_da"].to_numpy(dtype=float)
    regular_positions = np.asarray(regular_support_mz, dtype=float) if regular_support_mz.size else np.asarray([], dtype=float)

    def has_local_support(position_da: float) -> bool:
        if candidate_positions.size:
            candidate_support = int(np.sum(np.abs(candidate_positions - float(position_da)) <= support_radius_da)) - 1
            if candidate_support > 0:
                return True
        if regular_positions.size and bool(np.min(np.abs(regular_positions - float(position_da))) <= support_radius_da):
            return True
        return False

    def has_isolated_support(position_da: float) -> bool:
        if candidate_positions.size:
            candidate_support = int(np.sum(np.abs(candidate_positions - float(position_da)) <= isolated_support_radius_da)) - 1
            if candidate_support > 0:
                return True
        if regular_positions.size and bool(np.min(np.abs(regular_positions - float(position_da))) <= isolated_support_radius_da):
            return True
        return False

    support_mask = rescue["peak_mz_da"].map(has_local_support).to_numpy(dtype=bool)
    general_cluster_mask = (
        (rescue["peak_mz_da"].fillna(0.0) <= float(analysis_cfg.get("custom_log_rescue_general_cluster_max_mz_da", 40.0)))
        & (rescue["peakiness"].fillna(0.0) >= float(analysis_cfg.get("custom_log_rescue_general_cluster_min_peakiness", 1.5)))
        & (rescue["peak_height"].fillna(0.0) >= float(analysis_cfg.get("custom_log_rescue_general_cluster_min_count", 180.0)))
        & (rescue["prominence"].fillna(0.0) >= float(analysis_cfg.get("custom_log_rescue_general_cluster_min_prominence", 0.14)))
        & support_mask
    )
    dense_cluster_mask = (
        (rescue["peak_mz_da"].fillna(0.0) <= float(analysis_cfg.get("custom_log_rescue_dense_cluster_max_mz_da", 40.0)))
        & (rescue["peakiness"].fillna(0.0) >= float(analysis_cfg.get("custom_log_rescue_dense_cluster_min_peakiness", 0.8)))
        & (rescue["peak_height"].fillna(0.0) >= float(analysis_cfg.get("custom_log_rescue_dense_cluster_min_count", 1000.0)))
        & (rescue["prominence"].fillna(0.0) >= float(analysis_cfg.get("custom_log_rescue_dense_cluster_min_prominence", 0.10)))
        & support_mask
    )
    high_mz_mask = (
        (rescue["peak_mz_da"].fillna(0.0) >= float(analysis_cfg.get("custom_log_rescue_high_mz_threshold_da", 80.0)))
        & (rescue["peakiness"].fillna(0.0) >= float(analysis_cfg.get("custom_log_rescue_high_mz_min_peakiness", 4.5)))
        & (rescue["peak_height"].fillna(0.0) >= float(analysis_cfg.get("custom_log_rescue_high_mz_min_count", 35.0)))
        & (rescue["prominence"].fillna(0.0) >= float(analysis_cfg.get("custom_log_rescue_high_mz_min_prominence", 0.45)))
    )
    broad_high_mz_mask = (
        (rescue["peak_mz_da"].fillna(0.0) >= float(analysis_cfg.get("custom_log_rescue_broad_high_mz_threshold_da", 80.0)))
        & (rescue["peakiness"].fillna(0.0) >= float(analysis_cfg.get("custom_log_rescue_broad_min_peakiness", 3.7)))
        & (rescue["peak_height"].fillna(0.0) >= float(analysis_cfg.get("custom_log_rescue_broad_min_count", 250.0)))
        & (rescue["prominence"].fillna(0.0) >= float(analysis_cfg.get("custom_log_rescue_broad_min_prominence", 0.25)))
    )
    rescue = rescue[strong_mask | relaxed_mask | general_cluster_mask | dense_cluster_mask | high_mz_mask | broad_high_mz_mask].copy()
    if rescue.empty:
        return rescue
    low_count_limit = float(analysis_cfg.get("custom_log_rescue_isolated_low_count_max", 120.0))
    isolated_peakiness_min = float(analysis_cfg.get("custom_log_rescue_isolated_min_peakiness", 7.0))
    isolated_prominence_min = float(analysis_cfg.get("custom_log_rescue_isolated_min_prominence", 0.35))
    isolated_low_mz_threshold = float(analysis_cfg.get("custom_log_rescue_isolated_low_mz_threshold_da", 10.0))
    isolated_low_mz_prominence_min = float(analysis_cfg.get("custom_log_rescue_isolated_low_mz_min_prominence", 0.45))
    support_mask = rescue["peak_mz_da"].map(has_isolated_support).to_numpy(dtype=bool)
    rescue_high_mz_mask = (
        (rescue["peak_mz_da"].fillna(0.0).to_numpy(dtype=float) >= float(analysis_cfg.get("custom_log_rescue_high_mz_threshold_da", 80.0)))
        & (rescue["peakiness"].fillna(0.0).to_numpy(dtype=float) >= float(analysis_cfg.get("custom_log_rescue_high_mz_min_peakiness", 4.5)))
        & (rescue["peak_height"].fillna(0.0).to_numpy(dtype=float) >= float(analysis_cfg.get("custom_log_rescue_high_mz_min_count", 35.0)))
        & (rescue["prominence"].fillna(0.0).to_numpy(dtype=float) >= float(analysis_cfg.get("custom_log_rescue_high_mz_min_prominence", 0.45)))
    )
    isolated_mask = (
        (~support_mask)
        & (rescue["peak_height"].fillna(0.0).to_numpy(dtype=float) < low_count_limit)
        & (~rescue_high_mz_mask)
    )
    low_mz_isolated = isolated_mask & (rescue["peak_mz_da"].fillna(0.0).to_numpy(dtype=float) < isolated_low_mz_threshold)
    isolated_accept_mask = (
        (rescue["peakiness"].fillna(0.0).to_numpy(dtype=float) >= isolated_peakiness_min)
        & (rescue["prominence"].fillna(0.0).to_numpy(dtype=float) >= isolated_prominence_min)
    )
    isolated_accept_mask[low_mz_isolated] = (
        (rescue["peakiness"].fillna(0.0).to_numpy(dtype=float)[low_mz_isolated] >= isolated_peakiness_min)
        & (rescue["prominence"].fillna(0.0).to_numpy(dtype=float)[low_mz_isolated] >= isolated_low_mz_prominence_min)
    )
    rescue = rescue.loc[
        (~isolated_mask)
        | isolated_accept_mask
    ].copy()
    if rescue.empty:
        return rescue
    rescue_window_da = float(analysis_cfg.get("custom_log_rescue_window_da", 5.0))
    rescue["rescue_window"] = np.floor(rescue["peak_mz_da"].to_numpy(dtype=float) / max(rescue_window_da, 1.0e-6)).astype(int)
    rescue = (
        rescue.sort_values(["rescue_window", "peakiness", "peak_height"], ascending=[True, False, False])
        .groupby("rescue_window", sort=True, as_index=False)
        .head(int(analysis_cfg.get("custom_log_rescue_max_per_window", 4)))
        .drop(columns=["rescue_window"])
        .reset_index(drop=True)
    )
    if not rescue.empty:
        rescue["detection_source"] = "log-rescue"
        rescue["source_bin_width_da"] = float(analysis_cfg["default_zoom_bin_width_da"])
    return rescue


def _suppress_single_source_shoulders(
    peaks,
    *,
    exclusion_da: float,
    prominence_ratio_max: float,
    min_neighbor_da: float,
    neighbor_fwhm_scale: float,
):
    if peaks.empty or "detection_source_count" not in peaks.columns:
        return peaks.copy()
    peaks = peaks.sort_values("peak_mz_da").reset_index(drop=True).copy()
    keep_mask = np.ones(peaks.shape[0], dtype=bool)
    mz_values = peaks["peak_mz_da"].to_numpy(dtype=float)
    prominences = peaks["prominence"].to_numpy(dtype=float)
    source_counts = peaks["detection_source_count"].to_numpy(dtype=int)
    fwhm_values = peaks["fwhm_da"].to_numpy(dtype=float) if "fwhm_da" in peaks.columns else np.zeros(peaks.shape[0], dtype=float)
    for idx in range(peaks.shape[0]):
        if source_counts[idx] > 1:
            continue
        for other_idx in range(peaks.shape[0]):
            if idx == other_idx:
                continue
            local_exclusion = max(
                float(min_neighbor_da),
                float(neighbor_fwhm_scale) * max(float(fwhm_values[idx]), float(fwhm_values[other_idx]), 0.0),
            )
            local_exclusion = min(float(exclusion_da), local_exclusion)
            if abs(mz_values[idx] - mz_values[other_idx]) > local_exclusion:
                continue
            if prominences[other_idx] <= prominences[idx]:
                continue
            if prominences[idx] / max(prominences[other_idx], 1.0) <= prominence_ratio_max:
                keep_mask[idx] = False
                break
    return peaks.loc[keep_mask].reset_index(drop=True)


def _snap_and_merge_local_maxima(
    peaks,
    *,
    centers,
    counts,
    search_da: float,
    merge_tolerance_da: float,
    smooth_window_bins: int,
):
    if peaks.empty:
        return peaks.copy()
    smoothed = smooth_counts(counts, window_length=int(smooth_window_bins), polyorder=2)
    maxima_idx, _ = find_peaks(smoothed, distance=1, height=0)
    if maxima_idx.size == 0:
        return peaks.copy()
    maxima_mz = centers[maxima_idx]
    maxima_height = smoothed[maxima_idx]
    snapped = peaks.copy()
    snapped_centers = []
    for peak_mz in snapped["peak_mz_da"].to_numpy(dtype=float):
        local_mask = np.abs(maxima_mz - peak_mz) <= float(search_da)
        if not np.any(local_mask):
            snapped_centers.append(float(peak_mz))
            continue
        candidate_mz = maxima_mz[local_mask]
        candidate_height = maxima_height[local_mask]
        best_idx = int(np.argmax(candidate_height))
        snapped_centers.append(float(candidate_mz[best_idx]))
    snapped["peak_mz_da"] = snapped_centers
    snapped = snapped.sort_values(["peak_mz_da", "prominence"], ascending=[True, False]).reset_index(drop=True)
    clusters: list[list[int]] = []
    current: list[int] = []
    mz_values = snapped["peak_mz_da"].to_numpy(dtype=float)
    for idx in range(snapped.shape[0]):
        if not current:
            current = [idx]
            continue
        if abs(mz_values[idx] - mz_values[current[-1]]) <= float(merge_tolerance_da):
            current.append(idx)
        else:
            clusters.append(current)
            current = [idx]
    if current:
        clusters.append(current)
    rows = []
    for cluster in clusters:
        cluster_rows = snapped.iloc[cluster].copy()
        rank_columns = [column for column in ["detection_source_count", "prominence", "peak_height"] if column in cluster_rows.columns]
        representative_idx = cluster_rows.sort_values(
            rank_columns,
            ascending=[False] * len(rank_columns),
        ).index[0]
        row = snapped.loc[representative_idx].copy()
        row["peak_mz_da"] = float(np.average(cluster_rows["peak_mz_da"].to_numpy(dtype=float)))
        if "prominence" in cluster_rows.columns:
            row["prominence"] = float(cluster_rows["prominence"].max())
        if "smoothed_height" in cluster_rows.columns:
            row["smoothed_height"] = float(cluster_rows["smoothed_height"].max())
        if "peak_height" in cluster_rows.columns:
            row["peak_height"] = float(cluster_rows["peak_height"].max())
        if "fwhm_da" in cluster_rows.columns:
            row["fwhm_da"] = float(cluster_rows["fwhm_da"].min())
        if "detection_sources" in cluster_rows.columns:
            merged_sources = []
            for value in cluster_rows["detection_sources"].tolist():
                if isinstance(value, list):
                    merged_sources.extend(str(item) for item in value)
                else:
                    merged_sources.append(str(value))
            row["detection_sources"] = sorted(set(merged_sources))
            row["detection_source_count"] = int(len(row["detection_sources"]))
        rows.append(row.to_dict())
    return pd.DataFrame(rows).sort_values("peak_mz_da", ignore_index=True)


def _suppress_tail_artifacts(
    peaks: pd.DataFrame,
    *,
    centers: np.ndarray,
    counts: np.ndarray,
    max_distance_da: float,
    height_ratio_min: float,
    valley_ratio_min: float,
) -> pd.DataFrame:
    """Drop peaks that are just the thermal tail of a much larger peak.

    Field-evaporation peaks have a one-sided high-mass tail; the detector can
    split that tail into a spurious secondary maximum. A real neighbor separates
    from the parent through a valley, whereas a tail artifact rides on a
    monotone-ish decay, so the minimum between the two peaks stays comparable to
    the artifact's own height. The distance cap stays below the 1/3 Da spacing of
    3+ isotope families so genuine family members are never eligible.
    """
    if peaks.empty or peaks.shape[0] < 2:
        return peaks.copy()
    ordered = peaks.sort_values("peak_mz_da").reset_index(drop=True)
    mz_values = ordered["peak_mz_da"].to_numpy(dtype=float)
    heights = ordered["peak_height"].fillna(0.0).to_numpy(dtype=float)
    keep = np.ones(ordered.shape[0], dtype=bool)
    for idx in range(1, ordered.shape[0]):
        parent_idx = idx - 1
        gap = float(mz_values[idx] - mz_values[parent_idx])
        if gap <= 0 or gap > float(max_distance_da):
            continue
        if heights[parent_idx] < float(height_ratio_min) * max(heights[idx], 1.0):
            continue
        lo = int(np.searchsorted(centers, mz_values[parent_idx], side="left"))
        hi = int(np.searchsorted(centers, mz_values[idx], side="right"))
        if hi <= lo + 1:
            continue
        valley = float(np.min(counts[lo:hi]))
        if valley >= float(valley_ratio_min) * max(heights[idx], 1.0):
            keep[idx] = False
    return ordered.loc[keep].reset_index(drop=True)


def _crowded_peak_clusters(
    peaks: pd.DataFrame,
    *,
    cluster_gap_da: float,
    min_peaks: int,
    max_window_span_da: float,
) -> list[list[str]]:
    if peaks.empty:
        return []
    ordered = peaks.sort_values("peak_mz_da").reset_index(drop=True)
    mz_values = ordered["peak_mz_da"].to_numpy(dtype=float)
    peak_ids = ordered["peak_id"].astype(str).to_numpy()
    clusters: list[list[str]] = []
    current: list[str] = [str(peak_ids[0])]
    current_mz: list[float] = [float(mz_values[0])]
    for position in range(1, ordered.shape[0]):
        gap = float(mz_values[position] - mz_values[position - 1])
        candidate_mz = current_mz + [float(mz_values[position])]
        if gap <= float(cluster_gap_da) and (max(candidate_mz) - min(candidate_mz)) <= float(max_window_span_da):
            current.append(str(peak_ids[position]))
            current_mz.append(float(mz_values[position]))
            continue
        if len(current) >= int(min_peaks):
            clusters.append(current.copy())
        current = [str(peak_ids[position])]
        current_mz = [float(mz_values[position])]
    if len(current) >= int(min_peaks):
        clusters.append(current.copy())
    return clusters


def _select_fit_seed_rows(
    cluster_rows: pd.DataFrame,
    *,
    relative_prominence_min: float,
    min_detection_sources: int,
    min_peaks: int,
) -> pd.DataFrame:
    if cluster_rows.empty:
        return cluster_rows.copy()
    prominences = cluster_rows["prominence"].fillna(0.0)
    source_counts = (
        cluster_rows["detection_source_count"].fillna(0).astype(int)
        if "detection_source_count" in cluster_rows.columns
        else pd.Series(np.ones(cluster_rows.shape[0], dtype=int), index=cluster_rows.index)
    )
    prominence_cutoff = float(prominences.max()) * float(relative_prominence_min)
    seed_mask = (prominences >= prominence_cutoff) | (source_counts >= int(min_detection_sources))
    seed_rows = cluster_rows.loc[seed_mask].copy()
    if seed_rows.shape[0] < int(min_peaks):
        return cluster_rows.copy()
    return seed_rows.sort_values("peak_mz_da").copy()


def _annotate_atomic_family_hints(
    peaks: pd.DataFrame,
    *,
    raw_mz: np.ndarray,
    config: dict,
) -> pd.DataFrame:
    if peaks.empty:
        return peaks.copy()
    analysis_cfg = config["analysis"]
    isotopes = isotope_table().copy()
    if isotopes.empty:
        return peaks.copy()
    raw_sorted = np.sort(np.asarray(raw_mz, dtype=np.float64))
    peaks = peaks.copy()
    if "peak_mz_raw_da" not in peaks.columns:
        peaks["peak_mz_raw_da"] = peaks["peak_mz_da"].astype(float)
    peaks["family_hint_element"] = pd.Series([None] * peaks.shape[0], dtype=object)
    peaks["family_hint_charge"] = pd.Series([np.nan] * peaks.shape[0], dtype=float)
    peaks["family_hint_isotope_label"] = pd.Series([None] * peaks.shape[0], dtype=object)
    peaks["family_hint_score"] = 0.0
    peaks["family_hint_match_count"] = 0
    peaks["family_hint_exclusivity"] = 0.0
    peaks["family_hint_candidates"] = pd.Series([[] for _ in range(peaks.shape[0])], dtype=object)
    peaks["family_hint_tier"] = pd.Series([None] * peaks.shape[0], dtype=object)
    peaks["family_hint_calibration_shift_da"] = np.nan
    peaks["family_fit_area"] = np.nan
    peak_positions = peaks["peak_mz_da"].to_numpy(dtype=float)
    peak_ids = peaks["peak_id"].astype(str).to_numpy()
    peak_prominence = peaks["prominence"].fillna(0.0).to_numpy(dtype=float)
    peak_fwhm = peaks["fwhm_da"].fillna(float(analysis_cfg["custom_local_fit_sigma_floor_da"]) * 2.354820045).to_numpy(dtype=float)
    peak_sources = (
        peaks["detection_source_count"].fillna(1).astype(int).to_numpy()
        if "detection_source_count" in peaks.columns
        else np.ones(peaks.shape[0], dtype=int)
    )
    min_abundance = float(analysis_cfg["custom_family_hint_min_abundance"])
    initial_tolerance = float(analysis_cfg["custom_family_hint_match_tolerance_da"])
    fit_tolerance = float(analysis_cfg["custom_family_hint_fit_tolerance_da"])
    match_ppm = float(analysis_cfg.get("custom_family_hint_match_ppm", 350.0))
    initial_tolerance_floor = float(analysis_cfg.get("custom_family_hint_match_tolerance_floor_da", 0.012))
    fit_tolerance_floor = float(analysis_cfg.get("custom_family_hint_fit_tolerance_floor_da", 0.010))
    dominant_min_ratio = float(analysis_cfg.get("custom_family_hint_dominant_min_ratio", 0.5))

    def initial_match_tolerance(mz: float) -> float:
        return float(np.clip(match_ppm * 1.0e-6 * mz, initial_tolerance_floor, initial_tolerance))

    def fit_match_tolerance(mz: float) -> float:
        return float(np.clip(match_ppm * 1.0e-6 * mz, fit_tolerance_floor, fit_tolerance))

    shift_ppm = float(analysis_cfg.get("custom_family_hint_shift_ppm", 300.0))
    shift_floor_da = float(analysis_cfg.get("custom_family_hint_shift_floor_da", 0.006))
    shift_ceiling_da = float(analysis_cfg["custom_family_hint_shift_tolerance_da"])

    def shift_limit(mz: float) -> float:
        # A shared family shift models calibration error, which is a ppm-scale
        # effect; an anchor alignment that needs more than this is a different
        # species' peak, not a shifted family.
        return float(np.clip(shift_ppm * 1.0e-6 * mz, shift_floor_da, shift_ceiling_da))

    # Calibration-recovery tier: real datasets can carry locally large m/z
    # calibration errors (tens of mDa). A shift far beyond the ppm budget is
    # only accepted when the full isotope fingerprint is unmistakable: at least
    # three matched isotopes, high abundance coherence, and a dominant isotope
    # that actually dominates the fit.
    recovery_enabled = bool(analysis_cfg.get("custom_family_hint_recovery_enabled", True))
    recovery_shift_ppm = float(analysis_cfg.get("custom_family_hint_recovery_shift_ppm", 4000.0))
    recovery_min_matches = int(analysis_cfg.get("custom_family_hint_recovery_min_matches", 3))
    recovery_min_coherence = float(analysis_cfg.get("custom_family_hint_recovery_min_coherence", 0.75))
    recovery_dominant_min_ratio = float(analysis_cfg.get("custom_family_hint_recovery_dominant_min_ratio", 0.7))
    recovery_score_scale = float(analysis_cfg.get("custom_family_hint_recovery_score_scale", 0.85))

    def recovery_shift_limit(mz: float) -> float:
        return float(max(recovery_shift_ppm * 1.0e-6 * mz, shift_floor_da))

    score_min = float(analysis_cfg["custom_family_hint_score_min"])
    min_counts = float(analysis_cfg["custom_family_hint_min_total_counts"])
    max_charge = int(analysis_cfg["custom_family_hint_max_charge"])
    element_support = infer_element_support(
        peaks,
        max_charge=max_charge,
        ppm_tolerance=float(analysis_cfg["ppm_tolerance"]),
        min_absolute_tolerance_da=float(analysis_cfg["min_absolute_tolerance_da"]),
        max_absolute_tolerance_da=float(analysis_cfg["max_absolute_tolerance_da"]),
    )
    max_elements = int(analysis_cfg.get("custom_family_hint_max_elements", 12))
    min_element_support = float(analysis_cfg.get("custom_family_hint_min_element_support", 0.02))
    allowed_elements = {
        element
        for element, _ in sorted(
            element_support.items(),
            key=lambda item: float(item[1]),
            reverse=True,
        )[:max_elements]
    }
    allowed_elements.update(
        {
            element
            for element, score in element_support.items()
            if float(score) >= min_element_support
        }
    )
    anchor_peak_limit = int(analysis_cfg.get("custom_family_hint_anchor_peak_limit", 24))
    family_candidates: list[dict[str, object]] = []
    for element, family in isotopes.groupby("symbol", sort=True):
        if allowed_elements and str(element) not in allowed_elements:
            continue
        family = family.sort_values("mass_number").reset_index(drop=True)
        if family.shape[0] < 2:
            continue
        abundance_mask = family["natural_abundance"].to_numpy(dtype=float) >= min_abundance
        abundance_mask[int(np.argmax(family["natural_abundance"].to_numpy(dtype=float)))] = True
        family = family.loc[abundance_mask].reset_index(drop=True)
        if family.shape[0] < 2:
            continue
        dominant_idx = int(np.argmax(family["natural_abundance"].to_numpy(dtype=float)))
        isotope_labels = family["isotope_label"].astype(str).to_numpy()
        natural_fraction = family["natural_abundance"].to_numpy(dtype=float)
        for charge in range(1, max_charge + 1):
            expected_mz = family["mass_da"].to_numpy(dtype=float) / float(charge)
            if expected_mz.max() < peak_positions.min() - 0.2 or expected_mz.min() > peak_positions.max() + 0.2:
                continue
            best_family: dict[str, object] | None = None
            dominant_expected_mz = float(expected_mz[dominant_idx])
            anchor_order = np.argsort(-peak_prominence)
            anchor_peak_indices = anchor_order[: min(anchor_peak_limit, anchor_order.size)]
            tight_limit = shift_limit(dominant_expected_mz)
            recovery_limit = recovery_shift_limit(dominant_expected_mz) if recovery_enabled else tight_limit
            candidate_shifts: list[tuple[float, str]] = []
            for peak_idx in anchor_peak_indices:
                shift = float(peak_positions[peak_idx] - dominant_expected_mz)
                if abs(shift) <= tight_limit:
                    candidate_shifts.append((shift, "tight"))
                elif recovery_enabled and abs(shift) <= recovery_limit:
                    candidate_shifts.append((shift, "recovery"))
            for shift, shift_tier in candidate_shifts:
                used_peak_idx: set[int] = set()
                matches: list[tuple[int, int]] = []
                for isotope_idx, target_mz in enumerate(expected_mz + float(shift)):
                    candidate_order = np.argsort(np.abs(peak_positions - target_mz))
                    match_idx = None
                    for peak_idx in candidate_order:
                        if peak_idx in used_peak_idx:
                            continue
                        if abs(float(peak_positions[peak_idx]) - float(target_mz)) <= initial_match_tolerance(float(target_mz)):
                            match_idx = int(peak_idx)
                            break
                    if match_idx is None:
                        continue
                    used_peak_idx.add(match_idx)
                    matches.append((match_idx, isotope_idx))
                if len(matches) < 2:
                    continue
                dominant_matches = [isotope_idx for _, isotope_idx in matches]
                if dominant_idx not in dominant_matches:
                    # Dominant rescue: local calibration wobble can push just
                    # the dominant isotope outside the strict window while its
                    # siblings match. With two strict siblings in hand, allow
                    # the dominant a widened window; the abundance-coherence
                    # gate after the fit still rejects impostor families.
                    rescue_scale = float(analysis_cfg.get("custom_family_completion_tolerance_scale", 2.5))
                    if len(matches) < 2 or rescue_scale <= 1.0:
                        continue
                    dominant_target = float(expected_mz[dominant_idx] + float(shift))
                    candidate_order = np.argsort(np.abs(peak_positions - dominant_target))
                    rescued = False
                    for peak_idx in candidate_order:
                        if peak_idx in used_peak_idx:
                            continue
                        if (
                            abs(float(peak_positions[peak_idx]) - dominant_target)
                            <= rescue_scale * initial_match_tolerance(dominant_target)
                        ):
                            used_peak_idx.add(int(peak_idx))
                            matches.append((int(peak_idx), dominant_idx))
                            rescued = True
                        break
                    if not rescued:
                        continue
                sigma_da_init = float(max(np.median(peak_fwhm[[peak_idx for peak_idx, _ in matches]]) / 2.354820045, 0.005))
                fit_shift_tolerance = shift_limit(float(np.median(expected_mz)))
                if shift_tier == "recovery":
                    fit_shift_tolerance = max(fit_shift_tolerance, abs(float(shift)) + 0.02)
                fit = fit_isotope_family_tof(
                    np.asarray(raw_mz, dtype=np.float64),
                    target_mz_da=expected_mz,
                    sigma_da_init=sigma_da_init,
                    window_margin_da=float(analysis_cfg["custom_family_hint_window_margin_da"]),
                    shift_tolerance_da=fit_shift_tolerance,
                    sigma_floor_da=float(analysis_cfg["custom_family_hint_sigma_floor_da"]),
                    sigma_ceiling_da=float(analysis_cfg["custom_family_hint_sigma_ceiling_da"]),
                    bins_per_sigma=float(analysis_cfg["custom_family_hint_bins_per_sigma"]),
                    shift_penalty=float(analysis_cfg["custom_family_hint_shift_penalty"]),
                    sigma_penalty=float(analysis_cfg["custom_family_hint_sigma_penalty"]),
                )
                if fit is None:
                    continue
                fitted_counts = np.asarray(fit["fitted_counts"], dtype=float)
                total_counts = float(np.sum(fitted_counts))
                if total_counts < min_counts:
                    continue
                fitted_fraction = np.asarray(fit["fitted_fraction"], dtype=float)
                expected_fraction = natural_fraction / max(float(np.sum(natural_fraction)), 1.0e-12)
                l1_distance = float(np.abs(fitted_fraction - expected_fraction).sum())
                # A family hypothesis whose dominant natural isotope does not
                # actually dominate the fit is a mass coincidence riding on other
                # species' peaks, not evidence for this element.
                if float(fitted_fraction[dominant_idx]) < dominant_min_ratio * float(expected_fraction[dominant_idx]):
                    continue
                nearest_matches: list[tuple[int, int]] = []
                used_peak_idx = set()
                for isotope_idx, fitted_mz in enumerate(np.asarray(fit["fitted_mz_da"], dtype=float)):
                    candidate_order = np.argsort(np.abs(peak_positions - float(fitted_mz)))
                    match_idx = None
                    for peak_idx in candidate_order:
                        if peak_idx in used_peak_idx:
                            continue
                        if abs(float(peak_positions[peak_idx]) - float(fitted_mz)) <= fit_match_tolerance(float(fitted_mz)):
                            match_idx = int(peak_idx)
                            break
                    if match_idx is None:
                        continue
                    used_peak_idx.add(match_idx)
                    nearest_matches.append((match_idx, isotope_idx))
                # A peak that passed the strict initial (shift-aligned) match
                # must not be lost because an un-modeled interference pulled
                # the shared fit a few mDa: keep initial matches whose isotope
                # the fitted re-match failed to claim.
                matched_isotope_set = {isotope_idx for _, isotope_idx in nearest_matches}
                for peak_idx, isotope_idx in matches:
                    if isotope_idx in matched_isotope_set or peak_idx in used_peak_idx:
                        continue
                    used_peak_idx.add(peak_idx)
                    nearest_matches.append((peak_idx, isotope_idx))
                    matched_isotope_set.add(isotope_idx)
                if len(nearest_matches) < 2:
                    continue
                if dominant_idx not in [isotope_idx for _, isotope_idx in nearest_matches]:
                    continue
                support = float(
                    sum(
                        peak_prominence[peak_idx] * max(peak_sources[peak_idx], 1)
                        for peak_idx, _ in nearest_matches
                    )
                )
                support_scale = max(float(np.max(peak_prominence) * max(np.max(peak_sources), 1)), 1.0)
                support_score = float(np.clip(support / support_scale, 0.0, 2.0))
                # Abundance coherence multiplies the whole score: a family that
                # cannot reproduce natural relative abundances should not be able
                # to buy its way back with match count or peak prominence.
                coherence = float(max(0.0, 1.0 - l1_distance / 2.0))
                score = coherence * (
                    float(len(nearest_matches)) + 1.25 * coherence + 0.35 * support_score
                )
                calibration_shift_da = float(
                    np.median(np.asarray(fit["fitted_mz_da"], dtype=float) - expected_mz)
                )
                if shift_tier == "recovery":
                    effective_matches = len(nearest_matches)
                    if effective_matches < recovery_min_matches:
                        # Small members riding a large neighbor's thermal tail
                        # are often invisible to blind detection; verify their
                        # predicted positions directly in the raw spectrum.
                        matched_iso_set = {iso for _, iso in nearest_matches}
                        dominant_peak_rows = [p for p, iso in nearest_matches if iso == dominant_idx]
                        fwhm_dom = (
                            float(peak_fwhm[dominant_peak_rows[0]])
                            if dominant_peak_rows
                            else float(expected_mz[dominant_idx]) / 800.0
                        )
                        effective_matches += _verify_missing_members(
                            raw_sorted,
                            expected_mz=expected_mz,
                            fractions=natural_fraction / max(float(np.sum(natural_fraction)), 1.0e-12),
                            matched_isotopes=matched_iso_set,
                            dominant_idx=dominant_idx,
                            shift=calibration_shift_da,
                            fwhm_dominant_da=max(fwhm_dom, 0.01),
                            config=config,
                        )
                    if effective_matches == 2 and len(nearest_matches) == 2:
                        # Two-member recovery: the minor member's measured
                        # fraction must reproduce its natural value tightly.
                        # This is sharp where plain coherence is not - the
                        # dominant swamps the L1, but the minor ratio
                        # separates e.g. O2+ (0.41% at +2 Da) from S+ (4.2%).
                        minor_rel_tol = float(
                            analysis_cfg.get("custom_family_recovery_two_member_minor_rel_tolerance", 0.4)
                        )
                        minor_indices = [
                            iso for _, iso in nearest_matches if iso != dominant_idx
                        ]
                        expected_fraction_all = natural_fraction / max(float(np.sum(natural_fraction)), 1.0e-12)
                        minor_ok = False
                        if minor_indices and minor_rel_tol > 0.0:
                            minor_idx = max(minor_indices, key=lambda iso: float(expected_fraction_all[iso]))
                            expected_minor = float(expected_fraction_all[minor_idx])
                            fitted_minor = float(fitted_fraction[minor_idx])
                            if expected_minor > 0 and abs(fitted_minor - expected_minor) <= minor_rel_tol * expected_minor:
                                minor_ok = True
                        if not minor_ok:
                            continue
                    elif effective_matches < recovery_min_matches:
                        continue
                    if coherence < recovery_min_coherence:
                        continue
                    if float(fitted_fraction[dominant_idx]) < recovery_dominant_min_ratio * float(
                        expected_fraction[dominant_idx]
                    ):
                        continue
                    score *= recovery_score_scale
                # Family completion: once the fingerprint is confirmed by
                # enough strict matches, remaining isotopes may claim the
                # nearest free peak within a widened window. Local mass-scale
                # stretch otherwise drops individual members (e.g. 182W3+)
                # whose peaks then fall through to molecular misassignment.
                # Completion happens after scoring so it cannot buy score.
                completion_min = int(analysis_cfg.get("custom_family_completion_min_matches", 3))
                completion_scale = float(analysis_cfg.get("custom_family_completion_tolerance_scale", 2.5))
                if len(nearest_matches) >= completion_min and completion_scale > 1.0:
                    matched_isotopes = {isotope_idx for _, isotope_idx in nearest_matches}
                    for isotope_idx, fitted_mz in enumerate(np.asarray(fit["fitted_mz_da"], dtype=float)):
                        if isotope_idx in matched_isotopes:
                            continue
                        candidate_order = np.argsort(np.abs(peak_positions - float(fitted_mz)))
                        for peak_idx in candidate_order:
                            if peak_idx in used_peak_idx:
                                continue
                            if (
                                abs(float(peak_positions[peak_idx]) - float(fitted_mz))
                                <= completion_scale * fit_match_tolerance(float(fitted_mz))
                            ):
                                used_peak_idx.add(int(peak_idx))
                                nearest_matches.append((int(peak_idx), isotope_idx))
                            break
                if best_family is None or score > float(best_family["score"]):
                    best_family = {
                        "element": str(element),
                        "charge": int(charge),
                        "score": score,
                        "matches": nearest_matches,
                        "fitted_mz_da": np.asarray(fit["fitted_mz_da"], dtype=float),
                        "fitted_fraction": np.asarray(fit["fitted_fraction"], dtype=float),
                        "isotope_labels": isotope_labels,
                        "tier": shift_tier,
                        "calibration_shift_da": calibration_shift_da,
                    }
            if best_family is not None and float(best_family["score"]) >= score_min:
                family_candidates.append(best_family)
    peak_family_candidates: dict[str, list[dict[str, object]]] = {}
    peak_best_family: dict[str, dict[str, object]] = {}
    for family in sorted(family_candidates, key=lambda item: float(item["score"]), reverse=True):
        current_area_by_peak = {}
        for peak_idx, _ in family["matches"]:
            peak_id = str(peak_ids[peak_idx])
            current_row = peaks.loc[peaks["peak_id"] == peak_id].iloc[0]
            current_area_by_peak[peak_id] = float(
                current_row["fit_peak_area"]
                if "fit_peak_area" in current_row.index and pd.notna(current_row["fit_peak_area"])
                else current_row.get("prominence", current_row.get("peak_height", 0.0))
            )
        current_total_area = float(sum(current_area_by_peak.values()))
        fitted_mz_all = np.asarray(family["fitted_mz_da"], dtype=float)
        fitted_fraction = np.asarray(family.get("fitted_fraction", []), dtype=float)
        for peak_idx, isotope_idx in family["matches"]:
            peak_id = str(peak_ids[peak_idx])
            candidate_payload = {
                "element": str(family["element"]),
                "charge": int(family["charge"]),
                "isotope_label": str(family["isotope_labels"][isotope_idx]),
                "score": float(family["score"]),
                "match_count": int(len(family["matches"])),
                "fitted_mz_da": float(fitted_mz_all[isotope_idx]),
                "tier": str(family.get("tier", "tight")),
                "calibration_shift_da": float(family.get("calibration_shift_da", 0.0)),
                "family_fit_area": (
                    current_total_area * float(fitted_fraction[isotope_idx])
                    if fitted_fraction.size
                    else None
                ),
            }
            peak_family_candidates.setdefault(peak_id, []).append(candidate_payload)
            current_best = peak_best_family.get(peak_id)
            if current_best is None or float(candidate_payload["score"]) > float(current_best["score"]):
                peak_best_family[peak_id] = candidate_payload
    for peak_id, candidates in peak_family_candidates.items():
        candidates = sorted(
            candidates,
            key=lambda item: (
                float(item["score"]),
                int(item["match_count"]),
            ),
            reverse=True,
        )
        peak_index = peaks.index[peaks["peak_id"] == peak_id]
        if peak_index.empty:
            continue
        peaks.at[int(peak_index[0]), "family_hint_candidates"] = candidates
        best = candidates[0]
        second_score = float(candidates[1]["score"]) if len(candidates) > 1 else 0.0
        exclusivity = (
            float(np.clip((float(best["score"]) - second_score) / max(float(best["score"]), 1.0e-9), 0.0, 1.0))
            if len(candidates) > 1
            else 1.0
        )
        peaks.loc[peaks["peak_id"] == peak_id, "family_hint_element"] = str(best["element"])
        peaks.loc[peaks["peak_id"] == peak_id, "family_hint_charge"] = int(best["charge"])
        peaks.loc[peaks["peak_id"] == peak_id, "family_hint_isotope_label"] = str(best["isotope_label"])
        peaks.loc[peaks["peak_id"] == peak_id, "family_hint_score"] = float(best["score"])
        peaks.loc[peaks["peak_id"] == peak_id, "family_hint_match_count"] = int(best["match_count"])
        peaks.loc[peaks["peak_id"] == peak_id, "family_hint_exclusivity"] = exclusivity
        if best.get("family_fit_area") is not None:
            peaks.loc[peaks["peak_id"] == peak_id, "family_fit_area"] = float(best["family_fit_area"])
        peaks.loc[peaks["peak_id"] == peak_id, "family_hint_tier"] = str(best.get("tier", "tight"))
        fitted_mz = float(best["fitted_mz_da"])
        current_mz = float(peaks.loc[peaks["peak_id"] == peak_id, "peak_mz_da"].iloc[0])
        # Move the peak onto the reference mass scale: the family fingerprint
        # identified the species, so the corrected center is the fitted
        # position minus the shared shift. The shared shift absorbs both
        # calibration error and the apex pull of the one-sided thermal tail
        # (a proportional ~hundreds-of-ppm bias every histogram detector
        # shares); subtracting it is what makes hinted peaks land on the
        # reference ladder. Applies to both tiers - the recovery tier just
        # allows a larger shift.
        calibration_shift = float(best.get("calibration_shift_da", 0.0))
        if str(best.get("tier", "tight")) == "recovery" or abs(current_mz - fitted_mz) <= float(
            analysis_cfg["custom_family_hint_recentering_max_da"]
        ):
            peaks.loc[peaks["peak_id"] == peak_id, "peak_mz_da"] = fitted_mz - calibration_shift
            peaks.loc[peaks["peak_id"] == peak_id, "family_hint_calibration_shift_da"] = calibration_shift
    return peaks.sort_values("peak_mz_da", ignore_index=True)


def _raw_member_excess(
    raw_sorted: np.ndarray,
    target_mz: float,
    *,
    fwhm_da: float,
) -> tuple[float, float]:
    """Local hypothesis test at a predicted family-member position: excess
    event count in a core window over the sideband-estimated baseline, and
    its Poisson z-score. Used to verify members the blind detector missed
    (small isotopes riding on the thermal tail of a large neighbor)."""
    half = float(max(0.6 * fwhm_da, 0.02))
    core_lo = float(target_mz - half)
    core_hi = float(target_mz + half)
    core = int(np.searchsorted(raw_sorted, core_hi) - np.searchsorted(raw_sorted, core_lo))
    side_counts = 0
    side_width = 0.0
    for lo, hi in ((core_lo - 4.0 * half, core_lo - 1.5 * half), (core_hi + 1.5 * half, core_hi + 4.0 * half)):
        side_counts += int(np.searchsorted(raw_sorted, hi) - np.searchsorted(raw_sorted, lo))
        side_width += hi - lo
    baseline_rate = side_counts / max(side_width, 1.0e-9)
    expected = baseline_rate * (core_hi - core_lo)
    excess = core - expected
    z = excess / np.sqrt(max(expected, 1.0))
    return float(excess), float(z)


def _verify_missing_members(
    raw_sorted: np.ndarray,
    *,
    expected_mz: np.ndarray,
    fractions: np.ndarray,
    matched_isotopes: set[int],
    dominant_idx: int,
    shift: float,
    fwhm_dominant_da: float,
    config: dict,
) -> int:
    """Count family members absent from the peak list but verifiably present
    as a raw-spectrum excess at their predicted (shifted) positions."""
    analysis_cfg = config["analysis"]
    if not bool(analysis_cfg.get("custom_family_member_verify_enabled", True)):
        return 0
    min_z = float(analysis_cfg.get("custom_family_member_verify_min_z", 4.0))
    min_excess = float(analysis_cfg.get("custom_family_member_verify_min_excess", 60.0))
    dominant_mz = float(expected_mz[dominant_idx] + shift)
    dominant_excess, dominant_z = _raw_member_excess(
        raw_sorted, dominant_mz, fwhm_da=fwhm_dominant_da
    )
    if dominant_excess <= 0:
        return 0
    verified = 0
    for isotope_idx in range(expected_mz.size):
        if isotope_idx in matched_isotopes:
            continue
        expected_counts = dominant_excess * float(fractions[isotope_idx]) / max(
            float(fractions[dominant_idx]), 1.0e-9
        )
        if expected_counts < min_excess:
            continue
        target = float(expected_mz[isotope_idx] + shift)
        member_fwhm = fwhm_dominant_da * float(
            np.sqrt(max(target, 0.5) / max(dominant_mz, 0.5))
        )
        excess, z = _raw_member_excess(raw_sorted, target, fwhm_da=member_fwhm)
        if z >= min_z and excess >= min_excess:
            verified += 1
    return verified


def _molecular_isotopologues(formula: dict[str, int], *, min_probability: float = 5.0e-4) -> list[tuple[float, float]]:
    """Expand a neutral formula into (mass, probability) isotopologue lines,
    merging lines closer than 10 mDa."""
    from itertools import product as _product

    table = isotope_table()
    per_atom: list[list[tuple[float, float]]] = []
    for element, count in formula.items():
        family = table[table["symbol"] == element]
        options = [
            (float(row["mass_da"]), float(row["natural_abundance"]))
            for _, row in family.iterrows()
            if float(row["natural_abundance"]) >= 1.0e-4
        ]
        if not options:
            return []
        for _ in range(int(count)):
            per_atom.append(options)
    merged: dict[float, float] = {}
    for combo in _product(*per_atom):
        mass = round(sum(item[0] for item in combo), 5)
        probability = float(np.prod([item[1] for item in combo]))
        merged[mass] = merged.get(mass, 0.0) + probability
    items = sorted(merged.items())
    out: list[list[float]] = []
    for mass, probability in items:
        if out and mass - out[-1][0] <= 0.010:
            prev_mass, prev_prob = out[-1]
            total = prev_prob + probability
            out[-1] = [(prev_mass * prev_prob + mass * probability) / total, total]
        else:
            out.append([mass, probability])
    return [(mass, prob) for mass, prob in out if prob >= min_probability]


def _annotate_molecular_family_recovery(
    peaks: pd.DataFrame,
    *,
    raw_mz: np.ndarray,
    config: dict,
) -> pd.DataFrame:
    """Fingerprint recovery for molecular families (oxides, dimers, nitrides,
    carbides of elements already anchored by atomic fingerprints).

    Real datasets can carry large local mass-scale errors at high m/z where
    molecular species live (an FeO+ ladder shifted by tens of mDa). Atomic
    families recover from this via the calibration-recovery tier; this pass
    gives the same treatment to the most common molecular series, then moves
    matched peaks onto the reference mass scale so the assignment stage sees
    calibrated positions. Acceptance mirrors the atomic recovery tier:
    dominant isotopologue matched, at least three members matched, and the
    event-level family fit must reproduce the expected isotopologue pattern.
    """
    analysis_cfg = config["analysis"]
    if peaks.empty or not bool(analysis_cfg.get("custom_molecular_family_recovery_enabled", True)):
        return peaks
    anchors: list[str] = []
    if "family_hint_element" in peaks.columns:
        anchors = [
            str(element)
            for element in peaks["family_hint_element"].dropna().unique().tolist()
            if str(element)
        ]
    if not anchors:
        return peaks
    peaks = peaks.sort_values("peak_mz_da").reset_index(drop=True).copy()
    raw_sorted_molecular = np.sort(np.asarray(raw_mz, dtype=np.float64))
    if "molecular_family_label" not in peaks.columns:
        peaks["molecular_family_label"] = pd.Series([None] * peaks.shape[0], dtype=object)
        peaks["molecular_family_shift_da"] = np.nan
        peaks["molecular_family_formula"] = pd.Series([None] * peaks.shape[0], dtype=object)
        peaks["molecular_family_charge"] = np.nan
    peak_positions = peaks["peak_mz_da"].to_numpy(dtype=float)
    peak_prominence = peaks["prominence"].fillna(0.0).to_numpy(dtype=float)
    hinted_mask = (
        peaks["family_hint_element"].notna().to_numpy()
        if "family_hint_element" in peaks.columns
        else np.zeros(peaks.shape[0], dtype=bool)
    )
    match_ppm = float(analysis_cfg.get("custom_family_hint_match_ppm", 350.0))
    tolerance_floor = float(analysis_cfg.get("custom_family_hint_match_tolerance_floor_da", 0.012))
    tolerance_cap = float(analysis_cfg.get("custom_family_hint_match_tolerance_da", 0.04))
    recovery_shift_ppm = float(analysis_cfg.get("custom_family_hint_recovery_shift_ppm", 4000.0))
    min_matches = int(analysis_cfg.get("custom_molecular_family_min_matches", 3))
    min_coherence = float(analysis_cfg.get("custom_molecular_family_min_coherence", 0.75))
    min_counts = float(analysis_cfg.get("custom_family_hint_min_total_counts", 200.0))
    max_anchors = int(analysis_cfg.get("custom_molecular_family_max_anchor_elements", 6))
    partner_formulas = [
        {"O": 1}, {"O": 2}, {"O": 3}, {"N": 1}, {"C": 1},
    ]

    def match_tolerance(mz: float) -> float:
        return float(np.clip(match_ppm * 1.0e-6 * mz, tolerance_floor, tolerance_cap))

    candidates: list[dict[str, object]] = []
    for anchor in anchors[:max_anchors]:
        formulas = [{anchor: 2}, {anchor: 2, "O": 1}, {anchor: 2, "O": 2}]
        for partner in partner_formulas:
            formulas.append({anchor: 1, **partner})
        for formula in formulas:
            lines = _molecular_isotopologues(formula)
            if len(lines) < 2:
                continue
            masses = np.asarray([line[0] for line in lines], dtype=float)
            fractions = np.asarray([line[1] for line in lines], dtype=float)
            fractions = fractions / fractions.sum()
            dominant_idx = int(np.argmax(fractions))
            for charge in (1, 2):
                expected_mz = masses / float(charge)
                if expected_mz[dominant_idx] < peak_positions.min() - 0.2 or expected_mz[dominant_idx] > peak_positions.max() + 0.2:
                    continue
                dominant_mz = float(expected_mz[dominant_idx])
                shift_limit = recovery_shift_ppm * 1.0e-6 * dominant_mz
                near = np.abs(peak_positions - dominant_mz) <= shift_limit
                if not near.any():
                    continue
                anchor_order = np.argsort(-peak_prominence * near)
                for anchor_idx in anchor_order[:4]:
                    if not near[anchor_idx]:
                        break
                    shift = float(peak_positions[anchor_idx] - dominant_mz)
                    used: set[int] = set()
                    matches: list[tuple[int, int]] = []
                    for isotope_idx, target in enumerate(expected_mz + shift):
                        order = np.argsort(np.abs(peak_positions - target))
                        for peak_idx in order[:4]:
                            if int(peak_idx) in used:
                                continue
                            if abs(float(peak_positions[peak_idx]) - float(target)) <= match_tolerance(float(target)):
                                used.add(int(peak_idx))
                                matches.append((int(peak_idx), isotope_idx))
                            break
                    matched_isotopes = {iso for _, iso in matches}
                    if dominant_idx not in matched_isotopes:
                        continue
                    effective_matches = len(matches)
                    if effective_matches < min_matches:
                        dominant_rows = [p for p, iso in matches if iso == dominant_idx]
                        fwhm_dom = float(
                            peaks["fwhm_da"].iloc[dominant_rows[0]]
                            if dominant_rows and pd.notna(peaks["fwhm_da"].iloc[dominant_rows[0]])
                            else dominant_mz / 800.0
                        )
                        effective_matches += _verify_missing_members(
                            raw_sorted_molecular,
                            expected_mz=expected_mz,
                            fractions=fractions,
                            matched_isotopes=matched_isotopes,
                            dominant_idx=dominant_idx,
                            shift=shift,
                            fwhm_dominant_da=max(fwhm_dom, 0.01),
                            config=config,
                        )
                    if effective_matches < 2:
                        continue
                    fit = fit_isotope_family_tof(
                        np.asarray(raw_mz, dtype=np.float64),
                        target_mz_da=expected_mz,
                        sigma_da_init=max(0.01, dominant_mz / 800.0 / 2.354820045),
                        window_margin_da=float(analysis_cfg["custom_family_hint_window_margin_da"]),
                        shift_tolerance_da=abs(shift) + 0.03,
                        sigma_floor_da=float(analysis_cfg["custom_family_hint_sigma_floor_da"]),
                        sigma_ceiling_da=float(analysis_cfg["custom_family_hint_sigma_ceiling_da"]),
                        bins_per_sigma=float(analysis_cfg["custom_family_hint_bins_per_sigma"]),
                        shift_penalty=0.0,
                        sigma_penalty=float(analysis_cfg["custom_family_hint_sigma_penalty"]),
                    )
                    if fit is None:
                        continue
                    fitted_counts = np.asarray(fit["fitted_counts"], dtype=float)
                    if float(fitted_counts.sum()) < min_counts:
                        continue
                    fitted_fraction = np.asarray(fit["fitted_fraction"], dtype=float)
                    l1_distance = float(np.abs(fitted_fraction - fractions).sum())
                    coherence = float(max(0.0, 1.0 - l1_distance / 2.0))
                    if coherence < min_coherence:
                        continue
                    if float(fitted_fraction[dominant_idx]) < 0.7 * float(fractions[dominant_idx]):
                        continue
                    if effective_matches < min_matches and not (len(matches) == 2 and effective_matches == 2):
                        continue
                    if len(matches) == 2 and effective_matches == 2:
                        # Two-member molecular family: require the minor
                        # isotopologue's fitted fraction to reproduce its
                        # natural value tightly (see atomic counterpart).
                        minor_rel_tol = float(
                            analysis_cfg.get("custom_family_recovery_two_member_minor_rel_tolerance", 0.4)
                        )
                        minor_indices = [iso for _, iso in matches if iso != dominant_idx]
                        minor_idx = max(minor_indices, key=lambda iso: float(fractions[iso]))
                        expected_minor = float(fractions[minor_idx])
                        fitted_minor = float(fitted_fraction[minor_idx])
                        if not (
                            minor_rel_tol > 0.0
                            and expected_minor > 0
                            and abs(fitted_minor - expected_minor) <= minor_rel_tol * expected_minor
                        ):
                            continue
                    label = "".join(
                        f"{symbol}{count if count > 1 else ''}" for symbol, count in formula.items()
                    ) + f"^{charge}+"
                    candidates.append(
                        {
                            "label": label,
                            "formula": dict(formula),
                            "charge": int(charge),
                            "coherence": coherence,
                            "score": coherence * len(matches),
                            "shift": shift,
                            "matches": matches,
                            "expected_mz": expected_mz,
                        }
                    )
                    break
    claimed: set[int] = set()
    for candidate in sorted(candidates, key=lambda item: float(item["score"]), reverse=True):
        member_indices = [peak_idx for peak_idx, _ in candidate["matches"]]
        if any(idx in claimed or hinted_mask[idx] for idx in member_indices):
            continue
        for peak_idx, isotope_idx in candidate["matches"]:
            claimed.add(peak_idx)
            reference_mz = float(np.asarray(candidate["expected_mz"], dtype=float)[isotope_idx])
            peaks.at[peak_idx, "molecular_family_label"] = str(candidate["label"])
            peaks.at[peak_idx, "molecular_family_shift_da"] = float(candidate["shift"])
            peaks.at[peak_idx, "molecular_family_formula"] = dict(candidate["formula"])
            peaks.at[peak_idx, "molecular_family_charge"] = int(candidate["charge"])
            if "peak_mz_raw_da" not in peaks.columns:
                peaks["peak_mz_raw_da"] = peaks["peak_mz_da"].astype(float)
            if pd.isna(peaks.at[peak_idx, "peak_mz_raw_da"]):
                peaks.at[peak_idx, "peak_mz_raw_da"] = float(peaks.at[peak_idx, "peak_mz_da"])
            peaks.at[peak_idx, "peak_mz_da"] = reference_mz
    return peaks.sort_values("peak_mz_da", ignore_index=True)


def _reduce_fitted_components(
    amplitudes: np.ndarray,
    centers_fit: np.ndarray,
    sigma_fit: np.ndarray,
    *,
    merge_tolerance_da: float,
    min_relative_amplitude: float,
    min_absolute_amplitude: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if amplitudes.size == 0:
        return amplitudes, centers_fit, sigma_fit
    keep_mask = amplitudes >= max(float(min_absolute_amplitude), float(np.max(amplitudes)) * float(min_relative_amplitude))
    amplitudes = amplitudes[keep_mask]
    centers_fit = centers_fit[keep_mask]
    sigma_fit = sigma_fit[keep_mask]
    if amplitudes.size <= 1:
        return amplitudes, centers_fit, sigma_fit
    order = np.argsort(centers_fit)
    amplitudes = amplitudes[order]
    centers_fit = centers_fit[order]
    sigma_fit = sigma_fit[order]
    merged_amplitudes: list[float] = []
    merged_centers: list[float] = []
    merged_sigma: list[float] = []
    idx = 0
    while idx < amplitudes.size:
        current_amp = float(amplitudes[idx])
        current_center = float(centers_fit[idx])
        current_sigma = float(sigma_fit[idx])
        total_weight = max(current_amp, 1.0e-6)
        idx += 1
        while idx < amplitudes.size and abs(float(centers_fit[idx]) - current_center) <= float(merge_tolerance_da):
            neighbor_amp = float(amplitudes[idx])
            neighbor_center = float(centers_fit[idx])
            neighbor_sigma = float(sigma_fit[idx])
            total_weight += max(neighbor_amp, 1.0e-6)
            current_center = (current_center * max(current_amp, 1.0e-6) + neighbor_center * max(neighbor_amp, 1.0e-6)) / total_weight
            current_sigma = (current_sigma * max(current_amp, 1.0e-6) + neighbor_sigma * max(neighbor_amp, 1.0e-6)) / total_weight
            current_amp += neighbor_amp
            idx += 1
        merged_amplitudes.append(current_amp)
        merged_centers.append(current_center)
        merged_sigma.append(current_sigma)
    return (
        np.asarray(merged_amplitudes, dtype=np.float64),
        np.asarray(merged_centers, dtype=np.float64),
        np.asarray(merged_sigma, dtype=np.float64),
    )


def _fit_crowded_peak_windows(
    peaks: pd.DataFrame,
    *,
    raw_mz: np.ndarray,
    config: dict,
    cluster_gap_da: float,
    min_peaks: int,
    max_window_span_da: float,
    window_margin_da: float,
    center_tolerance_da: float,
    sigma_floor_da: float,
    sigma_ceiling_da: float,
    min_relative_amplitude: float,
    min_absolute_amplitude: float,
    merge_tolerance_da: float,
    max_fwhm_da: float,
):
    if peaks.empty:
        return peaks.copy(), []
    fitted = peaks.copy()
    crowded_clusters = _crowded_peak_clusters(
        fitted,
        cluster_gap_da=float(cluster_gap_da),
        min_peaks=int(min_peaks),
        max_window_span_da=float(max_window_span_da),
    )
    fit_diagnostics: list[dict[str, object]] = []
    for cluster_number, cluster_peak_ids in enumerate(crowded_clusters, start=1):
        cluster_rows = fitted[fitted["peak_id"].astype(str).isin(cluster_peak_ids)].sort_values("peak_mz_da").copy()
        if cluster_rows.empty:
            continue
        seed_rows = _select_fit_seed_rows(
            cluster_rows,
            relative_prominence_min=float(config["analysis"].get("custom_local_fit_seed_relative_prominence_min", 0.01)),
            min_detection_sources=int(config["analysis"].get("custom_local_fit_seed_min_detection_sources", 2)),
            min_peaks=int(min_peaks),
        )
        if seed_rows.empty:
            continue
        seed_peak_ids = seed_rows["peak_id"].astype(str).tolist()
        seed_centers = seed_rows["peak_mz_da"].to_numpy(dtype=np.float64)
        seed_fwhms = seed_rows["fwhm_da"].fillna(0.0).to_numpy(dtype=np.float64)
        if seed_centers.size > 1:
            min_gap = float(np.min(np.diff(np.sort(seed_centers))))
            overlap_gap_ratio = float(config["analysis"].get("custom_local_fit_max_gap_fwhm_ratio", 3.0))
            # If every seed is separated by several FWHMs there is nothing to
            # deblend, and forcing a shared-width mixture through well-separated
            # peaks of very different widths can only corrupt their areas.
            if min_gap > overlap_gap_ratio * max(float(np.max(seed_fwhms)), 1.0e-6):
                continue
        seed_heights_check = seed_rows["peak_height"].fillna(0.0).to_numpy(dtype=np.float64)
        significant_ratio = float(config["analysis"].get("custom_local_fit_min_significant_height_ratio", 0.02))
        significant_count = int(np.sum(seed_heights_check >= significant_ratio * max(float(np.max(seed_heights_check)), 1.0)))
        # Deblending one real peak against near-baseline junk is pointless and
        # lets the shared-width model damage the real peak's area.
        if significant_count < 2:
            continue
        fit_result = fit_local_tof_peak_mixture(
            np.asarray(raw_mz, dtype=np.float64),
            seed_mz_da=seed_centers,
            seed_fwhm_da=np.maximum(seed_rows["fwhm_da"].to_numpy(dtype=np.float64), float(sigma_floor_da) * 2.354820045),
            window_margin_da=float(window_margin_da),
            center_tolerance_da=float(center_tolerance_da),
            sigma_floor_da=float(sigma_floor_da),
            sigma_ceiling_da=float(sigma_ceiling_da),
            bins_per_sigma=float(config["analysis"].get("custom_local_fit_bins_per_sigma", 5.0)),
            offset_penalty=float(config["analysis"].get("custom_local_fit_offset_penalty", 0.15)),
            sigma_penalty=float(config["analysis"].get("custom_local_fit_sigma_penalty", 0.08)),
        )
        if fit_result is None:
            continue
        window_signal = float(
            np.sum(
                np.maximum(
                    np.asarray(fit_result["y_t"], dtype=np.float64)
                    - np.asarray(fit_result["baseline_curve"], dtype=np.float64),
                    0.0,
                )
            )
        )
        fitted_total = float(np.sum(np.asarray(fit_result["fitted_counts"], dtype=np.float64)))
        # A mixture that explains less than half of the above-baseline signal has
        # failed (e.g. a spike narrower than the width floor); keep the seeds.
        if fitted_total < 0.5 * max(window_signal, 1.0):
            fit_diagnostics.append(
                {
                    "cluster_id": int(cluster_number),
                    "fit_space": "tof",
                    "seed_peak_count": int(seed_rows.shape[0]),
                    "rejected_underfit": True,
                    "fitted_total": fitted_total,
                    "window_signal": window_signal,
                }
            )
            continue
        # The Gaussian mixture has no tail component, so its absolute areas
        # run ~10-25% low on tailed peaks while isolated peaks elsewhere keep
        # full integrals - an inconsistent scale that skews element ratios.
        # Use the mixture for apportionment only: rescale components so the
        # window total equals the above-baseline signal.
        area_scale = float(np.clip(window_signal / max(fitted_total, 1.0), 0.8, 2.0))
        fitted_counts, centers_fit, fwhm_fit = _reduce_fitted_components(
            area_scale * np.asarray(fit_result["fitted_counts"], dtype=np.float64),
            np.asarray(fit_result["fitted_mz_da"], dtype=np.float64),
            np.asarray(fit_result["fitted_fwhm_da"], dtype=np.float64),
            merge_tolerance_da=float(merge_tolerance_da),
            min_relative_amplitude=float(min_relative_amplitude),
            min_absolute_amplitude=float(min_absolute_amplitude),
        )
        if fitted_counts.size == 0:
            continue
        ordered_seed_centers = np.sort(seed_centers)
        seed_span_da = float(ordered_seed_centers[-1] - ordered_seed_centers[0]) if ordered_seed_centers.size > 1 else 0.0
        seed_min_gap_da = (
            float(np.min(np.diff(ordered_seed_centers)))
            if ordered_seed_centers.size > 1
            else 0.0
        )
        fallback_seed_count_min = int(config["analysis"].get("custom_local_fit_fallback_seed_count_min", 3))
        fallback_max_component_ratio = float(config["analysis"].get("custom_local_fit_fallback_max_component_ratio", 0.5))
        fallback_min_seed_gap_da = float(config["analysis"].get("custom_local_fit_fallback_min_seed_gap_da", 0.30))
        fallback_min_seed_span_da = float(config["analysis"].get("custom_local_fit_fallback_min_seed_span_da", 0.80))
        if (
            seed_rows.shape[0] >= fallback_seed_count_min
            and fitted_counts.size < max(1, int(np.ceil(seed_rows.shape[0] * fallback_max_component_ratio)))
            and seed_min_gap_da >= fallback_min_seed_gap_da
            and seed_span_da >= fallback_min_seed_span_da
        ):
            fit_diagnostics.append(
                {
                    "cluster_id": int(cluster_number),
                    "window_left_da": float(np.min(seed_centers) - float(window_margin_da)),
                    "window_right_da": float(np.max(seed_centers) + float(window_margin_da)),
                    "fit_space": "tof",
                    "seed_peak_ids": seed_peak_ids,
                    "seed_peak_count": int(seed_rows.shape[0]),
                    "seed_span_da": seed_span_da,
                    "seed_min_gap_da": seed_min_gap_da,
                    "reduced_component_count": int(fitted_counts.size),
                    "fallback_to_seed_rows": True,
                }
            )
            continue
        preserve_unmatched_seed_gap_da = float(
            config["analysis"].get("custom_local_fit_preserve_seed_min_gap_da", max(float(merge_tolerance_da) * 2.0, 0.14))
        )
        preserve_unmatched_seed_relative_prominence = float(
            config["analysis"].get("custom_local_fit_preserve_seed_relative_prominence_min", 0.05)
        )
        preserve_unmatched_seed_min_prominence = float(
            config["analysis"].get("custom_local_fit_preserve_seed_min_prominence", 150.0)
        )
        preserve_unmatched_seed_min_height = float(
            config["analysis"].get("custom_local_fit_preserve_seed_min_height", 150.0)
        )
        cluster_prominence_max = float(cluster_rows["prominence"].fillna(0.0).max()) if "prominence" in cluster_rows.columns else 0.0
        seed_prominences = seed_rows["prominence"].fillna(0.0).to_numpy(dtype=np.float64)
        seed_heights = seed_rows["peak_height"].fillna(0.0).to_numpy(dtype=np.float64)
        prominence_support_mask = (
            (seed_prominences >= preserve_unmatched_seed_min_prominence)
            | (seed_prominences >= cluster_prominence_max * preserve_unmatched_seed_relative_prominence)
        )
        min_seed_distances = (
            np.min(np.abs(seed_centers[:, None] - centers_fit[None, :]), axis=1) if centers_fit.size else np.full(seed_centers.shape[0], np.inf)
        )
        preserve_seed_mask = (
            (min_seed_distances >= preserve_unmatched_seed_gap_da)
            & prominence_support_mask
            & (seed_heights >= preserve_unmatched_seed_min_height)
        )
        preserved_seed_indices = seed_rows.index.to_numpy(dtype=int)[preserve_seed_mask]
        preserved_seed_ids = set(seed_rows.loc[preserve_seed_mask, "peak_id"].astype(str).tolist())
        assigned_groups: list[list[int]] = [[] for _ in range(fitted_counts.size)]
        original_positions = seed_rows.index.to_list()
        for original_index, original_center in zip(original_positions, seed_centers, strict=True):
            if int(original_index) in preserved_seed_indices:
                continue
            target = int(np.argmin(np.abs(centers_fit - float(original_center))))
            assigned_groups[target].append(int(original_index))
        replacement_rows: list[dict[str, object]] = []
        replaced_peak_ids: set[str] = set()
        for component_idx in range(fitted_counts.size):
            assigned = assigned_groups[component_idx]
            if not assigned:
                assigned = [int(seed_rows.index[int(np.argmin(np.abs(seed_centers - centers_fit[component_idx])))] )]
            source_rows = fitted.loc[assigned].copy()
            representative = source_rows.sort_values(
                [column for column in ["detection_source_count", "prominence", "peak_height"] if column in source_rows.columns],
                ascending=[False] * len([column for column in ["detection_source_count", "prominence", "peak_height"] if column in source_rows.columns]),
            ).iloc[0].copy()
            seed_center = float(representative["peak_mz_da"])
            fitted_center = float(centers_fit[component_idx])
            if abs(fitted_center - seed_center) <= float(config["analysis"].get("custom_local_fit_center_replace_max_shift_da", 0.008)):
                peak_center = fitted_center
            else:
                peak_center = seed_center
            fwhm_value = float(min(max_fwhm_da, float(fwhm_fit[component_idx])))
            model_area = float(max(fitted_counts[component_idx], 0.0))
            detection_sources: list[str] = []
            for value in source_rows.get("detection_sources", pd.Series(dtype=object)).tolist():
                if isinstance(value, list):
                    detection_sources.extend(str(item) for item in value)
                elif pd.notna(value):
                    detection_sources.append(str(value))
            if not detection_sources and "detection_source" in source_rows.columns:
                detection_sources = [str(value) for value in source_rows["detection_source"].tolist() if pd.notna(value)]
            representative["peak_mz_da"] = peak_center
            representative["peak_height"] = float(max(source_rows["peak_height"].max(), 0.0))
            representative["smoothed_height"] = float(max(source_rows["smoothed_height"].max(), representative["peak_height"]))
            representative["prominence"] = float(max(float(source_rows["prominence"].max()) if "prominence" in source_rows.columns else 0.0, model_area))
            representative["left_fwhm_mz_da"] = float(peak_center - fwhm_value / 2.0)
            representative["right_fwhm_mz_da"] = float(peak_center + fwhm_value / 2.0)
            representative["fwhm_da"] = fwhm_value
            representative["fit_peak_area"] = model_area
            representative["fit_cluster_id"] = int(cluster_number)
            representative["fit_space"] = "tof"
            representative["fit_sigma_t"] = float(fit_result["sigma_t"])
            representative["detection_sources"] = sorted(set(detection_sources))
            representative["detection_source_count"] = int(len(representative["detection_sources"]))
            replacement_rows.append(representative.to_dict())
            replaced_peak_ids.update(source_rows["peak_id"].astype(str).tolist())
        replaced_peak_ids.difference_update(preserved_seed_ids)
        fitted = fitted.loc[~fitted["peak_id"].astype(str).isin(replaced_peak_ids)].copy()
        if replacement_rows:
            fitted = pd.concat([fitted, pd.DataFrame(replacement_rows)], ignore_index=True, sort=False)
        fit_diagnostics.append(
            {
                "cluster_id": int(cluster_number),
                "window_left_da": float(np.min(seed_centers) - float(window_margin_da)),
                "window_right_da": float(np.max(seed_centers) + float(window_margin_da)),
                "cluster_peak_count": int(cluster_rows.shape[0]),
                "seed_peak_count": int(seed_rows.shape[0]),
                "fit_peak_count": int(len(replacement_rows)),
                "preserved_seed_count": int(len(preserved_seed_ids)),
                "preserved_seed_ids": sorted(preserved_seed_ids),
                "fit_space": "tof",
                "fit_sigma_t": float(fit_result["sigma_t"]),
            }
        )
    fitted = fitted.sort_values("peak_mz_da", ignore_index=True)
    return fitted, fit_diagnostics


def _analyze_sample_regions(sample_data, output_dir, config, artifacts):
    """Segment the specimen into composition regions and, when it is not
    homogeneous, run the full custom analysis independently on each region's
    own mass spectrum under output_dir/regions/."""
    import json

    segmentation = segment_sample_regions(
        x_nm=sample_data.x_nm,
        y_nm=sample_data.y_nm,
        z_nm=sample_data.z_nm,
        m_over_z_da=sample_data.m_over_z_da,
        peaks=artifacts.peaks,
        assignments=artifacts.assignments,
        config=config,
    )
    if segmentation is None:
        return artifacts
    voxel_table = segmentation.pop("voxel_table", None)
    event_region = segmentation.pop("event_region", None)
    if voxel_table is not None and not voxel_table.empty:
        voxel_table.to_csv(output_dir / "segmentation_voxels.csv", index=False)
    region_entries = list(segmentation.get("region_summary", []))
    if int(segmentation.get("n_regions", 1)) > 1 and event_region is not None:
        from rangefinder.io.pos_loader import PosSampleData, PosSampleMetadata
        from rangefinder.utils.paths import slugify

        for entry in region_entries:
            region = int(entry["region"])
            mask = event_region == region
            region_name = f"{sample_data.metadata.sample_name} region {region + 1}"
            region_metadata = PosSampleMetadata(
                path=sample_data.metadata.path,
                sample_name=region_name,
                sample_slug=slugify(region_name),
                event_count=int(mask.sum()),
                file_size_bytes=int(mask.sum()) * 16,
                verified_big_endian_float32=sample_data.metadata.verified_big_endian_float32,
                verified_columns=sample_data.metadata.verified_columns,
            )
            region_data = PosSampleData(
                metadata=region_metadata,
                x_nm=sample_data.x_nm[mask],
                y_nm=sample_data.y_nm[mask],
                z_nm=sample_data.z_nm[mask],
                m_over_z_da=sample_data.m_over_z_da[mask],
            )
            region_dir = output_dir / "regions" / f"region-{region + 1}"
            region_artifacts = run_custom_pipeline(
                region_data,
                region_dir,
                config,
                enable_segmentation=False,
            )
            elements = region_artifacts.elemental_composition
            if not elements.empty:
                entry["top_elements_at_pct"] = {
                    str(row["element"]): round(float(row["atomic_percent_weighted"]), 3)
                    for _, row in elements.sort_values(
                        "atomic_percent_weighted", ascending=False
                    ).head(8).iterrows()
                    if float(row["atomic_percent_weighted"]) >= 0.1
                }
            entry["output_dir"] = str(region_dir)
    segmentation["region_summary"] = region_entries
    (output_dir / "region_summary.json").write_text(
        json.dumps(segmentation, indent=2, default=str),
        encoding="utf-8",
    )
    artifacts.diagnostics["segmentation"] = segmentation
    (output_dir / "diagnostics.json").write_text(
        json.dumps(artifacts.diagnostics, indent=2, default=str),
        encoding="utf-8",
    )
    return artifacts


def detect_custom_peaks(sample_data, config):
    """Rangefinder's detection stage: multi-resolution + log-peakiness peak
    detection, crowded-window deblending, isotope- and molecular-family
    fingerprint recovery, area integration, and the final significance gate.
    Returns (centers, counts, peaks, detection_diagnostics). The full pipeline
    and the detection-ablation harness call this so there is exactly one
    Rangefinder detector implementation."""
    analysis_cfg = config["analysis"]
    base_bin_width = detection_bin_width(
        sample_data.m_over_z_da,
        list(analysis_cfg["bin_width_candidates_da"]),
        min_mz=float(analysis_cfg["min_mz"]),
        max_mz=float(analysis_cfg["zoom_max_mz"]),
        max_detection_bin_width_da=float(analysis_cfg["max_detection_bin_width_da"]),
    )
    detection_bin_widths = sorted(
        {
            float(candidate)
            for candidate in analysis_cfg["bin_width_candidates_da"]
            if float(candidate) <= float(analysis_cfg["max_detection_bin_width_da"])
        }
        | {float(base_bin_width)}
    )
    integration_bin_width = float(min(detection_bin_widths))
    centers, counts, _edges = build_histogram(
        sample_data.m_over_z_da,
        min_mz=float(analysis_cfg["min_mz"]),
        max_mz=float(analysis_cfg["max_mz"]),
        bin_width_da=integration_bin_width,
    )
    detected_tables = []
    detection_diagnostics = []
    for detect_bin_width in detection_bin_widths:
        detect_centers, detect_counts, _ = build_histogram(
            sample_data.m_over_z_da,
            min_mz=float(analysis_cfg["min_mz"]),
            max_mz=float(analysis_cfg["max_mz"]),
            bin_width_da=float(detect_bin_width),
        )
        prominence = default_prominence(
            detect_counts,
            quantile=float(analysis_cfg["pyccapt_prominence_quantile"]),
            minimum=float(analysis_cfg["pyccapt_min_prominence"]),
        )
        detected = detect_peaks_from_histogram(
            detect_centers,
            detect_counts,
            prominence=prominence,
            distance_bins=max(1, int(round(int(analysis_cfg["pyccapt_distance_bins"]) * detect_bin_width / integration_bin_width))),
            rel_height=0.5,
            smooth_window_bins=int(analysis_cfg["custom_savgol_window_bins"]),
            smooth_polyorder=int(analysis_cfg["custom_savgol_polyorder"]),
            method_label="custom",
            max_fwhm_da=float(analysis_cfg["custom_max_fwhm_da"]),
        )
        detected = refine_peak_centers_parabolic(detected, detect_centers, detect_counts)
        if bool(analysis_cfg.get("custom_raw_refine_enabled", True)):
            detected = refine_peak_centers_raw_local_mode(
                detected,
                sample_data.m_over_z_da,
                search_half_width_da=float(analysis_cfg["custom_raw_refine_window_half_width_da"]),
                local_bin_width_da=float(analysis_cfg["custom_raw_refine_local_bin_width_da"]),
                smooth_sigma_da=float(analysis_cfg["custom_raw_refine_smooth_sigma_da"]),
                min_events=int(analysis_cfg["custom_raw_refine_min_events"]),
                max_shift_da=float(analysis_cfg["custom_raw_refine_max_shift_da"]),
            )
        if not detected.empty:
            detected["detection_source"] = f"{detect_bin_width:.4f}Da"
            detected["source_bin_width_da"] = float(detect_bin_width)
            detected_tables.append(detected)
        detection_diagnostics.append(
            {
                "bin_width_da": float(detect_bin_width),
                "prominence": float(prominence),
                "peak_count": int(detected.shape[0]),
            }
        )
    if bool(analysis_cfg.get("custom_log_peakiness_enabled", False)):
        log_candidates = detect_peaks_log_peakiness_candidates(
            centers,
            counts,
            method_label="custom",
            smooth_sigma_bins=float(analysis_cfg["custom_log_smooth_sigma_bins"]),
            baseline_window_bins=int(analysis_cfg["custom_log_baseline_window_bins"]),
            min_excess=float(analysis_cfg["custom_log_min_excess"]),
            min_prominence=float(analysis_cfg["custom_log_min_prominence"]),
            min_count=float(analysis_cfg["custom_log_min_count"]),
            min_distance_bins=int(analysis_cfg["custom_log_min_distance_bins"]),
            min_detection_percentile=float(analysis_cfg["custom_log_min_detection_percentile"]),
            max_detection_percentile=float(analysis_cfg["custom_log_max_detection_percentile"]),
            tail_strength_center=float(analysis_cfg["custom_log_tail_strength_center"]),
            tail_strength_scale=float(analysis_cfg["custom_log_tail_strength_scale"]),
            max_fwhm_da=float(analysis_cfg["custom_max_fwhm_da"]),
        )
        log_detected = detect_peaks_log_peakiness(
            centers,
            counts,
            method_label="custom",
            smooth_sigma_bins=float(analysis_cfg["custom_log_smooth_sigma_bins"]),
            baseline_window_bins=int(analysis_cfg["custom_log_baseline_window_bins"]),
            min_excess=float(analysis_cfg["custom_log_min_excess"]),
            min_prominence=float(analysis_cfg["custom_log_min_prominence"]),
            min_count=float(analysis_cfg["custom_log_min_count"]),
            min_distance_bins=int(analysis_cfg["custom_log_min_distance_bins"]),
            min_detection_percentile=float(analysis_cfg["custom_log_min_detection_percentile"]),
            max_detection_percentile=float(analysis_cfg["custom_log_max_detection_percentile"]),
            tail_strength_center=float(analysis_cfg["custom_log_tail_strength_center"]),
            tail_strength_scale=float(analysis_cfg["custom_log_tail_strength_scale"]),
            max_fwhm_da=float(analysis_cfg["custom_max_fwhm_da"]),
        )
        merge_tolerance_da = float(max(integration_bin_width * 1.5, 0.015))
        regular_peak_mz = (
            np.concatenate([table["peak_mz_da"].to_numpy(dtype=float) for table in detected_tables if not table.empty])
            if detected_tables
            else np.asarray([], dtype=float)
        )
        regular_support_mz = np.asarray([], dtype=float)
        if detected_tables:
            merged_regular_support = merge_peak_tables(
                detected_tables,
                merge_tolerance_da=merge_tolerance_da,
                method_label="custom",
            )
            if not merged_regular_support.empty and "detection_source_count" in merged_regular_support.columns:
                regular_support_mz = merged_regular_support.loc[
                    merged_regular_support["detection_source_count"].fillna(0).astype(int) >= 2,
                    "peak_mz_da",
                ].to_numpy(dtype=float)
        rescue_detected = pd.DataFrame()
        if not log_candidates.empty and regular_peak_mz.size > 0:
            rescue_detected = _rescue_log_only_peaks(
                log_candidates,
                regular_peak_mz=regular_peak_mz,
                regular_support_mz=regular_support_mz,
                merge_tolerance_da=merge_tolerance_da,
                config=config,
            )
            log_detected = log_detected[
                log_detected["peak_mz_da"].map(
                    lambda value: bool(np.min(np.abs(regular_peak_mz - float(value))) <= merge_tolerance_da)
                )
            ].reset_index(drop=True)
        if not log_detected.empty:
            log_detected["detection_source"] = "log-peakiness"
            log_detected["source_bin_width_da"] = float(integration_bin_width)
            detected_tables.append(log_detected)
        if not rescue_detected.empty:
            detected_tables.append(rescue_detected)
        detection_diagnostics.append(
            {
                "bin_width_da": float(integration_bin_width),
                "prominence": None,
                "peak_count": int(log_detected.shape[0]),
                "detection_source": "log-peakiness",
                "rescue_peak_count": int(rescue_detected.shape[0]),
            }
        )
    peaks = merge_peak_tables(
        detected_tables,
        merge_tolerance_da=float(max(integration_bin_width * 1.5, 0.015)),
        method_label="custom",
    )
    peaks = _suppress_single_source_shoulders(
        peaks,
        exclusion_da=float(analysis_cfg["custom_single_source_neighbor_exclusion_da"]),
        prominence_ratio_max=float(analysis_cfg["custom_single_source_prominence_ratio_max"]),
        min_neighbor_da=float(analysis_cfg.get("custom_single_source_neighbor_min_da", 0.04)),
        neighbor_fwhm_scale=float(analysis_cfg.get("custom_single_source_neighbor_fwhm_scale", 1.25)),
    )
    if bool(analysis_cfg.get("custom_tail_artifact_suppression_enabled", True)):
        peaks = _suppress_tail_artifacts(
            peaks,
            centers=centers,
            counts=counts,
            max_distance_da=float(analysis_cfg.get("custom_tail_artifact_max_distance_da", 0.30)),
            height_ratio_min=float(analysis_cfg.get("custom_tail_artifact_height_ratio_min", 12.0)),
            valley_ratio_min=float(analysis_cfg.get("custom_tail_artifact_valley_ratio_min", 0.5)),
        )
    if bool(analysis_cfg.get("custom_local_snap_enabled", False)):
        peaks = _snap_and_merge_local_maxima(
            peaks,
            centers=centers,
            counts=counts,
            search_da=float(analysis_cfg["custom_local_snap_search_da"]),
            merge_tolerance_da=float(analysis_cfg["custom_local_snap_merge_tolerance_da"]),
            smooth_window_bins=int(analysis_cfg["custom_local_snap_smooth_window_bins"]),
        )
    local_fit_diagnostics = []
    if bool(analysis_cfg.get("custom_local_fit_enabled", False)):
        peaks, local_fit_diagnostics = _fit_crowded_peak_windows(
            peaks,
            raw_mz=sample_data.m_over_z_da,
            config=config,
            cluster_gap_da=float(analysis_cfg["custom_local_fit_cluster_gap_da"]),
            min_peaks=int(analysis_cfg["custom_local_fit_min_peaks"]),
            max_window_span_da=float(analysis_cfg["custom_local_fit_max_window_span_da"]),
            window_margin_da=float(analysis_cfg["custom_local_fit_window_margin_da"]),
            center_tolerance_da=float(analysis_cfg["custom_local_fit_center_tolerance_da"]),
            sigma_floor_da=float(analysis_cfg["custom_local_fit_sigma_floor_da"]),
            sigma_ceiling_da=float(analysis_cfg["custom_local_fit_sigma_ceiling_da"]),
            min_relative_amplitude=float(analysis_cfg["custom_local_fit_min_relative_amplitude"]),
            min_absolute_amplitude=float(analysis_cfg["custom_local_fit_min_absolute_amplitude"]),
            merge_tolerance_da=float(analysis_cfg["custom_local_fit_merge_tolerance_da"]),
            max_fwhm_da=float(analysis_cfg["custom_max_fwhm_da"]),
        )
    if bool(analysis_cfg.get("custom_family_hint_enabled", True)):
        peaks = _annotate_atomic_family_hints(
            peaks,
            raw_mz=sample_data.m_over_z_da,
            config=config,
        )
        peaks = _annotate_molecular_family_recovery(
            peaks,
            raw_mz=sample_data.m_over_z_da,
            config=config,
        )
    if not peaks.empty:
        nearest_indices = np.searchsorted(centers, peaks["peak_mz_da"].to_numpy(dtype=float), side="left").clip(0, centers.size - 1)
        peaks["peak_index"] = nearest_indices.astype(int)
        peaks["peak_height"] = counts[nearest_indices].astype(float)
        peaks["smoothed_height"] = np.maximum(
            peaks.get("smoothed_height", peaks["peak_height"]).to_numpy(dtype=float),
            counts[nearest_indices].astype(float),
        )
    peaks = integrate_peak_areas(
        peaks,
        centers,
        counts,
        integration_half_width_da=float(analysis_cfg["peak_integration_half_width_da"]),
        baseline_half_width_da=float(analysis_cfg["local_baseline_half_width_da"]),
        max_integration_half_width_da=float(analysis_cfg["custom_max_integration_half_width_da"]),
    )
    if "fit_peak_area" in peaks.columns:
        fit_mask = peaks["fit_peak_area"].notna()
        peaks.loc[fit_mask, "integrated_area"] = peaks.loc[fit_mask, "fit_peak_area"].astype(float)
    peaks, adaptive_ranging_diagnostics = apply_adaptive_ranging(
        peaks,
        centers,
        counts,
        config=config,
    )
    if bool(analysis_cfg.get("custom_peak_significance_filter_enabled", True)) and not peaks.empty:
        # Final significance gate: a reported peak must be a significant
        # local Poisson excess over its own sidebands. Background wiggles
        # that survive prominence thresholds die here; confirmed family
        # members are exempt (their evidence is the fingerprint).
        significance_min_z = float(analysis_cfg.get("custom_peak_significance_min_z", 5.0))
        raw_sorted_significance = np.sort(np.asarray(sample_data.m_over_z_da, dtype=np.float64))
        keep_mask: list[bool] = []
        for _, row in peaks.iterrows():
            if pd.notna(row.get("family_hint_element")) or (
                "molecular_family_label" in peaks.columns and pd.notna(row.get("molecular_family_label"))
            ):
                keep_mask.append(True)
                continue
            fwhm = float(row["fwhm_da"]) if pd.notna(row.get("fwhm_da")) else float(row["peak_mz_da"]) / 800.0
            _, significance_z = _raw_member_excess(
                raw_sorted_significance,
                float(row["peak_mz_da"]),
                fwhm_da=max(fwhm, 0.02),
            )
            keep_mask.append(bool(significance_z >= significance_min_z))
        dropped_insignificant = int(len(keep_mask) - int(np.sum(keep_mask)))
        peaks = peaks.loc[np.asarray(keep_mask, dtype=bool)].reset_index(drop=True)
    else:
        dropped_insignificant = 0
    detection_diagnostics_payload = {
        "integration_bin_width_da": integration_bin_width,
        "base_detection_bin_width_da": base_bin_width,
        "detection_passes": detection_diagnostics,
        "smoothing": {
            "window_bins": int(analysis_cfg["custom_savgol_window_bins"]),
            "polyorder": int(analysis_cfg["custom_savgol_polyorder"]),
        },
        "centroid_refinement": {
            "method": "raw-event local density mode with parabolic fallback",
            "window_half_width_da": float(analysis_cfg["custom_raw_refine_window_half_width_da"]),
            "local_bin_width_da": float(analysis_cfg["custom_raw_refine_local_bin_width_da"]),
            "smooth_sigma_da": float(analysis_cfg["custom_raw_refine_smooth_sigma_da"]),
            "min_events": int(analysis_cfg["custom_raw_refine_min_events"]),
            "max_shift_da": float(analysis_cfg["custom_raw_refine_max_shift_da"]),
        },
        "peak_detection": "multi-resolution histogram detection with crowded-window local Gaussian-mixture fitting",
        "crowded_window_fits": local_fit_diagnostics,
        "adaptive_ranging": adaptive_ranging_diagnostics,
        "dropped_insignificant_peaks": dropped_insignificant,
    }
    return centers, counts, peaks, detection_diagnostics_payload


def run_custom_pipeline(sample_data, output_dir, config, *, enable_segmentation: bool | None = None):
    analysis_cfg = config["analysis"]
    centers, counts, peaks, detection_diagnostics_payload = detect_custom_peaks(sample_data, config)
    artifacts = finalize_method_analysis(
        method_name="custom",
        method_label="Custom implementation",
        output_dir=output_dir,
        centers=centers,
        counts=counts,
        peaks=peaks,
        raw_mz=sample_data.m_over_z_da,
        sample_name=sample_data.metadata.sample_name,
        config=config,
        positions=(sample_data.x_nm, sample_data.y_nm, sample_data.z_nm),
        diagnostics=detection_diagnostics_payload,
    )
    segmentation_allowed = (
        bool(analysis_cfg.get("segmentation_enabled", True))
        if enable_segmentation is None
        else bool(enable_segmentation)
    )
    if segmentation_allowed:
        artifacts = _analyze_sample_regions(sample_data, output_dir, config, artifacts)
    return artifacts
