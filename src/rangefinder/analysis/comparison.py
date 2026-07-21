from __future__ import annotations

import numpy as np
import pandas as pd

from rangefinder.analysis.common import pairwise_peak_matches


def compare_method_pair(left_method, right_method) -> pd.DataFrame:
    peak_matches = pairwise_peak_matches(
        left_method.peaks,
        right_method.peaks,
        tolerance_da=0.1,
        left_label=left_method.method_name,
        right_label=right_method.method_name,
    )
    matched_fraction = float(peak_matches["matched"].mean()) if not peak_matches.empty else 0.0
    median_delta = float(peak_matches.loc[peak_matches["matched"], "delta_da"].median()) if peak_matches["matched"].any() else np.nan
    left_elements = left_method.elemental_composition.set_index("element")["atomic_percent_weighted"] if not left_method.elemental_composition.empty else pd.Series(dtype=float)
    right_elements = right_method.elemental_composition.set_index("element")["atomic_percent_weighted"] if not right_method.elemental_composition.empty else pd.Series(dtype=float)
    all_elements = sorted(set(left_elements.index) | set(right_elements.index))
    left_vector = left_elements.reindex(all_elements).fillna(0.0)
    right_vector = right_elements.reindex(all_elements).fillna(0.0)
    l1_distance = float(np.abs(left_vector - right_vector).sum())
    left_anomalies = _anomaly_labels(left_method.isotope_anomalies)
    right_anomalies = _anomaly_labels(right_method.isotope_anomalies)
    anomaly_overlap = len(left_anomalies & right_anomalies)
    return pd.DataFrame(
        [
            {
                "method_left": left_method.method_name,
                "method_right": right_method.method_name,
                "left_peak_count": int(left_method.peaks.shape[0]),
                "right_peak_count": int(right_method.peaks.shape[0]),
                "matched_peak_fraction": matched_fraction,
                "median_matched_delta_da": median_delta,
                "elemental_l1_distance_at_pct": l1_distance,
                "shared_isotope_anomaly_count": anomaly_overlap,
            }
        ]
    )


def _anomaly_labels(frame: pd.DataFrame) -> set[str]:
    if frame.empty or "status" not in frame.columns or "isotope_label" not in frame.columns:
        return set()
    return set(frame.loc[frame["status"].isin(["slight", "strong"]), "isotope_label"])


def compare_methods(method_results, *, anchor_method_name: str = "custom") -> pd.DataFrame:
    anchor = next((method for method in method_results if method.method_name == anchor_method_name), None)
    if anchor is None:
        return pd.DataFrame()
    rows = []
    for method in method_results:
        if method.method_name == anchor_method_name:
            continue
        rows.append(compare_method_pair(method, anchor))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def compare_methods_pairwise(method_results) -> pd.DataFrame:
    """Full symmetric pairwise agreement matrix across all methods."""
    rows = []
    for left_idx in range(len(method_results)):
        for right_idx in range(left_idx + 1, len(method_results)):
            rows.append(compare_method_pair(method_results[left_idx], method_results[right_idx]))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def consensus_peak_clusters(method_results, *, tolerance_da: float = 0.1) -> pd.DataFrame:
    """Cluster all detected peaks across methods and count method support per cluster."""
    records: list[dict[str, object]] = []
    for method in method_results:
        if method.peaks.empty:
            continue
        for mz in method.peaks["peak_mz_da"].to_numpy(dtype=np.float64):
            records.append({"method": method.method_name, "mz_da": float(mz)})
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records).sort_values("mz_da").reset_index(drop=True)
    cluster_ids = np.zeros(frame.shape[0], dtype=int)
    current_cluster = 0
    mz_values = frame["mz_da"].to_numpy(dtype=np.float64)
    for idx in range(1, frame.shape[0]):
        if mz_values[idx] - mz_values[idx - 1] > tolerance_da:
            current_cluster += 1
        cluster_ids[idx] = current_cluster
    frame["cluster_id"] = cluster_ids
    clusters = (
        frame.groupby("cluster_id")
        .agg(
            mz_da=("mz_da", "median"),
            method_count=("method", "nunique"),
            methods=("method", lambda values: sorted(set(values))),
        )
        .reset_index()
    )
    return clusters


def method_consensus_metrics(
    method_results,
    *,
    tolerance_da: float = 0.1,
    consensus_min_methods: int = 3,
) -> pd.DataFrame:
    """Per-method quality metrics against the cross-method consensus.

    - peak_precision_vs_consensus: fraction of the method's peaks that fall in a
      cluster confirmed by at least `consensus_min_methods` distinct methods.
    - peak_recall_vs_consensus: fraction of consensus clusters this method found.
    - elemental_l1_to_median: L1 distance of the method's elemental composition
      to the element-wise cross-method median composition.
    - isotope_rms_to_median: RMS deviation of resolved isotope fractions from the
      cross-method median fraction of the same isotope.
    - assigned_area_fraction / high_confidence_area_fraction: how much of the
      method's total peak area ends up assigned (at all / with high confidence).
    - median_abs_ppm: median absolute mass residual of high-confidence
      assignments, a proxy for centroid accuracy.
    """
    clusters = consensus_peak_clusters(method_results, tolerance_da=tolerance_da)
    consensus = clusters[clusters["method_count"] >= consensus_min_methods] if not clusters.empty else pd.DataFrame()
    consensus_mz = consensus["mz_da"].to_numpy(dtype=np.float64) if not consensus.empty else np.asarray([])
    element_frames = []
    for method in method_results:
        if method.elemental_composition.empty:
            continue
        series = method.elemental_composition.set_index("element")["atomic_percent_weighted"]
        series.name = method.method_name
        element_frames.append(series)
    element_matrix = pd.concat(element_frames, axis=1).fillna(0.0) if element_frames else pd.DataFrame()
    median_composition = element_matrix.median(axis=1) if not element_matrix.empty else pd.Series(dtype=float)
    isotope_frames = []
    for method in method_results:
        frame = method.isotope_composition
        if frame.empty or "isotope_fraction_weighted" not in frame.columns:
            continue
        subset = frame
        if "isotope_resolved" in frame.columns:
            subset = frame[frame["isotope_resolved"] == True]  # noqa: E712
        if subset.empty:
            continue
        series = subset.set_index("isotope_label")["isotope_fraction_weighted"]
        series = series[~series.index.duplicated(keep="first")]
        series.name = method.method_name
        isotope_frames.append(series)
    isotope_matrix = pd.concat(isotope_frames, axis=1) if isotope_frames else pd.DataFrame()
    median_isotope = isotope_matrix.median(axis=1, skipna=True) if not isotope_matrix.empty else pd.Series(dtype=float)
    rows = []
    for method in method_results:
        peaks = method.peaks
        peak_mz = peaks["peak_mz_da"].to_numpy(dtype=np.float64) if not peaks.empty else np.asarray([])
        if peak_mz.size and consensus_mz.size:
            matched_peak = np.min(np.abs(consensus_mz[None, :] - peak_mz[:, None]), axis=1) <= tolerance_da
            precision = float(np.mean(matched_peak))
            matched_consensus = np.min(np.abs(peak_mz[None, :] - consensus_mz[:, None]), axis=1) <= tolerance_da
            recall = float(np.mean(matched_consensus))
        else:
            precision = np.nan
            recall = np.nan
        if not element_matrix.empty and method.method_name in element_matrix.columns:
            l1_to_median = float(np.abs(element_matrix[method.method_name] - median_composition).sum())
        else:
            l1_to_median = np.nan
        isotope_rms = np.nan
        if not isotope_matrix.empty and method.method_name in isotope_matrix.columns:
            own = isotope_matrix[method.method_name].dropna()
            if not own.empty:
                deltas = own - median_isotope.reindex(own.index)
                deltas = deltas.dropna()
                if not deltas.empty:
                    isotope_rms = float(np.sqrt(np.mean(np.square(deltas.to_numpy(dtype=np.float64)))))
        assigned_area_fraction = np.nan
        high_confidence_area_fraction = np.nan
        if not peaks.empty and "integrated_area" in peaks.columns and "top_species" in peaks.columns:
            areas = peaks["integrated_area"].fillna(0.0).to_numpy(dtype=np.float64)
            total_area = float(areas.sum())
            if total_area > 0:
                assigned_mask = peaks["top_species"].fillna("unassigned").astype(str) != "unassigned"
                assigned_area_fraction = float(areas[assigned_mask.to_numpy()].sum() / total_area)
                if "confidence" in peaks.columns:
                    high_mask = peaks["confidence"].fillna("").astype(str) == "high"
                    high_confidence_area_fraction = float(areas[high_mask.to_numpy()].sum() / total_area)
        median_abs_ppm = np.nan
        mass_scale = method.diagnostics.get("mass_scale") if isinstance(method.diagnostics, dict) else None
        if isinstance(mass_scale, dict) and mass_scale.get("n_high_confidence", 0):
            median_error = mass_scale.get("median_abs_error_da")
            median_ppm = mass_scale.get("median_ppm")
            if median_ppm is not None:
                median_abs_ppm = abs(float(median_ppm))
            elif median_error is not None:
                median_abs_ppm = np.nan
        anomaly_count = 0
        if not method.isotope_anomalies.empty and "status" in method.isotope_anomalies.columns:
            anomaly_count = int(
                method.isotope_anomalies["status"].isin(["slight", "strong"]).sum()
            )
        rows.append(
            {
                "method": method.method_name,
                "method_label": method.method_label,
                "peak_count": int(peaks.shape[0]),
                "consensus_peak_count": int(consensus.shape[0]) if not consensus.empty else 0,
                "peak_precision_vs_consensus": precision,
                "peak_recall_vs_consensus": recall,
                "elemental_l1_to_median": l1_to_median,
                "isotope_rms_to_median": isotope_rms,
                "assigned_area_fraction": assigned_area_fraction,
                "high_confidence_area_fraction": high_confidence_area_fraction,
                "median_abs_mass_residual_ppm": median_abs_ppm,
                "flagged_isotope_count": anomaly_count,
            }
        )
    return pd.DataFrame(rows)
