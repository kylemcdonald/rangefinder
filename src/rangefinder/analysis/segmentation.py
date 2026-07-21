from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2
from scipy.spatial import cKDTree

RNG_SEED = 20260720


def _confident_windows(
    peaks: pd.DataFrame,
    assignments: pd.DataFrame,
) -> list[tuple[float, float, dict[str, float]]]:
    """Non-overlap-resolved m/z windows of confidently assigned peaks with their
    element stoichiometry, sorted by position."""
    if peaks.empty or assignments.empty:
        return []
    top = assignments[
        (assignments["rank"] == 1)
        & (assignments["species_label"] != "unassigned")
        & (assignments["confidence"].isin(["high", "ambiguous"]))
    ]
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
    windows.sort(key=lambda item: item[0])
    resolved: list[tuple[float, float, dict[str, float]]] = []
    for left, right, counts in windows:
        if resolved and left < resolved[-1][1]:
            left = resolved[-1][1]
            if right <= left:
                continue
        resolved.append((left, right, counts))
    return resolved


def _event_element_matrix(
    m_over_z_da: np.ndarray,
    windows: list[tuple[float, float, dict[str, float]]],
    elements: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Map each event to a window (or -1) and return per-window element rows."""
    edges: list[float] = []
    window_of_interval: list[int] = []
    for idx, (left, right, _counts) in enumerate(windows):
        edges.extend([left, right])
        window_of_interval.append(idx)
    edge_array = np.asarray(edges, dtype=np.float64)
    interval = np.searchsorted(edge_array, m_over_z_da, side="right") - 1
    inside = (interval >= 0) & (interval % 2 == 0)
    event_window = np.where(inside, interval // 2, -1)
    element_index = {element: pos for pos, element in enumerate(elements)}
    window_rows = np.zeros((len(windows), len(elements)), dtype=np.float64)
    for idx, (_l, _r, counts) in enumerate(windows):
        for element, count in counts.items():
            if element in element_index:
                window_rows[idx, element_index[element]] = float(count)
    return event_window, window_rows


def _majority_filter_labels(
    labels: np.ndarray,
    voxel_ijk: np.ndarray,
    n_labels: int,
) -> np.ndarray:
    """One pass of 3x3x3 majority smoothing over occupied voxels."""
    index = {tuple(ijk): pos for pos, ijk in enumerate(map(tuple, voxel_ijk))}
    smoothed = labels.copy()
    for pos, ijk in enumerate(voxel_ijk):
        votes = np.zeros(n_labels, dtype=np.int64)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for dk in (-1, 0, 1):
                    neighbor = (ijk[0] + di, ijk[1] + dj, ijk[2] + dk)
                    neighbor_pos = index.get(neighbor)
                    if neighbor_pos is not None:
                        votes[labels[neighbor_pos]] += 1
        smoothed[pos] = int(np.argmax(votes))
    return smoothed


def _neighbor_agreement(labels: np.ndarray, voxel_ijk: np.ndarray) -> float:
    index = {tuple(ijk): pos for pos, ijk in enumerate(map(tuple, voxel_ijk))}
    agree = 0
    total = 0
    for pos, ijk in enumerate(voxel_ijk):
        for di, dj, dk in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            neighbor_pos = index.get((ijk[0] + di, ijk[1] + dj, ijk[2] + dk))
            if neighbor_pos is None:
                continue
            total += 1
            if labels[pos] == labels[neighbor_pos]:
                agree += 1
    return float(agree) / max(total, 1)


def _silhouette(features: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette over voxels (small n, so the O(n^2) form is fine)."""
    n = features.shape[0]
    if n > 4000:
        rng = np.random.default_rng(RNG_SEED)
        keep = rng.choice(n, size=4000, replace=False)
        features = features[keep]
        labels = labels[keep]
        n = 4000
    distances = np.sqrt(
        np.maximum(
            np.sum(features**2, axis=1)[:, None]
            + np.sum(features**2, axis=1)[None, :]
            - 2.0 * features @ features.T,
            0.0,
        )
    )
    unique = np.unique(labels)
    if unique.size < 2:
        return 0.0
    scores = np.zeros(n, dtype=np.float64)
    for pos in range(n):
        own = labels[pos]
        same = labels == own
        same[pos] = False
        a = float(distances[pos, same].mean()) if same.any() else 0.0
        b = np.inf
        for other in unique:
            if other == own:
                continue
            mask = labels == other
            if mask.any():
                b = min(b, float(distances[pos, mask].mean()))
        denom = max(a, b)
        scores[pos] = 0.0 if denom <= 0 or not np.isfinite(b) else (b - a) / denom
    return float(scores.mean())


def segment_sample_regions(
    *,
    x_nm: np.ndarray,
    y_nm: np.ndarray,
    z_nm: np.ndarray,
    m_over_z_da: np.ndarray,
    peaks: pd.DataFrame,
    assignments: pd.DataFrame,
    config: dict,
) -> dict[str, object] | None:
    """Decide whether the specimen is compositionally homogeneous and, if not,
    partition its events into spatially coherent composition regions.

    Events are voxelized, each voxel is described by the local elemental
    composition of confidently assigned peak windows, and voxels are clustered
    with k-means. Extra regions are only accepted when they are simultaneously
    populated (minimum event fraction), compositionally distinct (minimum
    centroid L1 separation in at%), well separated in feature space
    (silhouette), and spatially coherent (neighbor agreement) -- otherwise the
    verdict is homogeneous. Composition noise alone cannot create regions.
    """
    analysis_cfg = config["analysis"]
    if not bool(analysis_cfg.get("segmentation_enabled", True)):
        return None
    x = np.asarray(x_nm, dtype=np.float64)
    y = np.asarray(y_nm, dtype=np.float64)
    z = np.asarray(z_nm, dtype=np.float64)
    mz = np.asarray(m_over_z_da, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & np.isfinite(mz)
    min_total = int(analysis_cfg.get("segmentation_min_total_events", 200000))
    if int(finite.sum()) < min_total:
        return {"enabled": True, "skipped": "too few events", "n_regions": 1, "homogeneous": True}
    windows = _confident_windows(peaks, assignments)
    if len(windows) < 2:
        return {"enabled": True, "skipped": "too few assigned windows", "n_regions": 1, "homogeneous": True}
    element_totals: dict[str, float] = {}
    for _l, _r, counts in windows:
        for element, count in counts.items():
            element_totals[element] = element_totals.get(element, 0.0) + count
    max_features = int(analysis_cfg.get("segmentation_max_feature_elements", 6))
    elements = [e for e, _ in sorted(element_totals.items(), key=lambda kv: -kv[1])[:max_features]]
    event_window, window_rows = _event_element_matrix(mz, windows, elements)

    target_per_voxel = float(analysis_cfg.get("segmentation_target_events_per_voxel", 3000.0))
    min_per_voxel = int(analysis_cfg.get("segmentation_min_events_per_voxel", 500))
    x_lo, x_hi = np.quantile(x[finite], [0.005, 0.995])
    y_lo, y_hi = np.quantile(y[finite], [0.005, 0.995])
    z_lo, z_hi = np.quantile(z[finite], [0.005, 0.995])
    volume = max((x_hi - x_lo) * (y_hi - y_lo) * (z_hi - z_lo), 1.0)
    # The reconstruction fills roughly a cone/cylinder inside the bounding box.
    fill_factor = 0.6
    edge = float(np.cbrt(volume * fill_factor * target_per_voxel / max(float(finite.sum()), 1.0)))
    edge = float(np.clip(edge, 1.0, 50.0))
    ix = np.floor((x - x_lo) / edge).astype(np.int64)
    iy = np.floor((y - y_lo) / edge).astype(np.int64)
    iz = np.floor((z - z_lo) / edge).astype(np.int64)
    # Pack ijk into one int64 key; offsets of +4096 need 8192-wide (2^13) fields.
    key = (ix.astype(np.int64) + 4096) * 67108864 + (iy + 4096) * 8192 + (iz + 4096)
    key[~finite] = -1

    unique_keys, inverse, counts_per_voxel = np.unique(key, return_inverse=True, return_counts=True)
    keep_voxel = (unique_keys >= 0) & (counts_per_voxel >= min_per_voxel)
    if int(keep_voxel.sum()) < 8:
        return {"enabled": True, "skipped": "too few populated voxels", "n_regions": 1, "homogeneous": True}

    n_voxels = unique_keys.size
    n_elements = len(elements)
    voxel_features = np.zeros((n_voxels, n_elements), dtype=np.float64)
    in_window = event_window >= 0
    if not in_window.any():
        return {"enabled": True, "skipped": "no events in assigned windows", "n_regions": 1, "homogeneous": True}
    contrib = window_rows[event_window[in_window]]
    np.add.at(voxel_features, inverse[in_window], contrib)
    row_sums = voxel_features.sum(axis=1, keepdims=True)
    voxel_composition = np.divide(
        voxel_features, np.maximum(row_sums, 1.0e-9), out=np.zeros_like(voxel_features)
    )

    kept_index = np.flatnonzero(keep_voxel)
    features = voxel_composition[kept_index]
    kept_keys = unique_keys[kept_index]
    kept_ijk = np.column_stack(
        [
            kept_keys // 67108864 - 4096,
            (kept_keys % 67108864) // 8192 - 4096,
            kept_keys % 8192 - 4096,
        ]
    )
    kept_counts = counts_per_voxel[kept_index]

    max_regions = int(analysis_cfg.get("segmentation_max_regions", 4))
    min_region_fraction = float(analysis_cfg.get("segmentation_min_region_fraction", 0.05))
    min_separation = float(analysis_cfg.get("segmentation_min_composition_separation_at_pct", 10.0))
    min_silhouette = float(analysis_cfg.get("segmentation_min_silhouette", 0.35))
    min_agreement = float(analysis_cfg.get("segmentation_min_neighbor_agreement", 0.60))

    best = {"k": 1, "labels": np.zeros(kept_index.size, dtype=int), "silhouette": 0.0, "separation": 0.0, "agreement": 1.0}
    total_kept_events = float(kept_counts.sum())
    for k in range(2, max_regions + 1):
        if kept_index.size < k * 8:
            break
        centroids, labels = kmeans2(features, k, minit="++", seed=RNG_SEED, iter=50)
        labels = _majority_filter_labels(labels, kept_ijk, k)
        populated = np.unique(labels)
        if populated.size < k:
            continue
        fractions = np.asarray(
            [kept_counts[labels == label].sum() / total_kept_events for label in populated]
        )
        if float(fractions.min()) < min_region_fraction:
            continue
        centroid_comp = np.stack(
            [
                np.average(features[labels == label], axis=0, weights=kept_counts[labels == label])
                for label in populated
            ]
        )
        separation = 0.0
        for a_pos in range(populated.size):
            for b_pos in range(a_pos + 1, populated.size):
                separation = max(
                    separation,
                    100.0 * float(np.abs(centroid_comp[a_pos] - centroid_comp[b_pos]).sum()),
                )
        if separation < min_separation:
            continue
        silhouette = _silhouette(features, labels)
        if silhouette < min_silhouette:
            continue
        agreement = _neighbor_agreement(labels, kept_ijk)
        if agreement < min_agreement:
            continue
        if silhouette > best["silhouette"] or best["k"] == 1:
            best = {
                "k": int(populated.size),
                "labels": labels,
                "silhouette": float(silhouette),
                "separation": float(separation),
                "agreement": float(agreement),
            }

    k = int(best["k"])
    labels = np.asarray(best["labels"], dtype=int)
    # Relabel regions by descending event count for stable, meaningful ids.
    order = np.argsort(
        [-float(kept_counts[labels == label].sum()) for label in range(labels.max() + 1 if labels.size else 1)]
    )
    remap = {int(old): int(new) for new, old in enumerate(order)}
    labels = np.asarray([remap.get(int(v), 0) for v in labels], dtype=int)

    # Every event gets a region: events in kept voxels inherit the voxel label,
    # everything else takes the label of the nearest kept voxel center.
    voxel_label = np.full(n_voxels, -1, dtype=int)
    voxel_label[kept_index] = labels
    event_label = voxel_label[inverse]
    if k > 1 and (event_label < 0).any():
        centers = np.column_stack(
            [
                x_lo + (kept_ijk[:, 0] + 0.5) * edge,
                y_lo + (kept_ijk[:, 1] + 0.5) * edge,
                z_lo + (kept_ijk[:, 2] + 0.5) * edge,
            ]
        )
        tree = cKDTree(centers)
        missing = np.flatnonzero(event_label < 0)
        _dist, nearest = tree.query(np.column_stack([x[missing], y[missing], z[missing]]), k=1)
        event_label[missing] = labels[nearest]
    if k == 1:
        event_label = np.zeros(mz.size, dtype=int)

    voxel_rows = []
    for pos in range(kept_index.size):
        row = {
            "ix": int(kept_ijk[pos, 0]),
            "iy": int(kept_ijk[pos, 1]),
            "iz": int(kept_ijk[pos, 2]),
            "x_nm": float(x_lo + (kept_ijk[pos, 0] + 0.5) * edge),
            "y_nm": float(y_lo + (kept_ijk[pos, 1] + 0.5) * edge),
            "z_nm": float(z_lo + (kept_ijk[pos, 2] + 0.5) * edge),
            "event_count": int(kept_counts[pos]),
            "region": int(labels[pos]) if k > 1 else 0,
        }
        for element_pos, element in enumerate(elements):
            row[f"{element}_at_frac"] = float(features[pos, element_pos])
        voxel_rows.append(row)

    region_summary = []
    for region in range(k):
        mask = event_label == region
        region_z = z[mask & finite]
        region_summary.append(
            {
                "region": region,
                "event_count": int(mask.sum()),
                "event_fraction": float(mask.sum()) / float(mz.size),
                "z_nm_p05": float(np.quantile(region_z, 0.05)) if region_z.size else None,
                "z_nm_p95": float(np.quantile(region_z, 0.95)) if region_z.size else None,
            }
        )

    return {
        "enabled": True,
        "homogeneous": bool(k == 1),
        "n_regions": k,
        "voxel_edge_nm": edge,
        "voxel_count": int(kept_index.size),
        "feature_elements": elements,
        "silhouette": best["silhouette"],
        "composition_separation_at_pct": best["separation"],
        "neighbor_agreement": best["agreement"],
        "event_region": event_label,
        "voxel_table": pd.DataFrame(voxel_rows),
        "region_summary": region_summary,
    }
