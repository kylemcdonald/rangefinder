from __future__ import annotations

from collections import defaultdict
import json

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from rangefinder.analysis.common import assignment_softmax, build_species_library, top_assignment_table
from rangefinder.references.isotopes import isotope_table, natural_abundance_by_element


def _normalized_natural_probability(values: pd.Series) -> pd.Series:
    clipped = values.clip(lower=1e-12)
    return (np.log10(clipped) + 12.0) / 12.0


def peak_tolerance_da(
    peak_mz_da: float,
    *,
    ppm_tolerance: float,
    min_absolute_tolerance_da: float,
    max_absolute_tolerance_da: float,
) -> float:
    ppm_component = peak_mz_da * ppm_tolerance * 1e-6
    return float(np.clip(max(ppm_component, min_absolute_tolerance_da), min_absolute_tolerance_da, max_absolute_tolerance_da))


def _normalize_score_map(scores: dict[object, float]) -> dict[object, float]:
    if not scores:
        return {}
    maximum = max(float(value) for value in scores.values())
    if maximum <= 0:
        return {key: 0.0 for key in scores}
    return {key: float(value) / maximum for key, value in scores.items()}


def _blend_support_maps(
    base_support: dict[object, float],
    pass_support: dict[object, float],
    *,
    pass_weight: float,
) -> dict[object, float]:
    keys = set(base_support) | set(pass_support)
    blended = {
        key: (1.0 - pass_weight) * float(base_support.get(key, 0.0))
        + pass_weight * float(pass_support.get(key, 0.0))
        for key in keys
    }
    return _normalize_score_map(blended)


def _atomic_family_support_from_peaks(
    peaks: pd.DataFrame,
    *,
    max_charge: int,
    ppm_tolerance: float,
    min_absolute_tolerance_da: float,
    max_absolute_tolerance_da: float,
) -> dict[tuple[str, int], float]:
    if peaks.empty:
        return {}
    isotopes = isotope_table()
    peak_positions = peaks["peak_mz_da"].to_numpy(dtype=np.float64)
    area_column = (
        "integrated_area"
        if "integrated_area" in peaks.columns
        else ("peak_height" if "peak_height" in peaks.columns else None)
    )
    if area_column is None:
        return {}
    peak_areas = peaks[area_column].to_numpy(dtype=np.float64)
    scores: dict[tuple[str, int], float] = {}
    for symbol, family in isotopes.groupby("symbol", sort=True):
        family = family.sort_values("mass_da").reset_index(drop=True)
        expected_prob = family["natural_abundance"].to_numpy(dtype=np.float64)
        dominant_idx = int(np.argmax(expected_prob))
        for charge in range(1, max_charge + 1):
            expected_mz = family["mass_da"].to_numpy(dtype=np.float64) / float(charge)
            observed = []
            for mz in expected_mz:
                tolerance = peak_tolerance_da(
                    float(mz),
                    ppm_tolerance=ppm_tolerance,
                    min_absolute_tolerance_da=min_absolute_tolerance_da,
                    max_absolute_tolerance_da=max_absolute_tolerance_da,
                )
                if peak_positions.size:
                    idx = int(np.argmin(np.abs(peak_positions - mz)))
                    if abs(float(peak_positions[idx]) - float(mz)) <= tolerance:
                        observed.append(float(peak_areas[idx]))
                    else:
                        observed.append(0.0)
                else:
                    observed.append(0.0)
            observed_array = np.asarray(observed, dtype=np.float64)
            if family.shape[0] == 1:
                scores[(str(symbol), int(charge))] = 1.0
                continue
            if observed_array.sum() <= 0.0:
                scores[(str(symbol), int(charge))] = 0.05
                continue
            observed_fraction = observed_array / observed_array.sum()
            expected_fraction = expected_prob / max(expected_prob.sum(), 1.0e-12)
            l1_distance = float(np.abs(observed_fraction - expected_fraction).sum())
            completeness = float(np.count_nonzero(observed_array > 0.0)) / max(float(family.shape[0]), 1.0)
            dominant_present = bool(observed_array[dominant_idx] > 0.0)
            dominant_penalty = 1.0 if dominant_present else 0.35
            scores[(str(symbol), int(charge))] = float(
                np.exp(-2.2 * l1_distance) * (0.5 + 0.5 * completeness) * dominant_penalty
            )
    return scores


def infer_element_support(
    peaks: pd.DataFrame,
    *,
    max_charge: int,
    ppm_tolerance: float,
    min_absolute_tolerance_da: float,
    max_absolute_tolerance_da: float,
) -> dict[str, float]:
    isotopes = isotope_table()
    family_support = _atomic_family_support_from_peaks(
        peaks,
        max_charge=max_charge,
        ppm_tolerance=ppm_tolerance,
        min_absolute_tolerance_da=min_absolute_tolerance_da,
        max_absolute_tolerance_da=max_absolute_tolerance_da,
    )
    element_scores: defaultdict[str, float] = defaultdict(float)
    for _, peak in peaks.iterrows():
        mz = float(peak["peak_mz_da"])
        intensity = float(peak.get("integrated_area", peak.get("peak_height", 0.0)))
        tolerance = peak_tolerance_da(
            mz,
            ppm_tolerance=ppm_tolerance,
            min_absolute_tolerance_da=min_absolute_tolerance_da,
            max_absolute_tolerance_da=max_absolute_tolerance_da,
        )
        # Same ppm-scale mass model as assignment scoring: without it, an element
        # 10-20 mDa off a large peak accumulates "support" from that peak and then
        # legitimizes molecular candidates containing it.
        support_sigma = float(np.clip(300.0e-6 * mz, 0.005, 0.025))
        for charge in range(1, max_charge + 1):
            expected = isotopes["mass_da"] / charge
            mask = np.abs(expected - mz) <= tolerance
            if not mask.any():
                continue
            candidates = isotopes.loc[mask]
            for _, candidate in candidates.iterrows():
                mass_error = abs(candidate["mass_da"] / charge - mz)
                family_score = float(family_support.get((str(candidate["symbol"]), int(charge)), 1.0))
                mass_weight = float(np.exp(-0.5 * (mass_error / support_sigma) ** 2))
                score = intensity * candidate["natural_abundance"] * family_score * mass_weight
                element_scores[str(candidate["symbol"])] += float(score)
    total = max(element_scores.values(), default=1.0)
    return {element: score / total for element, score in element_scores.items()}


def atomic_family_consistency_scores(peaks: pd.DataFrame, species_library: pd.DataFrame, *, config: dict) -> dict[str, float]:
    if peaks.empty or species_library.empty:
        return {}
    analysis_cfg = config["analysis"]
    peak_positions = peaks["peak_mz_da"].to_numpy(dtype=np.float64)
    peak_areas = peaks["integrated_area"].to_numpy(dtype=np.float64)
    atomic = species_library[species_library["category"] == "atomic"].copy()
    atomic["family_key"] = atomic["symbols"].map(tuple)
    scores: dict[str, float] = {}
    for _, group in atomic.groupby(["family_key", "charge"]):
        if group.shape[0] <= 1:
            row = group.iloc[0]
            scores[str(row["candidate_id"])] = 1.0
            continue
        observed = []
        expected = []
        for _, row in group.sort_values("m_over_z_da").iterrows():
            mz = float(row["m_over_z_da"])
            tolerance = peak_tolerance_da(
                mz,
                ppm_tolerance=float(analysis_cfg["ppm_tolerance"]),
                min_absolute_tolerance_da=float(analysis_cfg["min_absolute_tolerance_da"]),
                max_absolute_tolerance_da=float(analysis_cfg["max_absolute_tolerance_da"]),
            )
            idx = int(np.argmin(np.abs(peak_positions - mz))) if peak_positions.size else -1
            if idx >= 0 and abs(float(peak_positions[idx]) - mz) <= tolerance:
                observed.append(float(peak_areas[idx]))
            else:
                observed.append(0.0)
            expected.append(float(row["natural_probability"]))
        observed_array = np.asarray(observed, dtype=np.float64)
        expected_array = np.asarray(expected, dtype=np.float64)
        if observed_array.sum() <= 0 or expected_array.sum() <= 0:
            family_score = float(analysis_cfg["family_missing_score"])
        else:
            observed_fraction = observed_array / observed_array.sum()
            expected_fraction = expected_array / expected_array.sum()
            l1_distance = float(np.abs(observed_fraction - expected_fraction).sum())
            completeness = float(np.count_nonzero(observed_array > 0.0)) / max(float(group.shape[0]), 1.0)
            dominant_idx = int(np.argmax(expected_array))
            dominant_present = bool(observed_array[dominant_idx] > 0.0)
            dominant_penalty = 1.0 if dominant_present else float(analysis_cfg["family_missing_dominant_penalty"])
            family_score = float(
                np.exp(-float(analysis_cfg["family_l1_scale"]) * l1_distance)
                * (0.5 + 0.5 * completeness)
                * dominant_penalty
            )
        for _, row in group.iterrows():
            scores[str(row["candidate_id"])] = family_score
    return scores


def _candidate_element_prior_score(symbols: list[str], element_prior: dict[str, float] | None) -> float:
    if not element_prior or not symbols:
        return 0.0
    return float(np.mean([float(element_prior.get(symbol, 0.0)) for symbol in symbols]))


def _candidate_charge_prior_score(charge: int, charge_prior: dict[int, float] | None) -> float:
    if not charge_prior:
        return 0.0
    return float(charge_prior.get(int(charge), 0.0))


def _candidate_family_prior_score(
    match: pd.Series,
    family_prior: dict[tuple[str, int], float] | None,
) -> float:
    if not family_prior:
        return 0.0
    family_key = _atomic_family_from_row(match)
    if family_key is None:
        return 0.0
    return float(family_prior.get(family_key, 0.0))


def _family_hint_match_score(
    match: pd.Series,
    *,
    hint_element: str | None,
    hint_charge: int | None,
    hint_isotope_label: str | None,
) -> float:
    if not hint_element and hint_charge is None and not hint_isotope_label:
        return 0.0
    score = 0.0
    if str(match.get("category", "")) == "atomic":
        symbols = match.get("symbols")
        if isinstance(symbols, list) and len(symbols) == 1 and hint_element and str(symbols[0]) == str(hint_element):
            score += 0.5
        charge = match.get("charge")
        if hint_charge is not None and pd.notna(charge) and int(charge) == int(hint_charge):
            score += 0.35
        isotope_counts = match.get("isotope_counts")
        if hint_isotope_label and isinstance(isotope_counts, dict) and isotope_counts.get(str(hint_isotope_label), 0) == 1:
            score += 1.0
    return float(score)


def _coerce_family_hint_candidates(value: object) -> list[dict[str, object]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
    return []


def _family_hint_candidate_support(
    match: pd.Series,
    *,
    family_hint_candidates: list[dict[str, object]],
) -> float:
    if not family_hint_candidates:
        return 0.0
    best_support = 0.0
    for candidate in family_hint_candidates:
        match_score = _family_hint_match_score(
            match,
            hint_element=str(candidate.get("element")) if candidate.get("element") is not None else None,
            hint_charge=int(candidate["charge"]) if candidate.get("charge") is not None else None,
            hint_isotope_label=(
                str(candidate.get("isotope_label"))
                if candidate.get("isotope_label") is not None
                else None
            ),
        )
        if match_score <= 0.0:
            continue
        best_support = max(best_support, float(candidate.get("score", 0.0) or 0.0) * match_score)
    return float(best_support)


def _charge_state_penalty(charge: int, *, config: dict) -> float:
    penalties = config["analysis"].get("charge_state_penalties", {})
    return float(penalties.get(str(int(charge)), 0.0))


def _unsupported_charge_penalty(
    charge: int,
    *,
    charge_prior_score: float,
    config: dict,
) -> float:
    analysis_cfg = config["analysis"]
    threshold = float(analysis_cfg.get("unsupported_charge_prior_threshold", 0.0))
    if charge_prior_score > threshold:
        return 0.0
    penalties = analysis_cfg.get("unsupported_charge_penalties", {})
    return float(penalties.get(str(int(charge)), 0.0))


def _min_assignment_score(charge: int, *, config: dict) -> float:
    minimums = config["analysis"].get("min_assignment_score_by_charge", {})
    return float(minimums.get(str(int(charge)), minimums.get("default", float("-inf"))))


def _hinted_elements_from_peaks(
    peaks: pd.DataFrame,
    *,
    config: dict,
) -> list[str]:
    if peaks.empty or "family_hint_element" not in peaks.columns:
        return []
    hinted = peaks.copy()
    hinted = hinted[hinted["family_hint_element"].notna()].copy()
    if hinted.empty:
        return []
    hinted["family_hint_score"] = hinted.get("family_hint_score", 0.0)
    hinted["family_hint_match_count"] = hinted.get("family_hint_match_count", 0)
    hinted["integrated_area"] = hinted.get("integrated_area", hinted.get("peak_height", 0.0))
    score_min = float(config["analysis"].get("custom_family_hint_score_min", 0.0))
    hinted = hinted[
        (hinted["family_hint_score"].fillna(0.0) >= score_min)
        & (hinted["family_hint_match_count"].fillna(0).astype(int) >= 2)
    ].copy()
    if hinted.empty:
        return []
    grouped = (
        hinted.groupby("family_hint_element", dropna=True)
        .agg(
            peak_count=("peak_id", "nunique"),
            total_area=("integrated_area", "sum"),
            max_match_count=("family_hint_match_count", "max"),
        )
        .reset_index()
    )
    qualifying = grouped[
        (grouped["peak_count"] >= 2)
        & (grouped["max_match_count"].astype(int) >= 2)
        & (grouped["total_area"].fillna(0.0) > 0.0)
    ].copy()
    return [str(value) for value in qualifying["family_hint_element"].tolist()]


def _priors_from_assignments(assignments: pd.DataFrame, peaks: pd.DataFrame) -> tuple[dict[str, float], dict[int, float]]:
    if assignments.empty or peaks.empty:
        return {}, {}
    peak_areas = peaks.set_index("peak_id")["integrated_area"].fillna(0.0)
    weighted = assignments.copy()
    weighted = weighted[weighted["species_label"] != "unassigned"].copy()
    if weighted.empty:
        return {}, {}
    weighted["peak_area"] = weighted["peak_id"].map(peak_areas).fillna(0.0)
    weighted["weighted_area"] = weighted["peak_area"] * weighted["probability"]
    element_scores: defaultdict[str, float] = defaultdict(float)
    charge_scores: defaultdict[int, float] = defaultdict(float)
    for _, row in weighted.iterrows():
        area = float(row["weighted_area"])
        for element, count in row["element_counts"].items():
            element_scores[str(element)] += area * float(count)
        if pd.notna(row["charge"]):
            charge_scores[int(row["charge"])] += area
    return _normalize_score_map(dict(element_scores)), _normalize_score_map(dict(charge_scores))


def _family_priors_from_assignments(assignments: pd.DataFrame, peaks: pd.DataFrame) -> dict[tuple[str, int], float]:
    if assignments.empty or peaks.empty:
        return {}
    peak_areas = peaks.set_index("peak_id")["integrated_area"].fillna(0.0)
    weighted = assignments.copy()
    weighted = weighted[
        (weighted["species_label"] != "unassigned")
        & (weighted["category"] == "atomic")
    ].copy()
    if weighted.empty:
        return {}
    weighted["peak_area"] = weighted["peak_id"].map(peak_areas).fillna(0.0)
    weighted["weighted_area"] = weighted["peak_area"] * weighted["probability"]
    family_scores: defaultdict[tuple[str, int], float] = defaultdict(float)
    for _, row in weighted.iterrows():
        family_key = _atomic_family_from_row(row)
        if family_key is None:
            continue
        family_scores[family_key] += float(row["weighted_area"])
    return _normalize_score_map(dict(family_scores))


def _atomic_family_from_row(row: pd.Series) -> tuple[str, int] | None:
    if str(row.get("category", "")) != "atomic":
        return None
    charge = row.get("charge")
    if charge is None or pd.isna(charge):
        return None
    element_counts = row.get("element_counts")
    if not isinstance(element_counts, dict) or len(element_counts) != 1:
        return None
    element = next(iter(element_counts))
    return str(element), int(charge)


def _crowded_peak_windows(peaks: pd.DataFrame, assignments: pd.DataFrame, *, config: dict) -> list[list[str]]:
    analysis_cfg = config["analysis"]
    if peaks.empty or assignments.empty or not bool(analysis_cfg.get("local_fit_enabled", False)):
        return []
    top = assignments[assignments["rank"] == 1][["peak_id", "confidence"]].copy()
    merged = peaks.merge(top, on="peak_id", how="left").sort_values("peak_mz_da").reset_index(drop=True)
    if merged.empty:
        return []
    candidate_windows: list[list[str]] = []
    gap_threshold = float(analysis_cfg["local_fit_cluster_gap_da"])
    max_span = float(analysis_cfg["local_fit_max_window_span_da"])
    min_peaks = int(analysis_cfg["local_fit_min_peaks"])
    min_total_area = float(analysis_cfg["local_fit_min_total_area"])
    max_rank = int(analysis_cfg["local_fit_max_rank"])
    peak_rows = [merged.iloc[idx].to_dict() for idx in range(merged.shape[0])]
    for start_idx in range(len(peak_rows)):
        end_idx = start_idx
        while end_idx + 1 < len(peak_rows):
            next_gap = float(peak_rows[end_idx + 1]["peak_mz_da"]) - float(peak_rows[end_idx]["peak_mz_da"])
            next_span = float(peak_rows[end_idx + 1]["peak_mz_da"]) - float(peak_rows[start_idx]["peak_mz_da"])
            if next_gap > gap_threshold or next_span > max_span:
                break
            end_idx += 1
        cluster_rows = peak_rows[start_idx : end_idx + 1]
        if len(cluster_rows) < min_peaks:
            continue
        total_area = float(sum(float(row.get("integrated_area", 0.0)) for row in cluster_rows))
        if total_area < min_total_area:
            continue
        cluster_peak_ids = [str(row["peak_id"]) for row in cluster_rows]
        cluster_assignments = assignments[
            (assignments["peak_id"].isin(cluster_peak_ids))
            & (assignments["rank"] <= max_rank)
        ].copy()
        family_keys = {
            _atomic_family_from_row(row)
            for _, row in cluster_assignments.iterrows()
        }
        family_keys.discard(None)
        if len(family_keys) < 2:
            continue
        candidate_windows.append(cluster_peak_ids)
    unique_candidates = []
    seen = set()
    for window in candidate_windows:
        key = tuple(window)
        if key in seen:
            continue
        seen.add(key)
        unique_candidates.append(window)
    maximal_windows = []
    for window in unique_candidates:
        window_set = set(window)
        if any(window_set < set(other) for other in unique_candidates):
            continue
        maximal_windows.append(window)
    return maximal_windows


def _family_template_matrix(window_peaks: pd.DataFrame, family_keys: list[tuple[str, int]], *, config: dict) -> tuple[np.ndarray, list[tuple[str, int]]]:
    if window_peaks.empty or not family_keys:
        return np.zeros((0, 0), dtype=np.float64), []
    analysis_cfg = config["analysis"]
    isotopes = isotope_table()
    peak_mz = window_peaks["peak_mz_da"].to_numpy(dtype=np.float64)
    peak_fwhm = window_peaks["fwhm_da"].fillna(0.0).to_numpy(dtype=np.float64)
    sigma = np.maximum(peak_fwhm / 2.355, float(analysis_cfg["local_fit_sigma_floor_da"]))
    window_margin = float(analysis_cfg["local_fit_window_margin_da"])
    lower = float(peak_mz.min() - window_margin)
    upper = float(peak_mz.max() + window_margin)
    columns = []
    kept_keys: list[tuple[str, int]] = []
    for element, charge in family_keys:
        family = isotopes[isotopes["symbol"] == element].copy()
        if family.empty:
            continue
        family["family_mz_da"] = family["mass_da"] / charge
        family = family[(family["family_mz_da"] >= lower) & (family["family_mz_da"] <= upper)].copy()
        if family.empty:
            continue
        template = np.zeros(window_peaks.shape[0], dtype=np.float64)
        for _, isotope in family.iterrows():
            isotope_mz = float(isotope["family_mz_da"])
            abundance = float(isotope["natural_abundance"])
            kernel = np.exp(-0.5 * ((peak_mz - isotope_mz) / sigma) ** 2)
            template += abundance * kernel
        max_value = float(template.max())
        if max_value <= 0.0:
            continue
        columns.append(template / max_value)
        kept_keys.append((element, charge))
    if not columns:
        return np.zeros((window_peaks.shape[0], 0), dtype=np.float64), []
    return np.column_stack(columns), kept_keys


def _select_joint_species_labels(
    assignments: pd.DataFrame,
    peaks: pd.DataFrame,
    *,
    max_rank: int,
    max_species: int,
    min_vote_area: float,
) -> list[str]:
    if assignments.empty or peaks.empty:
        return []
    peak_areas = peaks.set_index("peak_id")["integrated_area"].fillna(0.0)
    ranked = assignments[
        (assignments["rank"] <= int(max_rank))
        & (assignments["species_label"] != "unassigned")
    ].copy()
    if ranked.empty:
        return []
    ranked["peak_area"] = ranked["peak_id"].map(peak_areas).fillna(0.0)
    ranked["vote_area"] = ranked["peak_area"] * ranked["probability"].clip(lower=0.05)
    species_votes = (
        ranked.groupby("species_label", as_index=False)["vote_area"]
        .sum()
        .sort_values(["vote_area", "species_label"], ascending=[False, True])
    )
    selected = species_votes[species_votes["vote_area"] >= float(min_vote_area)]["species_label"].astype(str).tolist()
    if len(selected) < int(max_species):
        top_rows = (
            assignments[(assignments["rank"] == 1) & (assignments["species_label"] != "unassigned")]
            ["species_label"]
            .drop_duplicates()
            .astype(str)
            .tolist()
        )
        for label in top_rows:
            if label not in selected:
                selected.append(label)
            if len(selected) >= int(max_species):
                break
    return selected[: int(max_species)]


def _species_template_matrix(
    window_peaks: pd.DataFrame,
    species_library: pd.DataFrame,
    species_labels: list[str],
    *,
    config: dict,
) -> tuple[np.ndarray, list[str]]:
    if window_peaks.empty or species_library.empty or not species_labels:
        return np.zeros((0, 0), dtype=np.float64), []
    analysis_cfg = config["analysis"]
    peak_mz = window_peaks["peak_mz_da"].to_numpy(dtype=np.float64)
    peak_fwhm = window_peaks["fwhm_da"].fillna(0.0).to_numpy(dtype=np.float64)
    sigma = np.maximum(
        peak_fwhm / 2.355,
        float(analysis_cfg.get("joint_species_sigma_floor_da", analysis_cfg.get("local_fit_sigma_floor_da", 0.035))),
    )
    window_margin = float(
        analysis_cfg.get("joint_species_window_margin_da", analysis_cfg.get("local_fit_window_margin_da", 0.25))
    )
    lower = float(peak_mz.min() - window_margin)
    upper = float(peak_mz.max() + window_margin)
    subset = species_library[species_library["species_label"].isin(species_labels)].copy()
    subset = subset[(subset["m_over_z_da"] >= lower) & (subset["m_over_z_da"] <= upper)].copy()
    if subset.empty:
        return np.zeros((window_peaks.shape[0], 0), dtype=np.float64), []
    columns = []
    kept_labels: list[str] = []
    for species_label, group in subset.groupby("species_label", sort=False):
        mz_values = group["m_over_z_da"].to_numpy(dtype=np.float64)
        weights = group["natural_probability"].clip(lower=1.0e-12).to_numpy(dtype=np.float64)
        weights = weights / max(float(weights.sum()), 1.0e-12)
        template = np.zeros(window_peaks.shape[0], dtype=np.float64)
        for mz_value, weight in zip(mz_values, weights, strict=True):
            template += weight * np.exp(-0.5 * ((peak_mz - mz_value) / sigma) ** 2)
        max_value = float(template.max())
        if max_value <= 0.0:
            continue
        template = template / max_value
        columns.append(template)
        kept_labels.append(str(species_label))
    if not columns:
        return np.zeros((window_peaks.shape[0], 0), dtype=np.float64), []
    return np.column_stack(columns), kept_labels


def _joint_species_support_map(
    window_peaks: pd.DataFrame,
    *,
    species_library: pd.DataFrame,
    species_labels: list[str],
    config: dict,
) -> tuple[dict[tuple[str, str], dict[str, float]], list[str]]:
    if window_peaks.empty or not species_labels:
        return {}, []
    design_matrix, kept_labels = _species_template_matrix(
        window_peaks,
        species_library,
        species_labels,
        config=config,
    )
    if design_matrix.shape[0] == 0 or design_matrix.shape[1] < 2:
        return {}, kept_labels
    observed = window_peaks["integrated_area"].to_numpy(dtype=np.float64)
    if not np.isfinite(observed).all() or observed.sum() <= 0.0:
        return {}, kept_labels
    amplitudes, _ = nnls(design_matrix, observed)
    fitted = design_matrix * amplitudes[np.newaxis, :]
    predicted = fitted.sum(axis=1)
    support_map: dict[tuple[str, str], dict[str, float]] = {}
    for peak_pos, peak_id in enumerate(window_peaks["peak_id"].astype(str).tolist()):
        observed_area = max(float(observed[peak_pos]), 1.0)
        coverage = float(np.clip(predicted[peak_pos] / observed_area, 0.0, 1.0))
        for species_pos, species_label in enumerate(kept_labels):
            contribution = float(np.clip(fitted[peak_pos, species_pos] / observed_area, 0.0, 1.0))
            if contribution <= 0.0:
                continue
            support_map[(peak_id, species_label)] = {
                "score": contribution * coverage,
                "coverage": coverage,
            }
    return support_map, kept_labels


def _window_atomic_family_keys(
    window_peaks: pd.DataFrame,
    species_library: pd.DataFrame,
    *,
    config: dict,
) -> list[tuple[str, int]]:
    if window_peaks.empty or species_library.empty:
        return []
    atomic = species_library[species_library["category"] == "atomic"].copy()
    if atomic.empty:
        return []
    analysis_cfg = config["analysis"]
    margin = float(analysis_cfg["local_fit_window_margin_da"])
    lower = float(window_peaks["peak_mz_da"].min() - margin)
    upper = float(window_peaks["peak_mz_da"].max() + margin)
    atomic = atomic[(atomic["m_over_z_da"] >= lower) & (atomic["m_over_z_da"] <= upper)].copy()
    if atomic.empty:
        return []
    family_keys: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for _, peak in window_peaks.iterrows():
        tolerance = peak_tolerance_da(
            float(peak["peak_mz_da"]),
            ppm_tolerance=float(analysis_cfg["ppm_tolerance"]),
            min_absolute_tolerance_da=float(analysis_cfg["min_absolute_tolerance_da"]),
            max_absolute_tolerance_da=float(analysis_cfg["max_absolute_tolerance_da"]),
        )
        matches = atomic[np.abs(atomic["m_over_z_da"] - float(peak["peak_mz_da"])) <= tolerance]
        for _, row in matches.iterrows():
            symbols = row.get("symbols")
            if not isinstance(symbols, list) or len(symbols) != 1:
                continue
            family_key = (str(symbols[0]), int(row["charge"]))
            if family_key in seen:
                continue
            seen.add(family_key)
            family_keys.append(family_key)
    return family_keys


def _rerank_assignment_group(group: pd.DataFrame, *, config: dict) -> pd.DataFrame:
    group = group.copy()
    if group.empty:
        return group
    if group.shape[0] == 1 and str(group.iloc[0]["species_label"]) == "unassigned":
        return group
    group = group.sort_values(["score", "natural_probability"], ascending=[False, False]).reset_index(drop=True)
    probabilities = assignment_softmax(group["score"].to_numpy(dtype=np.float64))
    group["probability"] = probabilities
    group["rank"] = np.arange(1, group.shape[0] + 1)
    group["candidate_count"] = int(group.shape[0])
    top_probability = float(group.iloc[0]["probability"])
    score_gap = float(group.iloc[0]["score"] - group.iloc[1]["score"]) if group.shape[0] > 1 else float("inf")
    analysis_cfg = config["analysis"]
    if top_probability >= float(analysis_cfg["high_confidence_min_probability"]) and score_gap >= float(
        analysis_cfg["ambiguous_score_delta"]
    ):
        top_confidence = "high"
    elif group.shape[0] > 1:
        top_confidence = "ambiguous"
    else:
        top_confidence = "tentative"
    group["confidence"] = ["candidate"] * group.shape[0]
    group.loc[group.index[0], "confidence"] = top_confidence
    return group


def _apply_local_family_fit(
    peaks: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    species_library: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    analysis_cfg = config["analysis"]
    if peaks.empty or assignments.empty or not bool(analysis_cfg.get("local_fit_enabled", False)):
        return assignments
    adjusted = assignments.copy()
    adjusted["base_score"] = adjusted["score"]
    adjusted["local_fit_score"] = 0.0
    adjusted["local_fit_coverage"] = 0.0
    adjusted["local_fit_window_id"] = np.nan
    windows = _crowded_peak_windows(peaks, adjusted, config=config)
    if not windows:
        return adjusted
    max_rank = int(analysis_cfg["local_fit_max_rank"])
    local_fit_weight = float(analysis_cfg["local_fit_weight"])
    peaks_indexed = peaks.set_index("peak_id")
    for window_idx, peak_ids in enumerate(windows, start=1):
        window_peaks = peaks_indexed.loc[peak_ids].reset_index().sort_values("peak_mz_da").reset_index(drop=True)
        if window_peaks.empty:
            continue
        window_assignments = adjusted[
            (adjusted["peak_id"].isin(peak_ids))
            & (adjusted["rank"] <= max_rank)
        ].copy()
        family_keys = _window_atomic_family_keys(window_peaks, species_library, config=config)
        seen = set(family_keys)
        for _, row in window_assignments.iterrows():
            family_key = _atomic_family_from_row(row)
            if family_key is None or family_key in seen:
                continue
            seen.add(family_key)
            family_keys.append(family_key)
        if len(family_keys) < 2:
            continue
        design_matrix, kept_families = _family_template_matrix(window_peaks, family_keys, config=config)
        if design_matrix.shape[1] < 2:
            continue
        observed = window_peaks["integrated_area"].to_numpy(dtype=np.float64)
        if not np.isfinite(observed).all() or observed.sum() <= 0.0:
            continue
        amplitudes, _ = nnls(design_matrix, observed)
        fitted = design_matrix * amplitudes[np.newaxis, :]
        predicted = fitted.sum(axis=1)
        family_support = {}
        for peak_pos, peak_id in enumerate(window_peaks["peak_id"].tolist()):
            coverage = float(np.clip(predicted[peak_pos] / max(observed[peak_pos], 1.0), 0.0, 1.0))
            for family_pos, family_key in enumerate(kept_families):
                contribution = float(np.clip(fitted[peak_pos, family_pos] / max(observed[peak_pos], 1.0), 0.0, 1.0))
                family_support[(peak_id, family_key)] = {
                    "score": contribution * coverage,
                    "coverage": coverage,
                }
        for idx, row in adjusted[adjusted["peak_id"].isin(peak_ids)].iterrows():
            family_key = _atomic_family_from_row(row)
            if family_key is None:
                continue
            support = family_support.get((str(row["peak_id"]), family_key))
            if not support:
                continue
            boost = float(support["score"])
            adjusted.at[idx, "local_fit_score"] = boost
            adjusted.at[idx, "local_fit_coverage"] = float(support["coverage"])
            adjusted.at[idx, "local_fit_window_id"] = window_idx
            adjusted.at[idx, "score"] = float(row["score"]) + local_fit_weight * boost
    groups = [
        _rerank_assignment_group(group, config=config)
        for _, group in adjusted.groupby("peak_id", sort=False)
    ]
    return pd.concat(groups, ignore_index=True) if groups else adjusted


def _apply_global_species_fit(
    peaks: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    species_library: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    analysis_cfg = config["analysis"]
    if peaks.empty or assignments.empty or not bool(analysis_cfg.get("joint_species_global_enabled", False)):
        return assignments
    adjusted = assignments.copy()
    adjusted["joint_species_global_score"] = 0.0
    adjusted["joint_species_global_coverage"] = 0.0
    species_labels = _select_joint_species_labels(
        adjusted,
        peaks,
        max_rank=int(analysis_cfg.get("joint_species_global_max_rank", 4)),
        max_species=int(analysis_cfg.get("joint_species_global_max_species", 48)),
        min_vote_area=float(analysis_cfg.get("joint_species_global_min_vote_area", 0.0)),
    )
    support_map, kept_labels = _joint_species_support_map(
        peaks,
        species_library=species_library,
        species_labels=species_labels,
        config=config,
    )
    if len(kept_labels) < 2 or not support_map:
        return assignments
    weight = float(analysis_cfg.get("joint_species_global_weight", 0.0))
    for idx, row in adjusted.iterrows():
        support = support_map.get((str(row["peak_id"]), str(row["species_label"])))
        if not support:
            continue
        adjusted.at[idx, "joint_species_global_score"] = float(support["score"])
        adjusted.at[idx, "joint_species_global_coverage"] = float(support["coverage"])
        adjusted.at[idx, "score"] = float(row["score"]) + weight * float(support["score"])
    groups = [
        _rerank_assignment_group(group, config=config)
        for _, group in adjusted.groupby("peak_id", sort=False)
    ]
    return pd.concat(groups, ignore_index=True) if groups else adjusted


def _apply_window_species_fit(
    peaks: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    species_library: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    analysis_cfg = config["analysis"]
    if peaks.empty or assignments.empty or not bool(analysis_cfg.get("joint_species_window_enabled", False)):
        return assignments
    adjusted = assignments.copy()
    adjusted["joint_species_window_score"] = 0.0
    adjusted["joint_species_window_coverage"] = 0.0
    adjusted["joint_species_window_id"] = np.nan
    windows = _crowded_peak_windows(peaks, adjusted, config=config)
    if not windows:
        return assignments
    peaks_indexed = peaks.set_index("peak_id")
    weight = float(analysis_cfg.get("joint_species_window_weight", 0.0))
    max_rank = int(analysis_cfg.get("joint_species_window_max_rank", 4))
    max_species = int(analysis_cfg.get("joint_species_window_max_species", 16))
    min_vote_area = float(analysis_cfg.get("joint_species_window_min_vote_area", 0.0))
    for window_idx, peak_ids in enumerate(windows, start=1):
        window_peaks = peaks_indexed.loc[peak_ids].reset_index().sort_values("peak_mz_da").reset_index(drop=True)
        if window_peaks.empty:
            continue
        window_assignments = adjusted[
            (adjusted["peak_id"].isin(peak_ids))
            & (adjusted["rank"] <= max_rank)
            & (adjusted["species_label"] != "unassigned")
        ].copy()
        if window_assignments.empty:
            continue
        species_labels = _select_joint_species_labels(
            window_assignments,
            window_peaks,
            max_rank=max_rank,
            max_species=max_species,
            min_vote_area=min_vote_area,
        )
        support_map, kept_labels = _joint_species_support_map(
            window_peaks,
            species_library=species_library,
            species_labels=species_labels,
            config=config,
        )
        if len(kept_labels) < 2 or not support_map:
            continue
        for idx, row in adjusted[adjusted["peak_id"].isin(peak_ids)].iterrows():
            support = support_map.get((str(row["peak_id"]), str(row["species_label"])))
            if not support:
                continue
            adjusted.at[idx, "joint_species_window_score"] = float(support["score"])
            adjusted.at[idx, "joint_species_window_coverage"] = float(support["coverage"])
            adjusted.at[idx, "joint_species_window_id"] = float(window_idx)
            adjusted.at[idx, "score"] = float(row["score"]) + weight * float(support["score"])
    groups = [
        _rerank_assignment_group(group, config=config)
        for _, group in adjusted.groupby("peak_id", sort=False)
    ]
    return pd.concat(groups, ignore_index=True) if groups else adjusted


def _average_element_support_from_row(
    row: pd.Series,
    *,
    element_support: dict[str, float],
) -> float:
    symbols = row.get("symbols")
    if not isinstance(symbols, list) or not symbols:
        return 0.0
    return float(np.mean([float(element_support.get(str(symbol), 0.0)) for symbol in symbols]))


def _joint_window_map(
    peaks: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    config: dict,
) -> tuple[dict[str, int], list[list[str]]]:
    windows = _crowded_peak_windows(peaks, assignments, config=config)
    peak_to_window: dict[str, int] = {}
    for window_idx, peak_ids in enumerate(windows, start=1):
        for peak_id in peak_ids:
            peak_to_window[str(peak_id)] = int(window_idx)
    return peak_to_window, windows


def _joint_optimizer_step(
    working: pd.DataFrame,
    peaks: pd.DataFrame,
    *,
    peak_to_window: dict[str, int],
    config: dict,
) -> pd.DataFrame:
    if working.empty:
        return working
    analysis_cfg = config["analysis"]
    updated = working.copy()
    updated["joint_species_global_score"] = 0.0
    updated["joint_species_window_score"] = 0.0
    updated["joint_element_global_score"] = 0.0
    updated["joint_family_global_score"] = 0.0
    updated["joint_species_window_id"] = updated["peak_id"].map(lambda value: peak_to_window.get(str(value), np.nan))

    candidate_rows = updated[updated["species_label"] != "unassigned"].copy()
    if candidate_rows.empty:
        return updated

    peak_areas = peaks.set_index("peak_id")["integrated_area"].fillna(0.0)
    candidate_rows["peak_area"] = candidate_rows["peak_id"].map(peak_areas).fillna(0.0)
    candidate_rows["weighted_area"] = candidate_rows["peak_area"] * candidate_rows["probability"].clip(lower=1.0e-9)
    support_probability_min = float(analysis_cfg.get("joint_optimizer_support_probability_min", 0.10))
    support_rows = candidate_rows[candidate_rows["probability"] >= support_probability_min].copy()

    species_area = candidate_rows.groupby("species_label")["weighted_area"].sum().to_dict()
    species_peak_counts = (
        support_rows.groupby("species_label")["peak_id"].nunique().to_dict()
        if not support_rows.empty
        else {}
    )
    min_global_species_peaks = int(analysis_cfg.get("joint_optimizer_global_species_min_peaks", 2))
    global_species_area_min = float(analysis_cfg.get("joint_optimizer_global_species_min_area", 0.0))
    filtered_species_area = {
        str(species_label): float(area)
        for species_label, area in species_area.items()
        if int(species_peak_counts.get(species_label, 0)) >= min_global_species_peaks
        and float(area) >= global_species_area_min
    }
    species_support = _normalize_score_map(filtered_species_area)

    element_scores: defaultdict[str, float] = defaultdict(float)
    family_scores: defaultdict[tuple[str, int], float] = defaultdict(float)
    for _, row in candidate_rows.iterrows():
        weighted_area = float(row["weighted_area"])
        element_counts = row.get("element_counts", {})
        if isinstance(element_counts, dict):
            for element, count in element_counts.items():
                element_scores[str(element)] += weighted_area * float(count)
        family_key = _atomic_family_from_row(row)
        if family_key is not None:
            family_scores[family_key] += weighted_area
    element_support = _normalize_score_map(dict(element_scores))
    family_support = _normalize_score_map(dict(family_scores))

    window_species_support: dict[tuple[int, str], float] = {}
    window_candidate_rows = candidate_rows[candidate_rows["joint_species_window_id"].notna()].copy()
    if not window_candidate_rows.empty:
        grouped = (
            window_candidate_rows.groupby(["joint_species_window_id", "species_label"])["weighted_area"]
            .sum()
            .reset_index()
        )
        window_support_rows = support_rows[support_rows["joint_species_window_id"].notna()].copy()
        window_species_peak_counts = (
            window_support_rows.groupby(["joint_species_window_id", "species_label"])["peak_id"]
            .nunique()
            .to_dict()
            if not window_support_rows.empty
            else {}
        )
        min_window_species_peaks = int(analysis_cfg.get("joint_optimizer_window_species_min_peaks", 2))
        window_species_area_min = float(analysis_cfg.get("joint_optimizer_window_species_min_area", 0.0))
        for window_id, group in grouped.groupby("joint_species_window_id", sort=False):
            support = _normalize_score_map(
                {
                    str(row["species_label"]): float(row["weighted_area"])
                    for _, row in group.iterrows()
                    if int(window_species_peak_counts.get((row["joint_species_window_id"], row["species_label"]), 0))
                    >= min_window_species_peaks
                    and float(row["weighted_area"]) >= window_species_area_min
                }
            )
            for species_label, score in support.items():
                window_species_support[(int(window_id), str(species_label))] = float(score)

    global_species_weight = float(analysis_cfg.get("joint_optimizer_global_species_weight", 0.0))
    window_species_weight = float(analysis_cfg.get("joint_optimizer_window_species_weight", 0.0))
    global_element_weight = float(analysis_cfg.get("joint_optimizer_global_element_weight", 0.0))
    global_family_weight = float(analysis_cfg.get("joint_optimizer_global_family_weight", 0.0))
    local_fit_weight = float(analysis_cfg.get("local_fit_weight", 0.0))

    adjusted_scores = []
    for idx, row in updated.iterrows():
        if str(row["species_label"]) == "unassigned":
            adjusted_scores.append(float(row["score"]))
            continue
        species_score = float(species_support.get(str(row["species_label"]), 0.0)) ** 0.5
        window_id = peak_to_window.get(str(row["peak_id"]))
        window_score = 0.0
        if window_id is not None:
            window_score = float(window_species_support.get((int(window_id), str(row["species_label"])), 0.0)) ** 0.5
        element_score = _average_element_support_from_row(row, element_support=element_support) ** 0.5
        family_key = _atomic_family_from_row(row)
        family_score = float(family_support.get(family_key, 0.0)) ** 0.5 if family_key is not None else 0.0
        updated.at[idx, "joint_species_global_score"] = species_score
        updated.at[idx, "joint_species_window_score"] = window_score
        updated.at[idx, "joint_element_global_score"] = element_score
        updated.at[idx, "joint_family_global_score"] = family_score
        adjusted_scores.append(
            float(row["base_score"])
            + local_fit_weight * float(row.get("local_fit_score", 0.0) or 0.0)
            + global_species_weight * species_score
            + window_species_weight * window_score
            + global_element_weight * element_score
            + global_family_weight * family_score
        )
    updated["score"] = np.asarray(adjusted_scores, dtype=np.float64)
    groups = [
        _rerank_assignment_group(group, config=config)
        for _, group in updated.groupby("peak_id", sort=False)
    ]
    return pd.concat(groups, ignore_index=True) if groups else updated


def _optimize_joint_assignments(
    peaks: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    config: dict,
) -> pd.DataFrame:
    if assignments.empty:
        return assignments
    analysis_cfg = config["analysis"]
    peak_to_window, _ = _joint_window_map(peaks, assignments, config=config)
    working = assignments.copy()
    iterations = max(int(analysis_cfg.get("joint_optimizer_iterations", 3)), 1)
    for _ in range(iterations):
        next_working = _joint_optimizer_step(
            working,
            peaks,
            peak_to_window=peak_to_window,
            config=config,
        )
        if next_working.empty:
            break
        current_scores = (
            working[["peak_id", "candidate_id", "species_label", "score"]]
            .sort_values(["peak_id", "species_label", "candidate_id"], na_position="last")
            .reset_index(drop=True)
        )
        next_scores = (
            next_working[["peak_id", "candidate_id", "species_label", "score"]]
            .sort_values(["peak_id", "species_label", "candidate_id"], na_position="last")
            .reset_index(drop=True)
        )
        if current_scores.shape != next_scores.shape:
            score_delta = float("inf")
        else:
            current_array = current_scores["score"].to_numpy(dtype=np.float64)
            next_array = next_scores["score"].to_numpy(dtype=np.float64)
            both_infinite = ~np.isfinite(current_array) & ~np.isfinite(next_array)
            current_array = np.where(both_infinite, 0.0, current_array)
            next_array = np.where(both_infinite, 0.0, next_array)
            score_delta = float(
                np.nanmax(np.abs(next_array - current_array))
            )
        working = next_working
        if score_delta <= float(analysis_cfg.get("joint_optimizer_convergence_tol", 1.0e-4)):
            break
    return working


def _assign_peaks_once(
    peaks: pd.DataFrame,
    *,
    species_library: pd.DataFrame,
    config: dict,
    family_scores: dict[str, float],
    pass_name: str,
    element_prior: dict[str, float] | None = None,
    charge_prior: dict[int, float] | None = None,
    family_prior: dict[tuple[str, int], float] | None = None,
    use_family_hints: bool = True,
    element_support: dict[str, float] | None = None,
) -> pd.DataFrame:
    if peaks.empty:
        return pd.DataFrame()
    analysis_cfg = config["analysis"]
    library_mz = species_library["m_over_z_da"].to_numpy()
    rows: list[dict[str, object]] = []
    use_second_pass_priors = element_prior is not None or charge_prior is not None or family_prior is not None
    # Molecular isotopologue envelope: a molecular candidate matched through a
    # minor isotopologue is only physical if its dominant isotopologue shows a
    # correspondingly larger peak. Atomic families get this via family_score;
    # molecules need their own dominant-must-dominate check.
    dominant_isotopologue: dict[tuple[str, int], tuple[float, float]] = {}
    molecular_library = species_library[species_library["category"] == "molecular"]
    if not molecular_library.empty:
        dominant_indices = molecular_library.groupby(["species_label", "charge"])[
            "natural_probability"
        ].idxmax()
        for library_idx in dominant_indices:
            library_row = species_library.loc[library_idx]
            dominant_isotopologue[
                (str(library_row["species_label"]), int(library_row["charge"]))
            ] = (float(library_row["m_over_z_da"]), float(library_row["natural_probability"]))
    peak_mz_all = peaks["peak_mz_da"].to_numpy(dtype=np.float64)
    if "integrated_area" in peaks.columns and peaks["integrated_area"].notna().any():
        peak_intensity_all = peaks["integrated_area"].fillna(0.0).to_numpy(dtype=np.float64)
    else:
        peak_intensity_all = peaks.get(
            "peak_height", pd.Series(np.zeros(peaks.shape[0]))
        ).fillna(0.0).to_numpy(dtype=np.float64)
    envelope_penalty_weight = float(analysis_cfg.get("molecular_envelope_penalty", 2.5))
    envelope_min_area_ratio = float(analysis_cfg.get("molecular_envelope_min_area_ratio", 0.25))
    for _, peak in peaks.iterrows():
        mz = float(peak["peak_mz_da"])
        charge_hint = peak.get("charge_hint")
        neutral_mass_hint = peak.get("neutral_mass_hint")
        family_hint_element = peak.get("family_hint_element")
        family_hint_charge = peak.get("family_hint_charge")
        family_hint_isotope_label = peak.get("family_hint_isotope_label")
        family_hint_strength = float(peak.get("family_hint_score", 0.0) or 0.0)
        family_hint_exclusivity = float(peak.get("family_hint_exclusivity", 1.0) or 0.0)
        family_hint_candidates = _coerce_family_hint_candidates(peak.get("family_hint_candidates"))
        tolerance = peak_tolerance_da(
            mz,
            ppm_tolerance=float(analysis_cfg["ppm_tolerance"]),
            min_absolute_tolerance_da=float(analysis_cfg["min_absolute_tolerance_da"]),
            max_absolute_tolerance_da=float(analysis_cfg["max_absolute_tolerance_da"]),
        )
        matches = species_library.loc[np.abs(library_mz - mz) <= tolerance].copy()
        if matches.empty:
            rows.append(
                {
                    "peak_id": peak["peak_id"],
                    "assignment_pass": pass_name,
                    "rank": 1,
                    "candidate_count": 0,
                    "candidate_id": None,
                    "category": "unassigned",
                    "charge": np.nan,
                    "species_label": "unassigned",
                    "isotopologue_label": "unassigned",
                    "candidate_mz_da": np.nan,
                    "mass_error_da": np.nan,
                    "natural_probability": np.nan,
                    "mass_score": 0.0,
                    "natural_score": 0.0,
                    "family_score": 0.0,
                    "charge_penalty": 0.0,
                    "element_prior_score": 0.0,
                    "charge_prior_score": 0.0,
                    "family_prior_score": 0.0,
                    "unsupported_charge_penalty": 0.0,
                    "base_score": -np.inf,
                    "local_fit_score": 0.0,
                    "local_fit_coverage": 0.0,
                    "local_fit_window_id": np.nan,
                    "score": -np.inf,
                    "probability": 1.0,
                    "confidence": "unassigned",
                    "element_counts": {},
                    "isotope_counts": {},
                    "element_support_score": 0.0,
                }
            )
            continue
        mass_error = mz - matches["m_over_z_da"]
        # High-count APT peak centroids are statistically precise to well under a
        # mDa; what limits mass accuracy is ppm-scale calibration systematics.
        # Scoring with tolerance/3 (tens of mDa) let chemically wrong candidates
        # 10-25 mDa away stay competitive, so the score width is tied to a ppm
        # budget instead. The candidate *list* still uses the wide tolerance.
        mass_sigma = float(
            np.clip(
                float(analysis_cfg.get("mass_score_sigma_ppm", 300.0)) * 1.0e-6 * mz,
                float(analysis_cfg.get("mass_score_sigma_min_da", 0.005)),
                float(analysis_cfg.get("mass_score_sigma_max_da", 0.025)),
            )
        )
        mass_score = np.exp(-0.5 * (mass_error / max(mass_sigma, 1e-6)) ** 2)
        natural_score = _normalized_natural_probability(matches["natural_probability"])
        matches["family_score"] = matches["candidate_id"].map(lambda value: family_scores.get(str(value), 0.5))
        matches["mass_error_da"] = mass_error
        matches["mass_score"] = mass_score
        matches["natural_score"] = natural_score
        matches["charge_penalty"] = matches["charge"].map(lambda value: _charge_state_penalty(int(value), config=config))
        matches["element_prior_score"] = matches["symbols"].map(
            lambda symbols: _candidate_element_prior_score(symbols, element_prior)
        )
        matches["charge_prior_score"] = matches["charge"].map(
            lambda value: _candidate_charge_prior_score(int(value), charge_prior)
        )
        matches["family_prior_score"] = matches.apply(
            lambda row: _candidate_family_prior_score(row, family_prior),
            axis=1,
        )
        if pd.notna(charge_hint):
            hinted_charge = int(abs(charge_hint))
            matches["charge_hint_score"] = matches["charge"].map(
                lambda value: 1.0 if int(value) == hinted_charge else 0.0
            )
        else:
            matches["charge_hint_score"] = 0.0
        if use_family_hints:
            if use_second_pass_priors:
                family_hint_gate = np.sqrt(
                    np.maximum(
                        matches["element_prior_score"].to_numpy(dtype=np.float64),
                        matches["family_prior_score"].to_numpy(dtype=np.float64),
                    )
                ) * np.sqrt(np.clip(matches["family_score"].to_numpy(dtype=np.float64), 0.0, 1.0))
                matches["family_hint_gate"] = np.clip(family_hint_gate, 0.0, 1.0)
            else:
                matches["family_hint_gate"] = 1.0
            matches["family_hint_match_score"] = matches.apply(
                lambda row: _family_hint_match_score(
                    row,
                    hint_element=None if pd.isna(family_hint_element) else str(family_hint_element),
                    hint_charge=None if pd.isna(family_hint_charge) else int(family_hint_charge),
                    hint_isotope_label=None if pd.isna(family_hint_isotope_label) else str(family_hint_isotope_label),
                ),
                axis=1,
            )
            matches["family_hint_candidate_support"] = matches.apply(
                lambda row: _family_hint_candidate_support(
                    row,
                    family_hint_candidates=family_hint_candidates,
                ),
                axis=1,
            )
        else:
            matches["family_hint_gate"] = 0.0
            matches["family_hint_match_score"] = 0.0
            matches["family_hint_candidate_support"] = 0.0
        molecular_support_min = float(analysis_cfg.get("molecular_unsupported_element_min_support", 0.03))
        molecular_penalty_weight = float(analysis_cfg.get("molecular_unsupported_element_penalty", 0.9))
        atomic_support_min = float(analysis_cfg.get("atomic_unsupported_element_min_support", 0.05))
        atomic_penalty_weight = float(analysis_cfg.get("atomic_unsupported_element_penalty", 0.8))

        def _unsupported_element_penalty_value(row: pd.Series) -> float:
            # A candidate that invokes an element with no independent
            # sample-level evidence is a mass coincidence, not chemistry.
            if not element_support:
                return 0.0
            symbols = row.get("symbols")
            if not isinstance(symbols, list) or not symbols:
                return 0.0
            if str(row.get("category", "")) == "molecular" and len(symbols) >= 2:
                return molecular_penalty_weight * int(
                    sum(
                        1
                        for symbol in set(symbols)
                        if float(element_support.get(str(symbol), 0.0)) < molecular_support_min
                    )
                )
            # Atomic: smooth ramp so weak-but-real trace elements are dampened,
            # not erased, while zero-evidence elements pay the full penalty.
            support = float(element_support.get(str(symbols[0]), 0.0))
            if support >= atomic_support_min or atomic_support_min <= 0.0:
                return 0.0
            return atomic_penalty_weight * (1.0 - support / atomic_support_min)

        exotic_elements = set(analysis_cfg.get("exotic_elements", ["He", "Ne", "Ar", "Kr", "Xe", "Rn"]))
        exotic_penalty_weight = float(analysis_cfg.get("exotic_element_penalty", 1.5))

        def _exotic_element_penalty_value(row: pd.Series) -> float:
            # Noble gases essentially never field-evaporate as ions in APT
            # (unless deliberately implanted, in which case hint them); a
            # noble-gas candidate is nearly always a mass coincidence, e.g.
            # He+ (4.0026) capturing D2+ (4.028) on deuterium-charged steel.
            if exotic_penalty_weight <= 0.0:
                return 0.0
            symbols = row.get("symbols")
            if not isinstance(symbols, list):
                return 0.0
            count = sum(1 for symbol in set(symbols) if str(symbol) in exotic_elements)
            return exotic_penalty_weight * count

        matches["unsupported_element_penalty"] = matches.apply(
            _unsupported_element_penalty_value,
            axis=1,
        )
        peak_intensity_current = float(
            peak.get("integrated_area")
            if pd.notna(peak.get("integrated_area", np.nan))
            else peak.get("peak_height", 0.0) or 0.0
        )
        # Hydride-on-family-member guard: when this peak is a confirmed member
        # of element X's isotope family (fingerprint reproduces natural
        # abundances), an X-hydride candidate a few mDa away is the classic
        # +1 Da systematic, not independent evidence. Penalize it so the
        # family keeps the area unless abundances actually demand a hydride.
        hydride_penalty_weight = float(analysis_cfg.get("family_member_hydride_penalty", 1.1))

        def _hydride_on_member_penalty(row: pd.Series) -> float:
            if (
                hydride_penalty_weight <= 0.0
                or str(row.get("category", "")) != "molecular"
                or pd.isna(family_hint_element)
                or pd.isna(family_hint_charge)
            ):
                return 0.0
            counts = row.get("element_counts")
            if not isinstance(counts, dict) or "H" not in counts:
                return 0.0
            non_hydrogen = {symbol for symbol in counts if symbol != "H"}
            if non_hydrogen != {str(family_hint_element)}:
                return 0.0
            if int(row.get("charge", 0)) != int(family_hint_charge):
                return 0.0
            return hydride_penalty_weight

        matches["hydride_member_penalty"] = matches.apply(_hydride_on_member_penalty, axis=1)
        # Molecular-family fingerprint hint: a peak claimed as a member of a
        # verified molecular family (e.g. FeO+) boosts candidates with that
        # exact formula and charge, mirroring atomic family hints.
        molecular_formula = peak.get("molecular_family_formula")
        molecular_charge = peak.get("molecular_family_charge")
        molecular_hint_weight = float(analysis_cfg.get("molecular_family_hint_weight", 1.2))
        if (
            isinstance(molecular_formula, dict)
            and molecular_formula
            and pd.notna(molecular_charge)
            and molecular_hint_weight > 0.0
        ):
            normalized_formula = {str(k): int(v) for k, v in molecular_formula.items()}

            def _molecular_hint_match(row: pd.Series) -> float:
                counts = row.get("element_counts")
                if not isinstance(counts, dict):
                    return 0.0
                if int(row.get("charge", 0)) != int(molecular_charge):
                    return 0.0
                return 1.0 if {str(k): int(v) for k, v in counts.items()} == normalized_formula else 0.0

            matches["molecular_family_hint"] = matches.apply(_molecular_hint_match, axis=1)
        else:
            matches["molecular_family_hint"] = 0.0

        def _envelope_penalty_value(row: pd.Series) -> float:
            if str(row.get("category", "")) != "molecular" or envelope_penalty_weight <= 0.0:
                return 0.0
            key = (str(row["species_label"]), int(row["charge"]))
            dominant = dominant_isotopologue.get(key)
            if dominant is None:
                return 0.0
            dominant_mz, dominant_probability = dominant
            row_probability = float(row.get("natural_probability", 0.0) or 0.0)
            if row_probability <= 0.0 or row_probability >= 0.9 * dominant_probability:
                return 0.0
            expected_dominant = peak_intensity_current * dominant_probability / row_probability
            dominant_tolerance = peak_tolerance_da(
                dominant_mz,
                ppm_tolerance=float(analysis_cfg["ppm_tolerance"]),
                min_absolute_tolerance_da=float(analysis_cfg["min_absolute_tolerance_da"]),
                max_absolute_tolerance_da=float(analysis_cfg["max_absolute_tolerance_da"]),
            )
            nearby = np.abs(peak_mz_all - dominant_mz) <= dominant_tolerance
            observed = float(peak_intensity_all[nearby].max()) if nearby.any() else 0.0
            required = envelope_min_area_ratio * expected_dominant
            if required <= 0.0 or observed >= required:
                return 0.0
            return envelope_penalty_weight * (1.0 - min(observed / required, 1.0))

        matches["envelope_penalty"] = matches.apply(_envelope_penalty_value, axis=1)
        matches["exotic_penalty"] = matches.apply(_exotic_element_penalty_value, axis=1)
        matches["unsupported_charge_penalty"] = [
            _unsupported_charge_penalty(
                int(charge),
                charge_prior_score=float(charge_prior_score),
                config=config,
            )
            for charge, charge_prior_score in zip(matches["charge"], matches["charge_prior_score"], strict=False)
        ]
        score = (
            float(analysis_cfg["mass_score_weight"]) * matches["mass_score"]
            + float(analysis_cfg["support_score_weight"]) * matches["support_score"]
            + float(analysis_cfg["natural_score_weight"]) * matches["natural_score"]
            + float(analysis_cfg["family_score_weight"]) * matches["family_score"]
            + float(analysis_cfg.get("charge_hint_weight", 0.0)) * matches["charge_hint_score"]
            # Hint bonuses are gated by the candidate's own mass agreement so a
            # family fit whose shared shift absorbed a large mass error cannot
            # override a chemically simpler candidate that actually sits on the
            # measured centroid.
            + float(analysis_cfg.get("family_hint_weight", 0.0))
            * family_hint_strength
            * family_hint_exclusivity
            * matches["family_hint_gate"]
            * matches["family_hint_match_score"]
            * matches["mass_score"]
            + float(
                analysis_cfg.get(
                    "family_hint_multi_weight",
                    0.35 * float(analysis_cfg.get("family_hint_weight", 0.0)),
                )
            )
            * matches["family_hint_gate"]
            * matches["family_hint_candidate_support"]
            * matches["mass_score"]
            - matches["charge_penalty"]
            - matches["unsupported_charge_penalty"]
            - matches["unsupported_element_penalty"]
            - matches["envelope_penalty"]
            - matches["hydride_member_penalty"]
            - matches["exotic_penalty"]
            + float(analysis_cfg.get("molecular_family_hint_weight", 1.2))
            * matches["molecular_family_hint"]
            * matches["mass_score"]
        )
        if use_second_pass_priors:
            score = (
                score
                + float(analysis_cfg["element_prior_weight"]) * matches["element_prior_score"]
                + float(analysis_cfg["charge_prior_weight"]) * matches["charge_prior_score"]
                + float(analysis_cfg.get("family_prior_weight", 0.0)) * matches["family_prior_score"]
            )
        if pd.notna(neutral_mass_hint):
            neutral_mass_hint = float(neutral_mass_hint)
            neutral_weight = float(analysis_cfg.get("neutral_hint_weight", 0.0))
            if neutral_weight > 0.0:
                neutral_error = np.abs(matches["neutral_mass_da"] - neutral_mass_hint)
                neutral_tolerance = max(tolerance * max(int(abs(charge_hint)) if pd.notna(charge_hint) else 1, 1), 0.05)
                neutral_score = np.exp(-0.5 * (neutral_error / max(neutral_tolerance / 3.0, 1e-6)) ** 2)
                if bool(analysis_cfg.get("neutral_hint_requires_charge_match", False)) and pd.notna(charge_hint):
                    neutral_score = neutral_score * matches["charge_hint_score"]
                score = score + neutral_weight * neutral_score
        matches["score"] = score
        matches = matches.sort_values(["score", "natural_probability"], ascending=[False, False]).head(8)
        top_score = float(matches.iloc[0]["score"])
        top_charge = int(matches.iloc[0]["charge"])
        if top_score < _min_assignment_score(top_charge, config=config):
            rows.append(
                {
                    "peak_id": peak["peak_id"],
                    "assignment_pass": pass_name,
                    "rank": 1,
                    "candidate_count": int(matches.shape[0]),
                    "candidate_id": None,
                    "category": "unassigned",
                    "charge": np.nan,
                    "species_label": "unassigned",
                    "isotopologue_label": "unassigned",
                    "candidate_mz_da": np.nan,
                    "mass_error_da": np.nan,
                    "natural_probability": np.nan,
                    "mass_score": 0.0,
                    "natural_score": 0.0,
                    "family_score": 0.0,
                    "charge_penalty": 0.0,
                    "element_prior_score": 0.0,
                    "charge_prior_score": 0.0,
                    "family_prior_score": 0.0,
                    "unsupported_charge_penalty": 0.0,
                    "base_score": top_score,
                    "local_fit_score": 0.0,
                    "local_fit_coverage": 0.0,
                    "local_fit_window_id": np.nan,
                    "score": top_score,
                    "probability": 1.0,
                    "confidence": "unassigned",
                    "element_counts": {},
                    "isotope_counts": {},
                    "element_support_score": 0.0,
                }
            )
            continue
        probabilities = assignment_softmax(matches["score"].to_numpy(dtype=np.float64))
        matches["probability"] = probabilities
        candidate_count = int(matches.shape[0])
        top_probability = float(matches.iloc[0]["probability"])
        score_gap = (
            float(matches.iloc[0]["score"] - matches.iloc[1]["score"])
            if matches.shape[0] > 1
            else float("inf")
        )
        if top_probability >= float(analysis_cfg["high_confidence_min_probability"]) and score_gap >= float(
            analysis_cfg["ambiguous_score_delta"]
        ):
            confidence = "high"
        elif candidate_count > 1:
            confidence = "ambiguous"
        else:
            confidence = "tentative"
        for rank, (_, match) in enumerate(matches.iterrows(), start=1):
            rows.append(
                {
                    "peak_id": peak["peak_id"],
                    "assignment_pass": pass_name,
                    "rank": rank,
                    "candidate_count": candidate_count,
                    "candidate_id": match["candidate_id"],
                    "category": match["category"],
                    "charge": int(match["charge"]),
                    "species_label": match["species_label"],
                    "isotopologue_label": match["isotopologue_label"],
                    "candidate_mz_da": float(match["m_over_z_da"]),
                    "mass_error_da": float(match["mass_error_da"]),
                    "natural_probability": float(match["natural_probability"]),
                    "mass_score": float(match["mass_score"]),
                    "natural_score": float(match["natural_score"]),
                    "family_score": float(match["family_score"]),
                    "charge_penalty": float(match["charge_penalty"]),
                    "element_prior_score": float(match["element_prior_score"]),
                    "charge_prior_score": float(match["charge_prior_score"]),
                    "family_prior_score": float(match["family_prior_score"]),
                    "unsupported_charge_penalty": float(match["unsupported_charge_penalty"]),
                    "base_score": float(match["score"]),
                    "local_fit_score": 0.0,
                    "local_fit_coverage": 0.0,
                    "local_fit_window_id": np.nan,
                    "score": float(match["score"]),
                    "probability": float(match["probability"]),
                    "confidence": confidence if rank == 1 else "candidate",
                    "element_counts": match["element_counts"],
                    "isotope_counts": match["isotope_counts"],
                    "element_support_score": float(match["support_score"]),
                }
            )
    return pd.DataFrame(rows)


def _extra_candidates_from_molecular_families(
    peaks: pd.DataFrame,
    library: pd.DataFrame,
) -> pd.DataFrame:
    """Inject candidates for formulas confirmed by molecular-family
    fingerprint recovery that the combinatorial library cannot represent
    (e.g. 4-atom FeO3). Expansion mirrors build_species_library rows."""
    from itertools import product as _product

    from rangefinder.analysis.common import formula_from_element_counter, isotopologue_label
    from collections import Counter

    if "molecular_family_formula" not in peaks.columns:
        return pd.DataFrame()
    claimed: list[tuple[dict, int]] = []
    seen: set[tuple] = set()
    for _, row in peaks.iterrows():
        formula = row.get("molecular_family_formula")
        charge = row.get("molecular_family_charge")
        if not isinstance(formula, dict) or not formula or pd.isna(charge):
            continue
        key = (tuple(sorted(formula.items())), int(charge))
        if key in seen:
            continue
        seen.add(key)
        claimed.append((formula, int(charge)))
    if not claimed:
        return pd.DataFrame()
    existing = {
        (tuple(sorted(dict(counts).items())), int(charge))
        for counts, charge in zip(library["element_counts"], library["charge"], strict=False)
        if isinstance(counts, dict)
    }
    table = isotope_table()
    rows: list[dict[str, object]] = []
    candidate_number = 1
    for formula, charge in claimed:
        if (tuple(sorted(formula.items())), charge) in existing:
            continue
        per_atom = []
        for element, count in formula.items():
            family = table[table["symbol"] == element]
            options = [
                (str(r["isotope_label"]), str(r["symbol"]), float(r["mass_da"]), float(r["natural_abundance"]))
                for _, r in family.iterrows()
                if float(r["natural_abundance"]) >= 1.0e-3
            ]
            if not options:
                per_atom = []
                break
            for _ in range(int(count)):
                per_atom.append(options)
        if not per_atom:
            continue
        combos: dict[tuple, tuple[float, float]] = {}
        for combo in _product(*per_atom):
            key = tuple(sorted(item[0] for item in combo))
            mass = float(sum(item[2] for item in combo))
            probability = float(np.prod([item[3] for item in combo]))
            if key in combos:
                combos[key] = (combos[key][0], combos[key][1] + probability)
            else:
                combos[key] = (mass, probability)
        element_counter = Counter()
        for element, count in formula.items():
            element_counter[str(element)] += int(count)
        atom_count = sum(element_counter.values())
        for key, (mass, probability) in combos.items():
            if probability < 1.0e-4:
                continue
            isotope_counter = Counter(key)
            rows.append(
                {
                    "candidate_id": f"cand-mf-{candidate_number:05d}",
                    "category": "molecular",
                    "charge": int(charge),
                    "neutral_mass_da": mass,
                    "m_over_z_da": mass / charge,
                    "species_label": formula_from_element_counter(element_counter, charge),
                    "isotopologue_label": isotopologue_label(isotope_counter, charge),
                    "element_counts": dict(element_counter),
                    "isotope_counts": dict(isotope_counter),
                    "symbols": sorted(element_counter),
                    "natural_probability": probability,
                    "complexity_penalty": 0.12 + 0.08 * (atom_count - 1) + 0.03 * (charge - 1),
                    "support_score": 0.6,
                }
            )
            candidate_number += 1
    return pd.DataFrame(rows)


def _filter_species_library(library: pd.DataFrame, excluded_elements: set[str] | None) -> pd.DataFrame:
    if not excluded_elements or library.empty:
        return library
    mask = library["symbols"].map(
        lambda symbols: not any(str(symbol) in excluded_elements for symbol in symbols)
    )
    return library.loc[mask].reset_index(drop=True)


def assign_peaks_two_pass(
    peaks: pd.DataFrame,
    *,
    config: dict,
    excluded_elements: set[str] | None = None,
    library_cache: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if peaks.empty:
        return pd.DataFrame(), peaks.copy()
    analysis_cfg = config["analysis"]
    base_element_support = infer_element_support(
        peaks,
        max_charge=int(analysis_cfg["max_charge"]),
        ppm_tolerance=float(analysis_cfg["ppm_tolerance"]),
        min_absolute_tolerance_da=float(analysis_cfg["min_absolute_tolerance_da"]),
        max_absolute_tolerance_da=float(analysis_cfg["max_absolute_tolerance_da"]),
    )
    always_include = list(analysis_cfg["always_include_elements"])
    always_include.extend(_hinted_elements_from_peaks(peaks, config=config))
    always_include = list(dict.fromkeys(always_include))
    pass1_library = _cached_species_library(
        library_cache,
        base_element_support,
        always_include,
        max_candidate_elements=int(analysis_cfg["max_candidate_elements"]),
        max_charge=int(analysis_cfg["pass1_max_charge"]),
        max_molecular_charge=int(analysis_cfg["pass1_max_molecular_charge"]),
        max_molecular_atoms=int(analysis_cfg["pass1_max_molecular_atoms"]),
    )
    pass1_library = _filter_species_library(pass1_library, excluded_elements)
    pass1_family_scores = atomic_family_consistency_scores(peaks, pass1_library, config=config)
    pass1_assignments = _assign_peaks_once(
        peaks,
        species_library=pass1_library,
        config=config,
        family_scores=pass1_family_scores,
        pass_name="pass1",
        use_family_hints=False,
        element_support=base_element_support,
    )
    pass1_element_prior, pass1_charge_prior = _priors_from_assignments(pass1_assignments, peaks)
    pass1_family_prior = _family_priors_from_assignments(pass1_assignments, peaks)
    element_support = _blend_support_maps(
        base_element_support,
        pass1_element_prior,
        pass_weight=float(analysis_cfg["pass1_support_blend_weight"]),
    )
    species_library = _cached_species_library(
        library_cache,
        element_support,
        always_include,
        max_candidate_elements=int(analysis_cfg["max_candidate_elements"]),
        max_charge=int(analysis_cfg["max_charge"]),
        max_molecular_charge=int(analysis_cfg["max_molecular_charge"]),
        max_molecular_atoms=int(analysis_cfg["max_molecular_atoms"]),
    )
    species_library = _filter_species_library(species_library, excluded_elements)
    family_scores = atomic_family_consistency_scores(peaks, species_library, config=config)
    assignments = _assign_peaks_once(
        peaks,
        species_library=species_library,
        config=config,
        family_scores=family_scores,
        pass_name="pass2",
        element_prior=pass1_element_prior,
        charge_prior=pass1_charge_prior,
        family_prior=pass1_family_prior,
        use_family_hints=True,
        element_support=element_support,
    )
    assignments = _apply_local_family_fit(peaks, assignments, species_library=species_library, config=config)
    assignments["assignment_backend"] = "two_pass"
    return assignments, top_assignment_table(assignments, peaks)


def _cached_species_library(
    cache: dict | None,
    element_support: dict[str, float],
    always_include: list[str],
    *,
    max_candidate_elements: int,
    max_charge: int,
    max_molecular_charge: int,
    max_molecular_atoms: int,
) -> pd.DataFrame:
    """Build the (unfiltered) species library, memoized within one assign_peaks
    call. The library is a pure function of the isotope table plus these
    arguments; element pruning re-invokes the backend up to three times with the
    same peaks (hence the same element support), so the ~400k-row combinatorial
    build would otherwise run identically three times. excluded_elements is
    applied afterwards by the caller, so it is not part of the key."""
    key = (
        tuple(sorted(element_support.items())),
        tuple(always_include),
        max_candidate_elements,
        max_charge,
        max_molecular_charge,
        max_molecular_atoms,
    )
    if cache is not None and key in cache:
        return cache[key].copy()
    built = build_species_library(
        isotope_table(),
        element_support,
        always_include=always_include,
        max_candidate_elements=max_candidate_elements,
        max_charge=max_charge,
        max_molecular_charge=max_molecular_charge,
        max_molecular_atoms=max_molecular_atoms,
    )
    if cache is not None:
        # Store the pristine build; hand callers a copy so downstream mutation
        # (added score columns, concatenated extra candidates) never corrupts it.
        cache[key] = built
        return built.copy()
    return built


def assign_peaks_joint(
    peaks: pd.DataFrame,
    *,
    config: dict,
    excluded_elements: set[str] | None = None,
    library_cache: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if peaks.empty:
        return pd.DataFrame(), peaks.copy()
    analysis_cfg = config["analysis"]
    base_element_support = infer_element_support(
        peaks,
        max_charge=int(analysis_cfg["max_charge"]),
        ppm_tolerance=float(analysis_cfg["ppm_tolerance"]),
        min_absolute_tolerance_da=float(analysis_cfg["min_absolute_tolerance_da"]),
        max_absolute_tolerance_da=float(analysis_cfg["max_absolute_tolerance_da"]),
    )
    always_include = list(analysis_cfg["always_include_elements"])
    always_include.extend(_hinted_elements_from_peaks(peaks, config=config))
    always_include = list(dict.fromkeys(always_include))
    species_library = _cached_species_library(
        library_cache,
        base_element_support,
        always_include,
        max_candidate_elements=int(analysis_cfg["max_candidate_elements"]),
        max_charge=int(analysis_cfg["max_charge"]),
        max_molecular_charge=int(analysis_cfg["max_molecular_charge"]),
        max_molecular_atoms=int(analysis_cfg["max_molecular_atoms"]),
    )
    species_library = _filter_species_library(species_library, excluded_elements)
    extra_candidates = _extra_candidates_from_molecular_families(peaks, species_library)
    if not extra_candidates.empty:
        species_library = pd.concat(
            [species_library, _filter_species_library(extra_candidates, excluded_elements)],
            ignore_index=True,
        ).sort_values("m_over_z_da", ignore_index=True)
    family_scores = atomic_family_consistency_scores(peaks, species_library, config=config)
    assignments = _assign_peaks_once(
        peaks,
        species_library=species_library,
        config=config,
        family_scores=family_scores,
        pass_name="joint_init",
        use_family_hints=True,
        element_support=base_element_support,
    )
    if assignments.empty:
        return assignments, top_assignment_table(assignments, peaks)
    assignments = _apply_local_family_fit(
        peaks,
        assignments,
        species_library=species_library,
        config=config,
    )
    assignments = _optimize_joint_assignments(
        peaks,
        assignments,
        config=config,
    )
    assignments["assignment_pass"] = "joint"
    assignments["assignment_backend"] = "joint_optimizer"
    return assignments, top_assignment_table(assignments, peaks)


def _anchored_elements(
    peaks: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    config: dict,
) -> set[str]:
    """Elements with independent evidence: an atomic rank-1 assignment at
    decent probability, a confirmed isotope-family fingerprint, or a user
    hint. Molecular species can then vouch for their partners, but only from
    already-anchored ground (a confident SiO+ anchors O when Si is anchored;
    an exotic N2Ni2+ anchors nothing)."""
    analysis_cfg = config["analysis"]
    anchor_min_probability = float(analysis_cfg.get("element_pruning_anchor_min_probability", 0.35))
    molecular_anchor_min_probability = float(
        analysis_cfg.get("element_pruning_molecular_anchor_min_probability", 0.6)
    )
    anchored: set[str] = set(_hinted_elements_from_peaks(peaks, config=config))
    if "family_hint_element" in peaks.columns:
        anchored |= {
            str(element)
            for element in peaks["family_hint_element"].dropna().tolist()
            if str(element)
        }
    rank1 = assignments[
        (assignments["rank"] == 1) & (assignments["species_label"] != "unassigned")
    ]
    for _, row in rank1.iterrows():
        counts = row.get("element_counts")
        if not isinstance(counts, dict) or not counts:
            continue
        if str(row.get("category", "")) == "atomic" and float(row.get("probability", 0.0)) >= anchor_min_probability:
            anchored.update(str(element) for element in counts)
    # Molecular vouching, iterated to fixpoint (an anchored partner chain).
    for _ in range(3):
        added = False
        for _, row in rank1.iterrows():
            if str(row.get("category", "")) != "molecular":
                continue
            if float(row.get("probability", 0.0)) < molecular_anchor_min_probability:
                continue
            if str(row.get("confidence", "")) != "high":
                continue
            counts = row.get("element_counts")
            if not isinstance(counts, dict) or not counts:
                continue
            elements = {str(element) for element in counts}
            unanchored = elements - anchored
            if len(unanchored) == 1 and len(elements) > 1:
                anchored |= unanchored
                added = True
        if not added:
            break
    return anchored


def _elements_in_use(assignments: pd.DataFrame) -> set[str]:
    used: set[str] = set()
    for counts in assignments.loc[
        assignments["species_label"] != "unassigned", "element_counts"
    ].tolist():
        if isinstance(counts, dict):
            used.update(str(element) for element in counts)
    return used


def assign_peaks(
    peaks: pd.DataFrame,
    *,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    analysis_cfg = config["analysis"]
    backend = str(analysis_cfg.get("assignment_backend", "joint_optimizer"))
    backend_runner = assign_peaks_two_pass if backend == "two_pass" else assign_peaks_joint
    # The combinatorial species library is identical across pruning rounds
    # (same peaks -> same element support); build it once and reuse.
    library_cache: dict = {}
    assignments, summary = backend_runner(peaks, config=config, library_cache=library_cache)
    if assignments.empty or not bool(analysis_cfg.get("element_pruning_enabled", True)):
        return assignments, summary
    excluded: set[str] = set()
    for _ in range(int(analysis_cfg.get("element_pruning_max_rounds", 2))):
        anchored = _anchored_elements(peaks, assignments, config=config)
        unanchored = _elements_in_use(assignments) - anchored
        unanchored -= excluded
        if not unanchored:
            break
        excluded |= unanchored
        assignments, summary = backend_runner(
            peaks, config=config, excluded_elements=excluded, library_cache=library_cache
        )
        if assignments.empty:
            break
    if excluded and not assignments.empty:
        assignments["pruned_elements"] = ";".join(sorted(excluded))
    return assignments, summary


def composition_tables(
    peaks: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if peaks.empty or assignments.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty, empty
    peak_areas = peaks.set_index("peak_id")["integrated_area"].fillna(0.0)
    ranked = assignments.copy()
    ranked["peak_area"] = ranked["peak_id"].map(peak_areas).fillna(0.0)
    probability_floor = float(config["analysis"].get("composition_probability_floor", 0.0))
    if probability_floor > 0.0:
        # Losing candidates below the floor otherwise siphon a few percent of
        # every strong peak into phantom species; drop them and renormalize
        # so each peak's area is conserved among the surviving candidates.
        keep = (
            (ranked["rank"] == 1)
            | (ranked["species_label"] == "unassigned")
            | (ranked["probability"] >= probability_floor)
        )
        ranked = ranked.loc[keep].copy()
        assigned_mask = ranked["species_label"] != "unassigned"
        probability_sums = (
            ranked.loc[assigned_mask].groupby("peak_id")["probability"].transform("sum")
        )
        ranked.loc[assigned_mask, "probability"] = ranked.loc[
            assigned_mask, "probability"
        ] / probability_sums.clip(lower=1.0e-9)
    ranked["weighted_area"] = ranked["peak_area"] * ranked["probability"]
    top = ranked[ranked["rank"] == 1].copy()
    top["robust_area"] = np.where(top["confidence"] == "high", top["peak_area"], 0.0)
    ambiguous_peaks = top[top["confidence"] == "ambiguous"][
        [
            "peak_id",
            "species_label",
            "isotopologue_label",
            "probability",
            "candidate_count",
            "mass_error_da",
        ]
    ].copy()

    species_rows = []
    element_rows = []
    atomic_isotope_rows = []
    top_atomic = top[(top["species_label"] != "unassigned") & (top["category"] == "atomic")].copy()
    for _, row in ranked.iterrows():
        if row["species_label"] == "unassigned":
            continue
        species_rows.append(
            {
                "species_label": row["species_label"],
                "category": row["category"],
                "charge": row["charge"],
                "weighted_counts": row["weighted_area"],
                "robust_counts": row["peak_area"] if row["rank"] == 1 and row["confidence"] == "high" else 0.0,
            }
        )
        for element, count in row["element_counts"].items():
            element_rows.append(
                {
                    "element": element,
                    "weighted_counts": row["weighted_area"] * count,
                    "robust_counts": (
                        row["peak_area"] * count
                        if row["rank"] == 1 and row["confidence"] == "high"
                        else 0.0
                    ),
                }
            )
    for _, row in top_atomic.iterrows():
        for isotope, count in row["isotope_counts"].items():
            atomic_isotope_rows.append(
                {
                    "isotope_label": isotope,
                    "element": "".join(filter(str.isalpha, isotope)),
                    "charge": int(row["charge"]) if pd.notna(row["charge"]) else np.nan,
                    "weighted_counts": row["peak_area"] * row["probability"] * count,
                    "robust_counts": row["peak_area"] * count if row["confidence"] == "high" else 0.0,
                }
            )
    species_frame = pd.DataFrame(species_rows)
    element_frame = pd.DataFrame(element_rows)
    atomic_isotope_frame = pd.DataFrame(atomic_isotope_rows)
    if species_frame.empty or element_frame.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty, ambiguous_peaks
    species = species_frame.groupby(["species_label", "category", "charge"], as_index=False).sum(
        numeric_only=True
    )
    species["atomic_percent_weighted"] = 100.0 * species["weighted_counts"] / max(
        float(species["weighted_counts"].sum()), 1.0
    )
    species["atomic_percent_robust"] = 100.0 * species["robust_counts"] / max(
        float(species["robust_counts"].sum()), 1.0
    )
    elements = element_frame.groupby(["element"], as_index=False).sum(numeric_only=True)
    elements["atomic_percent_weighted"] = 100.0 * elements["weighted_counts"] / max(
        float(elements["weighted_counts"].sum()), 1.0
    )
    elements["atomic_percent_robust"] = 100.0 * elements["robust_counts"] / max(
        float(elements["robust_counts"].sum()), 1.0
    )
    if atomic_isotope_frame.empty:
        isotopes = pd.DataFrame()
    else:
        family_totals = atomic_isotope_frame.groupby(["element", "charge"], as_index=False)[
            ["weighted_counts", "robust_counts"]
        ].sum()
        family_totals["selection_score"] = np.where(
            family_totals["robust_counts"] > 0.0,
            family_totals["robust_counts"],
            family_totals["weighted_counts"],
        )
        dominant_families = (
            family_totals.sort_values(["element", "selection_score", "weighted_counts"], ascending=[True, False, False])
            .groupby("element", as_index=False)
            .first()[["element", "charge"]]
            .rename(columns={"charge": "selected_charge"})
        )
        isotope_frame = atomic_isotope_frame.merge(dominant_families, on="element", how="inner")
        isotope_frame = isotope_frame[isotope_frame["charge"] == isotope_frame["selected_charge"]].copy()
        isotopes = isotope_frame.groupby(["element", "isotope_label"], as_index=False)[
            ["weighted_counts", "robust_counts"]
        ].sum()
        isotopes = isotopes.merge(dominant_families, on="element", how="left")
        isotopes = isotopes[(isotopes["weighted_counts"] > 0.0) | (isotopes["robust_counts"] > 0.0)].copy()
    if isotopes.empty:
        anomalies = pd.DataFrame()
        return species, elements, isotopes, anomalies, ambiguous_peaks
    isotopes["isotope_fraction_weighted"] = isotopes.groupby("element")["weighted_counts"].transform(
        lambda values: values / max(float(values.sum()), 1.0)
    )
    isotopes["isotope_fraction_robust"] = isotopes.groupby("element")["robust_counts"].transform(
        lambda values: values / max(float(values.sum()), 1.0)
    )
    natural = natural_abundance_by_element()
    resolution_rows: list[dict[str, object]] = []
    for element, group in isotopes.groupby("element", sort=True):
        natural_group = natural.get(element)
        dominant_natural_isotope = None
        dominant_present = False
        if natural_group is not None and not natural_group.empty:
            dominant_natural_isotope = str(
                natural_group.sort_values("natural_abundance", ascending=False).iloc[0]["isotope_label"]
            )
            dominant_present = bool(
                group.loc[group["isotope_label"] == dominant_natural_isotope, "weighted_counts"].sum() > 0.0
            )
        resolution_rows.append(
            {
                "element": element,
                "dominant_natural_isotope": dominant_natural_isotope,
                "dominant_natural_present": dominant_present,
                "isotope_resolved": bool(group.shape[0] >= 2 and dominant_present),
            }
        )
    resolution_frame = pd.DataFrame(resolution_rows)
    isotopes = isotopes.merge(resolution_frame, on="element", how="left")
    anomalies = isotope_anomaly_table(
        isotopes[isotopes["isotope_resolved"] == True].copy(),  # noqa: E712
        min_counts=int(config["analysis"]["min_isotope_counts_for_anomaly"]),
        slight_abs_diff=float(config["analysis"]["slight_anomaly_min_abs_fraction_diff"]),
        strong_abs_diff=float(config["analysis"]["strong_anomaly_min_abs_fraction_diff"]),
        slight_z=float(config["analysis"]["slight_anomaly_min_z"]),
        strong_z=float(config["analysis"]["strong_anomaly_min_z"]),
    )
    return species, elements, isotopes, anomalies, ambiguous_peaks


def isotope_anomaly_table(
    isotopes: pd.DataFrame,
    *,
    min_counts: int,
    slight_abs_diff: float,
    strong_abs_diff: float,
    slight_z: float,
    strong_z: float,
) -> pd.DataFrame:
    if isotopes.empty:
        return pd.DataFrame()
    natural = natural_abundance_by_element()
    rows: list[dict[str, object]] = []
    for element, group in isotopes.groupby("element", sort=True):
        observed_counts = group["robust_counts"].to_numpy(dtype=np.float64)
        if observed_counts.sum() < min_counts or group.shape[0] < 2:
            continue
        natural_group = natural.get(element)
        if natural_group is None:
            continue
        natural_group = natural_group[natural_group["isotope_label"].isin(group["isotope_label"])]
        if natural_group.empty or natural_group.shape[0] < 2:
            continue
        observed = group.set_index("isotope_label")["robust_counts"]
        expected = natural_group.set_index("isotope_label")["natural_abundance"]
        expected = expected / expected.sum()
        observed = observed.reindex(expected.index).fillna(0.0)
        total = float(observed.sum())
        observed_fraction = observed / max(total, 1.0)
        for isotope_label, expected_fraction in expected.items():
            obs = float(observed_fraction[isotope_label])
            exp = float(expected_fraction)
            variance = max(exp * (1.0 - exp) / max(total, 1.0), 1e-12)
            z = (obs - exp) / np.sqrt(variance)
            abs_diff = abs(obs - exp)
            if abs_diff >= strong_abs_diff and abs(z) >= strong_z:
                status = "strong"
            elif abs_diff >= slight_abs_diff and abs(z) >= slight_z:
                status = "slight"
            else:
                status = "none"
            rows.append(
                {
                    "element": element,
                    "isotope_label": isotope_label,
                    "observed_fraction": obs,
                    "expected_fraction": exp,
                    "fraction_difference": obs - exp,
                    "observed_to_natural_ratio": obs / exp if exp > 0 else np.nan,
                    "z_score": z,
                    "status": status,
                }
            )
    return pd.DataFrame(rows)
