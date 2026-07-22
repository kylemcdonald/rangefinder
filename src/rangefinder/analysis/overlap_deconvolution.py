"""Deterministic isotope-envelope deconvolution for composition quantification.

Assignment remains responsible for deciding which ionic species are plausible.
This module performs a separate clean-room quantification step: it constructs
natural-isotope envelopes for those species, identifies connected overlap
components, and fits nonnegative abundances with a Poisson count model.  A fit
is used only when the component is identifiable and passes an explicit residual
gate; otherwise the assignment probabilities remain unchanged.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import product

import numpy as np
import pandas as pd
from scipy.optimize import minimize, nnls

from rangefinder.references.isotopes import isotope_table


def _peak_tolerance_da(
    peak_mz_da: float,
    *,
    ppm_tolerance: float,
    min_absolute_tolerance_da: float,
    max_absolute_tolerance_da: float,
) -> float:
    ppm_component = peak_mz_da * ppm_tolerance * 1.0e-6
    return float(
        np.clip(
            max(ppm_component, min_absolute_tolerance_da),
            min_absolute_tolerance_da,
            max_absolute_tolerance_da,
        )
    )


def _coerce_element_counts(value: object) -> dict[str, int]:
    if isinstance(value, dict):
        return {str(key): int(count) for key, count in value.items() if int(count) > 0}
    return {}


def isotopologue_envelope(
    element_counts: dict[str, int],
    charge: int,
    *,
    min_probability: float,
) -> list[tuple[float, float]]:
    """Expand a formula into a normalized ``(m/z, probability)`` envelope."""
    if not element_counts or int(charge) <= 0:
        return []
    isotopes = isotope_table()
    per_atom: list[list[tuple[float, float]]] = []
    for element, count in sorted(element_counts.items()):
        family = isotopes[isotopes["symbol"] == element]
        if family.empty:
            return []
        options = [
            (float(row["mass_da"]), float(row["natural_abundance"]))
            for _, row in family.iterrows()
            if float(row["natural_abundance"]) > 0.0
        ]
        per_atom.extend([options] * int(count))

    merged: defaultdict[float, float] = defaultdict(float)
    for combination in product(*per_atom):
        mass = round(sum(item[0] for item in combination), 6)
        probability = float(np.prod([item[1] for item in combination]))
        merged[mass] += probability
    kept = [
        (mass / float(charge), probability)
        for mass, probability in sorted(merged.items())
        if probability >= float(min_probability)
    ]
    total = float(sum(probability for _, probability in kept))
    if total <= 0.0:
        return []
    return [(mass, probability / total) for mass, probability in kept]


def _design_matrix(
    peaks: pd.DataFrame,
    species: pd.DataFrame,
    *,
    config: dict,
) -> tuple[np.ndarray, list[str], dict[str, float]]:
    analysis_cfg = config["analysis"]
    peak_mz = peaks["peak_mz_da"].to_numpy(dtype=np.float64)
    matrix = np.zeros((peaks.shape[0], species.shape[0]), dtype=np.float64)
    kept_labels: list[str] = []
    coverage: dict[str, float] = {}
    columns: list[np.ndarray] = []
    probability_floor = float(
        analysis_cfg.get("overlap_deconvolution_isotopologue_probability_floor", 1.0e-5)
    )
    tolerance_scale = float(analysis_cfg.get("overlap_deconvolution_match_tolerance_scale", 1.0))

    for _, row in species.iterrows():
        label = str(row["species_label"])
        counts = _coerce_element_counts(row["element_counts"])
        try:
            charge = int(row["charge"])
        except (TypeError, ValueError):
            continue
        envelope = isotopologue_envelope(
            counts,
            charge,
            min_probability=probability_floor,
        )
        if not envelope:
            continue
        column = np.zeros(peaks.shape[0], dtype=np.float64)
        for expected_mz, probability in envelope:
            if peak_mz.size == 0:
                continue
            nearest = int(np.argmin(np.abs(peak_mz - expected_mz)))
            tolerance = tolerance_scale * _peak_tolerance_da(
                float(expected_mz),
                ppm_tolerance=float(analysis_cfg["ppm_tolerance"]),
                min_absolute_tolerance_da=float(analysis_cfg["min_absolute_tolerance_da"]),
                max_absolute_tolerance_da=float(analysis_cfg["max_absolute_tolerance_da"]),
            )
            if abs(float(peak_mz[nearest] - expected_mz)) <= tolerance:
                column[nearest] += probability
        observed_coverage = float(column.sum())
        if observed_coverage <= 0.0:
            continue
        columns.append(column / observed_coverage)
        kept_labels.append(label)
        coverage[label] = observed_coverage

    if columns:
        matrix = np.column_stack(columns)
    else:
        matrix = np.zeros((peaks.shape[0], 0), dtype=np.float64)
    return matrix, kept_labels, coverage


def _connected_components(matrix: np.ndarray) -> list[tuple[list[int], list[int]]]:
    """Connected row/column components of a nonzero bipartite matrix."""
    row_neighbors = [set(np.flatnonzero(matrix[row] > 0.0).tolist()) for row in range(matrix.shape[0])]
    column_neighbors = [
        set(np.flatnonzero(matrix[:, column] > 0.0).tolist())
        for column in range(matrix.shape[1])
    ]
    components: list[tuple[list[int], list[int]]] = []
    unseen_columns = set(range(matrix.shape[1]))
    while unseen_columns:
        seed = min(unseen_columns)
        component_columns: set[int] = {seed}
        component_rows: set[int] = set()
        pending_columns = [seed]
        while pending_columns:
            column = pending_columns.pop()
            for row in sorted(column_neighbors[column]):
                if row in component_rows:
                    continue
                component_rows.add(row)
                for neighbor in sorted(row_neighbors[row]):
                    if neighbor not in component_columns:
                        component_columns.add(neighbor)
                        pending_columns.append(neighbor)
        unseen_columns -= component_columns
        components.append((sorted(component_rows), sorted(component_columns)))
    return components


def _poisson_abundances(matrix: np.ndarray, observed: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    initial, _ = nnls(matrix, observed)
    if not np.any(initial > 0.0):
        initial = np.full(matrix.shape[1], max(float(observed.sum()), 1.0) / matrix.shape[1])
    scale = max(float(observed.sum()), 1.0)
    initial_scaled = initial / scale
    epsilon = max(scale * 1.0e-12, 1.0e-9)

    def objective(scaled: np.ndarray) -> tuple[float, np.ndarray]:
        abundance = scaled * scale
        expected = np.clip(matrix @ abundance, epsilon, None)
        value = float(np.sum(expected - observed * np.log(expected)))
        gradient = scale * (matrix.T @ (1.0 - observed / expected))
        return value, gradient

    result = minimize(
        objective,
        initial_scaled,
        method="L-BFGS-B",
        jac=True,
        bounds=[(0.0, None)] * matrix.shape[1],
        options={"ftol": 1.0e-12, "gtol": 1.0e-8, "maxiter": 1000},
    )
    abundance = np.clip(np.asarray(result.x, dtype=np.float64) * scale, 0.0, None)
    expected = np.clip(matrix @ abundance, epsilon, None)
    positive = observed > 0.0
    deviance_terms = np.empty_like(observed)
    deviance_terms[positive] = 2.0 * (
        observed[positive] * np.log(observed[positive] / expected[positive])
        - (observed[positive] - expected[positive])
    )
    deviance_terms[~positive] = 2.0 * expected[~positive]
    relative_rmse = float(
        np.linalg.norm(expected - observed) / max(np.linalg.norm(observed), 1.0)
    )
    fisher_information = matrix.T @ (matrix / expected[:, np.newaxis])
    covariance = np.linalg.pinv(fisher_information, rcond=1.0e-12)
    abundance_std = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    return abundance, {
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "poisson_deviance": float(np.sum(np.clip(deviance_terms, 0.0, None))),
        "relative_rmse": relative_rmse,
        "abundance_std_counts": abundance_std.tolist(),
        "iterations": int(getattr(result, "nit", 0)),
    }


def deconvolve_peak_overlaps(
    peaks: pd.DataFrame,
    ranked_assignments: pd.DataFrame,
    *,
    config: dict,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return assignments with a quantification probability per peak.

    The original assignment ``probability`` is never overwritten.  Accepted
    overlap fits populate ``quantification_probability``; rejected or
    underdetermined components retain their original probabilities.
    """
    ranked = ranked_assignments.copy()
    ranked["quantification_probability"] = ranked.get(
        "probability", pd.Series(0.0, index=ranked.index)
    ).fillna(0.0)
    ranked["quantification_method"] = "assignment_probability"
    analysis_cfg = config["analysis"]
    enabled = bool(analysis_cfg.get("overlap_deconvolution_enabled", False))
    diagnostics: dict[str, object] = {
        "enabled": enabled,
        "method": "connected natural-isotope envelopes with nonnegative Poisson fitting",
        "accepted_component_count": 0,
        "rejected_component_count": 0,
        "deconvolved_peak_count": 0,
        "components": [],
    }
    if not enabled or peaks.empty or ranked.empty:
        return ranked, diagnostics

    peak_area = peaks.set_index("peak_id")["integrated_area"].fillna(0.0)
    candidates = ranked[
        (ranked["species_label"] != "unassigned")
        & (
            (ranked["rank"] <= int(analysis_cfg.get("overlap_deconvolution_max_rank", 4)))
            | (
                ranked["probability"]
                >= float(analysis_cfg.get("overlap_deconvolution_min_probability", 0.05))
            )
        )
    ].copy()
    if candidates.empty:
        return ranked, diagnostics
    candidates["peak_area"] = candidates["peak_id"].map(peak_area).fillna(0.0)
    candidates["vote"] = candidates["peak_area"] * candidates["probability"].clip(lower=0.0)
    votes = candidates.groupby("species_label", as_index=False)["vote"].sum()
    anchor_rows = candidates[candidates["rank"] == 1]
    if bool(analysis_cfg.get("overlap_deconvolution_require_high_confidence_anchor", True)):
        confidence = anchor_rows.get(
            "confidence", pd.Series("high", index=anchor_rows.index, dtype=object)
        )
        anchor_rows = anchor_rows[
            (confidence == "high")
            & (
                anchor_rows["probability"]
                >= float(
                    analysis_cfg.get("overlap_deconvolution_anchor_min_probability", 0.72)
                )
            )
        ]
    top_labels = set(anchor_rows["species_label"].astype(str).tolist())
    min_vote = float(analysis_cfg.get("overlap_deconvolution_min_vote_area", 100.0))
    voted_labels = set(votes.loc[votes["vote"] >= min_vote, "species_label"].astype(str))
    selected_labels = top_labels & voted_labels if top_labels else set()
    max_species = int(analysis_cfg.get("overlap_deconvolution_max_species", 64))
    ordered_labels = (
        votes[votes["species_label"].astype(str).isin(selected_labels)]
        .sort_values(["vote", "species_label"], ascending=[False, True])["species_label"]
        .astype(str)
        .tolist()[:max_species]
    )
    representatives = (
        candidates[candidates["species_label"].astype(str).isin(ordered_labels)]
        .sort_values(["species_label", "rank", "probability"], ascending=[True, True, False])
        .drop_duplicates("species_label")
        [["species_label", "element_counts", "charge"]]
    )
    ordered_peaks = peaks.sort_values("peak_mz_da").reset_index(drop=True)
    matrix, labels, coverage = _design_matrix(ordered_peaks, representatives, config=config)
    if matrix.shape[1] == 0:
        return ranked, diagnostics

    components = _connected_components(matrix)
    accepted_peaks: set[str] = set()
    max_condition = float(analysis_cfg.get("overlap_deconvolution_max_condition_number", 1.0e6))
    max_relative_rmse = float(analysis_cfg.get("overlap_deconvolution_max_relative_rmse", 0.35))
    min_coverage = float(analysis_cfg.get("overlap_deconvolution_min_envelope_coverage", 0.50))
    max_component_species = int(
        analysis_cfg.get("overlap_deconvolution_max_component_species", 8)
    )

    for component_id, (row_indices, column_indices) in enumerate(components, start=1):
        component_labels = [labels[column] for column in column_indices]
        component_peak_ids = ordered_peaks.iloc[row_indices]["peak_id"].astype(str).tolist()
        component_matrix = matrix[np.ix_(row_indices, column_indices)]
        observed = ordered_peaks.iloc[row_indices]["integrated_area"].fillna(0.0).to_numpy(dtype=np.float64)
        overlapping = bool(np.any(np.count_nonzero(component_matrix > 0.0, axis=1) >= 2))
        normalized = component_matrix / np.maximum(
            np.linalg.norm(component_matrix, axis=0, keepdims=True), 1.0e-12
        )
        rank = int(np.linalg.matrix_rank(normalized))
        condition = float(np.linalg.cond(normalized)) if normalized.size else float("inf")
        component_diagnostic: dict[str, object] = {
            "component_id": component_id,
            "peak_ids": component_peak_ids,
            "species_labels": component_labels,
            "peak_count": len(row_indices),
            "species_count": len(column_indices),
            "matrix_rank": rank,
            "condition_number": condition,
            "envelope_coverage": {label: float(coverage[label]) for label in component_labels},
            "overlapping": overlapping,
            "accepted": False,
        }
        if not overlapping:
            component_diagnostic["reason"] = "no shared peak"
            diagnostics["components"].append(component_diagnostic)
            continue
        if len(column_indices) > max_component_species:
            component_diagnostic["reason"] = "component exceeds species-count gate"
            diagnostics["rejected_component_count"] += 1
            diagnostics["components"].append(component_diagnostic)
            continue
        if rank < len(column_indices) or len(row_indices) < len(column_indices):
            component_diagnostic["reason"] = "underdetermined envelope matrix"
            diagnostics["rejected_component_count"] += 1
            diagnostics["components"].append(component_diagnostic)
            continue
        if not np.isfinite(condition) or condition > max_condition:
            component_diagnostic["reason"] = "ill-conditioned envelope matrix"
            diagnostics["rejected_component_count"] += 1
            diagnostics["components"].append(component_diagnostic)
            continue
        if min(coverage[label] for label in component_labels) < min_coverage:
            component_diagnostic["reason"] = "insufficient observed envelope coverage"
            diagnostics["rejected_component_count"] += 1
            diagnostics["components"].append(component_diagnostic)
            continue

        abundance, fit_diagnostics = _poisson_abundances(component_matrix, observed)
        component_diagnostic |= fit_diagnostics
        if not bool(fit_diagnostics["optimizer_success"]):
            component_diagnostic["reason"] = "optimizer did not converge"
            diagnostics["rejected_component_count"] += 1
            diagnostics["components"].append(component_diagnostic)
            continue
        if float(fit_diagnostics["relative_rmse"]) > max_relative_rmse:
            component_diagnostic["reason"] = "residual gate"
            diagnostics["rejected_component_count"] += 1
            diagnostics["components"].append(component_diagnostic)
            continue

        contributions = component_matrix * abundance[np.newaxis, :]
        for local_row, peak_id in enumerate(component_peak_ids):
            total = float(contributions[local_row].sum())
            if total <= 0.0:
                continue
            species_fractions = {
                component_labels[local_column]: float(contributions[local_row, local_column] / total)
                for local_column in range(len(component_labels))
                if contributions[local_row, local_column] > 0.0
            }
            peak_rows = ranked[ranked["peak_id"].astype(str) == peak_id]
            new_probabilities = pd.Series(0.0, index=peak_rows.index, dtype=np.float64)
            for label, fraction in species_fractions.items():
                label_rows = peak_rows[peak_rows["species_label"].astype(str) == label]
                if label_rows.empty:
                    continue
                within = label_rows["probability"].clip(lower=0.0).to_numpy(dtype=np.float64)
                if float(within.sum()) <= 0.0:
                    within = np.ones(label_rows.shape[0], dtype=np.float64)
                within = within / float(within.sum())
                new_probabilities.loc[label_rows.index] = fraction * within
            probability_sum = float(new_probabilities.sum())
            if probability_sum <= 0.0:
                continue
            ranked.loc[peak_rows.index, "quantification_probability"] = (
                new_probabilities / probability_sum
            )
            ranked.loc[peak_rows.index, "quantification_method"] = "isotope_envelope_poisson"
            accepted_peaks.add(peak_id)

        component_diagnostic["accepted"] = True
        component_diagnostic["reason"] = "accepted"
        abundance_std = np.asarray(
            component_diagnostic.pop("abundance_std_counts"), dtype=np.float64
        )
        component_diagnostic["fitted_observed_counts"] = {
            label: float(value) for label, value in zip(component_labels, abundance, strict=True)
        }
        component_diagnostic["fitted_counting_std"] = {
            label: float(value)
            for label, value in zip(component_labels, abundance_std, strict=True)
        }
        diagnostics["accepted_component_count"] += 1
        diagnostics["components"].append(component_diagnostic)

    diagnostics["deconvolved_peak_count"] = len(accepted_peaks)
    diagnostics["selected_species_count"] = len(labels)
    diagnostics["candidate_component_count"] = len(components)
    diagnostics["changed_area_fraction"] = float(
        sum(float(peak_area.get(peak_id, 0.0)) for peak_id in accepted_peaks)
        / max(float(peak_area.sum()), 1.0)
    )
    return ranked, diagnostics
