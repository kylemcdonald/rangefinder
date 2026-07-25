"""Fixed-ranging regional composition quantification.

Peak detection, species identification, overlap deconvolution, and charge
family selection belong to the whole specimen. Regional analysis only
partitions those globally quantified peak contributions according to the raw
events inside each already-established integration window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rangefinder.analysis.assignment import (
    composition_contributions,
    composition_tables_from_contributions,
)

_COUNT_FIELDS = ("weighted_counts", "assignment_weighted_counts", "robust_counts")


def _peak_region_allocations(
    *,
    mz_values: np.ndarray,
    event_region: np.ndarray,
    peaks: pd.DataFrame,
    region_ids: list[int],
) -> pd.DataFrame:
    event_totals = np.asarray(
        [int(np.sum(event_region == region)) for region in region_ids],
        dtype=np.float64,
    )
    overall = event_totals / max(float(event_totals.sum()), 1.0)
    rows: list[dict[str, object]] = []
    for _, peak in peaks.iterrows():
        left = float(peak.get("integration_left_da", np.nan))
        right = float(peak.get("integration_right_da", np.nan))
        if np.isfinite(left) and np.isfinite(right) and right >= left:
            in_range = (mz_values >= left) & (mz_values <= right)
        else:
            in_range = np.zeros(mz_values.shape[0], dtype=bool)
        labels_in_range = event_region[in_range]
        counts = np.asarray(
            [int(np.sum(labels_in_range == region)) for region in region_ids],
            dtype=np.float64,
        )
        if float(counts.sum()) > 0.0:
            fractions = counts / float(counts.sum())
            method = "events_in_global_integration_window"
        else:
            fractions = overall
            method = "overall_event_fraction_fallback"
        for index, region in enumerate(region_ids):
            rows.append(
                {
                    "peak_id": peak["peak_id"],
                    "region": region,
                    "event_count_in_range": int(counts[index]),
                    "allocation_fraction": float(fractions[index]),
                    "allocation_method": method,
                }
            )
    return pd.DataFrame(rows)


def _recombination_diagnostics(
    global_table: pd.DataFrame,
    regional_tables: list[pd.DataFrame],
    *,
    keys: list[str],
) -> dict[str, object]:
    fields = [field for field in _COUNT_FIELDS if field in global_table.columns]
    if global_table.empty or not fields:
        return {"row_count": 0, "max_absolute_error": 0.0, "max_relative_error": 0.0}
    global_counts = global_table.groupby(keys, as_index=False)[fields].sum()
    regional_counts = pd.concat(regional_tables, ignore_index=True).groupby(
        keys, as_index=False
    )[fields].sum()
    compared = global_counts.merge(
        regional_counts,
        on=keys,
        how="outer",
        suffixes=("_global", "_regional"),
    ).fillna(0.0)
    max_absolute = 0.0
    max_relative = 0.0
    for field in fields:
        difference = (
            compared[f"{field}_regional"] - compared[f"{field}_global"]
        ).abs()
        denominator = compared[f"{field}_global"].abs().clip(lower=1.0)
        max_absolute = max(max_absolute, float(difference.max()))
        max_relative = max(max_relative, float((difference / denominator).max()))
    return {
        "row_count": int(compared.shape[0]),
        "max_absolute_error": max_absolute,
        "max_relative_error": max_relative,
    }


def partition_composition_by_region(
    *,
    mz_values: np.ndarray,
    event_region: np.ndarray,
    peaks: pd.DataFrame,
    assignments: pd.DataFrame,
    config: dict,
) -> dict[str, object]:
    """Partition one global ranging solution into spatial region compositions.

    The returned regional peak areas and assignment contributions sum back to
    the whole-sample solution. No regional spectrum is independently ranged.
    """
    mz_values = np.asarray(mz_values, dtype=np.float64)
    event_region = np.asarray(event_region)
    if mz_values.ndim != 1 or event_region.ndim != 1:
        raise ValueError("mz_values and event_region must be one-dimensional")
    if mz_values.shape[0] != event_region.shape[0]:
        raise ValueError("mz_values and event_region must have equal length")
    if event_region.size == 0:
        raise ValueError("event_region must not be empty")
    if not np.issubdtype(event_region.dtype, np.integer):
        if not np.all(np.isfinite(event_region)) or not np.all(event_region == np.floor(event_region)):
            raise ValueError("event_region must contain integer labels")
        event_region = event_region.astype(np.int64)
    region_ids = sorted(int(value) for value in np.unique(event_region))
    if region_ids[0] < 0:
        raise ValueError("event_region labels must be non-negative")

    ranked, ambiguous, overlap_diagnostics = composition_contributions(
        peaks,
        assignments,
        config=config,
    )
    global_species, global_elements, global_isotopes, _, _ = (
        composition_tables_from_contributions(
            ranked,
            config=config,
            ambiguous_peaks=ambiguous,
            overlap_diagnostics=overlap_diagnostics,
        )
    )
    selected_isotope_charges = (
        global_isotopes[["element", "selected_charge"]]
        .drop_duplicates("element")
        .set_index("element")["selected_charge"]
        .to_dict()
        if not global_isotopes.empty
        else {}
    )
    allocations = _peak_region_allocations(
        mz_values=mz_values,
        event_region=event_region,
        peaks=peaks,
        region_ids=region_ids,
    )

    regions: list[dict[str, object]] = []
    for region in region_ids:
        region_allocations = allocations[allocations["region"] == region].copy()
        fractions = region_allocations.set_index("peak_id")["allocation_fraction"]
        scaled = ranked.copy()
        scaled["regional_allocation_fraction"] = (
            scaled["peak_id"].map(fractions).fillna(0.0)
        )
        for field in ("peak_area", "assignment_weighted_area", "weighted_area"):
            scaled[field] = scaled[field] * scaled["regional_allocation_fraction"]
        species, elements, isotopes, anomalies, regional_ambiguous = (
            composition_tables_from_contributions(
                scaled,
                config=config,
                ambiguous_peaks=ambiguous,
                overlap_diagnostics=overlap_diagnostics,
                selected_isotope_charges=selected_isotope_charges,
            )
        )
        regional_peaks = peaks.copy()
        regional_peaks["global_integrated_area"] = regional_peaks["integrated_area"]
        regional_peaks["regional_allocation_fraction"] = (
            regional_peaks["peak_id"].map(fractions).fillna(0.0)
        )
        regional_peaks["integrated_area"] = (
            regional_peaks["global_integrated_area"]
            * regional_peaks["regional_allocation_fraction"]
        )
        regional_assignments = assignments.copy()
        regional_assignments["regional_allocation_fraction"] = (
            regional_assignments["peak_id"].map(fractions).fillna(0.0)
        )
        if not regional_ambiguous.empty:
            regional_ambiguous["regional_allocation_fraction"] = (
                regional_ambiguous["peak_id"].map(fractions).fillna(0.0)
            )
        regions.append(
            {
                "region": region,
                "event_count": int(np.sum(event_region == region)),
                "peaks": regional_peaks,
                "assignments": regional_assignments,
                "contributions": scaled,
                "species": species,
                "elements": elements,
                "isotopes": isotopes,
                "anomalies": anomalies,
                "ambiguous_peaks": regional_ambiguous,
                "peak_allocations": region_allocations,
            }
        )

    recombination = {
        "elements": _recombination_diagnostics(
            global_elements,
            [region["elements"] for region in regions],
            keys=["element"],
        ),
        "species": _recombination_diagnostics(
            global_species,
            [region["species"] for region in regions],
            keys=["species_label", "category", "charge"],
        ),
        "isotopes": _recombination_diagnostics(
            global_isotopes,
            [region["isotopes"] for region in regions],
            keys=["element", "isotope_label", "selected_charge"],
        ),
    }
    worst_absolute = max(
        float(table["max_absolute_error"]) for table in recombination.values()
    )
    worst_relative = max(
        float(table["max_relative_error"]) for table in recombination.values()
    )
    recombination["max_absolute_error"] = worst_absolute
    recombination["max_relative_error"] = worst_relative
    recombination["passed"] = bool(worst_absolute <= 1.0e-6 or worst_relative <= 1.0e-12)
    if not recombination["passed"]:
        raise RuntimeError(
            "regional composition failed to recombine with the whole-sample "
            f"solution (absolute={worst_absolute:.6g}, relative={worst_relative:.6g})"
        )

    return {
        "quantification_mode": "fixed_global_ranging",
        "region_ids": region_ids,
        "peak_allocations": allocations,
        "regions": regions,
        "recombination": recombination,
    }
