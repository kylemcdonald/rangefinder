from __future__ import annotations

from collections import Counter, defaultdict
import re

import numpy as np
import pandas as pd

from rangefinder.analysis.assignment import infer_element_support
from rangefinder.analysis.common import build_species_library, neutral_formula_from_element_counter
from rangefinder.references.isotopes import isotope_table


_FORMULA_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


def parse_formula(formula: str) -> dict[str, int]:
    text = str(formula).strip()
    if not text:
        raise ValueError("Formula must be non-empty.")
    counts: dict[str, int] = {}
    position = 0
    for match in _FORMULA_RE.finditer(text):
        if match.start() != position:
            raise ValueError(f"Unsupported formula syntax: {formula!r}")
        symbol = str(match.group(1))
        count_text = str(match.group(2))
        count = int(count_text) if count_text else 1
        counts[symbol] = counts.get(symbol, 0) + count
        position = match.end()
    if position != len(text):
        raise ValueError(f"Unsupported formula syntax: {formula!r}")
    return counts


def formula_category(formula: str) -> str:
    counts = parse_formula(formula)
    return "atomic" if len(counts) == 1 and next(iter(counts.values())) == 1 else "molecular"


def charged_formula_label(formula: str, charge: int) -> str:
    return f"{formula}{'+' if int(charge) == 1 else f'{int(charge)}+'}"


def select_apyt_species_candidates(peaks: pd.DataFrame, *, config: dict) -> pd.DataFrame:
    if peaks.empty:
        return pd.DataFrame(
            columns=[
                "formula",
                "charge",
                "category",
                "support_score",
                "natural_probability",
                "m_over_z_da",
            ]
        )
    analysis_cfg = config["analysis"]
    element_support = infer_element_support(
        peaks,
        max_charge=int(analysis_cfg["max_charge"]),
        ppm_tolerance=float(analysis_cfg["ppm_tolerance"]),
        min_absolute_tolerance_da=float(analysis_cfg["min_absolute_tolerance_da"]),
        max_absolute_tolerance_da=float(analysis_cfg["max_absolute_tolerance_da"]),
    )
    always_include = list(dict.fromkeys(list(analysis_cfg["always_include_elements"])))
    species_library = build_species_library(
        isotope_table(),
        element_support,
        always_include=always_include,
        max_candidate_elements=int(
            analysis_cfg.get("apyt_max_candidate_elements", analysis_cfg["max_candidate_elements"])
        ),
        max_charge=int(analysis_cfg.get("apyt_max_charge", analysis_cfg["max_charge"])),
        max_molecular_charge=int(
            analysis_cfg.get("apyt_max_molecular_charge", analysis_cfg["max_molecular_charge"])
        ),
        max_molecular_atoms=int(
            analysis_cfg.get("apyt_max_molecular_atoms", analysis_cfg["max_molecular_atoms"])
        ),
    )
    if species_library.empty:
        return pd.DataFrame()
    selected = species_library.copy()
    selected["formula"] = selected["element_counts"].map(
        lambda counts: neutral_formula_from_element_counter(Counter(dict(counts)))
    )
    min_fit_mz = float(
        analysis_cfg.get(
            "apyt_min_fit_mz",
            min(float(analysis_cfg["default_zoom_bin_width_da"]), 0.01),
        )
    )
    selected = selected[
        (selected["m_over_z_da"] >= min_fit_mz)
        & (selected["m_over_z_da"] <= float(analysis_cfg["max_mz"]))
    ].copy()
    if selected.empty:
        return pd.DataFrame()
    atomic = (
        selected[selected["category"] == "atomic"][
            ["formula", "charge", "category", "support_score", "natural_probability", "m_over_z_da"]
        ]
        .groupby(["formula", "charge", "category"], as_index=False)
        .max(numeric_only=True)
    )
    molecular = selected[selected["category"] == "molecular"].copy()
    if not molecular.empty:
        molecular = molecular.sort_values(
            ["support_score", "natural_probability", "m_over_z_da"],
            ascending=[False, False, True],
        )
        molecular = molecular.drop_duplicates(["formula", "charge"], keep="first")
        minimum_support = float(analysis_cfg.get("apyt_min_support_score", 0.0))
        minimum_probability = float(analysis_cfg.get("apyt_min_natural_probability", 0.0))
        molecular = molecular[
            (molecular["support_score"] >= minimum_support)
            & (molecular["natural_probability"] >= minimum_probability)
        ]
        molecular = molecular.head(int(analysis_cfg.get("apyt_max_molecular_species", 0)))
        molecular = molecular[
            ["formula", "charge", "category", "support_score", "natural_probability", "m_over_z_da"]
        ]
    selected = pd.concat([atomic, molecular], ignore_index=True)
    if selected.empty:
        return pd.DataFrame()
    return selected.sort_values(
        ["category", "support_score", "natural_probability", "formula", "charge"],
        ascending=[True, False, False, True, True],
        ignore_index=True,
    )


def build_apyt_species_dictionary(
    selection: pd.DataFrame,
    *,
    atomic_volume_nm3: float,
) -> dict[str, tuple[tuple[int, ...], float]]:
    if selection.empty:
        return {}
    species: dict[str, tuple[tuple[int, ...], float]] = {}
    for formula, group in selection.groupby("formula", sort=True):
        charges = tuple(sorted({int(value) for value in group["charge"].tolist()}))
        species[str(formula)] = (charges, float(atomic_volume_nm3))
    return species


def apyt_species_tables_from_counts(
    counts_list: list[dict[str, float | int | str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    species_rows: list[dict[str, object]] = []
    element_totals: defaultdict[str, float] = defaultdict(float)
    for item in counts_list:
        formula = str(item["element"])
        charge = int(item["charge"])
        count = float(item["count"])
        if not np.isfinite(count) or count <= 0.0:
            continue
        category = formula_category(formula)
        species_rows.append(
            {
                "species_label": charged_formula_label(formula, charge),
                "category": category,
                "charge": charge,
                "weighted_counts": count,
                "robust_counts": count,
            }
        )
        for symbol, atom_count in parse_formula(formula).items():
            element_totals[symbol] += count * float(atom_count)
    species = pd.DataFrame(species_rows)
    if species.empty:
        return pd.DataFrame(), pd.DataFrame()
    species = species.groupby(["species_label", "category", "charge"], as_index=False).sum(
        numeric_only=True
    )
    total_species = max(float(species["weighted_counts"].sum()), 1.0)
    species["atomic_percent_weighted"] = 100.0 * species["weighted_counts"] / total_species
    species["atomic_percent_robust"] = 100.0 * species["robust_counts"] / total_species
    elements = pd.DataFrame(
        [
            {
                "element": symbol,
                "weighted_counts": count,
                "robust_counts": count,
            }
            for symbol, count in sorted(element_totals.items())
        ]
    )
    total_elements = max(float(elements["weighted_counts"].sum()), 1.0)
    elements["atomic_percent_weighted"] = 100.0 * elements["weighted_counts"] / total_elements
    elements["atomic_percent_robust"] = 100.0 * elements["robust_counts"] / total_elements
    return (
        species.sort_values("atomic_percent_weighted", ascending=False, ignore_index=True),
        elements.sort_values("atomic_percent_weighted", ascending=False, ignore_index=True),
    )
