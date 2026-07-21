"""Parsers for APT range files (.rrng and Felfer .rng.fig.txt lists).

A range file is the human/expert answer key shipped with a public dataset: a
set of [lo, hi] m/z windows, each mapped to an ion with a stoichiometry.
Benchmarks use them two ways: the intervals act as reference peak locations,
and applying them to the raw spectrum yields a reference composition.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

_ELEMENT_COUNT_RE = re.compile(r"^([A-Z][a-z]?)(\d*)$")


def _parse_rrng(text: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    in_ranges = False
    for line in text.splitlines():
        line = line.strip()
        if line.lower() == "[ranges]":
            in_ranges = True
            continue
        if line.startswith("["):
            in_ranges = in_ranges and not line.startswith("[")
            continue
        if not in_ranges or "=" not in line:
            continue
        _, _, payload = line.partition("=")
        tokens = payload.split()
        if len(tokens) < 2:
            continue
        try:
            lo = float(tokens[0])
            hi = float(tokens[1])
        except ValueError:
            continue
        formula: dict[str, int] = {}
        for token in tokens[2:]:
            if ":" not in token:
                continue
            key, _, value = token.partition(":")
            if key in {"Vol", "Color", "Name"}:
                continue
            match = _ELEMENT_COUNT_RE.match(key)
            if match is None:
                continue
            try:
                count = int(value)
            except ValueError:
                continue
            if count > 0:
                formula[match.group(1)] = formula.get(match.group(1), 0) + count
        if not formula:
            continue
        rows.append({"mz_lo_da": lo, "mz_hi_da": hi, "formula": formula})
    return pd.DataFrame(rows)


_FIG_ION_RE = re.compile(r"^(?:\d+)?([A-Z][a-z]?)")


def _parse_rng_fig(text: str) -> pd.DataFrame:
    """Parse the Felfer Atom-Probe-Toolbox range list: lines like
    '56Fe++ 2.775530e+01 2.832031e+01' or '16O 1H2+ 1.79e+01 1.80e+01'."""
    rows: list[dict[str, object]] = []
    for line in text.splitlines():
        tokens = line.strip().split()
        if len(tokens) < 3:
            continue
        try:
            lo = float(tokens[-2])
            hi = float(tokens[-1])
        except ValueError:
            continue
        formula: dict[str, int] = {}
        for token in tokens[:-2]:
            token = token.rstrip("+-")
            match = re.match(r"^(?:\d+)?([A-Z][a-z]?)(\d*)$", token)
            if match is None:
                continue
            element = match.group(1)
            count = int(match.group(2)) if match.group(2) else 1
            formula[element] = formula.get(element, 0) + count
        if not formula:
            continue
        rows.append({"mz_lo_da": lo, "mz_hi_da": hi, "formula": formula})
    return pd.DataFrame(rows)


def load_range_file(path: Path) -> pd.DataFrame:
    """Return a DataFrame with mz_lo_da, mz_hi_da, formula ({element: count})."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() == ".rrng":
        frame = _parse_rrng(text)
    else:
        frame = _parse_rng_fig(text)
    if frame.empty:
        raise ValueError(f"No ranges parsed from {path}")
    frame = frame.sort_values("mz_lo_da").reset_index(drop=True)
    frame["mz_center_da"] = 0.5 * (frame["mz_lo_da"] + frame["mz_hi_da"])
    return frame


def range_reference_composition(
    ranges: pd.DataFrame,
    mz_values: np.ndarray,
) -> dict[str, float]:
    """Apply reference ranges to raw events -> atomic percent per element.

    This reproduces what an expert following the shipped range file would
    report (no background correction), which is the standard 'answer' for a
    public dataset without a certified bulk composition.
    """
    totals: dict[str, float] = {}
    mz_sorted = np.sort(np.asarray(mz_values, dtype=np.float64))
    for _, row in ranges.iterrows():
        lo = float(row["mz_lo_da"])
        hi = float(row["mz_hi_da"])
        count = int(
            np.searchsorted(mz_sorted, hi, side="right")
            - np.searchsorted(mz_sorted, lo, side="left")
        )
        if count <= 0:
            continue
        for element, multiplicity in dict(row["formula"]).items():
            totals[element] = totals.get(element, 0.0) + count * int(multiplicity)
    denominator = sum(totals.values())
    if denominator <= 0:
        return {}
    return {element: 100.0 * value / denominator for element, value in totals.items()}
