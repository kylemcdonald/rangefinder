"""Benchmark dataset registry: synthetic materials and real reference data.

Each dataset carries the best available ground truth:
- synthetic: exact composition and exact emitted line list (mz, weight);
- real controls with a certified/nominal bulk composition (nominal.yaml);
- real controls with an expert range file: reference composition obtained by
  applying the shipped ranges to the raw spectrum, plus the range intervals
  as reference peak locations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from rangefinder.benchmark.ranges import load_range_file, range_reference_composition
from rangefinder.validation.synthetic import (
    SyntheticMaterial,
    _isotopologues,
    default_validation_materials,
    synthesize_material_events,
    write_pos_file,
)


@dataclass
class BenchmarkDataset:
    name: str
    kind: str  # "synthetic" | "real"
    pos_path: Path
    truth_at_pct: dict[str, float] = field(default_factory=dict)
    truth_source: str = ""  # "synthetic" | "nominal" | "range-file"
    truth_lines: pd.DataFrame | None = None
    ranges: pd.DataFrame | None = None
    mass_resolving_power: float = 800.0
    arrays: tuple | None = None  # (x, y, z, mz) lazily populated


def truth_lines_for_material(material: SyntheticMaterial) -> pd.DataFrame:
    """All emitted isotopologue lines with their fraction of signal ions."""
    rows: list[dict[str, float]] = []
    for species, weight in material._species_weighted():
        for mass, probability in _isotopologues(species.formula):
            rows.append(
                {
                    "mz_da": mass / float(species.charge),
                    "weight": float(weight) * float(probability),
                }
            )
    frame = pd.DataFrame(rows).sort_values("mz_da").reset_index(drop=True)
    # Merge lines that are unresolvable at the simulated resolving power.
    merged: list[dict[str, float]] = []
    for _, row in frame.iterrows():
        fwhm = row["mz_da"] / material.mass_resolving_power
        if merged and abs(row["mz_da"] - merged[-1]["mz_da"]) <= max(0.6 * fwhm, 0.02):
            total = merged[-1]["weight"] + row["weight"]
            merged[-1]["mz_da"] = (
                merged[-1]["mz_da"] * merged[-1]["weight"] + row["mz_da"] * row["weight"]
            ) / total
            merged[-1]["weight"] = total
        else:
            merged.append({"mz_da": float(row["mz_da"]), "weight": float(row["weight"])})
    return pd.DataFrame(merged)


def load_pos_arrays(path: Path):
    if path.suffix.lower() == ".epos":
        raw = np.fromfile(path, dtype=">f4").reshape(-1, 11)
    else:
        raw = np.fromfile(path, dtype=">f4").reshape(-1, 4)
    return (
        raw[:, 0].astype(np.float32),
        raw[:, 1].astype(np.float32),
        raw[:, 2].astype(np.float32),
        raw[:, 3].astype(np.float32),
    )


def synthetic_datasets(
    controls_dir: Path,
    *,
    rng_seed: int = 20260720,
    materials: list[SyntheticMaterial] | None = None,
) -> list[BenchmarkDataset]:
    rng = np.random.default_rng(rng_seed)
    datasets: list[BenchmarkDataset] = []
    for material in materials if materials is not None else default_validation_materials():
        pos_path = controls_dir / "synthetic" / f"{material.name}.POS"
        x, y, z, mz = synthesize_material_events(material, rng=rng)
        write_pos_file(pos_path, x, y, z, mz)
        datasets.append(
            BenchmarkDataset(
                name=material.name,
                kind="synthetic",
                pos_path=pos_path,
                truth_at_pct={
                    element: 100.0 * fraction
                    for element, fraction in material.true_atomic_fractions().items()
                },
                truth_source="synthetic",
                truth_lines=truth_lines_for_material(material),
                mass_resolving_power=float(material.mass_resolving_power),
                arrays=(x, y, z, mz),
            )
        )
    return datasets


_RANGE_FILE_SUFFIXES = (".rrng", ".RRNG", ".rng.fig.txt")


def real_control_datasets(controls_dir: Path, nominal: dict[str, dict[str, float]]) -> list[BenchmarkDataset]:
    suffix_priority = {".pos": 0, ".epos": 1}
    search_dirs = [controls_dir]
    if (controls_dir / "extra").is_dir():
        search_dirs.append(controls_dir / "extra")
    paths = sorted(
        {
            path
            for directory in search_dirs
            for pattern in ("*.pos", "*.POS", "*.epos", "*.EPOS")
            for path in directory.glob(pattern)
        },
        key=lambda path: (path.stem, suffix_priority.get(path.suffix.lower(), 2)),
    )
    datasets: list[BenchmarkDataset] = []
    seen: set[str] = set()
    for path in paths:
        if path.stem in seen:
            continue
        seen.add(path.stem)
        ranges = None
        for suffix in _RANGE_FILE_SUFFIXES:
            candidate = path.with_suffix("") if suffix == ".rng.fig.txt" else path
            range_path = (
                candidate.parent / f"{path.stem}{suffix}"
                if suffix == ".rng.fig.txt"
                else path.with_suffix(suffix)
            )
            if range_path.exists():
                ranges = load_range_file(range_path)
                break
        truth: dict[str, float] = {}
        truth_source = ""
        nominal_entry = nominal.get(path.name) or nominal.get(path.stem)
        if isinstance(nominal_entry, dict) and nominal_entry:
            truth = {str(element): float(pct) for element, pct in nominal_entry.items()}
            truth_source = "nominal"
        datasets.append(
            BenchmarkDataset(
                name=path.stem,
                kind="real",
                pos_path=path,
                truth_at_pct=truth,
                truth_source=truth_source,
                ranges=ranges,
            )
        )
    return datasets


# Datasets whose shipped range file is documented as incomplete for
# composition (the APAV Si example unambiguously contains Ni silicide, but
# the shipped RRNG does not range Ni): use their ranges for detection
# scoring only, never as a composition truth.
RANGE_COMPOSITION_UNRELIABLE = {"control_Si_apav_usa_denton_smith"}


def attach_range_reference_truth(dataset: BenchmarkDataset, mz: np.ndarray) -> None:
    """For real controls without a nominal composition, derive the reference
    composition by applying the shipped expert ranges to the raw spectrum."""
    if dataset.truth_at_pct or dataset.ranges is None or dataset.ranges.empty:
        return
    if dataset.name in RANGE_COMPOSITION_UNRELIABLE:
        return
    dataset.truth_at_pct = range_reference_composition(dataset.ranges, mz)
    dataset.truth_source = "range-file"


def allowed_extra_elements(dataset: BenchmarkDataset) -> set[str]:
    """Elements the expert range file acknowledges beyond the nominal truth
    (e.g. Si/Ga/O in Ck10): counting them as 'fabricated' would penalize a
    method for finding real minor species."""
    if dataset.ranges is None or dataset.ranges.empty:
        return set()
    allowed: set[str] = set()
    for formula in dataset.ranges["formula"].tolist():
        allowed.update(str(element) for element in dict(formula))
    return allowed
