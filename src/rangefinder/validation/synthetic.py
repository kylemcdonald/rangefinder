from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

import numpy as np

from rangefinder.references.isotopes import isotope_table


@dataclass(frozen=True)
class SyntheticSpecies:
    """One ionic species emitted by the synthetic specimen."""

    formula: dict[str, int]
    charge: int
    ion_fraction: float


@dataclass(frozen=True)
class SyntheticZone:
    """A depth band of the specimen with its own species mix.

    z fractions are relative to the 0-100 nm synthetic depth; signal_fraction
    is the share of all signal ions emitted by this zone.
    """

    z_min_frac: float
    z_max_frac: float
    signal_fraction: float
    species: list[SyntheticSpecies]


@dataclass(frozen=True)
class SyntheticMaterial:
    """A material with known ground-truth composition for pipeline validation.

    Homogeneous materials define `species`; layered materials define `zones`
    instead, each zone carrying its own species mix and depth band.
    """

    name: str
    species: list[SyntheticSpecies] = field(default_factory=list)
    zones: list[SyntheticZone] = field(default_factory=list)
    total_events: int = 1_500_000
    background_fraction: float = 0.02
    mass_resolving_power: float = 800.0
    tail_fraction: float = 0.35
    tail_tau_sigma_scale: float = 2.0
    max_background_mz: float = 236.0
    extra: dict[str, object] = field(default_factory=dict)

    def _species_weighted(self) -> list[tuple[SyntheticSpecies, float]]:
        if self.zones:
            weighted = []
            zone_total = sum(zone.signal_fraction for zone in self.zones)
            for zone in self.zones:
                zone_species_total = sum(s.ion_fraction for s in zone.species)
                for species in zone.species:
                    weighted.append(
                        (species, (zone.signal_fraction / zone_total) * (species.ion_fraction / zone_species_total))
                    )
            return weighted
        total = sum(s.ion_fraction for s in self.species)
        return [(species, species.ion_fraction / total) for species in self.species]

    def true_atomic_fractions(self) -> dict[str, float]:
        totals: Counter[str] = Counter()
        for species, weight in self._species_weighted():
            for element, count in species.formula.items():
                totals[element] += weight * count
        denom = sum(totals.values())
        return {element: value / denom for element, value in totals.items()}

    def zone_atomic_fractions(self) -> list[dict[str, float]]:
        fractions: list[dict[str, float]] = []
        for zone in self.zones:
            totals: Counter[str] = Counter()
            zone_species_total = sum(s.ion_fraction for s in zone.species)
            for species in zone.species:
                for element, count in species.formula.items():
                    totals[element] += (species.ion_fraction / zone_species_total) * count
            denom = sum(totals.values())
            fractions.append({element: value / denom for element, value in totals.items()})
        return fractions


def _isotopologues(formula: dict[str, int], *, min_probability: float = 1.0e-5) -> list[tuple[float, float]]:
    """Expand a neutral formula into (mass, probability) isotopologues."""
    table = isotope_table()
    per_atom: list[list[tuple[float, float]]] = []
    for element, count in formula.items():
        family = table[table["symbol"] == element]
        if family.empty:
            raise ValueError(f"Unknown element {element}")
        options = [
            (float(row["mass_da"]), float(row["natural_abundance"]))
            for _, row in family.iterrows()
        ]
        for _ in range(int(count)):
            per_atom.append(options)
    merged: Counter[float] = Counter()
    for combo in product(*per_atom):
        mass = sum(item[0] for item in combo)
        probability = float(np.prod([item[1] for item in combo]))
        merged[round(mass, 6)] += probability
    return [
        (mass, probability)
        for mass, probability in sorted(merged.items())
        if probability >= min_probability
    ]


def _species_mz_events(
    species: SyntheticSpecies,
    n_ions: int,
    material: SyntheticMaterial,
    rng: np.random.Generator,
) -> np.ndarray:
    isotopologues = _isotopologues(species.formula)
    masses = np.asarray([item[0] for item in isotopologues], dtype=np.float64)
    probabilities = np.asarray([item[1] for item in isotopologues], dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    iso_counts = rng.multinomial(int(n_ions), probabilities)
    chunks: list[np.ndarray] = []
    for mass, iso_n in zip(masses, iso_counts, strict=True):
        if iso_n <= 0:
            continue
        mz0 = mass / float(species.charge)
        t0 = np.sqrt(mz0)
        # m/dm (FWHM) = MRP implies dt_FWHM = t0 / (2 MRP) in t = sqrt(m/z).
        sigma_t = t0 / (2.0 * material.mass_resolving_power) / 2.354820045
        t = t0 + rng.normal(0.0, sigma_t, size=int(iso_n))
        tail_mask = rng.random(int(iso_n)) < material.tail_fraction
        n_tail = int(tail_mask.sum())
        if n_tail:
            t[tail_mask] += rng.exponential(material.tail_tau_sigma_scale * sigma_t, size=n_tail)
        chunks.append(np.square(t))
    if not chunks:
        return np.asarray([], dtype=np.float64)
    return np.concatenate(chunks)


def synthesize_material_events(
    material: SyntheticMaterial,
    *,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate (x_nm, y_nm, z_nm, m_over_z_da) event arrays with realistic
    normalized-TOF peak widths and one-sided thermal tails. Layered materials
    (zones) emit each zone's species mix within its own depth band."""
    n_background = int(round(material.total_events * material.background_fraction))
    n_signal = material.total_events - n_background
    if material.zones:
        groups = [
            (species, weight, 100.0 * zone.z_min_frac, 100.0 * zone.z_max_frac)
            for zone in material.zones
            for species, weight in (
                lambda zone=zone: [
                    (
                        s,
                        (zone.signal_fraction / sum(zz.signal_fraction for zz in material.zones))
                        * (s.ion_fraction / sum(sp.ion_fraction for sp in zone.species)),
                    )
                    for s in zone.species
                ]
            )()
        ]
    else:
        total = sum(s.ion_fraction for s in material.species)
        groups = [(s, s.ion_fraction / total, 0.0, 100.0) for s in material.species]
    weights = np.asarray([weight for _s, weight, _lo, _hi in groups], dtype=np.float64)
    weights = weights / weights.sum()
    counts = rng.multinomial(n_signal, weights)
    mz_chunks: list[np.ndarray] = []
    z_chunks: list[np.ndarray] = []
    for (species, _weight, z_lo, z_hi), n_ions in zip(groups, counts, strict=True):
        if n_ions <= 0:
            continue
        mz_values = _species_mz_events(species, int(n_ions), material, rng)
        mz_chunks.append(mz_values)
        z_chunks.append(rng.uniform(z_lo, z_hi, size=mz_values.size))
    if n_background > 0:
        t_lo = np.sqrt(0.5)
        t_hi = np.sqrt(material.max_background_mz)
        t_bg = rng.uniform(t_lo, t_hi, size=n_background)
        mz_chunks.append(np.square(t_bg))
        z_chunks.append(rng.uniform(0.0, 100.0, size=n_background))
    m_over_z = np.concatenate(mz_chunks)
    z_nm = np.concatenate(z_chunks)
    order = rng.permutation(m_over_z.size)
    m_over_z = m_over_z[order]
    z_nm = z_nm[order]
    n_events = m_over_z.size
    radius = 30.0 * np.sqrt(rng.random(n_events))
    angle = rng.uniform(0.0, 2.0 * np.pi, size=n_events)
    x_nm = radius * np.cos(angle)
    y_nm = radius * np.sin(angle)
    return (
        x_nm.astype(np.float32),
        y_nm.astype(np.float32),
        z_nm.astype(np.float32),
        m_over_z.astype(np.float32),
    )


def write_pos_file(
    path: Path,
    x_nm: np.ndarray,
    y_nm: np.ndarray,
    z_nm: np.ndarray,
    m_over_z_da: np.ndarray,
) -> None:
    stacked = np.column_stack(
        [
            np.asarray(x_nm, dtype=np.float32),
            np.asarray(y_nm, dtype=np.float32),
            np.asarray(z_nm, dtype=np.float32),
            np.asarray(m_over_z_da, dtype=np.float32),
        ]
    ).astype(">f4")
    path.parent.mkdir(parents=True, exist_ok=True)
    stacked.tofile(path)


def default_validation_materials() -> list[SyntheticMaterial]:
    """Three ground-truth controls spanning the alloy classes seen in the report:
    an Al alloy, a Si/O sample, and a Zn/Cu/Al metallic sample."""
    return [
        SyntheticMaterial(
            name="synthetic_al_mg_si",
            species=[
                SyntheticSpecies({"Al": 1}, 2, 0.545),
                SyntheticSpecies({"Al": 1}, 1, 0.360),
                SyntheticSpecies({"Al": 1, "H": 1}, 1, 0.015),
                SyntheticSpecies({"Mg": 1}, 2, 0.022),
                SyntheticSpecies({"Mg": 1}, 1, 0.008),
                SyntheticSpecies({"Si": 1}, 2, 0.008),
                SyntheticSpecies({"Si": 1}, 1, 0.004),
                SyntheticSpecies({"H": 1}, 1, 0.038),
            ],
        ),
        SyntheticMaterial(
            name="synthetic_si_o",
            species=[
                SyntheticSpecies({"Si": 1}, 2, 0.33),
                SyntheticSpecies({"Si": 1}, 1, 0.27),
                SyntheticSpecies({"Si": 1, "O": 1}, 1, 0.12),
                SyntheticSpecies({"O": 1}, 1, 0.09),
                SyntheticSpecies({"O": 2}, 1, 0.06),
                SyntheticSpecies({"Si": 1, "O": 2}, 1, 0.05),
                SyntheticSpecies({"H": 1}, 1, 0.08),
            ],
        ),
        SyntheticMaterial(
            name="synthetic_zn_cu_al",
            species=[
                SyntheticSpecies({"Zn": 1}, 1, 0.30),
                SyntheticSpecies({"Zn": 1}, 2, 0.28),
                SyntheticSpecies({"Cu": 1}, 1, 0.17),
                SyntheticSpecies({"Cu": 1}, 2, 0.06),
                SyntheticSpecies({"Al": 1}, 1, 0.10),
                SyntheticSpecies({"Al": 1}, 2, 0.06),
                SyntheticSpecies({"H": 1}, 1, 0.03),
            ],
        ),
        # Layered control: a Zn-rich coating over an Al-rich substrate with a
        # sharp interface at 45 nm -- the geometry the segmentation stage must
        # detect, mirroring the layered Zn-Al specimen among the unknowns.
        SyntheticMaterial(
            name="synthetic_zn_layer_on_al",
            zones=[
                SyntheticZone(
                    0.0,
                    0.45,
                    0.45,
                    [
                        SyntheticSpecies({"Zn": 1}, 1, 0.50),
                        SyntheticSpecies({"Zn": 1}, 2, 0.44),
                        SyntheticSpecies({"Al": 1}, 1, 0.03),
                        SyntheticSpecies({"H": 1}, 1, 0.03),
                    ],
                ),
                SyntheticZone(
                    0.45,
                    1.0,
                    0.55,
                    [
                        SyntheticSpecies({"Al": 1}, 2, 0.55),
                        SyntheticSpecies({"Al": 1}, 1, 0.36),
                        SyntheticSpecies({"Zn": 1}, 1, 0.03),
                        SyntheticSpecies({"Mg": 1}, 2, 0.02),
                        SyntheticSpecies({"H": 1}, 1, 0.04),
                    ],
                ),
            ],
            extra={
                "expected_regions": 2,
                "expected_region_top_elements": ["Zn", "Al"],
            },
        ),
    ]
