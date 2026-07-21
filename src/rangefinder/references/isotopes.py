from __future__ import annotations

from functools import lru_cache

import pandas as pd
from ase.data import chemical_symbols
from ifes_apt_tc_data_modeling.utils.nist_isotope_data import isotopes as nist_isotopes


@lru_cache(maxsize=1)
def isotope_table() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for atomic_number, isotope_map in nist_isotopes.items():
        if atomic_number >= len(chemical_symbols):
            continue
        symbol = chemical_symbols[atomic_number]
        if not symbol:
            continue
        for mass_number, isotope_info in isotope_map.items():
            abundance = float(isotope_info["composition"])
            if abundance <= 0.0:
                continue
            rows.append(
                {
                    "atomic_number": int(atomic_number),
                    "symbol": symbol,
                    "mass_number": int(mass_number),
                    "isotope_label": f"{mass_number}{symbol}",
                    "mass_da": float(isotope_info["mass"]),
                    "natural_abundance": abundance,
                }
            )
    frame = pd.DataFrame(rows)
    return frame.sort_values(
        ["atomic_number", "mass_number"], ignore_index=True
    )


def natural_abundance_by_element() -> dict[str, pd.DataFrame]:
    table = isotope_table()
    return {
        symbol: group.reset_index(drop=True)
        for symbol, group in table.groupby("symbol", sort=True)
    }
