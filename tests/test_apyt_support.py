from __future__ import annotations

import unittest

from rangefinder.analysis.apyt_support import (
    apyt_species_tables_from_counts,
    build_apyt_species_dictionary,
    charged_formula_label,
    formula_category,
    parse_formula,
)

import pandas as pd


class APyTSupportTests(unittest.TestCase):
    def test_parse_formula_handles_atomic_and_molecular_cases(self) -> None:
        self.assertEqual(parse_formula("Si"), {"Si": 1})
        self.assertEqual(parse_formula("Al2O3"), {"Al": 2, "O": 3})
        self.assertEqual(parse_formula("Gx2O"), {"Gx": 2, "O": 1})

    def test_formula_category_and_charge_label(self) -> None:
        self.assertEqual(formula_category("O"), "atomic")
        self.assertEqual(formula_category("Si2"), "molecular")
        self.assertEqual(charged_formula_label("AlO", 1), "AlO+")
        self.assertEqual(charged_formula_label("AlO", 2), "AlO2+")

    def test_species_dictionary_groups_charges_by_formula(self) -> None:
        selection = pd.DataFrame(
            [
                {"formula": "Si", "charge": 1},
                {"formula": "Si", "charge": 2},
                {"formula": "O2", "charge": 1},
            ]
        )
        species = build_apyt_species_dictionary(selection, atomic_volume_nm3=0.015)
        self.assertEqual(species["Si"], ((1, 2), 0.015))
        self.assertEqual(species["O2"], ((1,), 0.015))

    def test_species_tables_expand_molecules_into_element_totals(self) -> None:
        counts_list = [
            {"element": "Si", "charge": 2, "count": 80.0, "fraction": 0.4},
            {"element": "O2", "charge": 1, "count": 60.0, "fraction": 0.3},
            {"element": "SiO", "charge": 1, "count": 60.0, "fraction": 0.3},
        ]
        species, elements = apyt_species_tables_from_counts(counts_list)
        self.assertEqual(set(species["species_label"]), {"Si2+", "O2+", "SiO+"})
        element_percent = elements.set_index("element")["atomic_percent_weighted"].to_dict()
        self.assertAlmostEqual(sum(element_percent.values()), 100.0, places=6)
        self.assertGreater(element_percent["O"], element_percent["Si"])


if __name__ == "__main__":
    unittest.main()
