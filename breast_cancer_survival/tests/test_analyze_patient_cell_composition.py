from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


SURVIVAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVIVAL_DIR))

import analyze_patient_cell_composition as patient_analysis  # noqa: E402


class PatientCellCompositionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cells = pd.DataFrame(
            {
                "PID": ["p2", "p1", "p1", "p1", "p2"],
                "core": ["c3", "c1", "c1", "c2", "c3"],
                "cell_type": ["T cells", "tumor cells", "T cells", "unknown", "T cells"],
            }
        )
        self.cell_types = ["T cells", "tumor cells", "unknown"]

    def test_patient_counts_and_fractions_reconcile(self) -> None:
        wide, long, cores = patient_analysis.summarize_patients(
            self.cells, self.cell_types
        )
        indexed = wide.set_index("PID")
        self.assertEqual(indexed.loc["p1", "total_cells"], 3)
        self.assertEqual(indexed.loc["p1", "number_of_cores"], 2)
        self.assertEqual(indexed.loc["p2", "T cells"], 2)
        self.assertEqual(int(wide["total_cells"].sum()), len(self.cells))
        self.assertEqual(len(cores), 3)
        fractions = long.groupby("PID")["cell_fraction"].sum()
        self.assertTrue(fractions.eq(1.0).all())

    def test_patient_order_uses_total_cells_then_pid(self) -> None:
        wide, _, _ = patient_analysis.summarize_patients(self.cells, self.cell_types)
        self.assertEqual(patient_analysis.patient_order(wide), ["p2", "p1"])

    def test_plots_smoke(self) -> None:
        wide, long, _ = patient_analysis.summarize_patients(
            self.cells, self.cell_types
        )
        patients = patient_analysis.patient_order(wide)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            composition = root / "composition.png"
            counts = root / "counts.png"
            patient_analysis.plot_composition(
                long, patients, self.cell_types, composition, 80
            )
            patient_analysis.plot_cell_counts(wide, patients, counts, 80)
            self.assertGreater(composition.stat().st_size, 0)
            self.assertGreater(counts.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
