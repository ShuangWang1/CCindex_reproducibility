from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import plot_top30_motif_kaplan_meier as motif_km  # noqa: E402


class Top30MotifKaplanMeierTests(unittest.TestCase):
    def test_motif_name_uses_ordered_cell_type_names(self) -> None:
        observed = motif_km.motif_name(
            "4&5&6&4",
            {"4": "macrophages", "5": "stromal cells", "6": "tumor cells"},
        )
        self.assertEqual(
            observed,
            "macrophages&stromal cells&tumor cells&macrophages",
        )

    def test_duplicate_names_get_numeric_suffixes_in_rank_order(self) -> None:
        observed = motif_km.unique_plot_stems(["A&B&C", "D&E", "A&B&C", "A&B&C"])
        self.assertEqual(observed, ["A&B&C", "D&E", "A&B&C_2", "A&B&C_3"])

    def test_long_motif_name_wraps_at_cell_type_boundaries(self) -> None:
        name = "tumor cells&tumor cells&macrophages&stromal cells&tumor cells"
        observed = motif_km.wrap_motif_name(name, max_characters=35)
        self.assertEqual(observed.replace("\n", "&"), name)
        self.assertTrue(all(len(line) <= 35 for line in observed.splitlines()))

    def test_validate_reconstruction_accepts_strict_median_groups(self) -> None:
        counts = np.array([[0, 0], [0, 1], [2, 1], [4, 3]], dtype=float)
        medians = np.median(counts, axis=0)
        enriched = counts > medians
        event = np.array([1, 0, 1, 1], dtype=int)
        selected = pd.DataFrame(
            {
                "top_rank": [1, 2],
                "median_patient_raw_count": [1.0, 1.0],
                "n_enriched_unadjusted": [2, 1],
                "events_enriched_unadjusted": [2, 1],
                "n_non_enriched_unadjusted": [2, 3],
                "events_non_enriched_unadjusted": [1, 2],
            }
        )
        motif_km.validate_reconstruction(selected, medians, enriched, event)

    def test_plot_survival_writes_png(self) -> None:
        patients = pd.DataFrame(
            {
                "time_months": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
                "event": [1, 0, 1, 1, 0, 1],
            }
        )
        enriched = np.array([False, False, False, True, True, True])
        row = pd.Series(
            {
                "triangle_level": 2,
                "top_rank": 1,
                "wl_hash": "0123456789abcdef",
                "hazard_ratio_unadjusted": 1.5,
                "ci95_low_unadjusted": 0.7,
                "ci95_high_unadjusted": 3.2,
                "p_value_unadjusted": 0.25,
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "A&B&C.png"
            motif_km.plot_survival(patients, enriched, row, "A&B&C", output)
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)

    def test_threshold_summary_records_strict_median_split(self) -> None:
        row = pd.Series(
            {
                "triangle_level": 3,
                "top_rank": 2,
                "wl_hash": "abcdef",
            }
        )
        enriched = np.array([False, False, True, True])
        event = np.array([1, 0, 1, 0], dtype=int)
        observed = motif_km.threshold_summary(row, "A&B&C", 0.5, enriched, event)
        self.assertIn("enriched_definition: patient_raw_count > 0.5", observed)
        self.assertIn("non_enriched_definition: patient_raw_count <= 0.5", observed)
        self.assertIn("median_ties_assigned_to: non-enriched", observed)
        self.assertIn("enriched_patients: 2", observed)
        self.assertIn("non_enriched_patients: 2", observed)


if __name__ == "__main__":
    unittest.main()
