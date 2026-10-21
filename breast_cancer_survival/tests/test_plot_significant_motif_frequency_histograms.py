from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SURVIVAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVIVAL_DIR))

import plot_significant_motif_frequency_histograms as frequency  # noqa: E402


class SignificantMotifFrequencyHistogramTests(unittest.TestCase):
    def setUp(self) -> None:
        self.patients = pd.DataFrame({"PID": [3, 1, 4, 2]})
        self.counts = np.array([5.0, 0.0, 2.0, 2.0])
        self.median = 2.0

    def test_orders_counts_and_keeps_median_ties_non_enriched(self) -> None:
        observed = frequency.ordered_frequency_frame(
            self.patients, self.counts, self.median
        )
        self.assertEqual(observed["motif_count"].tolist(), [0.0, 2.0, 2.0, 5.0])
        self.assertEqual(
            observed["enrichment_group"].tolist(),
            ["Non-enriched", "Non-enriched", "Non-enriched", "Enriched"],
        )

    def test_validates_saved_median_and_group_sizes(self) -> None:
        ordered = frequency.ordered_frequency_frame(
            self.patients, self.counts, self.median
        )
        row = pd.Series(
            {
                "triangle_level": 2,
                "wl_hash": "abc",
                "median_patient_raw_count": 2.0,
                "n_enriched_unadjusted": 1,
                "n_non_enriched_unadjusted": 3,
            }
        )
        frequency.validate_split(row, ordered, self.median)
        with self.assertRaisesRegex(ValueError, "median mismatch"):
            frequency.validate_split(row, ordered, 2.5)

    def test_plot_smoke(self) -> None:
        ordered = frequency.ordered_frequency_frame(
            self.patients, self.counts, self.median
        )
        row = pd.Series(
            {
                "triangle_level": 2,
                "wl_hash": "abcdef123456",
                "cell_type_composition": "T cells; tumor cells",
                "median_patient_raw_count": 2.0,
                "p_value_unadjusted": 0.01,
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "plot.png"
            frequency.plot_frequency_histogram(ordered, row, output, 80)
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
