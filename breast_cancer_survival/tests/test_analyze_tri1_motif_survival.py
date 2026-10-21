from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze_tri1_motif_survival as tri1  # noqa: E402


class Tri1MotifSurvivalTests(unittest.TestCase):
    def test_analysis_uses_strict_median_and_existing_result_schema(self) -> None:
        patients = pd.DataFrame(
            {
                "time_months": [2, 3, 5, 6, 8, 9, 12, 15],
                "event": [1, 1, 0, 1, 0, 1, 0, 1],
            }
        )
        features = pd.DataFrame(
            {
                "triangle_level": [1],
                "wl_hash": ["abc"],
                "triangles": [1],
                "nodes": [3],
                "edges": [3],
                "representative_node_labels": ["4&5&6"],
                "representative_edge_list": ["0-1;0-2;1-2"],
                "cell_type_composition": ["macrophages; stromal cells; tumor cells"],
                "sample_document_frequency": [4],
            }
        )
        counts = np.array([[0], [0], [2], [2], [3], [4], [5], [9]], dtype=float)
        observed = tri1.analyze_tri1(features, counts, patients).iloc[0]
        self.assertEqual(observed.triangle_level, 1)
        self.assertEqual(observed.median_patient_raw_count, 2.5)
        self.assertEqual(observed.n_enriched_unadjusted, 4)
        self.assertEqual(observed.n_non_enriched_unadjusted, 4)
        self.assertIn("p_value_unadjusted", observed.index)
        self.assertIn("q_value_bh_within_level_unadjusted", observed.index)
        self.assertEqual(list(tri1.analyze_tri1(features, counts, patients)), tri1.OUTPUT_COLUMNS)


if __name__ == "__main__":
    unittest.main()
