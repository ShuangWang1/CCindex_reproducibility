from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import screen_all_rank_normalized_motifs as screen  # noqa: E402


class CompleteRankScreenTests(unittest.TestCase):
    def test_ranked_directory_name_sorts_by_within_level_p_value_rank(self) -> None:
        self.assertEqual(
            screen.ranked_motif_directory_name(7, "abcdef"), "007_abcdef"
        )
        self.assertEqual(
            screen.ranked_motif_directory_name(270, "abcdef"), "270_abcdef"
        )
        with self.assertRaises(ValueError):
            screen.ranked_motif_directory_name(0, "abcdef")

    def test_primary_screen_uses_rank_median_without_raw_p_filter(self) -> None:
        selected = pd.DataFrame(
            {
                "triangle_level": [2, 2],
                "wl_hash": ["a", "b"],
                "median_patient_raw_count": [1.0, 2.0],
                "p_value_unadjusted": [0.9, 0.8],
                "cell_type_composition": ["A", "B"],
            }
        )
        scores = np.array(
            [
                [0.1, 0.2],
                [0.2, 0.4],
                [0.3, 0.6],
                [0.4, 0.8],
                [0.5, 0.9],
                [0.6, 1.0],
            ]
        )
        result, groups = screen.primary_screen(
            selected,
            scores,
            np.arange(1, 7, dtype=float),
            np.array([1, 0, 1, 0, 1, 1]),
            vocabulary_size=10,
        )
        self.assertEqual(len(result), 2)
        np.testing.assert_array_equal(groups.sum(axis=0), [3, 3])
        self.assertTrue(result["median_patient_rank_score"].gt(0).all())
        self.assertEqual(result["p_value_unadjusted"].tolist(), [0.9, 0.8])

    def test_primary_screen_rejects_zero_rank_threshold(self) -> None:
        selected = pd.DataFrame(
            {
                "triangle_level": [2],
                "wl_hash": ["a"],
                "median_patient_raw_count": [1.0],
                "p_value_unadjusted": [0.9],
                "cell_type_composition": ["A"],
            }
        )
        with self.assertRaisesRegex(ValueError, "zero rank-score median"):
            screen.primary_screen(
                selected,
                np.zeros((6, 1)),
                np.arange(1, 7, dtype=float),
                np.array([1, 0, 1, 0, 1, 1]),
                vocabulary_size=10,
            )

    def test_delaunay_topology_plot_has_no_node_index_requirement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "motif.png"
            screen.plot_delaunay_topology(
                level=1,
                motif_hash="abcdef0123456789",
                composition="T cells; tumor cells x2",
                node_labels="2&6&6",
                edge_list="0-1;0-2;1-2",
                label_map={"2": "T cells", "6": "tumor cells"},
                output=output,
                dpi=90,
            )
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
