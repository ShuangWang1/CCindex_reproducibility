from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cluster_significant_motif_trajectories as cluster  # noqa: E402


class SignificantMotifTrajectoryTests(unittest.TestCase):
    def test_trend_labels(self) -> None:
        self.assertEqual(cluster.trend_label(np.arange(7)), "increasing_significance")
        self.assertEqual(cluster.trend_label(np.arange(7)[::-1]), "decreasing_significance")
        self.assertEqual(
            cluster.trend_label(np.array([0, 1, 2, 3, 2, 1, 0])),
            "increasing_then_decreasing",
        )
        self.assertEqual(
            cluster.trend_label(np.array([3, 2, 1, 0, 1, 2, 3])),
            "decreasing_then_increasing",
        )
        self.assertEqual(cluster.trend_label(np.ones(7)), "flat")

    def test_clustering_is_deterministic(self) -> None:
        matrix = np.vstack(
            [
                np.arange(4),
                np.arange(4) + 0.1,
                np.arange(4)[::-1],
                np.arange(4)[::-1] + 0.1,
            ]
        )
        first = cluster.choose_clustering(matrix, 2)
        second = cluster.choose_clustering(matrix, 2)
        np.testing.assert_array_equal(first[0], second[0])
        self.assertEqual(first[1:], second[1:])


if __name__ == "__main__":
    unittest.main()
