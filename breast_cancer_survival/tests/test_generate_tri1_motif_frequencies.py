from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import networkx as nx
import numpy as np


SURVIVAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVIVAL_DIR))

import generate_tri1_motif_frequencies as tri1  # noqa: E402


class Tri1FrequencyTests(unittest.TestCase):
    def test_counts_each_delaunay_simplex_once(self) -> None:
        coordinates = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        labels = np.array(["6", "6", "6", "6"])
        frequencies, simplex_count = tri1.delaunay_triangle_frequencies(
            coordinates, labels
        )
        self.assertEqual(simplex_count, 2)
        self.assertEqual(frequencies, {("6", "6", "6"): 2})

    def test_sorts_labels_numerically_within_motif(self) -> None:
        coordinates = np.array([[0, 0], [2, 0], [0, 2]], dtype=float)
        labels = np.array(["6", "2", "5"])
        frequencies, simplex_count = tri1.delaunay_triangle_frequencies(
            coordinates, labels
        )
        self.assertEqual(simplex_count, 1)
        self.assertEqual(frequencies, {("2", "5", "6"): 1})

    def test_collinear_sample_has_no_triangles(self) -> None:
        coordinates = np.array([[0, 0], [1, 0], [2, 0]], dtype=float)
        frequencies, simplex_count = tri1.delaunay_triangle_frequencies(
            coordinates, np.array(["1", "2", "3"])
        )
        self.assertEqual(frequencies, {})
        self.assertEqual(simplex_count, 0)

    def test_triangle_hash_matches_labeled_complete_graph(self) -> None:
        labels = ("2", "5", "6")
        graph = nx.complete_graph(3)
        nx.set_node_attributes(graph, dict(enumerate(labels)), "label")
        expected = nx.weisfeiler_lehman_graph_hash(graph, node_attr="label")
        self.assertEqual(tri1.triangle_hash(labels), expected)

    def test_result_csv_uses_existing_motif_schema(self) -> None:
        rows = tri1.result_rows({("2", "5", "6"): 7}, elapsed_seconds=0.25)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ccindex_tri_1.csv"
            tri1.write_result(path, rows)
            with path.open(encoding="utf-8", newline="") as source:
                observed = list(csv.DictReader(source))
            self.assertEqual(tuple(observed[0]), tri1.RESULT_FIELDS)
            self.assertEqual(observed[0]["pattern_id"], "tri_1_1")
            self.assertEqual(observed[0]["triangles"], "1")
            self.assertEqual(observed[0]["edge_list"], "0-1;0-2;1-2")
            self.assertEqual(observed[0]["matches"], "7")


if __name__ == "__main__":
    unittest.main()
