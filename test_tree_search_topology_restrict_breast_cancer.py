from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import igraph as ig
import numpy as np

import tree_search_topology_restrict_breast_cancer as search


class BreastCancerTopologyRestrictTests(unittest.TestCase):
    def test_strict_majority_cutoff(self) -> None:
        self.assertEqual(280, search.strict_support_cutoff(559, 0.5))
        self.assertEqual(3, search.strict_support_cutoff(4, 0.5))
        self.assertEqual(1, search.strict_support_cutoff(4, 0.0))

    def test_edge_pool_handles_repeated_labels(self) -> None:
        pool = search.build_edge_pool(["0&0&1", "0&1&2"], ["0", "1", "2"])
        self.assertEqual(["0&0&1"], pool["0&0"])
        self.assertEqual(["0&0&1", "0&1&2"], pool["0&1"])
        self.assertNotIn("1&1", pool)

    def test_exact_matcher_counts_distinct_node_sets(self) -> None:
        target = ig.Graph.Full(4)
        target.vs["label"] = [0, 0, 0, 0]
        pattern = search.triangle_graph("0&0&0")
        spec = search.pattern_spec(pattern)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample1.gml"
            target.write_gml(str(path))
            sample = search.SampleInfo(1, "sample1", "core1", path, 4, 6)
            search._init_match_worker([spec])
            sample_index, counts, seconds = search._exact_count_worker(sample)
        self.assertEqual(1, sample_index)
        self.assertEqual([4], counts.tolist())
        self.assertGreaterEqual(seconds[0], 0.0)

    def test_count_checkpoint_round_trip_and_digest_guard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample1.npz"
            counts = np.asarray([0, 3, 8], dtype=np.int64)
            seconds = np.asarray([0.1, 0.2, 0.3], dtype=np.float64)
            search._write_count_checkpoint(path, "correct", counts, seconds)
            restored = search._read_count_checkpoint(path, "correct", 3)
            self.assertIsNotNone(restored)
            assert restored is not None
            np.testing.assert_array_equal(counts, restored[0])
            np.testing.assert_allclose(seconds, restored[1])
            self.assertIsNone(search._read_count_checkpoint(path, "wrong", 3))
            self.assertIsNone(search._read_count_checkpoint(path, "correct", 2))

    def test_candidate_digest_ignores_equivalent_vertex_numbering(self) -> None:
        left = search.PatternSpec("same-hash", ("0", "1", "1"), ((0, 1), (1, 2)))
        right = search.PatternSpec("same-hash", ("1", "0", "1"), ((0, 2), (1, 2)))
        self.assertEqual(search.candidate_digest([left]), search.candidate_digest([right]))

    def test_presence_checkpoint_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample1.npz"
            indices = np.asarray([1, 4, 7], dtype=np.int32)
            present = np.asarray([True, False, True], dtype=np.bool_)
            seconds = np.asarray([0.01, 0.02, 0.03], dtype=np.float64)
            search._write_presence_checkpoint(
                path, "digest", indices, present, seconds
            )
            restored = search._read_presence_checkpoint(path, "digest", 8)
            self.assertIsNotNone(restored)
            assert restored is not None
            np.testing.assert_array_equal(indices, restored[0])
            np.testing.assert_array_equal(present, restored[1])
            np.testing.assert_allclose(seconds, restored[2])
            self.assertIsNone(search._read_presence_checkpoint(path, "wrong", 8))

    def test_prevalence_marks_rejected_exact_totals_incomplete(self) -> None:
        representatives = [
            search.triangle_graph("0&0&0"),
            search.triangle_graph("0&0&1"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prevalence.csv"
            search.write_prevalence(
                path,
                representatives,
                1,
                np.asarray([280, 19]),
                np.asarray([900, -1]),
                np.asarray([1.0, 0.5]),
                559,
                280,
                np.asarray([True, False]),
            )
            with path.open(newline="") as source:
                rows = list(csv.DictReader(source))
        self.assertEqual("900", rows[0]["total_matches"])
        self.assertEqual("True", rows[0]["sample_support_exact"])
        self.assertEqual("", rows[1]["total_matches"])
        self.assertEqual("False", rows[1]["sample_support_exact"])

    def test_sample_csvs_share_ids_and_preserve_zero_counts(self) -> None:
        representatives = [
            search.triangle_graph("0&0&0"),
            search.triangle_graph("0&0&1"),
        ]
        samples = [
            search.SampleInfo(1, "sample1", "core1", Path("unused1"), 3, 3),
            search.SampleInfo(2, "sample2", "core2", Path("unused2"), 3, 3),
        ]
        tri1_rows = [
            {"counts": {"0&0&1": 5}, "seconds": 0.1},
            {"counts": {}, "seconds": 0.2},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            search.write_sample_level_csvs(
                samples,
                representatives,
                retained_indices=[1],
                level=1,
                output_root=root,
                digest=None,
                tri1_rows=tri1_rows,
            )
            with (root / "sample1/ccindex_tri_1.csv").open(newline="") as source:
                first = list(csv.DictReader(source))
            with (root / "sample2/ccindex_tri_1.csv").open(newline="") as source:
                second = list(csv.DictReader(source))
        self.assertEqual("tri_1_2", first[0]["pattern_id"])
        self.assertEqual(first[0]["pattern_id"], second[0]["pattern_id"])
        self.assertEqual("5", first[0]["matches"])
        self.assertEqual("0", second[0]["matches"])

    def test_triangle_seed_round_trips_as_rooted_queue(self) -> None:
        graph = search.triangle_graph("0&1&2")
        graph["count"] = 280
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            search.write_queue_idempotent([graph], 1, root)
            restored = search.read_queue(root, 1)
            self.assertEqual(1, len(restored))
            self.assertEqual(graph["rootedhash"], restored[0]["rootedhash"])


if __name__ == "__main__":
    unittest.main()
