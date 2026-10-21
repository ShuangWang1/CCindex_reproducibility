from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import tree_search_topology_restrict as search

import igraph as ig
import pandas as pd


ONE_LABEL_POOL = {"0&0": ["0&0&0"]}


class TopologyRestrictedSearchTests(unittest.TestCase):
    def seed(self) -> ig.Graph:
        return search.construct_queue({"0&0&0": [100]})[0]

    def levels(self, maximum: int) -> list[list[ig.Graph]]:
        levels = [[self.seed()]]
        for _ in range(2, maximum + 1):
            levels.append(search.expand_level(levels[-1], ONE_LABEL_POOL))
        return levels

    def test_default_max_depth_is_tri7(self) -> None:
        self.assertEqual(7, search.parse_args([]).MaxDepth)

    def test_exact_restricted_topology_counts_through_tri7(self) -> None:
        levels = self.levels(7)
        self.assertEqual([1, 1, 1, 1, 2, 8, 12], [len(level) for level in levels])
        self.assertEqual(
            [1, 1, 1, 1, 2, 8, 12],
            [len(search.group_unrooted_states(level)[0]) for level in levels],
        )

    def test_nearer_shell_is_exhausted_before_hop_two(self) -> None:
        levels = self.levels(4)
        for parent in levels[1:3]:
            candidates = [
                candidate
                for graph in parent
                for candidate in search.growth_candidates(graph, ONE_LABEL_POOL)
            ]
            self.assertTrue(candidates)
            self.assertEqual({1}, {candidate.hop for candidate in candidates})

        candidates = search.growth_candidates(levels[3][0], ONE_LABEL_POOL)
        self.assertTrue(candidates)
        self.assertEqual({2}, {candidate.hop for candidate in candidates})

    def test_multi_digit_labels_do_not_match_by_substring(self) -> None:
        frame = pd.DataFrame({"A&A&A": [20], "B&B&B": [20]})
        indexed, edge_pool, _ = search.build_search_vocabulary(frame, 10)
        self.assertEqual({"0&0&0", "1&1&1"}, set(indexed))
        self.assertEqual(["0&0&0"], edge_pool["0&0"])
        self.assertEqual(["1&1&1"], edge_pool["1&1"])

    def test_rooted_queue_round_trip(self) -> None:
        graph = self.levels(5)[-1][0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = search.write_queue([graph], 5, root)
            self.assertIn(graph["rootedhash"], paths)
            restored = search.read_queue(root, 5)
            self.assertEqual(1, len(restored))
            self.assertEqual((0, 1, 2), search.parse_root_nodes(restored[0]))
            self.assertEqual(graph["rootedhash"], restored[0]["rootedhash"])
            self.assertEqual(
                set(graph["extendableedge"]), set(restored[0]["extendableedge"])
            )

    def test_match_count_propagates_to_rooted_state(self) -> None:
        graph = self.levels(2)[-1][0]

        def fake_matcher(target, patterns, *, threshold, result_path, workers):
            self.assertEqual(0, threshold)
            patterns[0]["count"] = 17
            patterns[0]["time"] = 0.25
            search._write_match_rows(result_path, [["2_0", 17, 0.25]])
            return [["2_0", 17, 0.25]]

        with tempfile.TemporaryDirectory() as directory, patch.object(
            search, "_ccindex_match_patterns", side_effect=fake_matcher
        ):
            result = Path(directory) / "tri2.csv"
            survivors, matched = search.match_and_prune(
                self.seed(),
                [graph],
                method="ccindex",
                threshold=10,
                stop_early=False,
                result_path=result,
            )
            self.assertEqual(1, matched)
            self.assertEqual(1, len(survivors))
            self.assertEqual(17, survivors[0]["count"])
            with result.open(newline="") as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(["Pattern", "Matches", "Time"], rows[0])

    def test_diverse_selection_and_visualization(self) -> None:
        states = self.levels(7)[-1]
        for index, graph in enumerate(states):
            graph["count"] = index + 10
            graph["time"] = 0.01
        chosen = search.select_diverse_examples(states, 6)
        self.assertEqual(6, len(chosen))
        self.assertEqual(6, len({search.exact_topology_key(graph) for graph in chosen}))

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            rows = search.visualize_examples(
                states,
                {"0": "cell"},
                output,
                sample_count=2,
                dpi=50,
                queue_paths={},
            )
            self.assertEqual(2, len(rows))
            self.assertTrue((output / "tri7_example_01.png").is_file())
            self.assertTrue((output / "tri7_example_02.png").is_file())
            self.assertTrue((output / "tri7_examples_contact_sheet.png").is_file())
            self.assertTrue((output / "tri7_examples.csv").is_file())

    def test_multitype_selection_prioritizes_most_distinct_labels(self) -> None:
        states = self.levels(7)[-1][:3]
        distinct_counts = (1, 2, 3)
        for graph, distinct_count in zip(states, distinct_counts):
            graph.vs["label"] = [
                str(index % distinct_count) for index in range(graph.vcount())
            ]
            graph["hashvalue"] = search.motif_hash(graph)
        chosen = search.select_multitype_examples(states, 2)
        self.assertEqual([3, 2], [len(set(graph.vs["label"])) for graph in chosen])


if __name__ == "__main__":
    unittest.main()
