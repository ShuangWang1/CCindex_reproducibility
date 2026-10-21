from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
import os
import pytest

import networkx as nx
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("sibling_topology_pipeline", PROJECT / "run_analysis.py")
pipeline = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = pipeline
SPEC.loader.exec_module(pipeline)


def labeled_graph(edges: list[tuple[int, int]], labels: list[str]) -> nx.Graph:
    graph = nx.Graph(edges)
    nx.set_node_attributes(graph, {index: label for index, label in enumerate(labels)}, "label")
    return graph


class SiblingTopologyPipelineTests(unittest.TestCase):
    def test_triangle_features(self) -> None:
        graph = labeled_graph([(0, 1), (0, 2), (1, 2)], ["2", "4", "6"])
        observed = pipeline.graph_features(graph)
        self.assertEqual(observed["graph_nodes"], 3)
        self.assertEqual(observed["triangle_fraction_nodes_participating"], 1)
        self.assertEqual(observed["edge_2_4_count"], 1)
        self.assertEqual(observed["labeled_triangle_2_4_6_count"], 1)
        self.assertEqual(observed["labeled_triangle_mean_distinct_labels"], 3)

    def test_overlapping_triangle_features(self) -> None:
        graph = labeled_graph(
            [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3)],
            ["2", "2", "4", "6"],
        )
        observed = pipeline.graph_features(graph)
        self.assertEqual(observed["triangle_pairs_share_edge"], 1)
        self.assertEqual(observed["triangle_largest_edge_overlap_component"], 2)
        self.assertEqual(observed["clique_max_size"], 3)

    def test_features_are_invariant_to_node_ids(self) -> None:
        graph = labeled_graph([(0, 1), (0, 2), (1, 2), (0, 3), (1, 3)], ["2", "4", "6", "5"])
        remapped = nx.relabel_nodes(graph, {0: 9, 1: 4, 2: 7, 3: 2})
        left = pipeline.graph_features(graph)
        right = pipeline.graph_features(remapped)
        for key in left:
            if np.isnan(left[key]) and np.isnan(right[key]):
                continue
            self.assertAlmostEqual(left[key], right[key], msg=key)

    def test_bh_and_strict_significance_boundary(self) -> None:
        values = pd.Series([0.01, 0.04, 0.20])
        np.testing.assert_allclose(pipeline.benjamini_hochberg(values), [0.03, 0.06, 0.20])
        frame = pd.DataFrame(
            {
                "fit_status_unadjusted": ["estimated"] * 3,
                "p_value_unadjusted": [0.049, 0.05, np.nan],
            }
        )
        eligible = frame.fit_status_unadjusted.eq("estimated") & frame.p_value_unadjusted.notna()
        significant = eligible & frame.p_value_unadjusted.lt(0.05)
        self.assertEqual(significant.tolist(), [True, False, False])

    def test_association_effect_direction(self) -> None:
        frame = pd.DataFrame(
            {
                "metric": np.r_[np.arange(20, 40), np.arange(20)],
                "significant_raw_p": [True] * 20 + [False] * 20,
            }
        )
        result, excluded = pipeline.association_for_metric(
            frame, "metric", "test", False, min_class_size=20, min_prevalence=0.01
        )
        self.assertIsNone(excluded)
        assert result is not None
        self.assertEqual(result["effect_direction"], "higher_in_significant")
        self.assertAlmostEqual(result["effect_size"], 1.0)

    def test_sparse_binary_metric_is_excluded(self) -> None:
        frame = pd.DataFrame(
            {
                "metric": [1.0] + [0.0] * 39,
                "significant_raw_p": [True] * 20 + [False] * 20,
            }
        )
        result, excluded = pipeline.association_for_metric(
            frame, "metric", "test", True, min_class_size=20, min_prevalence=0.05
        )
        self.assertIsNone(result)
        assert excluded is not None
        self.assertEqual(excluded["reason"], "sparse_pattern")

    def test_cross_level_consistency_rule(self) -> None:
        primary = pd.DataFrame(
            {
                "metric": ["metric_a"],
                "display_name": ["metric a"],
                "exploratory_candidate": [True],
                "is_representative": [True],
            }
        )
        associations = pd.DataFrame(
            {
                "triangle_level": [4, 5, 6, 7],
                "metric": ["metric_a"] * 4,
                "effect_size": [-0.1, -0.2, -0.3, 0.1],
                "topology_assoc_q_bh": [0.2, 0.01, 0.02, 0.2],
            }
        )
        observed = pipeline.cross_level_consistency(primary, associations, 5).iloc[0]
        self.assertEqual(observed["secondary_levels_same_direction"], 2)
        self.assertFalse(observed["significant_reverse_secondary"])
        self.assertTrue(observed["cross_level_consistent"])

    @pytest.mark.integration
    @unittest.skipUnless(os.environ.get("CCINDEX_RUN_INTEGRATION") == "1", "Requires saved study artifacts and CCINDEX_RUN_INTEGRATION=1")
    def test_generated_results_match_known_tri5_audit(self) -> None:
        path = PROJECT / "results" / "tables" / "level_summary.csv"
        if not path.is_file():
            self.skipTest("optional generated full-analysis results are unavailable")
        row = pd.read_csv(path).set_index("triangle_level").loc[5]
        self.assertEqual(row["total_lineage_motifs"], 14332)
        self.assertEqual(row["estimable_motifs"], 6099)
        self.assertEqual(row["significant_raw_p"], 472)
        self.assertEqual(row["nonsignificant_raw_p"], 5627)


if __name__ == "__main__":
    unittest.main()
