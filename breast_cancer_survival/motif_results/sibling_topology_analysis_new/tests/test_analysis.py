import importlib.util
import unittest
from pathlib import Path
import os
import pytest

import networkx as nx
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("new_topology_analysis", PROJECT / "run_analysis.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CriteriaTests(unittest.TestCase):
    def test_triangle_patch_boundary_and_composition(self):
        graph = nx.Graph()
        graph.add_nodes_from([(0, {"label": "2"}), (1, {"label": "2"}), (2, {"label": "4"})])
        graph.add_edges_from([(0, 1), (1, 2), (0, 2)])
        values, raw = MODULE.extra_features(graph, "test", 8, 1)
        self.assertEqual(values["size_triangles"], 1)
        self.assertEqual(values["boundary_edge_count"], 3)
        self.assertEqual(values["boundary_node_fraction"], 1)
        self.assertAlmostEqual(values["triangle_AAB_fraction"], 1)
        self.assertEqual(raw["degree"], [2.0, 2.0, 2.0])

    def test_label_permutation_is_deterministic(self):
        graph = nx.complete_graph(4)
        nx.set_node_attributes(graph, {0: "0", 1: "0", 2: "1", 3: "2"}, "label")
        first, _ = MODULE.extra_features(graph, "same", 16, 77)
        second, _ = MODULE.extra_features(graph, "same", 16, 77)
        keys = [key for key in first if key.endswith("permutation_z")]
        np.testing.assert_allclose([first[key] for key in keys], [second[key] for key in keys], equal_nan=True)

    def test_pca_display_jitter_preserves_stack_centroid(self):
        scores = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [2.0, 1.0]])
        displayed, points, stacks = MODULE.jitter_coincident_scores(scores, 9)
        self.assertEqual(points, 3)
        self.assertEqual(stacks, 1)
        np.testing.assert_allclose(displayed[:3].mean(axis=0), [0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(displayed[3], scores[3])

    def test_representative_pair_quota_covers_all_topologies(self):
        counts = pd.DataFrame(
            {
                "nonsignificant": [2070, 1738, 742, 936, 100, 41],
                "significant": [204, 118, 76, 64, 6, 4],
            },
            index=list("abcdef"),
        )
        quotas = MODULE.allocate_topology_quotas(counts, 12)
        self.assertEqual(sum(quotas.values()), 12)
        self.assertEqual(sorted(quotas.values(), reverse=True), [3, 3, 2, 2, 1, 1])

    def test_label_preserving_isomorphism(self):
        reference = nx.path_graph(4)
        comparison = nx.relabel_nodes(reference, {0: 3, 1: 2, 2: 1, 3: 0})
        nx.set_node_attributes(reference, {0: "2", 1: "4", 2: "4", 3: "6"}, "label")
        nx.set_node_attributes(comparison, {0: "6", 1: "4", 2: "4", 3: "2"}, "label")
        mapping = MODULE.best_label_preserving_isomorphism(reference, comparison)
        self.assertTrue(all(comparison.has_edge(mapping[u], mapping[v]) for u, v in reference.edges))

    @pytest.mark.integration
    @unittest.skipUnless(os.environ.get("CCINDEX_RUN_INTEGRATION") == "1", "Requires saved study artifacts and CCINDEX_RUN_INTEGRATION=1")
    def test_generated_representative_pairs_are_balanced_and_exact(self):
        tables = PROJECT / "results" / "tables"
        motifs_path = tables / "tri5_representative_motifs.csv"
        pairs_path = tables / "tri5_representative_pair_summary.csv"
        if not motifs_path.is_file() or not pairs_path.is_file():
            self.skipTest("Generated representative outputs are not present")
        motifs = pd.read_csv(motifs_path, dtype={"wl_hash": str})
        pairs = pd.read_csv(pairs_path)
        self.assertEqual(len(motifs), 24)
        self.assertEqual(motifs.wl_hash.nunique(), 24)
        self.assertEqual(
            motifs.groupby("significance_group").size().to_dict(),
            {"nonsignificant": 12, "significant": 12},
        )
        self.assertEqual(sorted(pairs.groupby("topology_id").size(), reverse=True), [3, 3, 2, 2, 1, 1])
        self.assertTrue(pairs.unlabeled_graph_isomorphic.all())

    @pytest.mark.integration
    @unittest.skipUnless(os.environ.get("CCINDEX_RUN_INTEGRATION") == "1", "Requires saved study artifacts and CCINDEX_RUN_INTEGRATION=1")
    def test_planar_layout_for_reported_problem_motif(self):
        source = pd.read_csv(
            PROJECT.parent / "significant_mtf_trace" / "tri1"
            / MODULE.TARGET_HASH / "trace_nodes.csv",
            dtype={"wl_hash": str, "node_labels": str, "edge_list": str},
        ).set_index("wl_hash")
        motif_hash = next(value for value in source.index if str(value).startswith("e959a91f"))
        row = source.loc[motif_hash]
        graph = MODULE.base.graph_from_encoding(row.node_labels, row.edge_list, motif_hash)
        positions = MODULE.triangulation_planar_layout(graph)

        def orientation(a, b, c):
            ab = np.asarray(b) - np.asarray(a)
            ac = np.asarray(c) - np.asarray(a)
            return float(ab[0] * ac[1] - ab[1] * ac[0])

        edges = list(graph.edges)
        for left_index, (a, b) in enumerate(edges):
            for c, d in edges[left_index + 1:]:
                if len({a, b, c, d}) < 4:
                    continue
                crosses = (
                    orientation(positions[a], positions[b], positions[c])
                    * orientation(positions[a], positions[b], positions[d]) < 0
                    and orientation(positions[c], positions[d], positions[a])
                    * orientation(positions[c], positions[d], positions[b]) < 0
                )
                self.assertFalse(crosses, f"Edges {(a, b)} and {(c, d)} cross")


if __name__ == "__main__":
    unittest.main()
