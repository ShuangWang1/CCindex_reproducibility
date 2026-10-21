from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
import json
from pathlib import Path

import networkx as nx
import pandas as pd


SURVIVAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVIVAL_DIR))

import trace_significant_motifs as trace  # noqa: E402


def labeled_graph(edges: list[tuple[int, int]], labels: dict[int, str]) -> nx.Graph:
    graph = nx.Graph(edges)
    nx.set_node_attributes(graph, labels, "label")
    return graph


class SignificantMotifTraceTests(unittest.TestCase):
    def test_pool_requires_raw_significance_and_positive_median(self) -> None:
        connection = sqlite3.connect(":memory:")
        connection.execute(
            "CREATE TABLE cox (triangle_level INTEGER, wl_hash TEXT, "
            "sample_document_frequency REAL, nodes REAL, edges REAL, "
            "representative_node_labels TEXT, cell_type_composition TEXT, "
            "median_patient_raw_count REAL, coef_unadjusted REAL, "
            "hazard_ratio_unadjusted REAL, ci95_low_unadjusted REAL, "
            "ci95_high_unadjusted REAL, p_value_unadjusted REAL, "
            "n_enriched_unadjusted REAL, events_enriched_unadjusted REAL, "
            "n_non_enriched_unadjusted REAL, events_non_enriched_unadjusted REAL, "
            "fit_status_unadjusted TEXT, n_unadjusted REAL, events_unadjusted REAL, "
            "q_value_bh_global_unadjusted REAL, "
            "q_value_bh_within_level_unadjusted REAL, "
            "rank_global_unadjusted REAL, rank_within_level_unadjusted REAL)"
        )
        empty = [None] * (len(trace.SQL_COLUMNS) - 2)
        connection.executemany(
            f"INSERT INTO cox VALUES ({','.join('?' for _ in trace.SQL_COLUMNS)})",
            [
                [7, "keep", *empty[:5], 1.0, *empty[6:10], 0.01, *empty[11:]],
                [7, "zero", *empty[:5], 0.0, *empty[6:10], 0.01, *empty[11:]],
                [7, "large_p", *empty[:5], 2.0, *empty[6:10], 0.10, *empty[11:]],
            ],
        )
        observed = trace.significant_pool(connection, 0.05)
        self.assertEqual(observed["wl_hash"].tolist(), ["keep"])
        self.assertEqual(observed["motif_id"].tolist(), ["tri_7:keep"])

    def test_direct_children_include_both_growth_operations(self) -> None:
        triangle = labeled_graph([(0, 1), (0, 2), (1, 2)], {0: "5", 1: "6", 2: "6"})
        vertex_child = triangle.copy()
        vertex_child.add_node(3, label="5")
        vertex_child.add_edges_from([(0, 3), (1, 3)])

        open_pair = labeled_graph(
            [(0, 1), (0, 2), (1, 2), (1, 3), (3, 4)],
            {0: "5", 1: "6", 2: "6", 3: "5", 4: "5"},
        )
        edge_child = open_pair.copy()
        edge_child.add_edge(1, 4)

        vertex_results = trace.potential_direct_children(
            triangle, {trace.motif_hash(vertex_child)}, ["5", "6"]
        )
        edge_results = trace.potential_direct_children(
            open_pair, {trace.motif_hash(edge_child)}, ["5", "6"]
        )
        self.assertIn("add_vertex", {value[0] for value in vertex_results})
        self.assertIn("add_edge", {value[0] for value in edge_results})
        self.assertTrue(
            all(trace.triangle_count(value[2]) == trace.triangle_count(triangle) + 1 for value in vertex_results)
        )

    def test_trace_lineage_runs_backward_and_forward(self) -> None:
        tri1 = labeled_graph([(0, 1), (0, 2), (1, 2)], {0: "6", 1: "6", 2: "5"})
        tri2 = tri1.copy()
        tri2.add_node(3, label="6")
        tri2.add_edges_from([(0, 3), (1, 3)])
        tri3 = tri2.copy()
        tri3.add_node(4, label="5")
        tri3.add_edges_from([(1, 4), (3, 4)])
        vocabularies = {level: set() for level in range(1, 8)}
        vocabularies[1].add(trace.motif_hash(tri1))
        vocabularies[2].add(trace.motif_hash(tri2))
        vocabularies[3].add(trace.motif_hash(tri3))

        graphs, edges = trace.trace_lineage(tri2, 2, vocabularies, ["5", "6"])

        self.assertEqual(set(graphs[1]), {trace.motif_hash(tri1)})
        self.assertEqual(set(graphs[3]), {trace.motif_hash(tri3)})
        self.assertEqual(len(edges), 2)
        self.assertTrue((edges["child_level"] - edges["parent_level"]).eq(1).all())

    def test_p_lineage_plot_smoke(self) -> None:
        nodes = pd.DataFrame(
            {
                "triangle_level": [1, 2],
                "wl_hash": ["parent", "target"],
                "lineage_role": ["ancestor", "target"],
                "p_value_unadjusted": [0.2, 0.01],
            }
        )
        edges = pd.DataFrame(
            {
                "parent_level": [1],
                "parent_hash": ["parent"],
                "child_level": [2],
                "child_hash": ["target"],
                "growth_operation": ["add_vertex"],
                "changed_node_or_edge": ["3:6@0-1"],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "cox.png"
            trace.plot_p_lineage(nodes, edges, 2, "target", output, 0.05)
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)

    def test_cumulative_manifest_preserves_prior_levels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prior = {
                "status": "complete",
                "motif_id": "tri_7:old",
                "target_level": 7,
                "target_hash": "old",
            }
            path = root / "tri7" / "old" / "trace_metadata.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(prior), encoding="utf-8")
            current = {
                "status": "complete",
                "motif_id": "tri_5:new",
                "target_level": 5,
                "target_hash": "new",
                "run_action": "generated",
            }

            observed = trace.cumulative_manifest(root, [current])

            self.assertEqual(set(observed["motif_id"]), {"tri_7:old", "tri_5:new"})
            actions = observed.set_index("motif_id")["run_action"]
            self.assertEqual(actions["tri_7:old"], "existing_complete")
            self.assertEqual(actions["tri_5:new"], "generated")


if __name__ == "__main__":
    unittest.main()
