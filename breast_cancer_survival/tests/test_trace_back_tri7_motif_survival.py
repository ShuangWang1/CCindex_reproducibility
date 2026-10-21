import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import pandas as pd


SURVIVAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVIVAL_DIR))

from trace_back_tri7_motif_survival import (  # noqa: E402
    fixed_target_layout,
    motif_hash,
    plot_growth_transition_gallery,
    potential_direct_parents,
    transition_graphs,
    triangle_count,
)


class TraceBackTests(unittest.TestCase):
    def test_reverses_vertex_on_edge_growth(self):
        graph = nx.Graph([(0, 1), (0, 2), (1, 2), (0, 3), (1, 3)])
        nx.set_node_attributes(graph, "6", "label")
        parents = potential_direct_parents(graph)
        vertex_parents = [parent for operation, removed, parent in parents if operation == "remove_vertex" and removed == "3"]
        self.assertEqual(len(vertex_parents), 1)
        self.assertEqual(triangle_count(vertex_parents[0]), 1)

    def test_reverses_edge_closing_one_triangle(self):
        graph = nx.Graph([(0, 1), (1, 2), (2, 0), (1, 3), (2, 3)])
        nx.set_node_attributes(graph, "5", "label")
        parents = potential_direct_parents(graph)
        edge_parents = [parent for operation, _, parent in parents if operation == "remove_edge"]
        self.assertTrue(edge_parents)
        self.assertTrue(all(triangle_count(parent) == 1 for parent in edge_parents))

    def test_reconstructs_vertex_transition_in_child_embedding(self):
        child = nx.Graph([(0, 1), (0, 2), (1, 2), (0, 3), (1, 3)])
        nx.set_node_attributes(child, "6", "label")
        expected_parent = child.copy()
        expected_parent.remove_node(3)
        relationship = SimpleNamespace(
            reverse_operation="remove_vertex",
            removed_node_or_edge="3",
            parent_hash=motif_hash(expected_parent),
        )

        parent, added_nodes, added_edges = transition_graphs(child, relationship)

        self.assertEqual(set(parent.nodes), {0, 1, 2})
        self.assertEqual(added_nodes, {3})
        self.assertEqual(added_edges, {(0, 3), (1, 3)})

    def test_reconstructs_edge_transition_in_child_embedding(self):
        child = nx.Graph([(0, 1), (1, 2), (2, 0), (1, 3), (2, 3)])
        nx.set_node_attributes(child, "5", "label")
        expected_parent = child.copy()
        expected_parent.remove_edge(1, 3)
        relationship = SimpleNamespace(
            reverse_operation="remove_edge",
            removed_node_or_edge="1-3",
            parent_hash=motif_hash(expected_parent),
        )

        parent, added_nodes, added_edges = transition_graphs(child, relationship)

        self.assertEqual(set(parent.edges), set(expected_parent.edges))
        self.assertEqual(added_nodes, set())
        self.assertEqual(added_edges, {(1, 3)})

    def test_transition_gallery_smoke(self):
        child = nx.Graph([(0, 1), (0, 2), (1, 2), (0, 3), (1, 3)])
        nx.set_node_attributes(child, "6", "label")
        parent = child.copy()
        parent.remove_node(3)
        child_hash = motif_hash(child)
        relationships = pd.DataFrame(
            [
                {
                    "parent_level": 1,
                    "parent_hash": motif_hash(parent),
                    "child_level": 2,
                    "child_hash": child_hash,
                    "reverse_operation": "remove_vertex",
                    "removed_node_or_edge": "3",
                }
            ]
        )
        coordinates = fixed_target_layout(child, child_hash)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "transitions.png"
            plot_growth_transition_gallery(
                relationships,
                {2: {child_hash: child}},
                child,
                coordinates,
                {"6": "tumor cells"},
                "Test transitions",
                output,
            )
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
