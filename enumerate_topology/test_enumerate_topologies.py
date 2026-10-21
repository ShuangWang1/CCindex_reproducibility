from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/enumerate_topology_matplotlib")

import igraph as ig

from enumerate_topology.enumerate_topologies import (
    ResourceLimits,
    _straight_line_layout_is_planar,
    boundary_cycle,
    canonical_code,
    decode_graph,
    expand_code,
    generate_next_level,
    graph_from_code,
    seed_code,
    topology_layout,
    validate_topology,
    write_sample,
)


EXPECTED_COUNTS = (1, 1, 1, 4, 6, 18, 41, 134, 390, 1339)


def literal_reference_children(code: bytes) -> set[bytes]:
    """Direct port of the 0806 tree_search loops for equivalence testing."""
    graph = graph_from_code(code)
    edge_list = []
    for edge in graph.get_edgelist():
        edge = tuple(sorted(edge))
        common = set(graph.neighbors(edge[0])).intersection(graph.neighbors(edge[1]))
        if len(common) == 1:
            edge_list.append(edge)

    children = set()
    done = set()
    for shared_edge in edge_list:
        if shared_edge in done:
            continue
        left, right = shared_edge
        neighbors = {
            tuple(sorted((neighbor, left)))
            for neighbor in graph.neighbors(left)
            if neighbor != right and not graph.are_adjacent(neighbor, right)
        }
        neighbors.update(
            tuple(sorted((neighbor, right)))
            for neighbor in graph.neighbors(right)
            if neighbor != left and not graph.are_adjacent(neighbor, left)
        )
        for neighbor_edge in neighbors.intersection(edge_list):
            done.add(neighbor_edge)
            endpoints = tuple(set(neighbor_edge).symmetric_difference(shared_edge))
            candidate = graph.copy()
            candidate.add_edge(*endpoints)
            if len(candidate.cliques(min=3, max=3)) == len(
                graph.cliques(min=3, max=3)
            ) + 1:
                children.add(canonical_code(candidate.vcount(), candidate.get_edgelist()))

    for left, right in edge_list:
        candidate = graph.copy()
        new_vertex = candidate.vcount()
        candidate.add_vertex()
        candidate.add_edges(((left, new_vertex), (right, new_vertex)))
        children.add(canonical_code(candidate.vcount(), candidate.get_edgelist()))
    return children


class EnumerationTests(unittest.TestCase):
    def test_canonical_code_ignores_vertex_ids(self) -> None:
        code = canonical_code(5, ((0, 1), (1, 2), (2, 0), (2, 3), (3, 4)))
        n, edges = decode_graph(code)
        permutation = (3, 0, 4, 1, 2)
        relabeled = [(permutation[u], permutation[v]) for u, v in edges]
        self.assertEqual(code, canonical_code(n, relabeled))

    def test_exact_counts_through_level_10(self) -> None:
        states = {seed_code()}
        observed = [len(states)]
        for _level in range(2, 11):
            children = set()
            for parent in states:
                local, _ = expand_code(parent)
                children.update(local)
            states = children
            observed.append(len(states))
        self.assertEqual(EXPECTED_COUNTS, tuple(observed))

    def test_optimized_growth_matches_literal_reference(self) -> None:
        states = {seed_code()}
        for _level in range(1, 7):
            optimized = set()
            literal = set()
            for parent in states:
                local, _ = expand_code(parent)
                optimized.update(local)
                literal.update(literal_reference_children(parent))
            self.assertEqual(literal, optimized)
            states = optimized

    def test_every_small_topology_satisfies_disk_invariants(self) -> None:
        states = {seed_code()}
        for level in range(1, 8):
            for code in states:
                validate_topology(code, level)
            children = set()
            for parent in states:
                local, _ = expand_code(parent)
                children.update(local)
            states = children

    def test_state_cap_marks_level_incomplete(self) -> None:
        states = {seed_code()}
        for _ in range(3):
            result = generate_next_level(
                states,
                workers=1,
                limits=ResourceLimits(max_states=2),
                batch_size=1,
            )
            if not result.complete:
                break
            states = result.children
        self.assertFalse(result.complete)
        self.assertEqual("state_count", result.cap_reason)

    def test_export_has_single_label_and_expected_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            row = write_sample(seed_code(), 1, 1, Path(directory))
            graph = ig.Graph.Read_GML(str(Path(directory) / row["gml_path"]))
            self.assertEqual({"1"}, set(graph.vs["label"]))
            self.assertEqual(1, int(graph["trianglecount"]))
            self.assertEqual(3, graph.vcount())

    def test_layout_places_every_vertex(self) -> None:
        states = {seed_code()}
        for _level in range(2, 8):
            children = set()
            for parent in states:
                local, _ = expand_code(parent)
                children.update(local)
            states = children
        for code in states:
            graph = graph_from_code(code)
            positions = topology_layout(graph)
            self.assertEqual(set(range(graph.vcount())), set(positions))
            self.assertTrue(
                _straight_line_layout_is_planar(graph.get_edgelist(), positions)
            )

    def test_layout_releases_asymmetric_boundary_from_circle(self) -> None:
        vertex_count = 8
        perimeter = [
            (vertex, (vertex + 1) % vertex_count)
            for vertex in range(vertex_count)
        ]
        fan_diagonals = [(0, vertex) for vertex in range(2, vertex_count - 1)]
        graph = ig.Graph(n=vertex_count, edges=perimeter + fan_diagonals)
        positions = topology_layout(graph)
        cycle = boundary_cycle(graph)
        center = (
            sum(positions[vertex][0] for vertex in cycle) / len(cycle),
            sum(positions[vertex][1] for vertex in cycle) / len(cycle),
        )
        radii = [
            ((positions[vertex][0] - center[0]) ** 2
             + (positions[vertex][1] - center[1]) ** 2)
            ** 0.5
            for vertex in cycle
        ]
        self.assertGreater(max(radii) - min(radii), 0.2)
        self.assertTrue(
            _straight_line_layout_is_planar(graph.get_edgelist(), positions)
        )

    def test_planarity_gate_rejects_crossing_segments(self) -> None:
        coordinates = ((0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (1.0, 0.0))
        self.assertFalse(
            _straight_line_layout_is_planar(((0, 1), (2, 3)), coordinates)
        )


if __name__ == "__main__":
    unittest.main()
