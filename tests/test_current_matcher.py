"""Check exact matching against independent labeled VF2 enumeration."""

import random

import igraph as ig
import pytest

import graph_pattern_cpp as cpp
from run_ccindex import run_cc_index


def graph(labels, edges):
    result = ig.Graph(n=len(labels), edges=edges)
    result.vs["label"] = labels
    return result


def native(g):
    result = cpp.Graph()
    for v in g.vs:
        result.addVertex(v.index, str(v["label"]))
    for left, right in g.get_edgelist():
        result.addEdge(left, right)
    return result


def independent_sets(target, pattern):
    palette = {value: i for i, value in enumerate(sorted(set(target.vs["label"] + pattern.vs["label"])))}
    mappings = target.get_subisomorphisms_vf2(
        pattern,
        color1=[palette[v] for v in target.vs["label"]],
        color2=[palette[v] for v in pattern.vs["label"]],
    )
    return {tuple(sorted(mapping)) for mapping in mappings}


def assert_matching(target, pattern):
    expected = independent_sets(target, pattern)
    cache = cpp.GraphCache(native(target))
    query = native(pattern)
    count, matches = cpp.find_pattern_matches(cache, query)
    assert count == len(expected)
    assert {tuple(match) for match in matches} == expected
    assert matches == sorted(matches)
    assert cpp.count_pattern_matches(cache, query) == len(expected)
    assert cpp.search_stats(cache, query)["sets"] == len(expected)
    for threshold in [1, 2, 4]:
        count, matches = cpp.find_pattern_matches(cache, query, threshold=threshold)
        assert count == min(threshold, len(expected))
        assert len(set(map(tuple, matches))) == count
        assert set(map(tuple, matches)).issubset(expected)
    count, matches = cpp.find_pattern_matches(cache, query, find_all=False)
    assert count == min(1, len(expected))
    assert set(map(tuple, matches)).issubset(expected)


@pytest.mark.parametrize("pattern", [
    graph(["a"] * 3, [(0, 1), (1, 2), (2, 0)]),
    graph(["a"] * 3, [(0, 1), (1, 2)]),
    graph(["a", "missing"], [(0, 1)]),
    graph(["a", "a"], []),
])
def test_symmetry_noninduced_disconnected_and_missing_labels(pattern):
    assert_matching(graph(["a"] * 5, [(i, j) for i in range(5) for j in range(i + 1, 5)]), pattern)


def test_seeded_small_graphs_match_vf2():
    rng = random.Random(20261020)
    for _ in range(40):
        n = rng.randrange(3, 9)
        target = graph([rng.choice(["a", "b"]) for _ in range(n)],
                       [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < 0.5])
        size = rng.randrange(1, min(n, 5) + 1)
        pattern = graph([rng.choice(["a", "b"]) for _ in range(size)],
                        [(i, j) for i in range(size) for j in range(i + 1, size) if rng.random() < 0.5])
        assert_matching(target, pattern)


def test_empty_target_has_no_matches():
    assert_matching(graph([], []), graph(["a"], []))


def test_duplicate_edges_do_not_change_occurrences():
    target = graph(["a"] * 3, [(0, 1), (1, 2), (0, 2), (0, 1)])
    query = graph(["a"] * 3, [(0, 1), (1, 2), (0, 2)])
    count, matches = cpp.find_pattern_matches(cpp.GraphCache(native(target)), native(query))
    assert (count, matches) == (1, [[0, 1, 2]])


def test_default_runner_and_empty_candidate_queue(tmp_path):
    target = graph(["a"] * 4, [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3)])
    pattern = graph(["a"] * 3, [(0, 1), (1, 2), (2, 0)])
    result = run_cc_index(target, [pattern], tmp_path / "matches.csv", "synthetic", False, perform_cpp=True)
    assert result[0]["count"] == 2
    assert run_cc_index(target, [], tmp_path / "empty.csv", "synthetic", False, perform_cpp=True) == []
