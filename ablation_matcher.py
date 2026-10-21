"""Instrumented Python/C++ runners used by tree_search_ablation.py."""

import copy
import csv
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path


VALID_VARIANTS = (
    "full",
    "no_cpp",
    "no_triangle_filter",
    "no_candidate_prefilter",
    "no_reorder",
)


@dataclass
class SearchStats:
    states_visited: int = 0
    vertices_examined: int = 0
    candidates_accepted: int = 0
    rejected_label: int = 0
    rejected_injective: int = 0
    rejected_degree: int = 0
    rejected_triangle_count: int = 0
    rejected_signature: int = 0
    rejected_adjacency: int = 0


def parse_graph(graph):
    adjacency = {vertex.index: set() for vertex in graph.vs}
    labels = {vertex.index: str(vertex["label"]) for vertex in graph.vs}
    for edge in graph.es:
        adjacency[edge.source].add(edge.target)
        adjacency[edge.target].add(edge.source)
    return adjacency, labels


def triangle_data(adjacency, labels):
    counts = {vertex: 0 for vertex in adjacency}
    signatures = {vertex: set() for vertex in adjacency}
    for vertex, neighbors in adjacency.items():
        neighbors = sorted(neighbors)
        for pos, left in enumerate(neighbors):
            for right in neighbors[pos + 1 :]:
                if right in adjacency[left]:
                    counts[vertex] += 1
                    signatures[vertex].add(tuple(sorted((labels[left], labels[right]))))
    return counts, signatures


class PythonGraphCache:
    def __init__(self, graph):
        self.adjacency, self.labels = parse_graph(graph)
        self.degrees = {vertex: len(neighbors) for vertex, neighbors in self.adjacency.items()}
        self.label_to_vertices = defaultdict(list)
        for vertex, label in self.labels.items():
            self.label_to_vertices[label].append(vertex)
        for vertices in self.label_to_vertices.values():
            vertices.sort()
        self.triangle_counts, self.triangle_signatures = triangle_data(
            self.adjacency, self.labels
        )


def _options(variant):
    if variant not in VALID_VARIANTS:
        raise ValueError(f"Unknown ablation variant: {variant}")
    return {
        "label_index": variant != "no_candidate_prefilter",
        "degree_filter": variant != "no_candidate_prefilter",
        "triangle_filter": variant != "no_triangle_filter",
        "reorder": variant != "no_reorder",
    }


def find_pattern_matches_python_with_options(
    cache, pattern, threshold=0, edge_dict=None, *, options
):
    """Run the Python matcher with an explicit set of optimization switches.

    This entry point is used by the cumulative v2 ablation.  Keeping the
    switches explicit avoids changing the semantics of the original
    one-at-a-time variants.
    """
    required = {"label_index", "degree_filter", "triangle_filter", "reorder"}
    missing = required.difference(options)
    if missing:
        raise ValueError(f"Missing matcher options: {sorted(missing)}")
    pattern_adj, pattern_labels = parse_graph(pattern)
    pattern_tri_counts, pattern_signatures = triangle_data(pattern_adj, pattern_labels)
    pattern_size = len(pattern_adj)
    mapping = {}
    reverse_mapping = {}
    solutions = set()
    stats = SearchStats()

    def select_vertex():
        remaining = [vertex for vertex in pattern_adj if vertex not in mapping]
        if not options["reorder"]:
            return min(remaining)
        frontier = [
            vertex
            for vertex in remaining
            if any(neighbor in mapping for neighbor in pattern_adj[vertex])
        ]
        candidates = frontier or remaining

        def score(vertex):
            mapped_neighbors = sum(neighbor in mapping for neighbor in pattern_adj[vertex])
            if frontier:
                return mapped_neighbors, -vertex
            if edge_dict is None:
                return len(pattern_adj[vertex]), -vertex
            rarity = 0.0
            for neighbor in pattern_adj[vertex]:
                key = "_".join(sorted((pattern_labels[vertex], pattern_labels[neighbor])))
                rarity += 1.0 / edge_dict.get(key, 1)
            return rarity, -vertex

        return max(candidates, key=score)

    def search():
        stats.states_visited += 1
        if len(mapping) == pattern_size:
            solutions.add(frozenset(mapping.values()))
            return bool(threshold and len(solutions) >= threshold)

        pattern_vertex = select_vertex()
        label = pattern_labels[pattern_vertex]
        degree = len(pattern_adj[pattern_vertex])
        mapped_neighbors = [
            mapping[neighbor]
            for neighbor in pattern_adj[pattern_vertex]
            if neighbor in mapping
        ]
        if options["label_index"]:
            pool = list(cache.label_to_vertices.get(label, ()))
        else:
            pool = sorted(cache.labels)

        accepted = []
        for graph_vertex in pool:
            stats.vertices_examined += 1
            if cache.labels[graph_vertex] != label:
                stats.rejected_label += 1
                continue
            if graph_vertex in reverse_mapping:
                stats.rejected_injective += 1
                continue
            if options["degree_filter"] and cache.degrees[graph_vertex] < degree:
                stats.rejected_degree += 1
                continue
            if (
                options["triangle_filter"]
                and cache.triangle_counts[graph_vertex] < pattern_tri_counts[pattern_vertex]
            ):
                stats.rejected_triangle_count += 1
                continue
            if any(neighbor not in cache.adjacency[graph_vertex] for neighbor in mapped_neighbors):
                stats.rejected_adjacency += 1
                continue
            if (
                options["triangle_filter"]
                and not pattern_signatures[pattern_vertex].issubset(
                    cache.triangle_signatures[graph_vertex]
                )
            ):
                stats.rejected_signature += 1
                continue
            accepted.append(graph_vertex)
            stats.candidates_accepted += 1

        if options["reorder"]:
            accepted.sort(
                key=lambda vertex: (-cache.triangle_counts[vertex], vertex)
            )
        else:
            accepted.sort()

        for graph_vertex in accepted:
            mapping[pattern_vertex] = graph_vertex
            reverse_mapping[graph_vertex] = pattern_vertex
            stop = search()
            del mapping[pattern_vertex]
            del reverse_mapping[graph_vertex]
            if stop:
                return True
        return False

    search()
    return len(solutions), [sorted(solution) for solution in solutions], asdict(stats)


def find_pattern_matches_python(cache, pattern, threshold=0, edge_dict=None, variant="full"):
    return find_pattern_matches_python_with_options(
        cache,
        pattern,
        threshold,
        edge_dict,
        options=_options(variant),
    )


def igraph_to_cpp(graph, cpp_module):
    converted = cpp_module.Graph()
    for vertex in graph.vs:
        converted.addVertex(vertex.index, str(vertex["label"]))
    for edge in graph.es:
        converted.addEdge(edge.source, edge.target)
    return converted


def run_cc_index_ablation(
    graph,
    patterns,
    variant,
    result_path,
    threshold=0,
    edge_dict=None,
):
    if variant not in VALID_VARIANTS:
        raise ValueError(f"Unknown ablation variant: {variant}")
    use_cpp = variant != "no_cpp"
    cache_started = time.perf_counter()
    if use_cpp:
        import graph_pattern_ablation_cpp as cpp

        cpp_graph = igraph_to_cpp(graph, cpp)
        cache = cpp.GraphCache(cpp_graph)
    else:
        cpp = None
        cache = PythonGraphCache(graph)
    cache_seconds = time.perf_counter() - cache_started

    rows = []
    updated_patterns = copy.deepcopy(patterns)
    for pattern in updated_patterns:
        started = time.perf_counter()
        if use_cpp:
            cpp_pattern = igraph_to_cpp(pattern, cpp)
            count, matches, stats = cpp.find_pattern_matches_ablation(
                cache, cpp_pattern, True, threshold, edge_dict, variant
            )
        else:
            count, matches, stats = find_pattern_matches_python(
                cache, pattern, threshold, edge_dict, "full"
            )
        elapsed = time.perf_counter() - started
        pattern["count"] = count
        pattern["mtf_ids"] = matches
        pattern["time"] = elapsed
        row = {
            "variant": variant,
            "level": pattern["ablation_level"],
            "pattern_id": pattern["ablation_id"],
            "matches": count,
            "threshold_reached": bool(threshold and count >= threshold),
            "search_seconds": elapsed,
            **dict(stats),
        }
        rows.append(row)

    result_path = Path(result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["variant", "level", "pattern_id", "matches"]
    with result_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return updated_patterns, rows, cache_seconds
