import csv
import multiprocessing
import os
import time
import warnings
from itertools import combinations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import igraph as ig
import argparse
warnings.filterwarnings("ignore")
from datetime import datetime
import resource

import random

# Separate, instrumented entry point used by tree_search_ablation.py.  Keeping it
# here makes the ablation runner discoverable beside the production run_cc_index
# API without changing that API's behavior.
from ablation_matcher import run_cc_index_ablation


def parse_graph(g: ig.Graph):
    """
    Parse an igraph object into an adjacency list and a label dictionary
    """
    adjacency = defaultdict(set)
    labels = {}
    for v in g.vs:
        labels[v.index] = v["label"]
        adjacency[v.index] = set()

    for e in g.es:
        s, t = e.source, e.target
        adjacency[s].add(t)
        adjacency[t].add(s)
    return adjacency, labels


def count_local_triangles(adjacency):
    """
    For each vertex in adjacency, compute how many triangles that vertex participates in and the triangle pairs.
    """
    tri_count = {v: 0 for v in adjacency}
    triangle_pairs = defaultdict(set)

    neighbor_lists = {v: list(neighbors) for v, neighbors in adjacency.items()}

    for v in adjacency:
        neighbors = neighbor_lists[v]
        for i, x in enumerate(neighbors):
            for y in neighbors[i+1:]:
                if y in adjacency[x]:
                    tri_count[v] += 1
                    pair = (min(x, y), max(x, y))
                    triangle_pairs[v].add(pair)

    return tri_count, triangle_pairs


def get_triangle_signature(triangle_pairs, vertex_labels):
    labeled_triangles = set()
    for v1, v2 in triangle_pairs:
        triangle_labels = sorted([vertex_labels[v1], vertex_labels[v2]])
        labeled_triangles.add(tuple(triangle_labels))
    return frozenset(labeled_triangles)


class GraphCache:
    """Cache for preprocessed graph data"""

    def __init__(self, graph: ig.Graph):
        self.adjacency, self.labels = parse_graph(graph)
        self.tri_count, self.tri_pairs = count_local_triangles(self.adjacency)
        self.label_to_vertices = defaultdict(list)
        self.degrees = {v: len(adj) for v, adj in self.adjacency.items()}

        self.adjacency_sets = {v: set(adj)
                               for v, adj in self.adjacency.items()}

        for v, label in self.labels.items():
            self.label_to_vertices[label].append(v)

        self.triangle_signatures = {}
        for v in self.adjacency:
            self.triangle_signatures[v] = get_triangle_signature(
                self.tri_pairs[v], self.labels)


class TriState:
    """
    Keeps track of the current state while finding matches
    """

    __slots__ = (
        "pattern_adj", "pattern_labels",
        "graph_cache",
        "mapping", "reverse_mapping",
        "p_next", "p_cur",
        "g_next", "g_cur",
        "pattern_size",
        # For triangle-based pruning:
        "pattern_tri_count", "pattern_triangle_pairs",
        "pattern_signatures"
    )

    def __init__(
        self,
        pattern_adj,
        pattern_labels,
        graph_cache: GraphCache,
        pattern_tri_count,
        pattern_triangle_pairs
    ):
        self.pattern_adj = pattern_adj
        self.pattern_labels = pattern_labels
        self.graph_cache = graph_cache

        self.pattern_tri_count = pattern_tri_count
        self.pattern_triangle_pairs = pattern_triangle_pairs

        self.pattern_size = len(pattern_adj)
        self.mapping = {}
        self.reverse_mapping = {}
        self.p_next = set()
        self.p_cur = set()
        self.g_next = set()
        self.g_cur = set()

        self.pattern_signatures = {}
        for v in pattern_adj:
            self.pattern_signatures[v] = get_triangle_signature(
                pattern_triangle_pairs[v], pattern_labels)

    def is_complete(self):
        """
        Whether matched all pattern vertices.
        """
        return len(self.mapping) == self.pattern_size

    def add_pair(self, p_v, g_v):
        """
        Add a new match: pattern vertex p_v -> graph vertex g_v
        """
        self.mapping[p_v] = g_v
        self.reverse_mapping[g_v] = p_v

        for p_u in self.pattern_adj[p_v]:
            if p_u not in self.mapping:
                if p_u not in self.p_next and p_u not in self.p_cur:
                    self.p_next.add(p_u)

        # p_v has just been matched, so remove it from p_next/p_cur
        self.p_next.discard(p_v)
        self.p_cur.discard(p_v)

        for g_u in self.graph_cache.adjacency[g_v]:
            if g_u not in self.reverse_mapping:
                if g_u not in self.g_next and g_u not in self.g_cur:
                    self.g_next.add(g_u)

        self.g_next.discard(g_v)
        self.g_cur.discard(g_v)

    def remove_pair(self, p_v, g_v):
        """
        Remove a match and update vertex sets for backtracking
        """
        invalid_p = []
        for p_u in self.pattern_adj[p_v]:
            if p_u not in self.mapping:
                keep = False
                for p_x in self.pattern_adj[p_u]:
                    if p_x in self.mapping and p_x != p_v:
                        keep = True
                        break
                if not keep:
                    invalid_p.append(p_u)

        invalid_g = []
        for g_u in self.graph_cache.adjacency[g_v]:
            if g_u not in self.reverse_mapping:
                keep = False
                for t_x in self.graph_cache.adjacency[g_u]:
                    if t_x in self.reverse_mapping and t_x != g_v:
                        keep = True
                        break
                if not keep:
                    invalid_g.append(g_u)

        del self.mapping[p_v]
        del self.reverse_mapping[g_v]

        for p_u in invalid_p:
            self.p_next.discard(p_u)
            self.p_cur.discard(p_u)

        for g_u in invalid_g:
            self.g_next.discard(g_u)
            self.g_cur.discard(g_u)


def find_pattern_triangles(state: TriState, solutions, allow_multiple=True, threshold = 0, st_time = time.time(), edge_dict = None):
    """
    Recursively attempt to extend a subgraph isomorphism.
    """
    if state.is_complete():
        sol = [None] * state.pattern_size
        for p_v, g_v in state.mapping.items():
            sol[p_v] = g_v
        solutions.append(sol)
        if threshold:
            if len(solutions) >= threshold:
                unique_solutions = set(frozenset(sol) for sol in solutions)
                if len(unique_solutions) >= threshold:
                    return True
        return (not allow_multiple)

    p_v = select_pattern_vertex(state, edge_dict)
    p_label = state.pattern_labels[p_v]
    p_tri_count = state.pattern_tri_count[p_v]
    p_degree = len(state.pattern_adj[p_v])

    connected_p_vertices = [
        p_u for p_u in state.pattern_adj[p_v] if p_u in state.mapping]
    connected_g_vertices = [state.mapping[p_u] for p_u in connected_p_vertices]

    candidates = []
    for g_v in state.graph_cache.label_to_vertices[p_label]:
        if g_v in state.reverse_mapping:
            continue

        if state.graph_cache.degrees[g_v] < p_degree:
            continue

        if state.graph_cache.tri_count[g_v] < p_tri_count:
            continue

        feasible = True
        for g_u in connected_g_vertices:
            if g_u not in state.graph_cache.adjacency_sets[g_v]:
                feasible = False
                break

        if not feasible:
            continue

        pattern_sig = state.pattern_signatures[p_v]
        graph_sig = state.graph_cache.triangle_signatures[g_v]

        if not pattern_sig.issubset(graph_sig):
            continue

        candidates.append(g_v)

    if len(candidates) > 1:
        candidates.sort(key=lambda g_v: (
            -sum(1 for g_u in connected_g_vertices if g_u in state.graph_cache.adjacency_sets[g_v]),
            -state.graph_cache.tri_count[g_v]
        ))

    # Recurse on the feasible candidates
    for g_v in candidates:
        state.add_pair(p_v, g_v)
        stop_early = find_pattern_triangles(state, solutions, allow_multiple, threshold, st_time = st_time, edge_dict = edge_dict)
        state.remove_pair(p_v, g_v)

        if stop_early:
            return True
    return False


def select_pattern_vertex(state, edge_dict = None):
    """
    Select the pattern vertex with minimum remaining candidates
    """
    if state.p_next:
        candidates = list(state.p_next)
        if len(candidates) > 1:
            return max(candidates, key=lambda v: sum(1 for u in state.pattern_adj[v] if u in state.mapping))
        return candidates[0]
    elif state.p_cur:
        return next(iter(state.p_cur))
    else:
        candidates = [v for v in state.pattern_adj if v not in state.mapping]

        if edge_dict == None:
            return max(candidates, key=lambda v: len(state.pattern_adj[v]))

        pattern_vertex_value = {k:0 for k in candidates}
        for k in candidates:
            for a in state.pattern_adj[k]:
                mtf = sorted([state.pattern_labels[k], state.pattern_labels[a]])
                mtf = '_'.join(mtf)
                if mtf not in edge_dict:
                    pattern_vertex_value[k] += 1
                else:
                    pattern_vertex_value[k] += 1 / edge_dict[mtf]
        return max(pattern_vertex_value, key=pattern_vertex_value.get)


def find_pattern_matches(graph_cache: GraphCache, pattern_graph: ig.Graph, find_all=True, threshold = 0, st_time = time.time(), edge_dict = None):
    """
    Main function that:
      1) Parses two graphs into adjacency + labels
      2) Computes local triangle counts for each vertex in both graphs
      3) Find subgraph matches
      4) Returns the number of matches and the matches
    """
    p_adj, p_labels = parse_graph(pattern_graph)
    p_tri_count, p_tri_pairs = count_local_triangles(p_adj)
    p_adj = {key: sorted(list(value), reverse=True) for key, value in p_adj.items()}

    state = TriState(
        p_adj, p_labels,
        graph_cache,
        p_tri_count, p_tri_pairs
    )

    solutions = []
    find_pattern_triangles(state, solutions, allow_multiple=find_all, threshold = threshold,  edge_dict = edge_dict)
    if len(solutions) and solutions[-1] == -1:
        return -1, []

    # Deduplicate
    unique_solutions = set(frozenset(sol) for sol in solutions)

    res = [list(sol) for sol in unique_solutions]
    return len(res), res


def igraph_to_custom_graph(ig_graph):
    """Convert igraph.Graph to custom Graph object for C++"""
    try:
        import graph_pattern_cpp
        custom_graph = graph_pattern_cpp.Graph()

        for i in range(ig_graph.vcount()):
            label = str(ig_graph.vs[i]['label'])
            custom_graph.addVertex(i, label)

        for edge in ig_graph.es:
            source = edge.source
            target = edge.target
            custom_graph.addEdge(source, target)

        return custom_graph
    except ImportError:
        raise ImportError("graph_pattern_cpp module not found. Please compile the C++ extension first.")


def process_sequential(graph, g_queue, threshold, perform_cpp, edge_dict):
    """Process patterns sequentially one by one"""

    results = []
    filter_g_queue = []
    cnt = 0

    if perform_cpp:
        # For C++ implementation
        import graph_pattern_cpp
        graph_cpp = igraph_to_custom_graph(graph)
        graph_cache = graph_pattern_cpp.GraphCache(graph_cpp)
    else:
        # For Python implementation
        graph_cache = GraphCache(graph)

    for pattern in g_queue:
        st = time.time()

        if perform_cpp == False:
            match_count, subisomorphisms = find_pattern_matches(
                graph_cache, pattern, find_all=True, threshold=threshold, st_time=st, edge_dict=edge_dict
            )
        else:
            import graph_pattern_cpp
            pattern_cpp = igraph_to_custom_graph(pattern)
            match_count, subisomorphisms = graph_pattern_cpp.find_pattern_matches(
                graph_cache, pattern_cpp, find_all=True, threshold=threshold, edge_dict=edge_dict
            )

        et = time.time()
        pattern['count'] = match_count
        pattern['mtf_ids'] = subisomorphisms
        pattern['time'] = et - st

        pattern_name = f'{len(pattern.cliques(min=3, max=3))}_{cnt}'
        cnt += 1

        results.append([pattern_name, match_count, et - st])

        if match_count != 0:
            filter_g_queue.append(pattern)

    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    return filter_g_queue, results, os.getpid(), mem


def run_cc_index(graph, pattern_queue, result_path, graph_name, use_multiprocessing, num_workers=1, threshold=0, perform_cpp=False, edge_dict=None):

    if not pattern_queue:
        os.makedirs(os.path.dirname(os.fspath(result_path)) or ".", exist_ok=True)
        with open(result_path, "a", newline=""):
            pass
        return []

    print("=" * 50)
    total_patterns = len(pattern_queue)
    print(f"Found {total_patterns} patterns to process")
    print(f"Using {'C++' if perform_cpp else 'Python'} implementation")

    total_start_time = time.time()
    match_buffer = []
    g_queue_buffer = []
    mem_used_buffer = []
    pid_buffer = []

    if use_multiprocessing:
        batch_size = 10
        print(f"Starting parallel processing with {num_workers} workers...")

        pattern_batches = [pattern_queue[i:i+batch_size]
                           for i in range(0, len(pattern_queue), batch_size)]

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_sequential, graph, batch, threshold, perform_cpp, edge_dict): batch
                       for batch in pattern_batches}

            for future in futures:
                try:
                    g_queue_res, match_res, pid, mem_used = future.result()
                    match_buffer.extend(match_res)
                    g_queue_buffer.extend(g_queue_res)
                    mem_used_buffer.append(mem_used)
                    pid_buffer.append(pid)
                except Exception as exc:
                    print(f"Processing batch generated an exception: {exc}")

    else:
        print("Starting sequential processing...")
        g_queue_buffer, match_buffer, pid_buffer, mem_used_buffer = process_sequential(graph, pattern_queue, threshold, perform_cpp, edge_dict)

    total_time = time.time() - total_start_time

    result_dir = os.path.dirname(result_path)
    match_buffer[0][0] = graph_name
    os.makedirs(result_dir, exist_ok=True)
    with open(result_path, mode='a', newline='') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerows(match_buffer)

    if use_multiprocessing:
        mem_max_by_pid = {}
        for p, m in zip(pid_buffer, mem_used_buffer):
            if p not in mem_max_by_pid or m > mem_max_by_pid[p]:
                mem_max_by_pid[p] = m

    print(f"Processing complete")
    print(f"Processing mode: {'Parallel' if use_multiprocessing else 'Sequential'}")
    print(f"Output File: {result_path}")
    print(f"Total patterns: {total_patterns}")
    print(f"Successfully processed: {len(match_buffer)}")
    print(f"Failed: {total_patterns - len(match_buffer)}")
    print(f"Total execution time: {total_time:.2f} seconds")
    if use_multiprocessing:
        print(f'Memory consumption {len(mem_max_by_pid)} CPUs: {list(mem_max_by_pid.values())} MB')
    print("="*50)

    return g_queue_buffer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pattern_type', type=str, default="1")
    parser.add_argument('--use_cpp', action='store_true', help='Use C++ implementation')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    pattern_type = args.pattern_type
    use_cpp = args.use_cpp

    number_of_vertices_labels_list = [8, 16, 32]
    pattern_random = 1
    pattern_dir = "../data/triangle" + pattern_type + "/patterns/"

    for p in range(pattern_random):
        for number_of_vertices_labels in number_of_vertices_labels_list:
            pattern = ig.read(
                pattern_dir + "P_triangle" + str(pattern_type) + "_NL" + str(number_of_vertices_labels) + "_" + str(p) + ".gml")
            graph_folder = "../data/triangle"+pattern_type+"/graphs/P_triangle"+str(pattern_type)+"_NL" + str(number_of_vertices_labels) + "_" + str(p) + "/"
            graph_files = os.listdir(graph_folder)

            result_path = "results/P_triangle" + str(pattern_type) + "_NL" + str(
                number_of_vertices_labels) + "_" + str(p) + ("_cpp" if use_cpp else "_py") + ".csv"

            with open(result_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['graph', 'Matches', 'Time'])

            for graph_file in graph_files:
                if graph_file.startswith("G_"):
                    graph = ig.read(graph_folder + graph_file)
                    pattern_queue = [pattern]
                    graph_name = graph_file.split(".gml")[0]

                    run_cc_index(graph, pattern_queue, result_path, graph_name,
                               use_multiprocessing=False, perform_cpp=use_cpp)
