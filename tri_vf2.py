import csv
import multiprocessing
import os
import time
import warnings
from itertools import combinations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import igraph as ig

warnings.filterwarnings("ignore")
from datetime import datetime
import psutil
import resource

import random


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
    '''
    if time.time() - st_time >= 10:
        if len(solutions) < threshold or threshold == 0:
            solutions.append(-1)
        return True
    '''
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

def find_pattern_triangles_debug(
    state: TriState,
    solutions,
    allow_multiple=True,
    threshold=0,
    st_time=time.time(),
    depth=0  # NEW: track recursion depth
):
    indent = "  " * depth  # For pretty printing

    # Debug: entering recursion
    print(f"{indent}Depth {depth}: Mapping so far {state.mapping}")

    if state.is_complete():
        sol = [None] * state.pattern_size
        for p_v, g_v in state.mapping.items():
            sol[p_v] = g_v
        solutions.append(sol)
        print(f"{indent}✅ Complete mapping found: {sol}")
        if threshold:
            if len(solutions) >= threshold:
                unique_solutions = set(frozenset(sol) for sol in solutions)
                if len(unique_solutions) >= threshold:
                    print(f"{indent}Reached threshold, stopping.")
                    return True
        return (not allow_multiple)

    # Choose next pattern vertex
    p_v = select_pattern_vertex(state)
    p_label = state.pattern_labels[p_v]
    p_tri_count = state.pattern_tri_count[p_v]
    p_degree = len(state.pattern_adj[p_v])

    connected_p_vertices = [
        p_u for p_u in state.pattern_adj[p_v] if p_u in state.mapping]
    connected_g_vertices = [state.mapping[p_u] for p_u in connected_p_vertices]

    print(f"{indent}→ Trying to match pattern vertex {p_v} "
          f"(label={p_label}, tri_count={p_tri_count}, degree={p_degree})")
    print(f"{indent}  Connected pattern vertices: {connected_p_vertices}")
    print(f"{indent}  Already mapped to: {connected_g_vertices}")

    candidates = []
    for g_v in state.graph_cache.label_to_vertices[p_label]:
        if g_v in state.reverse_mapping:
            print(f"{indent}  ✗ g_v={g_v} skipped (already mapped)")
            continue

        if state.graph_cache.degrees[g_v] < p_degree:
            print(f"{indent}  ✗ g_v={g_v} skipped (degree too low)")
            continue

        if state.graph_cache.tri_count[g_v] < p_tri_count:
            print(f"{indent}  ✗ g_v={g_v} skipped (tri_count too low)")
            continue

        feasible = True
        for g_u in connected_g_vertices:
            if g_u not in state.graph_cache.adjacency_sets[g_v]:
                feasible = False
                print(f"{indent}  ✗ g_v={g_v} skipped (not connected to {g_u})")
                break

        if not feasible:
            continue

        pattern_sig = state.pattern_signatures[p_v]
        graph_sig = state.graph_cache.triangle_signatures[g_v]

        if not pattern_sig.issubset(graph_sig):
            print(f"{indent}  ✗ g_v={g_v} skipped (signature mismatch)")
            continue

        print(f"{indent}  ✓ Candidate found: g_v={g_v}")
        candidates.append(g_v)

    if len(candidates) > 1:
        candidates.sort(key=lambda g_v: (
            -sum(1 for g_u in connected_g_vertices if g_u in state.graph_cache.adjacency_sets[g_v]),
            -state.graph_cache.tri_count[g_v]
        ))

    print(f"{indent}  Candidates at depth {depth}: {candidates}")

    # Recurse on the feasible candidates
    for g_v in candidates:
        print(f"{indent}→ Adding pair ({p_v}, {g_v}) and going deeper")
        state.add_pair(p_v, g_v)
        stop_early = find_pattern_triangles_debug(
            state, solutions, allow_multiple, threshold, st_time=st_time, depth=depth+1
        )
        state.remove_pair(p_v, g_v)
        print(f"{indent}← Backtracking from ({p_v}, {g_v})")

        if stop_early:
            return True

    return False

import time

def find_isomorphisms_with_profiling(
    state: TriState,
    solutions,
    profiling_data,  # NEW: Dictionary to store performance data
    allow_multiple=True,
    threshold=0,
    depth=0,
    edge_dict = None
):
    """
    Finds subgraph isomorphisms using a recursive backtracking algorithm,
    while profiling the search tree structure.
    """
    # ---- Profiling Logic ----
    # Initialize the entry for the current depth if it's the first time we see it
    if depth not in profiling_data:
        profiling_data[depth] = {'nodes_expanded': 0, 'children_generated': 0}

    # Count this function call as one "node expanded" at the current depth
    profiling_data[depth]['nodes_expanded'] += 1
    # -------------------------

    # Base case: A complete mapping has been found
    if state.is_complete():
        sol = [None] * state.pattern_size
        for p_v, g_v in state.mapping.items():
            sol[p_v] = g_v
        solutions.append(sol)

        # Stop early if the solution threshold is met
        if threshold and len(solutions) >= threshold:
            return True # Signal to stop
        return not allow_multiple

    # Core algorithm: Select next vertex and find candidates
    p_v = select_pattern_vertex(state, edge_dict)
    p_label = state.pattern_labels[p_v]
    p_tri_count = state.pattern_tri_count[p_v]
    p_degree = len(state.pattern_adj[p_v])

    connected_g_vertices = [
        state.mapping[p_u] for p_u in state.pattern_adj[p_v] if p_u in state.mapping
    ]

    # Filter candidates based on feasibility rules
    candidates = []
    lst = state.graph_cache.label_to_vertices.get(p_label, [])
    random.shuffle(lst)
    for g_v in lst:
        if g_v in state.reverse_mapping:
            continue
        if state.graph_cache.degrees[g_v] < p_degree:
            continue
        if state.graph_cache.tri_count[g_v] < p_tri_count:
            continue

        # Check connectivity to already mapped neighbors
        if not all(g_u in state.graph_cache.adjacency_sets[g_v] for g_u in connected_g_vertices):
            continue

        # Check signature compatibility
        if not state.pattern_signatures[p_v].issubset(state.graph_cache.triangle_signatures[g_v]):
            continue

        candidates.append(g_v)

    # ---- Profiling Logic ----
    # Record how many potential children (recursive calls) were generated from this node
    profiling_data[depth]['children_generated'] += len(candidates)
    # -------------------------

    # Optimization: Sort candidates to explore more promising branches first
    if len(candidates) > 1:
        candidates.sort(key=lambda g_v: -state.graph_cache.tri_count[g_v])

    # Recurse on the feasible candidates
    for g_v in candidates:
        state.add_pair(p_v, g_v)
        stop_early = find_isomorphisms_with_profiling(
            state, solutions, profiling_data, allow_multiple, threshold, depth + 1
        )
        state.remove_pair(p_v, g_v) # Backtrack

        if stop_early:
            return True # Propagate the stop signal up the recursion stack

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
    profiling_data={}
    find_pattern_triangles(state, solutions, allow_multiple=find_all, threshold = threshold,  edge_dict = edge_dict)
    if len(solutions) and solutions[-1] == -1:
        return -1, []

    # print("\n--- Search Tree Profile ---")
    # # Sort by depth for a clear, top-down view of the search tree
    # for depth, data in sorted(profiling_data.items()):
    #     nodes = data['nodes_expanded']
    #     children = data['children_generated']
    #     # Calculate the average branching factor at this level
    #     avg_branch_factor = (children / nodes) if nodes > 0 else 0

    #     print(
    #         f"Depth {depth}: "
    #         f"Nodes Expanded = {nodes}, "
    #         f"Children Generated = {children}, "
    #         f"Avg. Branch Factor = {avg_branch_factor:.2f}"
    #     )

    # Deduplicate
    unique_solutions = set(frozenset(sol) for sol in solutions)

    res = [list(sol) for sol in unique_solutions]
    return len(res), res

def process_sequential(graph, g_queue, threshold, perform_cpp, edge_dict):
    """Process patterns sequentially one by one"""

    results = []

    filter_g_queue = []

    cnt = 0
    if perform_cpp:
        #for CPython
        import graph_pattern_cpp
        def igraph_to_custom_graph(ig_graph):
            """Convert igraph.Graph to custom Graph object"""
            # Create a new custom graph object
            custom_graph = graph_pattern_cpp.Graph()  # Assuming Graph is exposed in your module

            for i in range(ig_graph.vcount()):
                # Get vertex label if it exists, otherwise use string representation of index
                label = str(ig_graph.vs[i]['label'])
                custom_graph.addVertex(i, label)

            for edge in ig_graph.es:
                source = edge.source
                target = edge.target
                custom_graph.addEdge(source, target)

            return custom_graph

        graph_cpp = igraph_to_custom_graph(graph)
        graph_cache = graph_pattern_cpp.GraphCache(graph_cpp)

    if perform_cpp == False:
        graph_cache = GraphCache(graph)

    for pattern in g_queue:

        st = time.time()
        if perform_cpp == False:
            st = time.time()
            match_count, subisomorphisms = find_pattern_matches(
                graph_cache, pattern, find_all=True, threshold=threshold, st_time = st, edge_dict = edge_dict
            )
        else:
            pattern_cpp = igraph_to_custom_graph(pattern)
            st = time.time()
            match_count, subisomorphisms = graph_pattern_cpp.find_pattern_matches(
                graph_cache, pattern_cpp, find_all=True, threshold=threshold, edge_dict = edge_dict
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

    return filter_g_queue, results,os.getpid(),mem


def run_cc_index(graph, pattern_queue, result_path, use_multiprocessing, num_workers, threshold, perform_cpp, edge_dict = None):

    total_start_time = time.time()
    print("="*50)

    with open(result_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Pattern', 'Matches', 'Time'])


    total_patterns = len(pattern_queue)
    print(f"Found {total_patterns} patterns to process")

    match_buffer = []
    g_queue_buffer = []
    mem_used_buffer = []
    pid_buffer = []

    if use_multiprocessing:
        batch_size = 10
        #num_workers = multiprocessing.cpu_count() - 1
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


    result_dir = os.path.dirname(result_path)
    os.makedirs(result_dir, exist_ok=True)
    with open(result_path, mode='a', newline='') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerows(match_buffer)
    total_time = time.time() - total_start_time

    if use_multiprocessing:
        mem_max_by_pid = {}
        for p, m in zip(pid_buffer, mem_used_buffer):
            if p not in mem_max_by_pid or m > mem_max_by_pid[p]:
                mem_max_by_pid[p] = m

    print(f"Processing complete")
    print(
        f"Processing mode: {'Parallel' if use_multiprocessing else 'Sequential'}")
    print(f"Output File: {result_path}")
    print(f"Total patterns: {total_patterns}")
    print(f"Successfully processed: {len(match_buffer)}")
    print(f"Failed: {total_patterns - len(match_buffer)}")
    print(f"Total execution time: {total_time:.2f} seconds")
    if use_multiprocessing:
        print(f'Memory consumption 6 CPUs {list(mem_max_by_pid.values())} MB')
    print("="*50)

    return g_queue_buffer
