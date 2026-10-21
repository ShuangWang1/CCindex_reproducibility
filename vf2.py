
import igraph as ig

import warnings

from collections import Counter
import time
from datetime import datetime
import os
import csv
from concurrent.futures import ProcessPoolExecutor
import resource

warnings.filterwarnings("ignore")

def retrieve_multiple_edges(graph, source=-1, target=-1):
    if source != -1:
        e = graph.incident(source, mode=ig.OUT)
        if target != -1:
            e = set(e).intersection(graph.incident(target, mode=ig.IN))
        return ig.EdgeSeq(graph, e)
    else:
        if target != -1:
            e = graph.incident(target, mode=ig.IN)
        else:
            e = list()
        return ig.EdgeSeq(graph, e)

class PatternChecker(object):
    def __init__(self):
        pass

    @classmethod
    def node_compat_fn(cls, g1, g2, v1, v2):
        vertex1 = g1.vs[v1]
        vertex2 = g2.vs[v2]
        return vertex1["label"] == vertex2["label"]

    @classmethod
    def edge_compat_fn(cls, g1, g2, e1, e2):
        edge1 = g1.es[e1]
        edge2 = g2.es[e2]
        if edge1.is_loop() != edge2.is_loop():
            return False
        # for multiedges
        edges1 = retrieve_multiple_edges(g1, edge1.source, edge1.target)
        edges2 = retrieve_multiple_edges(g2, edge2.source, edge2.target)
        if len(edges1) < len(edges2):
            return False
        edge1_labels = set(edges1["label"])
        for el in edges2["label"]:
            if el not in edge1_labels:
                return False
        return True

    @classmethod
    def get_vertex_color_vectors(cls, g1, g2, seed_v1=-1, seed_v2=-1):
        N1 = g1.vcount()
        N2 = g2.vcount()
        color_vectors = list()
        if seed_v1 == -1 and seed_v2 == -1:
            color_vectors.append((None, None))
        elif seed_v1 == -1 and seed_v2 != -1:
            vertex = g2.vs[seed_v2]
            seed_label = vertex["label"]
            for seed_v1, vertex in enumerate(g1.vs):
                if vertex["label"] == seed_label:
                    color1 = [0] * N1
                    color1[seed_v1] = 1
                    color2 = [0] * N2
                    color2[seed_v2] = 1
                    color_vectors.append((color1, color2))
        elif seed_v1 != -1 and seed_v2 == -1:
            seed_label = g1.vs[seed_v1]["label"]
            for seed_v2, vertex in enumerate(g2.vs):
                if vertex["label"] == seed_label:
                    color1 = [0] * N1
                    color1[seed_v1] = 1
                    color2 = [0] * N2
                    color2[seed_v2] = 1
                    color_vectors.append((color1, color2))
        else:  # seed_v1 != -1 and seed_v2 != -1:
            if g1.vs[seed_v1]["label"] == g2.vs[seed_v2]["label"]:
                color1 = [0] * N1
                color1[seed_v1] = 1
                color2 = [0] * N2
                color2[seed_v2] = 1
                color_vectors.append((color1, color2))
        return color_vectors

    @classmethod
    def get_edge_color_vectors(cls, g1, g2, seed_e1=-1, seed_e2=-1):
        E1 = len(g1.es)
        E2 = len(g2.es)
        edge_color_vectors = list()
        if seed_e1 == -1 and seed_e2 == -1:
            edge_color_vectors.append((None, None))
        elif seed_e1 == -1 and seed_e2 != -1:
            edge = g2.es[seed_e2]
            color2 = [0] * E2
            color2[seed_e2] = 1
            seed_label = edge["label"]
            is_loop = edge.is_loop()
            for seed_e1, edge in enumerate(g1.es):
                if edge["label"] == seed_label and is_loop == edge.is_loop():
                    color1 = [0] * E1
                    color1[seed_e1] = 1
                    edge_color_vectors.append((color1, color2))
        elif seed_e1 != -1 and seed_e2 == -1:
            edge = g1.es[seed_e1]
            color1 = [0] * E1
            color1[seed_e1] = 1
            seed_label = edge["label"]
            is_loop = edge.is_loop()
            for seed_e2, edge in enumerate(g2.es):
                if edge["label"] == seed_label and is_loop == edge.is_loop():
                    color2 = [0] * E2
                    color2[seed_e2] = 1
                    edge_color_vectors.append((color1, color2))
        else:  # seed_e1 != -1 and seed_e2 != -1:
            edge1 = g1.es[seed_e1]
            edge2 = g2.es[seed_e2]
            color1 = [0] * E1
            color1[seed_e1] = 1
            color2 = [0] * E2
            color2[seed_e2] = 1
            if edge1["label"] == edge2["label"] and edge1.is_loop() == edge2.is_loop():
                edge_color_vectors.append((color1, color2))
        return edge_color_vectors

    def check(self, graph, pattern, **kw):
        # valid or not
        if graph.vcount() < pattern.vcount():
            return False
        if graph.ecount() < pattern.ecount():
            return False

        graph_vlabels = Counter(graph.vs["label"])
        pattern_vlabels = Counter(pattern.vs["label"])
        if len(graph_vlabels) < len(pattern_vlabels):
            return False
        for vertex_label, pv_cnt in pattern_vlabels.most_common():
            diff = graph_vlabels[vertex_label] - pv_cnt
            if diff < 0:
                return False

        graph_elabels = Counter(graph.es["label"])
        pattern_elabels = Counter(pattern.es["label"])
        if len(graph_elabels) < len(pattern_elabels):
            return False
        for edge_label, pe_cnt in pattern_elabels.most_common():
            diff = graph_elabels[edge_label] - pe_cnt
            if diff < 0:
                return False
        return True

    def get_subisomorphisms(self, graph, pattern, **kw):
        if not self.check(graph, pattern):
            return list()

        seed_v1 = kw.get("seed_v1", -1)
        seed_v2 = kw.get("seed_v2", -1)
        seed_e1 = kw.get("seed_e1", -1)
        seed_e2 = kw.get("seed_e2", -1)

        vertex_color_vectors = PatternChecker.get_vertex_color_vectors(graph, pattern, seed_v1=seed_v1, seed_v2=seed_v2)
        edge_color_vectors = PatternChecker.get_edge_color_vectors(graph, pattern, seed_e1=seed_e1, seed_e2=seed_e2)

        vertices_in_graph = list()
        if seed_v1 != -1:
            vertices_in_graph.append(seed_v1)
        if seed_e1 != -1:
            vertices_in_graph.extend(graph.es[seed_e1].tuple)
        subisomorphisms = list()  # [(component, mapping), ...]
        '''
        for vertex_colors in vertex_color_vectors:
            for edge_colors in edge_color_vectors:
                for subisomorphism in graph.get_subisomorphisms_vf2(pattern,
                                                                    color1=vertex_colors[0], color2=vertex_colors[1],
                                                                    edge_color1=edge_colors[0],
                                                                    edge_color2=edge_colors[1],
                                                                    node_compat_fn=PatternChecker.node_compat_fn,
                                                                    edge_compat_fn=PatternChecker.edge_compat_fn):
                    if len(vertices_in_graph) == 0 or all([v in subisomorphism for v in vertices_in_graph]):
                        subisomorphisms.append(subisomorphism)
        '''
        for subisomorphism in graph.get_subisomorphisms_vf2(pattern,
                                                                    node_compat_fn=PatternChecker.node_compat_fn,
                                                                    edge_compat_fn=PatternChecker.edge_compat_fn):
                    if len(vertices_in_graph) == 0 or all([v in subisomorphism for v in vertices_in_graph]):
                        subisomorphisms.append(subisomorphism)
        return subisomorphisms

    def count_subisomorphisms(self, graph, pattern, **kw):
        if not self.check(graph, pattern):
            return 0

        seed_v1 = kw.get("seed_v1", -1)
        seed_v2 = kw.get("seed_v2", -1)
        seed_e1 = kw.get("seed_e1", -1)
        seed_e2 = kw.get("seed_e2", -1)

        vertex_color_vectors = PatternChecker.get_vertex_color_vectors(graph, pattern, seed_v1=seed_v1, seed_v2=seed_v2)
        edge_color_vectors = PatternChecker.get_edge_color_vectors(graph, pattern, seed_e1=seed_e1, seed_e2=seed_e2)

        vertices_in_graph = list()
        if seed_v1 != -1:
            vertices_in_graph.append(seed_v1)
        if seed_e1 != -1:
            vertices_in_graph.extend(graph.es[seed_e1].tuple)
        if len(vertices_in_graph) == 0:
            counts = 0
            for vertex_colors in vertex_color_vectors:
                for edge_colors in edge_color_vectors:
                    counts += graph.count_subisomorphisms_vf2(pattern,
                                                              color1=vertex_colors[0], color2=vertex_colors[1],
                                                              edge_color1=edge_colors[0], edge_color2=edge_colors[1],
                                                              node_compat_fn=PatternChecker.node_compat_fn,
                                                              edge_compat_fn=PatternChecker.edge_compat_fn)
            return counts
        else:
            counts = 0
            for vertex_colors in vertex_color_vectors:
                for edge_colors in edge_color_vectors:
                    for subisomorphism in graph.get_subisomorphisms_vf2(pattern,
                                                                        color1=vertex_colors[0],
                                                                        color2=vertex_colors[1],
                                                                        edge_color1=edge_colors[0],
                                                                        edge_color2=edge_colors[1],
                                                                        node_compat_fn=PatternChecker.node_compat_fn,
                                                                        edge_compat_fn=PatternChecker.edge_compat_fn):
                        if all([v in subisomorphism for v in vertices_in_graph]):
                            counts += 1
            return counts



def process_nested_list_column(column):


    sorted_list = [tuple(sorted(sublist)) for sublist in column]

    unique_sorted_list = list(set(sorted_list))

    result = [list(item) for item in sorted(unique_sorted_list)]
    return result

def vf2_count(graph, g_queue):
    results = []
    filter_g_queue = []

    cnt = 0
    for pattern in g_queue:
        pattern.es['label'] = [0 for i in pattern.es]

        pc = PatternChecker()
        st = time.time()
        subisomorphisms = pc.get_subisomorphisms(graph, pattern)
        et = time.time()


        subisomorphisms_unique = process_nested_list_column(subisomorphisms)

        pattern['count'] = len(subisomorphisms_unique)
        pattern['mtf_ids'] = subisomorphisms_unique
        pattern['time'] = et - st
        cnt += 1

        pattern_name = f'{len(pattern.cliques(min=3, max=3))}_{cnt}'
        match_count = len(subisomorphisms_unique)
        results.append([pattern_name, match_count, et - st])

        if match_count > 0:
            filter_g_queue.append(pattern)

    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    return filter_g_queue, results, os.getpid(), mem

def run_vf2(graph, pattern_queue, result_path, use_multiprocessing, num_workers):

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
        batch_size = 100
        print(f"Starting parallel processing with {num_workers} workers...")
        pattern_batches = [pattern_queue[i:i+batch_size]
                           for i in range(0, len(pattern_queue), batch_size)]

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(vf2_count, graph, pattern_queue)
                    for pattern_queue in pattern_batches]

            for future in futures:
                g_queue_res, match_res, pid, mem_used = future.result()
                match_buffer.extend(match_res)
                g_queue_buffer.extend(g_queue_res)
                mem_used_buffer.append(mem_used)
                pid_buffer.append(pid)


        mem_max_by_pid = {}
        for p, m in zip(pid_buffer, mem_used_buffer):
            if p not in mem_max_by_pid or m > mem_max_by_pid[p]:
                mem_max_by_pid[p] = m

        print(f'Memory consumption 6 CPUs {list(mem_max_by_pid.values())} MB')

    else:
        print("Starting sequential processing...")
        g_queue_buffer, match_buffer,pid_buffer, mem_used_buffer  = vf2_count(graph, pattern_queue)

    result_dir = os.path.dirname(result_path)
    os.makedirs(result_dir, exist_ok=True)
    with open(result_path, mode='a', newline='') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerows(match_buffer)
    total_time = time.time() - total_start_time


    total_time = time.time() - total_start_time
    print(f"Processing complete")
    print(
        f"Processing mode: {'Parallel' if use_multiprocessing else 'Sequential'}")
    print(f"Output File: {result_path}")
    print(f"Total patterns: {total_patterns}")
    print(f"Successfully processed: {len(match_buffer)}")
    print(f"Failed: {total_patterns - len(match_buffer)}")
    print(f"Total execution time: {total_time:.2f} seconds")
    print("="*50)

    return g_queue_buffer
