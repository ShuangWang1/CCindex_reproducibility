"""Run one-at-a-time CC-index ablations during motif tree search.

This is intentionally independent of tree_search_all_level_threshold_cpp_0806.py.
"""

import argparse
import csv
import json
import statistics
from collections import Counter
from datetime import datetime
from itertools import combinations_with_replacement
from pathlib import Path

import igraph as ig
import networkx as nx
import pandas as pd

from ablation_matcher import VALID_VARIANTS
from run_ccindex import run_cc_index_ablation


def parse_args():
    parser = argparse.ArgumentParser(description="CC-index tree-search ablation")
    parser.add_argument("--MaxDepth", type=int, default=10)
    parser.add_argument("--threshold", type=int, default=10)
    parser.add_argument("--SM_stop_early", action="store_true")
    parser.add_argument("--LevelStartFrom", type=int, default=0)
    parser.add_argument("--GraphQueuePath", default="")
    parser.add_argument("--data_csv", default="ad_data.csv")
    parser.add_argument(
        "--target_graph", default="spatial_13months-disease-replicate_1.gml"
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VALID_VARIANTS,
        default=list(VALID_VARIANTS),
    )
    parser.add_argument("--output_dir", default="ad_ablation_res")
    parser.add_argument("--run_name", default="")
    return parser.parse_args()


def to_networkx(graph):
    converted = nx.Graph()
    for vertex in graph.vs:
        converted.add_node(vertex.index, label=str(vertex["label"]))
    converted.add_edges_from(graph.get_edgelist())
    return converted


def graph_hash(graph):
    return nx.weisfeiler_lehman_graph_hash(to_networkx(graph), node_attr="label")


def construct_queue(triangles):
    queue = []
    for motif in triangles:
        graph = ig.Graph.Full(3)
        graph.vs["label"] = motif.split("&")
        graph["trinum"] = 1
        graph["extendableedge"] = graph.get_edgelist()
        graph["hashvalue"] = graph_hash(graph)
        queue.append(graph)
    return queue


def read_queue(graph_path, level):
    level_path = Path(graph_path) / str(level)
    if not level_path.is_dir():
        raise ValueError(f"Graph queue directory does not exist: {level_path}")
    queue = []
    for path in sorted(level_path.glob("*.gml")):
        graph = ig.read(path)
        value = graph["extendableedge"]
        if isinstance(value, str):
            import ast

            value = ast.literal_eval(value)
        graph["extendableedge"] = [tuple(edge) for edge in value]
        graph["hashvalue"] = graph_hash(graph)
        queue.append(graph)
    return queue


def extend_graph(graph, edge_pool, output):
    edge_list = [tuple(edge) for edge in graph["extendableedge"]]
    labels = [str(label) for label in graph.vs["label"]]
    edge_motifs = [
        f"{min(int(labels[u]), int(labels[v]))}&{max(int(labels[u]), int(labels[v]))}"
        for u, v in edge_list
    ]

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
            new_graph = graph.copy()
            endpoints = list(set(neighbor_edge).symmetric_difference(shared_edge))
            new_graph.add_edge(endpoints[0], endpoints[1])
            new_graph["trinum"] += 1
            extendable = edge_list.copy()
            common = next(iter(set(neighbor_edge).intersection(shared_edge)))
            extendable.append(tuple(sorted(endpoints)))
            extendable.remove(tuple(sorted((common, endpoints[0]))))
            extendable.remove(tuple(sorted((common, endpoints[1]))))
            new_graph["extendableedge"] = extendable
            if len(new_graph.cliques(min=3, max=3)) > len(graph.cliques(min=3, max=3)) + 1:
                continue
            new_graph["hashvalue"] = graph_hash(new_graph)
            output.setdefault(new_graph["hashvalue"], new_graph)

    for edge_motif, edge in zip(edge_motifs, edge_list):
        for triangle in edge_pool.get(edge_motif, ()):
            nodes = triangle.split("&")
            triangle_edges = {
                f"{nodes[left]}&{nodes[right]}" for left, right in ((0, 1), (0, 2), (1, 2))
            }
            if edge_motif not in triangle_edges:
                continue
            remaining = list((Counter(nodes) - Counter(edge_motif.split("&"))).elements())
            if not remaining:
                continue
            new_graph = graph.copy()
            new_index = new_graph.vcount()
            new_graph.add_vertex(label=remaining[0])
            new_graph.add_edges(((edge[0], new_index), (edge[1], new_index)))
            new_graph["trinum"] += 1
            extendable = edge_list.copy()
            extendable.remove(edge)
            extendable.extend(((edge[0], new_index), (edge[1], new_index)))
            new_graph["extendableedge"] = extendable
            new_graph["hashvalue"] = graph_hash(new_graph)
            output.setdefault(new_graph["hashvalue"], new_graph)
    return output


def prepare_triangle_index(data_csv, threshold):
    frame = pd.read_csv(data_csv)
    original = frame.to_dict(orient="list")
    labels = sorted({label for motif in original for label in motif.split("&")})
    label_mapping = {label: str(index) for index, label in enumerate(labels)}
    triangles = {}
    for motif, values in original.items():
        if values[0] > threshold:
            triangles["&".join(label_mapping[label] for label in motif.split("&"))] = values
    edge_pool = {
        "&".join(map(str, pair)): []
        for pair in combinations_with_replacement(range(len(labels)), 2)
    }
    for edge in edge_pool:
        left, right = edge.split("&")
        for triangle in triangles:
            parts = triangle.split("&")
            if (left != right and left in parts and right in parts) or (
                left == right and parts.count(left) >= 2
            ):
                edge_pool[edge].append(triangle)
    return triangles, {edge: motifs for edge, motifs in edge_pool.items() if motifs}


def target_edge_frequencies(graph):
    frequencies = Counter()
    for edge in graph.es:
        labels = sorted((str(graph.vs[edge.source]["label"]), str(graph.vs[edge.target]["label"])))
        frequencies["_".join(labels)] += 1
    return dict(frequencies)


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_level(variant, level, rows, cache_seconds):
    times = [row["search_seconds"] for row in rows]
    ordered = sorted(times)
    p95_index = max(0, min(len(ordered) - 1, int(0.95 * len(ordered)))) if ordered else 0
    summary = {
        "variant": variant,
        "level": level,
        "candidate_count": len(rows),
        "cache_seconds": cache_seconds,
        "search_seconds": sum(times),
        "mean_pattern_seconds": statistics.mean(times) if times else 0,
        "median_pattern_seconds": statistics.median(times) if times else 0,
        "p95_pattern_seconds": ordered[p95_index] if ordered else 0,
    }
    for key in (
        "states_visited", "vertices_examined", "candidates_accepted", "rejected_label",
        "rejected_injective", "rejected_degree", "rejected_triangle_count",
        "rejected_signature", "rejected_adjacency",
    ):
        summary[key] = sum(int(row.get(key, 0)) for row in rows)
    return summary


def main():
    args = parse_args()
    if "full" not in args.variants:
        raise ValueError("--variants must include full because full controls tree pruning")
    run_name = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    config = vars(args) | {"run_name": run_name, "counting": "threshold" if args.SM_stop_early else "exact"}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    triangles, edge_pool = prepare_triangle_index(args.data_csv, args.threshold)
    queue = construct_queue(triangles) if args.LevelStartFrom == 0 else read_queue(
        args.GraphQueuePath, args.LevelStartFrom
    )
    target = ig.read(args.target_graph)
    target.vs["label"] = [str(int(label)) for label in target.vs["label"]]
    edge_dict = target_edge_frequencies(target)
    effective_threshold = args.threshold if args.SM_stop_early else 0
    pattern_rows = []
    level_rows = []

    for loop_level in range(args.LevelStartFrom, args.MaxDepth - 1):
        level = loop_level + 2
        generated = {}
        for graph in queue:
            extend_graph(graph, edge_pool, generated)
        candidates = list(generated.values())
        for position, graph in enumerate(candidates):
            graph["ablation_level"] = level
            graph["ablation_id"] = f"L{level}-{position:07d}-{graph['hashvalue']}"

        variant_results = {}
        for variant in args.variants:
            result_path = run_dir / variant / f"ccindex_tri_{level}.csv"
            matched, rows, cache_seconds = run_cc_index_ablation(
                target, candidates, variant, result_path, effective_threshold, edge_dict
            )
            variant_results[variant] = matched
            pattern_rows.extend(rows)
            level_rows.append(summarize_level(variant, level, rows, cache_seconds))

        reference = {graph["ablation_id"]: graph["count"] for graph in variant_results["full"]}
        for variant, matched in variant_results.items():
            observed = {graph["ablation_id"]: graph["count"] for graph in matched}
            if observed != reference:
                raise RuntimeError(f"Count mismatch at level {level}: full versus {variant}")

        queue = [
            graph for graph in variant_results["full"] if graph["count"] >= args.threshold
        ]
        write_csv(run_dir / "pattern_results.csv", pattern_rows)
        write_csv(run_dir / "level_summary.csv", level_rows)
        print(f"level={level} candidates={len(candidates)} retained={len(queue)}")
        if not queue:
            break


if __name__ == "__main__":
    main()
