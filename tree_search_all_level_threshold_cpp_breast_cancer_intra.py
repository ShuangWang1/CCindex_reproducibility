#!/usr/bin/env python3
"""Run all-level motif tree search on each intra breast-cancer sample.

This is the breast-cancer, per-sample counterpart of
``tree_search_all_level_threshold_cpp_0806.py``.  It reads the dataset-native
graphs made by ``breast_cancer_preprocess_intra.py`` and builds the initial
triangle vocabulary independently for every sample.  It writes result CSVs
only; it never writes the candidate-pattern GML directories used by the
original script.

Default result layout::

    breast_cancer_meta/
      BZ_res_threshold_10_cpp_0811/sample1/ccindex_tri_2.csv
      metabric_res_threshold_10_cpp_0811/sample1/ccindex_tri_2.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import resource
import time
from collections import Counter
from datetime import datetime
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Iterable

import igraph as ig
import networkx as nx

from run_ccindex import run_cc_index
from vf2 import run_vf2


DATASET_DIRS = {
    "BZ": "BZ_samples_intra",
    "metabric": "metabric_samples_intra",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--MaxDepth", type=int, default=10)
    parser.add_argument(
        "--SM_method",
        choices=("ccindex", "vf2"),
        default="ccindex",
        help="Subgraph matching implementation",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=10,
        help="Keep patterns occurring at least this many times",
    )
    parser.add_argument(
        "--SM_stop_early",
        action="store_true",
        help="For ccindex, stop counting a pattern when threshold is reached",
    )
    parser.add_argument(
        "--dataset",
        choices=("both", "BZ", "metabric"),
        default="both",
        help="Process both datasets (default) or one dataset",
    )
    parser.add_argument(
        "--samples",
        nargs="*",
        default=None,
        metavar="SAMPLE",
        help="Optional sample names/numbers, e.g. sample1 sample12 or 1 12",
    )
    parser.add_argument(
        "--start-sample",
        type=int,
        default=None,
        help="First numeric sample index to process (inclusive)",
    )
    parser.add_argument(
        "--end-sample",
        type=int,
        default=None,
        help="Last numeric sample index to process (inclusive)",
    )
    parser.add_argument(
        "--breast-cancer-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "breast_cancer",
        help="Directory containing BZ_samples_intra and metabric_samples_intra",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "breast_cancer_meta",
        help="Root directory for result CSVs",
    )
    return parser.parse_args()


def convert_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds %= 60
    return f"{hours} : {minutes} : {seconds:.2f}"


def normalize_label(value: object) -> str:
    """Normalize GML numeric labels while rejecting non-integral values."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Node label is not numeric: {value!r}") from exc
    if not number.is_integer():
        raise ValueError(f"Node label is not an integer: {value!r}")
    return str(int(number))


def load_target_graph(path: Path) -> ig.Graph:
    graph = ig.read(path)
    if "label" not in graph.vs.attributes():
        raise ValueError(f"Graph contains no node labels: {path}")
    graph.vs["label"] = [normalize_label(value) for value in graph.vs["label"]]
    # Preprocessing intentionally retains repeated Delaunay edges.  All
    # matchers in the referenced pipeline use adjacency sets, so simplifying
    # here preserves its topology and avoids redundant work.
    graph.simplify(multiple=True, loops=True)
    return graph


def triangle_frequencies(graph: ig.Graph) -> Counter[str]:
    """Count labeled three-node cliques in one target sample."""
    labels = graph.vs["label"]
    frequencies: Counter[str] = Counter()
    for vertices in graph.cliques(min=3, max=3):
        motif = "&".join(sorted((labels[index] for index in vertices), key=int))
        frequencies[motif] += 1
    return frequencies


def graph_hash(graph: ig.Graph) -> str:
    converted = nx.Graph()
    for vertex in graph.vs:
        converted.add_node(vertex.index, label=str(vertex["label"]))
    converted.add_edges_from(graph.get_edgelist())
    return nx.weisfeiler_lehman_graph_hash(converted, node_attr="label")


def construct_queue(triangles: Iterable[str]) -> list[ig.Graph]:
    queue: list[ig.Graph] = []
    for motif in triangles:
        graph = ig.Graph.Full(3)
        graph.vs["label"] = motif.split("&")
        graph["trinum"] = 1
        graph["extendableedge"] = graph.get_edgelist()
        graph["hashvalue"] = graph_hash(graph)
        queue.append(graph)
    return queue


def build_edge_pool(triangles: Iterable[str], label_count: int) -> dict[str, list[str]]:
    triangle_list = list(triangles)
    edge_pool = {
        "&".join(map(str, pair)): []
        for pair in combinations_with_replacement(range(label_count), 2)
    }
    for edge, motifs in edge_pool.items():
        left, right = edge.split("&")
        for triangle in triangle_list:
            parts = triangle.split("&")
            if (left != right and left in parts and right in parts) or (
                left == right and parts.count(left) >= 2
            ):
                motifs.append(triangle)
    return {edge: motifs for edge, motifs in edge_pool.items() if motifs}


def extend_graph(
    graph: ig.Graph,
    edge_pool: dict[str, list[str]],
    output: dict[str, ig.Graph],
) -> None:
    """Generate unique one-triangle extensions, following the 0806 search."""
    edge_list = [tuple(sorted(edge)) for edge in graph["extendableedge"]]
    labels = [str(label) for label in graph.vs["label"]]
    edge_motifs = [
        f"{min(int(labels[left]), int(labels[right]))}&"
        f"{max(int(labels[left]), int(labels[right]))}"
        for left, right in edge_list
    ]

    done: set[tuple[int, int]] = set()
    for shared_edge in edge_list:
        if shared_edge in done:
            continue
        left, right = shared_edge
        neighbor_edges = {
            tuple(sorted((neighbor, left)))
            for neighbor in graph.neighbors(left)
            if neighbor != right and not graph.are_adjacent(neighbor, right)
        }
        neighbor_edges.update(
            tuple(sorted((neighbor, right)))
            for neighbor in graph.neighbors(right)
            if neighbor != left and not graph.are_adjacent(neighbor, left)
        )
        for neighbor_edge in neighbor_edges.intersection(edge_list):
            done.add(neighbor_edge)
            endpoints = list(set(neighbor_edge).symmetric_difference(shared_edge))
            if len(endpoints) != 2:
                continue
            new_graph = graph.copy()
            new_graph.add_edge(endpoints[0], endpoints[1])
            new_graph["trinum"] += 1

            extendable = edge_list.copy()
            common = next(iter(set(neighbor_edge).intersection(shared_edge)))
            extendable.append(tuple(sorted(endpoints)))
            extendable.remove(tuple(sorted((common, endpoints[0]))))
            extendable.remove(tuple(sorted((common, endpoints[1]))))
            new_graph["extendableedge"] = extendable

            if len(new_graph.cliques(min=3, max=3)) > graph["trinum"] + 1:
                continue
            digest = graph_hash(new_graph)
            new_graph["hashvalue"] = digest
            output.setdefault(digest, new_graph)

    for edge_motif, edge in zip(edge_motifs, edge_list):
        for triangle in edge_pool.get(edge_motif, ()):
            nodes = triangle.split("&")
            triangle_edges = {
                "&".join(sorted((nodes[left], nodes[right]), key=int))
                for left, right in ((0, 1), (0, 2), (1, 2))
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
            digest = graph_hash(new_graph)
            new_graph["hashvalue"] = digest
            output.setdefault(digest, new_graph)


def extend_queue(
    queue: Iterable[ig.Graph], edge_pool: dict[str, list[str]]
) -> list[ig.Graph]:
    output: dict[str, ig.Graph] = {}
    for graph in queue:
        extend_graph(graph, edge_pool, output)
    return list(output.values())


def write_result_csv(path: Path, candidates: list[ig.Graph], triangle_count: int) -> None:
    """Replace the matcher's terse CSV with a self-describing result CSV."""
    fields = (
        "pattern_id",
        "triangles",
        "nodes",
        "edges",
        "node_labels",
        "edge_list",
        "wl_hash",
        "matches",
        "seconds",
    )
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for index, graph in enumerate(candidates, start=1):
            writer.writerow(
                {
                    "pattern_id": f"tri_{triangle_count}_{index}",
                    "triangles": triangle_count,
                    "nodes": graph.vcount(),
                    "edges": graph.ecount(),
                    "node_labels": "&".join(map(str, graph.vs["label"])),
                    "edge_list": ";".join(
                        f"{left}-{right}" for left, right in graph.get_edgelist()
                    ),
                    "wl_hash": graph["hashvalue"],
                    "matches": graph["count"],
                    "seconds": graph["time"],
                }
            )


def match_candidates(
    target: ig.Graph,
    candidates: list[ig.Graph],
    result_path: Path,
    graph_name: str,
    method: str,
    stop_early: bool,
    threshold: int,
) -> list[ig.Graph]:
    """Match candidates and write one enriched CSV without pattern GMLs."""
    result_path.parent.mkdir(parents=True, exist_ok=True)
    if not candidates:
        write_result_csv(result_path, [], 0)
        return []

    if method == "ccindex":
        matched = run_cc_index(
            target,
            candidates,
            str(result_path),
            graph_name,
            use_multiprocessing=False,
            threshold=threshold if stop_early else 0,
            perform_cpp=True,
        )
    else:
        matched = run_vf2(
            target,
            candidates,
            str(result_path),
            use_multiprocessing=False,
            num_workers=1,
        )

    triangle_count = int(candidates[0]["trinum"])
    write_result_csv(result_path, candidates, triangle_count)
    return matched


def result_directory(output_root: Path, dataset: str, args: argparse.Namespace) -> Path:
    if args.SM_stop_early:
        suffix = f"res_threshold_{args.threshold}_stop_early_cpp_0811"
    else:
        suffix = f"res_threshold_{args.threshold}_cpp_0811"
    return output_root / f"{dataset}_{suffix}"


def process_sample(
    dataset: str,
    sample_name: str,
    source_sample: str,
    graph_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    started = time.time()
    target = load_target_graph(graph_path)
    frequencies = triangle_frequencies(target)
    # Preserve the referenced script's initial strict cutoff (> threshold).
    triangles = {
        motif: count for motif, count in frequencies.items() if count > args.threshold
    }
    label_count = max((int(label) for label in target.vs["label"]), default=-1) + 1
    edge_pool = build_edge_pool(triangles, label_count)
    queue = construct_queue(triangles)
    sample_output = output_dir / sample_name
    sample_output.mkdir(parents=True, exist_ok=True)

    print(f"[{dataset}/{sample_name}] {datetime.now().isoformat(timespec='seconds')}")
    print(
        f"  target nodes={target.vcount():,}, edges={target.ecount():,}, "
        f"cell types={len(set(target.vs['label']))}, retained triangles={len(queue):,}"
    )

    completed_level = 1
    for triangle_count in range(2, args.MaxDepth + 1):
        level_started = time.time()
        candidates = extend_queue(queue, edge_pool)
        print(
            f"  tri-{triangle_count}: generated {len(candidates):,} candidates "
            f"in {convert_time(time.time() - level_started)}"
        )
        result_path = sample_output / f"{args.SM_method}_tri_{triangle_count}.csv"
        matched = match_candidates(
            target,
            candidates,
            result_path,
            f"{dataset}_{sample_name}_tri_{triangle_count}",
            args.SM_method,
            args.SM_stop_early,
            args.threshold,
        )
        queue = [
            graph
            for graph in matched
            if graph["count"] >= args.threshold or graph["count"] == -1
        ]
        completed_level = triangle_count
        memory_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        print(
            f"  tri-{triangle_count}: retained {len(queue):,}; "
            f"elapsed={convert_time(time.time() - level_started)}; "
            f"peak memory={memory_mb:.2f} MB"
        )
        if not candidates or not queue:
            print(f"  search exhausted after tri-{triangle_count}")
            break

    summary = {
        "dataset": dataset,
        "sample": sample_name,
        "source_sample": source_sample,
        "graph_file": str(graph_path),
        "nodes": target.vcount(),
        "edges": target.ecount(),
        "source_triangle_types": len(frequencies),
        "retained_triangle_types": len(triangles),
        "completed_triangle_level": completed_level,
        "seconds": time.time() - started,
    }
    with (sample_output / "run_summary.json").open("w", encoding="utf-8") as output:
        json.dump(summary, output, indent=2)
        output.write("\n")
    return summary


def sample_number(name: str) -> int:
    stem = Path(name).stem
    if not stem.startswith("sample") or not stem[6:].isdigit():
        raise ValueError(f"Expected a name like sample12.gml; found {name!r}")
    return int(stem[6:])


def requested_sample_numbers(values: list[str] | None) -> set[int] | None:
    if values is None:
        return None
    numbers = set()
    for value in values:
        normalized = Path(value).stem
        if normalized.startswith("sample"):
            normalized = normalized[6:]
        if not normalized.isdigit() or int(normalized) <= 0:
            raise ValueError(f"Invalid sample selector: {value!r}")
        numbers.add(int(normalized))
    return numbers


def select_samples(input_dir: Path, args: argparse.Namespace) -> list[Path]:
    selected = requested_sample_numbers(args.samples)
    paths = sorted(input_dir.glob("sample*.gml"), key=lambda path: sample_number(path.name))
    if selected is not None:
        paths = [path for path in paths if sample_number(path.name) in selected]
        found = {sample_number(path.name) for path in paths}
        if missing := selected - found:
            raise FileNotFoundError(f"Samples not found in {input_dir}: {sorted(missing)}")
    if args.start_sample is not None:
        paths = [path for path in paths if sample_number(path.name) >= args.start_sample]
    if args.end_sample is not None:
        paths = [path for path in paths if sample_number(path.name) <= args.end_sample]
    return paths


def read_source_samples(input_dir: Path) -> dict[str, str]:
    """Read the preprocessing manifest's GML-to-original-sample mapping."""
    manifest_path = input_dir / "sample_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Sample manifest does not exist: {manifest_path}")
    with manifest_path.open(encoding="utf-8", newline="") as source:
        rows = csv.DictReader(source)
        return {row["gml_file"]: row["source_sample"] for row in rows}


def append_run_manifest(path: Path, summaries: list[dict[str, object]]) -> None:
    if not summaries:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)


def validate_args(args: argparse.Namespace) -> None:
    if args.MaxDepth < 2:
        raise ValueError("--MaxDepth must be at least 2")
    if args.threshold < 0:
        raise ValueError("--threshold cannot be negative")
    if args.SM_stop_early and args.SM_method != "ccindex":
        raise ValueError("--SM_stop_early is supported only with --SM_method ccindex")
    if args.start_sample is not None and args.start_sample <= 0:
        raise ValueError("--start-sample must be positive")
    if args.end_sample is not None and args.end_sample <= 0:
        raise ValueError("--end-sample must be positive")
    if (
        args.start_sample is not None
        and args.end_sample is not None
        and args.start_sample > args.end_sample
    ):
        raise ValueError("--start-sample cannot be greater than --end-sample")


def main() -> None:
    args = parse_args()
    validate_args(args)
    datasets = DATASET_DIRS if args.dataset == "both" else {args.dataset: DATASET_DIRS[args.dataset]}
    total_started = time.time()

    for dataset, directory_name in datasets.items():
        input_dir = args.breast_cancer_dir / directory_name
        if not input_dir.is_dir():
            raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
        samples = select_samples(input_dir, args)
        if not samples:
            raise ValueError(f"No samples selected from {input_dir}")
        source_samples = read_source_samples(input_dir)

        output_dir = result_directory(args.output_root, dataset, args)
        print(f"Processing {len(samples):,} {dataset} samples -> {output_dir}")
        summaries = []
        for position, graph_path in enumerate(samples, start=1):
            print(f"Sample {position:,}/{len(samples):,}")
            summaries.append(
                process_sample(
                    dataset,
                    graph_path.stem,
                    source_samples.get(graph_path.name, ""),
                    graph_path,
                    output_dir,
                    args,
                )
            )
            append_run_manifest(output_dir / "run_manifest.csv", summaries)

    print(f"All selected samples completed in {convert_time(time.time() - total_started)}")


if __name__ == "__main__":
    main()
