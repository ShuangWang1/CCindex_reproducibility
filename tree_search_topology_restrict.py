#!/usr/bin/env python3
"""Thresholded motif tree search with root-centered BFS topology growth.

This is the topology-restricted counterpart of
``tree_search_all_level_threshold_cpp_0806.py``.  The initial TRI-1 face is
the root.  A child may only add a triangle in the nearest face-hop shell that
still has a valid growth operation, preventing a branch from growing outward
on one side while a nearer attachment remains elsewhere.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import resource
import time
from ast import literal_eval
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations, combinations_with_replacement
from pathlib import Path
from typing import Iterable, Sequence

# Keep Matplotlib/igraph caches in a writable location on this workstation.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/tree_search_topology_restrict_mpl")

import igraph as ig
import networkx as nx
import pandas as pd
import graph_pattern_cpp

from enumerate_topology.enumerate_topologies import topology_layout
from vf2 import run_vf2


Edge = tuple[int, int]
Triangle = tuple[int, int, int]


def normalize_edge(edge: Iterable[int]) -> Edge:
    left, right = edge
    return (min(int(left), int(right)), max(int(left), int(right)))


def parse_root_nodes(graph: ig.Graph) -> Triangle:
    if "rootnodes" not in graph.attributes():
        raise ValueError("graph is missing rootnodes metadata")
    raw = graph["rootnodes"]
    if isinstance(raw, str):
        values = tuple(int(value) for value in raw.split(",") if value != "")
    else:
        values = tuple(int(value) for value in raw)
    if len(values) != 3:
        raise ValueError(f"rootnodes must contain three vertices, found {values}")
    root = tuple(sorted(values))
    if any(vertex < 0 or vertex >= graph.vcount() for vertex in root):
        raise ValueError(f"rootnodes contains an invalid vertex: {root}")
    return root  # type: ignore[return-value]


def convert_to_nx(graph: ig.Graph, *, rooted: bool = False) -> nx.Graph:
    topology = nx.Graph()
    root = set(parse_root_nodes(graph)) if rooted else set()
    for vertex, label in enumerate(graph.vs["label"]):
        value = str(label)
        topology.add_node(
            vertex,
            label=value,
            rooted_label=f"{value}|{int(vertex in root)}",
        )
    topology.add_edges_from(graph.get_edgelist())
    return topology


def motif_hash(graph: ig.Graph) -> str:
    """Match the reference search's label-aware WL deduplication key."""
    return nx.weisfeiler_lehman_graph_hash(convert_to_nx(graph), node_attr="label")


def rooted_motif_hash(graph: ig.Graph) -> str:
    """Deduplicate growth states while preserving the distinguished root face."""
    return nx.weisfeiler_lehman_graph_hash(
        convert_to_nx(graph, rooted=True), node_attr="rooted_label"
    )


@dataclass(frozen=True)
class FaceContext:
    hops: dict[Triangle, int]
    boundary_owner: dict[Edge, Triangle]


def face_context(graph: ig.Graph) -> FaceContext:
    """Return root distances in the triangle dual and each boundary edge owner."""
    triangles = [tuple(sorted(face)) for face in graph.cliques(min=3, max=3)]
    root = parse_root_nodes(graph)
    if root not in triangles:
        raise ValueError(f"root face {root} is not a triangle in the graph")

    dual = nx.Graph()
    dual.add_nodes_from(triangles)
    edge_faces: dict[Edge, list[Triangle]] = {}
    for face in triangles:
        for edge in combinations(face, 2):
            edge_faces.setdefault(normalize_edge(edge), []).append(face)
    for owners in edge_faces.values():
        for left, right in combinations(owners, 2):
            dual.add_edge(left, right)

    hops = dict(nx.single_source_shortest_path_length(dual, root))
    if len(hops) != len(triangles):
        raise ValueError("triangle dual is disconnected from the root face")
    boundary_owner = {
        edge: owners[0] for edge, owners in edge_faces.items() if len(owners) == 1
    }
    return FaceContext(hops=hops, boundary_owner=boundary_owner)


@dataclass
class GrowthCandidate:
    graph: ig.Graph
    hop: int


def _finalize_candidate(candidate: ig.Graph, hop: int) -> GrowthCandidate:
    candidate["newtrianglehop"] = int(hop)
    candidate["maxfacehop"] = max(face_context(candidate).hops.values())
    candidate["hashvalue"] = motif_hash(candidate)
    candidate["rootedhash"] = rooted_motif_hash(candidate)
    return GrowthCandidate(candidate, hop)


def growth_candidates(
    graph: ig.Graph, edge_pool: dict[str, list[str]]
) -> list[GrowthCandidate]:
    """Build all valid children, then retain only the nearest available shell."""
    edge_list = [normalize_edge(edge) for edge in graph["extendableedge"]]
    context = face_context(graph)
    missing = sorted(set(edge_list) - set(context.boundary_owner))
    if missing:
        raise ValueError(f"extendableedge contains non-boundary edges: {missing}")

    old_triangle_count = len(context.hops)
    candidates: list[GrowthCandidate] = []

    # Reference operation 1: close a wedge formed by consecutive boundary edges.
    done: set[Edge] = set()
    for shared_edge in edge_list:
        if shared_edge in done:
            continue
        left, right = shared_edge
        neighboring_edges = {
            normalize_edge((neighbor, left))
            for neighbor in graph.neighbors(left)
            if neighbor != right and not graph.are_adjacent(neighbor, right)
        }
        neighboring_edges.update(
            normalize_edge((neighbor, right))
            for neighbor in graph.neighbors(right)
            if neighbor != left and not graph.are_adjacent(neighbor, left)
        )
        for neighboring_edge in sorted(neighboring_edges.intersection(edge_list)):
            done.add(neighboring_edge)
            endpoints = sorted(set(neighboring_edge).symmetric_difference(shared_edge))
            if len(endpoints) != 2:
                continue
            candidate = graph.copy()
            candidate.add_edge(endpoints[0], endpoints[1])
            candidate["trinum"] = int(graph["trinum"]) + 1

            new_boundary = list(edge_list)
            center = next(iter(set(neighboring_edge).intersection(shared_edge)))
            new_edge = normalize_edge(endpoints)
            new_boundary.append(new_edge)
            new_boundary.remove(normalize_edge((center, endpoints[0])))
            new_boundary.remove(normalize_edge((center, endpoints[1])))
            candidate["extendableedge"] = new_boundary

            if len(candidate.cliques(min=3, max=3)) != old_triangle_count + 1:
                continue
            hop = 1 + min(
                context.hops[context.boundary_owner[shared_edge]],
                context.hops[context.boundary_owner[neighboring_edge]],
            )
            candidates.append(_finalize_candidate(candidate, hop))

    # Reference operation 2: attach a new vertex to a boundary edge.
    labels = [str(label) for label in graph.vs["label"]]
    for boundary_edge in edge_list:
        left, right = boundary_edge
        edge_motif = f"{min(int(labels[left]), int(labels[right]))}&{max(int(labels[left]), int(labels[right]))}"
        for triangle_motif in edge_pool.get(edge_motif, []):
            triangle_labels = triangle_motif.split("&")
            remainder = list(
                (Counter(triangle_labels) - Counter(edge_motif.split("&"))).elements()
            )
            if len(remainder) != 1:
                raise ValueError(
                    f"triangle {triangle_motif} does not extend edge {edge_motif}"
                )
            candidate = graph.copy()
            new_vertex = candidate.vcount()
            candidate.add_vertex(label=remainder[0])
            candidate.add_edges(((left, new_vertex), (right, new_vertex)))
            candidate["trinum"] = int(graph["trinum"]) + 1
            new_boundary = list(edge_list)
            new_boundary.remove(boundary_edge)
            new_boundary.extend(
                (normalize_edge((left, new_vertex)), normalize_edge((right, new_vertex)))
            )
            candidate["extendableedge"] = new_boundary
            hop = context.hops[context.boundary_owner[boundary_edge]] + 1
            candidates.append(_finalize_candidate(candidate, hop))

    if not candidates:
        return []
    nearest_hop = min(candidate.hop for candidate in candidates)
    unique: dict[str, ig.Graph] = {}
    for candidate in candidates:
        if candidate.hop == nearest_hop:
            unique.setdefault(candidate.graph["rootedhash"], candidate.graph)
    return [GrowthCandidate(graph=value, hop=nearest_hop) for value in unique.values()]


def expand_level(
    parents: Sequence[ig.Graph], edge_pool: dict[str, list[str]]
) -> list[ig.Graph]:
    children: dict[str, ig.Graph] = {}
    for parent in parents:
        for candidate in growth_candidates(parent, edge_pool):
            children.setdefault(candidate.graph["rootedhash"], candidate.graph)
    return list(children.values())


def construct_queue(filter_tri_dict: dict[str, Sequence[int]]) -> list[ig.Graph]:
    queue: list[ig.Graph] = []
    for motif, values in filter_tri_dict.items():
        labels = motif.split("&")
        graph = ig.Graph(n=3, edges=((0, 1), (0, 2), (1, 2)))
        graph.vs["label"] = labels
        graph["trinum"] = 1
        graph["extendableedge"] = [normalize_edge(edge) for edge in graph.get_edgelist()]
        graph["rootnodes"] = "0,1,2"
        graph["newtrianglehop"] = 0
        graph["maxfacehop"] = 0
        graph["count"] = int(values[0])
        graph["hashvalue"] = motif_hash(graph)
        graph["rootedhash"] = rooted_motif_hash(graph)
        queue.append(graph)
    return queue


def build_search_vocabulary(
    dataframe: pd.DataFrame, threshold: int
) -> tuple[dict[str, Sequence[int]], dict[str, list[str]], dict[str, str]]:
    original = dataframe.to_dict(orient="list")
    filtered = {key: value for key, value in original.items() if value[0] > threshold}
    cell_types = sorted(
        {cell_type for motif in original for cell_type in motif.split("&")}
    )
    label_to_index = {label: str(index) for index, label in enumerate(cell_types)}
    index_to_label = {value: key for key, value in label_to_index.items()}

    indexed: dict[str, Sequence[int]] = {}
    for motif, values in filtered.items():
        indexed["&".join(label_to_index[cell] for cell in motif.split("&"))] = values

    edge_pool = {
        "&".join(map(str, sorted(pair))): []
        for pair in combinations_with_replacement(range(len(cell_types)), 2)
    }
    for edge in edge_pool:
        required = Counter(edge.split("&"))
        for triangle in indexed:
            available = Counter(triangle.split("&"))
            if not (required - available):
                edge_pool[edge].append(triangle)
    edge_pool = {key: value for key, value in edge_pool.items() if value}
    return indexed, edge_pool, index_to_label


def group_unrooted_states(
    states: Sequence[ig.Graph],
) -> tuple[list[ig.Graph], dict[str, list[ig.Graph]]]:
    groups: dict[str, list[ig.Graph]] = {}
    for graph in states:
        key = motif_hash(graph)
        graph["hashvalue"] = key
        groups.setdefault(key, []).append(graph)
    representatives = [groups[key][0] for key in sorted(groups)]
    return representatives, groups


def _write_match_rows(path: Path, rows: Sequence[Sequence[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(
            f"refusing to append duplicate results to existing file: {path}"
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("Pattern", "Matches", "Time"))
        writer.writerows(rows)


def _existing_match_counts_equal(
    path: Path, rows: Sequence[Sequence[object]]
) -> bool:
    if not path.exists():
        return False
    with path.open(encoding="utf-8", newline="") as handle:
        existing = list(csv.DictReader(handle))
    if len(existing) != len(rows):
        return False
    return all(
        old["Pattern"] == str(new[0]) and int(old["Matches"]) == int(new[1])
        for old, new in zip(existing, rows)
    )


def igraph_to_custom_graph(graph: ig.Graph) -> object:
    custom = graph_pattern_cpp.Graph()
    for vertex, label in enumerate(graph.vs["label"]):
        custom.addVertex(vertex, str(label))
    for edge in graph.es:
        custom.addEdge(edge.source, edge.target)
    return custom


def _ccindex_worker_batch(
    target: ig.Graph, patterns: Sequence[ig.Graph], threshold: int
) -> list[tuple[str, int, float]]:
    """Count one independent batch in a worker while retaining no match lists."""
    target_cache = graph_pattern_cpp.GraphCache(igraph_to_custom_graph(target))
    results: list[tuple[str, int, float]] = []
    for pattern in patterns:
        started = time.time()
        custom_pattern = igraph_to_custom_graph(pattern)
        if threshold:
            match_count, matches = graph_pattern_cpp.find_pattern_matches(
                target_cache,
                custom_pattern,
                find_all=True,
                threshold=threshold,
                edge_dict=None,
            )
            del matches
        else:
            match_count = graph_pattern_cpp.count_pattern_matches(
                target_cache, custom_pattern
            )
        elapsed = time.time() - started
        results.append((str(pattern["hashvalue"]), int(match_count), elapsed))
    return results


def _ccindex_match_patterns(
    target: ig.Graph,
    patterns: Sequence[ig.Graph],
    *,
    threshold: int,
    result_path: Path,
    workers: int = 1,
) -> list[list[object]]:
    """Count patterns exactly without retaining every matched vertex set.

    The inherited matcher stores all subisomorphism lists on every pattern,
    causing later levels to accumulate several GiB.  Counts are checkpointed
    by motif hash and each returned match list is released immediately.
    """
    checkpoint_path = result_path.with_suffix(result_path.suffix + ".checkpoint")
    completed: dict[str, tuple[int, float]] = {}
    if checkpoint_path.exists():
        with checkpoint_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                completed[row["WLHash"]] = (int(row["Matches"]), float(row["Time"]))

    result_path.parent.mkdir(parents=True, exist_ok=True)
    new_checkpoint = not checkpoint_path.exists()
    checkpoint_handle = checkpoint_path.open("a", encoding="utf-8", newline="")
    checkpoint_writer = csv.DictWriter(
        checkpoint_handle, fieldnames=("WLHash", "Matches", "Time")
    )
    if new_checkpoint:
        checkpoint_writer.writeheader()
        checkpoint_handle.flush()

    try:
        remaining = [
            pattern for pattern in patterns if str(pattern["hashvalue"]) not in completed
        ]
        newly_completed = 0
        if workers <= 1:
            target_cache = graph_pattern_cpp.GraphCache(igraph_to_custom_graph(target))
            for pattern in remaining:
                key = str(pattern["hashvalue"])
                started = time.time()
                custom_pattern = igraph_to_custom_graph(pattern)
                if threshold:
                    match_count, matches = graph_pattern_cpp.find_pattern_matches(
                        target_cache,
                        custom_pattern,
                        find_all=True,
                        threshold=threshold,
                        edge_dict=None,
                    )
                    del matches
                else:
                    match_count = graph_pattern_cpp.count_pattern_matches(
                        target_cache, custom_pattern
                    )
                elapsed = time.time() - started
                completed[key] = (int(match_count), elapsed)
                checkpoint_writer.writerow(
                    {"WLHash": key, "Matches": int(match_count), "Time": elapsed}
                )
                checkpoint_handle.flush()
                newly_completed += 1
                total_done = len(completed)
                if total_done % 500 == 0 or total_done == len(patterns):
                    memory_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                    print(
                        f"    matched {total_done:,}/{len(patterns):,} patterns; "
                        f"peak RSS={memory_mb:.1f} MB",
                        flush=True,
                    )
        elif remaining:
            batch_size = 50
            batches = [
                remaining[index : index + batch_size]
                for index in range(0, len(remaining), batch_size)
            ]
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(_ccindex_worker_batch, target, batch, threshold)
                    for batch in batches
                ]
                for future in as_completed(futures):
                    for key, match_count, elapsed in future.result():
                        completed[key] = (match_count, elapsed)
                        checkpoint_writer.writerow(
                            {"WLHash": key, "Matches": match_count, "Time": elapsed}
                        )
                        newly_completed += 1
                    checkpoint_handle.flush()
                    total_done = len(completed)
                    if total_done // 500 != (total_done - len(future.result())) // 500 or total_done == len(patterns):
                        memory_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                        print(
                            f"    matched {total_done:,}/{len(patterns):,} patterns "
                            f"with {workers} workers; coordinator peak RSS={memory_mb:.1f} MB",
                            flush=True,
                        )
    finally:
        checkpoint_handle.close()

    rows: list[list[object]] = []
    for index, pattern in enumerate(patterns):
        key = str(pattern["hashvalue"])
        match_count, elapsed = completed[key]
        pattern["count"] = int(match_count)
        pattern["time"] = float(elapsed)
        rows.append([f"{int(pattern['trinum'])}_{index}", int(match_count), elapsed])

    if result_path.exists():
        if not _existing_match_counts_equal(result_path, rows):
            raise FileExistsError(
                f"existing result counts disagree with checkpoint: {result_path}"
            )
    else:
        _write_match_rows(result_path, rows)
    checkpoint_path.unlink()
    return rows


def match_and_prune(
    target: ig.Graph,
    rooted_states: Sequence[ig.Graph],
    *,
    method: str,
    threshold: int,
    stop_early: bool,
    result_path: Path,
    workers: int = 1,
) -> tuple[list[ig.Graph], int]:
    representatives, groups = group_unrooted_states(rooted_states)
    if not representatives:
        _write_match_rows(result_path, [])
        return [], 0

    if method == "ccindex":
        matcher_threshold = threshold if stop_early else 0
        _ccindex_match_patterns(
            target,
            representatives,
            threshold=matcher_threshold,
            result_path=result_path,
            workers=workers,
        )
    elif method == "vf2":
        if result_path.exists():
            raise FileExistsError(f"result file already exists: {result_path}")
        run_vf2(target, representatives, str(result_path), False, 1)
    else:
        raise ValueError("SM_method must be 'ccindex' or 'vf2'")

    surviving: list[ig.Graph] = []
    for representative in representatives:
        count = int(representative["count"])
        if count < threshold and count != -1:
            continue
        for state in groups[representative["hashvalue"]]:
            state["count"] = count
            state["time"] = float(representative["time"])
            if "mtf_ids" in state.attributes():
                del state["mtf_ids"]
            surviving.append(state)
    return surviving, len(representatives)


def write_queue(states: Sequence[ig.Graph], level: int, output_root: Path) -> dict[str, Path]:
    level_dir = output_root / str(level)
    if level_dir.exists() and any(level_dir.iterdir()):
        raise FileExistsError(f"queue directory is not empty: {level_dir}")
    level_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for index, graph in enumerate(sorted(states, key=lambda item: item["rootedhash"])):
        graph.vs["id"] = list(range(graph.vcount()))
        boundary = graph["extendableedge"]
        graph["extendableedge"] = repr([normalize_edge(edge) for edge in boundary])
        path = level_dir / f"{index:06d}_{graph['rootedhash'][:12]}.gml"
        graph.write_gml(str(path))
        graph["extendableedge"] = boundary
        paths[graph["rootedhash"]] = path
    return paths


def read_queue(graph_path: Path, level: int) -> list[ig.Graph]:
    level_dir = graph_path / str(level)
    if not level_dir.is_dir():
        raise FileNotFoundError(f"queue level does not exist: {level_dir}")
    queue: list[ig.Graph] = []
    for path in sorted(level_dir.glob("*.gml")):
        graph = ig.Graph.Read_GML(str(path))
        if "rootnodes" not in graph.attributes():
            raise ValueError(f"{path} predates root-aware topology restriction")
        graph["extendableedge"] = [
            normalize_edge(edge) for edge in literal_eval(graph["extendableedge"])
        ]
        parse_root_nodes(graph)
        graph["hashvalue"] = motif_hash(graph)
        graph["rootedhash"] = rooted_motif_hash(graph)
        queue.append(graph)
    if not queue:
        raise ValueError(f"no GML queue files found in {level_dir}")
    return queue


def exact_topology_key(graph: ig.Graph) -> str:
    """Exact canonical unlabeled topology identifier for sample diversity."""
    topology = ig.Graph(n=graph.vcount(), edges=graph.get_edgelist(), directed=False)
    permutation = topology.canonical_permutation()
    edges = sorted(
        (min(permutation[left], permutation[right]), max(permutation[left], permutation[right]))
        for left, right in topology.get_edgelist()
    )
    payload = json.dumps((topology.vcount(), edges), separators=(",", ":"))
    return hashlib.blake2b(payload.encode("ascii"), digest_size=10).hexdigest()


def select_diverse_examples(states: Sequence[ig.Graph], count: int) -> list[ig.Graph]:
    """Select count-spanning examples, preferring a new topology each time."""
    representatives, _ = group_unrooted_states(states)
    ordered = sorted(
        representatives,
        key=lambda graph: (int(graph["count"]), graph["hashvalue"], graph["rootedhash"]),
    )
    count = min(max(0, count), len(ordered))
    if count == 0:
        return []
    targets = [round(index * (len(ordered) - 1) / max(1, count - 1)) for index in range(count)]
    selected: list[ig.Graph] = []
    used_hashes: set[str] = set()
    used_topologies: set[str] = set()
    for target in targets:
        candidates = [
            (abs(index - target), exact_topology_key(graph), graph["hashvalue"], graph)
            for index, graph in enumerate(ordered)
            if graph["hashvalue"] not in used_hashes
            and exact_topology_key(graph) not in used_topologies
        ]
        if not candidates:
            candidates = [
                (abs(index - target), exact_topology_key(graph), graph["hashvalue"], graph)
                for index, graph in enumerate(ordered)
                if graph["hashvalue"] not in used_hashes
            ]
        _, topology_key, motif_key, chosen = min(candidates, key=lambda value: value[:3])
        selected.append(chosen)
        used_hashes.add(motif_key)
        used_topologies.add(topology_key)
    return selected


def select_multitype_examples(states: Sequence[ig.Graph], count: int) -> list[ig.Graph]:
    """Prefer diverse motifs with the largest number of distinct cell types."""
    representatives, _ = group_unrooted_states(states)
    multi_type = [
        graph for graph in representatives if len(set(graph.vs["label"])) >= 2
    ]
    selected: list[ig.Graph] = []
    used_hashes: set[str] = set()
    distinct_counts = sorted(
        {len(set(graph.vs["label"])) for graph in multi_type}, reverse=True
    )
    for distinct_count in distinct_counts:
        remaining = count - len(selected)
        if remaining <= 0:
            break
        tier = [
            graph
            for graph in multi_type
            if len(set(graph.vs["label"])) == distinct_count
            and str(graph["hashvalue"]) not in used_hashes
        ]
        chosen = select_diverse_examples(tier, remaining)
        selected.extend(chosen)
        used_hashes.update(str(graph["hashvalue"]) for graph in chosen)
    if len(selected) < count:
        fallback = [
            graph
            for graph in representatives
            if str(graph["hashvalue"]) not in used_hashes
        ]
        selected.extend(select_diverse_examples(fallback, count - len(selected)))
    return selected


def _draw_motif(
    axis: object,
    graph: ig.Graph,
    index_to_label: dict[str, str],
    color_by_label: dict[str, object],
    *,
    compact: bool,
    show_node_ids: bool = True,
) -> None:
    import matplotlib.pyplot as plt

    topology = nx.Graph()
    topology.add_nodes_from(range(graph.vcount()))
    topology.add_edges_from(graph.get_edgelist())
    positions = topology_layout(graph)
    root = parse_root_nodes(graph)
    root_center = (
        sum(positions[vertex][0] for vertex in root) / 3.0,
        sum(positions[vertex][1] for vertex in root) / 3.0,
    )
    positions = {
        vertex: (point[0] - root_center[0], point[1] - root_center[1])
        for vertex, point in positions.items()
    }
    labels = [index_to_label.get(str(value), str(value)) for value in graph.vs["label"]]
    label_counts = Counter(labels)
    composition = " · ".join(
        f"{label}×{amount}" if amount > 1 else label
        for label, amount in sorted(label_counts.items())
    )
    nx.draw_networkx_edges(topology, positions, ax=axis, edge_color="#6b7280", width=1.8)
    root_edges = [normalize_edge(edge) for edge in combinations(root, 2)]
    nx.draw_networkx_edges(
        topology, positions, ax=axis, edgelist=root_edges, edge_color="#d97706", width=4.0
    )
    nx.draw_networkx_nodes(
        topology,
        positions,
        ax=axis,
        node_color=[color_by_label[label] for label in labels],
        edgecolors=["#d97706" if vertex in root else "white" for vertex in topology],
        linewidths=2.4,
        node_size=500 if compact else 720,
    )
    if show_node_ids:
        nx.draw_networkx_labels(
            topology,
            positions,
            ax=axis,
            labels={vertex: str(vertex) for vertex in topology},
            font_size=7 if compact else 9,
            font_color="black",
        )
    axis.set_title(
        f"count={int(graph['count']):,} | topology={exact_topology_key(graph)[:8]}\n"
        + composition,
        fontsize=8 if compact else 11,
    )
    axis.margins(0.16)
    axis.set_aspect("equal")
    axis.set_axis_off()


def visualize_examples(
    states: Sequence[ig.Graph],
    index_to_label: dict[str, str],
    output_root: Path,
    *,
    sample_count: int,
    dpi: int,
    queue_paths: dict[str, Path],
    show_node_ids: bool = True,
    prefer_multitype: bool = False,
) -> list[dict[str, object]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    examples = (
        select_multitype_examples(states, sample_count)
        if prefer_multitype
        else select_diverse_examples(states, sample_count)
    )
    output_root.mkdir(parents=True, exist_ok=True)
    level = int(examples[0]["trinum"]) if examples else 0
    level_tag = f"tri{level}"
    all_labels = sorted(set(index_to_label.values()))
    palette = plt.get_cmap("tab20")
    colors = {label: palette(index % 20) for index, label in enumerate(all_labels)}
    rows: list[dict[str, object]] = []
    for index, graph in enumerate(examples, start=1):
        path = output_root / f"{level_tag}_example_{index:02d}.png"
        figure, axis = plt.subplots(figsize=(7.4, 6.6))
        _draw_motif(
            axis,
            graph,
            index_to_label,
            colors,
            compact=False,
            show_node_ids=show_node_ids,
        )
        figure.suptitle(
            f"Root-centered TRI-{level} motif (gold = seed triangle)", fontsize=13
        )
        figure.tight_layout()
        figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        rows.append(
            {
                "example": index,
                "matches": int(graph["count"]),
                "wl_hash": graph["hashvalue"],
                "rooted_hash": graph["rootedhash"],
                "topology_id": exact_topology_key(graph),
                "node_labels": "&".join(str(value) for value in graph.vs["label"]),
                "cell_types": "&".join(index_to_label[str(value)] for value in graph.vs["label"]),
                "gml_path": str(queue_paths.get(graph["rootedhash"], "")),
                "png_path": str(path),
            }
        )

    if examples:
        columns = 3
        rows_count = (len(examples) + columns - 1) // columns
        figure, axes = plt.subplots(rows_count, columns, figsize=(15, 5 * rows_count))
        axes_list = list(getattr(axes, "flat", [axes]))
        for axis, graph in zip(axes_list, examples):
            _draw_motif(
                axis,
                graph,
                index_to_label,
                colors,
                compact=True,
                show_node_ids=show_node_ids,
            )
        for axis in axes_list[len(examples) :]:
            axis.set_axis_off()
        figure.suptitle(
            f"Diverse root-centered TRI-{level} motifs (gold = seed triangle)",
            fontsize=16,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.97))
        figure.savefig(
            output_root / f"{level_tag}_examples_contact_sheet.png",
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(figure)

    manifest = output_root / f"{level_tag}_examples.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "example", "matches", "wl_hash", "rooted_hash", "topology_id",
            "node_labels", "cell_types", "gml_path", "png_path",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, found {value!r}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--MaxDepth", type=int, default=7)
    parser.add_argument("--SM_method", choices=("ccindex", "vf2"), default="ccindex")
    parser.add_argument("--threshold", type=int, default=10)
    parser.add_argument("--SM_stop_early", type=str_to_bool, default=False)
    parser.add_argument("--LevelStartFrom", type=int, default=0)
    parser.add_argument("--GraphQueuePath", type=Path, default=None)
    parser.add_argument("--DataPath", type=Path, default=Path("ad_data.csv"))
    parser.add_argument(
        "--TargetGraphPath",
        type=Path,
        default=Path("spatial_13months-disease-replicate_1.gml"),
    )
    parser.add_argument("--OutputGraphPath", type=Path, default=None)
    parser.add_argument("--ResultPath", type=Path, default=None)
    parser.add_argument("--SampleCount", type=int, default=6)
    parser.add_argument("--DPI", type=int, default=180)
    parser.add_argument("--Workers", type=int, default=1)
    args = parser.parse_args(argv)
    if args.MaxDepth < 1:
        parser.error("--MaxDepth must be at least 1")
    if args.LevelStartFrom < 0 or args.LevelStartFrom > args.MaxDepth:
        parser.error("--LevelStartFrom must be between 0 and MaxDepth")
    if args.LevelStartFrom and args.GraphQueuePath is None:
        parser.error("--GraphQueuePath is required when --LevelStartFrom is nonzero")
    if args.SampleCount < 0:
        parser.error("--SampleCount cannot be negative")
    if args.Workers < 1:
        parser.error("--Workers must be at least 1")
    return args


def convert_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours} : {minutes} : {seconds % 60:.2f}"


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    started = time.time()
    dataframe = pd.read_csv(args.DataPath)
    filtered_triangles, edge_pool, index_to_label = build_search_vocabulary(
        dataframe, args.threshold
    )
    output_root = args.OutputGraphPath or Path(
        f"ad_output_graph_{args.SM_method}_{args.threshold}_topology_restrict"
    )
    result_root = args.ResultPath or Path(
        f"ad_res_threshold_{args.threshold}_{args.SM_method}_topology_restrict"
    )

    if args.LevelStartFrom == 0:
        current_level = 1
        queue = construct_queue(filtered_triangles)
        queue_paths = write_queue(queue, current_level, output_root)
    else:
        current_level = args.LevelStartFrom
        queue = read_queue(args.GraphQueuePath, current_level)
        queue_paths = {}

    target = ig.Graph.Read_GML(str(args.TargetGraphPath))
    target.vs["label"] = [str(int(value)) for value in target.vs["label"]]
    print(datetime.now())
    print(
        f"Target node size {target.vcount()}    "
        f"Node Type size {len(set(target.vs['label']))}"
    )
    summary_path = output_root / "run_summary.json"
    prior_summary: dict[str, object] = {}
    if args.LevelStartFrom and summary_path.exists():
        with summary_path.open(encoding="utf-8") as handle:
            prior_summary = json.load(handle)
    level_summary: list[dict[str, object]] = [
        row
        for row in prior_summary.get("levels", [])
        if int(row["triangle_level"]) <= current_level
    ]
    prior_seconds = float(prior_summary.get("total_seconds", 0.0))

    def save_summary(visualized_examples: int | None = None) -> None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        if visualized_examples is None:
            visualized_examples = int(prior_summary.get("visualized_examples", 0))
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "definition": "strict triangle-face hop shells around the TRI-1 root",
                    "data_path": str(args.DataPath),
                    "target_graph_path": str(args.TargetGraphPath),
                    "sm_method": args.SM_method,
                    "threshold": args.threshold,
                    "max_depth": args.MaxDepth,
                    "stop_early": args.SM_stop_early,
                    "levels": level_summary,
                    "visualized_examples": visualized_examples,
                    "total_seconds": prior_seconds + time.time() - started,
                },
                handle,
                indent=2,
            )

    for level in range(current_level + 1, args.MaxDepth + 1):
        level_started = time.time()
        candidates = expand_level(queue, edge_pool)
        generated_seconds = time.time() - level_started
        result_file = result_root / f"{args.SM_method}_tri_{level}.csv"
        queue, matched_patterns = match_and_prune(
            target,
            candidates,
            method=args.SM_method,
            threshold=args.threshold,
            stop_early=args.SM_stop_early,
            result_path=result_file,
            workers=args.Workers,
        )
        queue_paths = write_queue(queue, level, output_root)
        elapsed = time.time() - level_started
        memory_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        summary = {
            "triangle_level": level,
            "rooted_candidates": len(candidates),
            "unrooted_patterns_matched": matched_patterns,
            "rooted_survivors": len(queue),
            "unrooted_survivors": len(group_unrooted_states(queue)[0]),
            "generation_seconds": generated_seconds,
            "level_seconds": elapsed,
            "memory_mb": memory_mb,
        }
        level_summary.append(summary)
        save_summary()
        print(
            f"{convert_time(time.time() - started)}  TRI-{level}: "
            f"rooted candidates={len(candidates):,}, matched={matched_patterns:,}, "
            f"rooted survivors={len(queue):,}, level={convert_time(elapsed)}"
        )
        if not queue:
            print(f"No motifs survived TRI-{level}; stopping early.")
            break

    visualization_rows: list[dict[str, object]] = []
    if queue and int(queue[0]["trinum"]) == 7 and args.SampleCount:
        visualization_rows = visualize_examples(
            queue,
            index_to_label,
            output_root / "visualizations",
            sample_count=args.SampleCount,
            dpi=args.DPI,
            queue_paths=queue_paths,
        )

    save_summary(
        len(visualization_rows)
        if visualization_rows
        else int(prior_summary.get("visualized_examples", 0))
    )


if __name__ == "__main__":
    main()
