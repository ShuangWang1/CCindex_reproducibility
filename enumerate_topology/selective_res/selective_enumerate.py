#!/usr/bin/env python3
"""Run a deterministic, diversity-preserving beam over triangle topologies."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import heapq
import json
import math
import os
import platform
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Hashable, Iterable, Iterator, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from enumerate_topology.enumerate_topologies import (
    GraphCode,
    adjacency_bits,
    decode_graph,
    expand_code,
    process_tree_rss_gib,
    seed_code,
    stable_rank,
    topology_id,
    validate_topology,
    write_sample,
)


ATTACH = "attach"
CLOSE = "close"


@dataclass(slots=True)
class ParentLinks:
    attach: set[int]
    close: set[int]

    @classmethod
    def empty(cls) -> "ParentLinks":
        return cls(set(), set())

    def parents(self) -> set[int]:
        return self.attach | self.close

    def operation_tokens(self) -> set[tuple[int, str]]:
        return {(parent, ATTACH) for parent in self.attach} | {
            (parent, CLOSE) for parent in self.close
        }


@dataclass(frozen=True, slots=True)
class TopologyFeatures:
    nodes: int
    edges: int
    boundary_vertices: int
    interior_vertices: int
    closable_wedges: int
    minimum_degree: int
    maximum_degree: int
    degree_two: int
    degree_three: int
    degree_four: int
    degree_five_plus: int
    graph_diameter: int
    graph_radius: int
    mean_distance_milli: int
    max_triangle_participation: int
    dual_edges: int
    dual_cycle_rank: int
    dual_leaves: int
    dual_branches: int
    dual_diameter: int
    degree_sequence: tuple[int, ...]
    boundary_degree_sequence: tuple[int, ...]
    triangle_participation_sequence: tuple[int, ...]
    dual_degree_sequence: tuple[int, ...]

    def coarse_signature(self) -> tuple[int, ...]:
        """A deliberately bounded-resolution structural stratum."""
        return (
            self.nodes,
            self.graph_diameter,
            self.graph_radius,
            self.maximum_degree // 2,
            self.closable_wedges // 2,
            self.dual_cycle_rank // 2,
            self.dual_leaves // 2,
            self.dual_branches // 2,
            self.dual_diameter,
        )

    def fine_signature(self) -> tuple[object, ...]:
        return (
            self.degree_sequence,
            self.boundary_degree_sequence,
            self.triangle_participation_sequence,
            self.dual_degree_sequence,
        )


@dataclass(slots=True)
class SelectionResult:
    selected: set[GraphCode]
    reasons: dict[GraphCode, str]
    parent_coverage: float
    operation_coverage: float
    signature_coverage: float
    pool_signatures: int
    selected_signatures: int
    seconds: float


def seeded_rank(code: bytes, seed: int) -> bytes:
    return hashlib.blake2b(
        seed.to_bytes(8, "little", signed=True) + code, digest_size=16
    ).digest()


def signature_rank(signature: Hashable, seed: int) -> bytes:
    return hashlib.blake2b(
        seed.to_bytes(8, "little", signed=True) + repr(signature).encode("utf-8"),
        digest_size=16,
    ).digest()


def _batched(values: Iterable[object], size: int) -> Iterator[tuple[object, ...]]:
    batch: list[object] = []
    for value in values:
        batch.append(value)
        if len(batch) == size:
            yield tuple(batch)
            batch.clear()
    if batch:
        yield tuple(batch)


def _expand_parent_batch(
    batch: tuple[tuple[int, GraphCode], ...]
) -> tuple[dict[GraphCode, tuple[set[int], set[int]]], int]:
    merged: dict[GraphCode, tuple[set[int], set[int]]] = {}
    relations = 0
    for parent_index, parent in batch:
        children, _ = expand_code(parent)
        parent_nodes = parent[0]
        for child in children:
            attach, close = merged.setdefault(child, (set(), set()))
            if child[0] == parent_nodes + 1:
                attach.add(parent_index)
            else:
                close.add(parent_index)
            relations += 1
    return merged, relations


def expand_candidate_pool(
    parents: Sequence[GraphCode], workers: int, batch_size: int
) -> tuple[dict[GraphCode, ParentLinks], int, float, float]:
    """Expand all retained parents and preserve child-to-parent provenance."""
    started = time.monotonic()
    peak_rss = process_tree_rss_gib()
    pool: dict[GraphCode, ParentLinks] = {}
    relation_count = 0
    indexed = tuple(enumerate(parents))

    def merge(local: dict[GraphCode, tuple[set[int], set[int]]]) -> None:
        for code, (attach, close) in local.items():
            links = pool.setdefault(code, ParentLinks.empty())
            links.attach.update(attach)
            links.close.update(close)

    if workers <= 1:
        for batch in _batched(indexed, batch_size):
            local, relations = _expand_parent_batch(batch)  # type: ignore[arg-type]
            merge(local)
            relation_count += relations
            peak_rss = max(peak_rss, process_tree_rss_gib())
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        batches = iter(_batched(indexed, batch_size))
        pending = set()
        exhausted = False
        last_progress = started
        try:
            while True:
                while not exhausted and len(pending) < workers * 2:
                    try:
                        batch = next(batches)
                    except StopIteration:
                        exhausted = True
                        break
                    pending.add(executor.submit(_expand_parent_batch, batch))
                if not pending:
                    break
                done, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
                for future in done:
                    local, relations = future.result()
                    merge(local)
                    relation_count += relations
                now = time.monotonic()
                if now - last_progress >= 10.0:
                    peak_rss = max(peak_rss, process_tree_rss_gib())
                    print(
                        f"    candidates={len(pool):,} relations={relation_count:,} "
                        f"rss={peak_rss:.2f} GiB",
                        flush=True,
                    )
                    last_progress = now
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
    peak_rss = max(peak_rss, process_tree_rss_gib())
    return pool, relation_count, time.monotonic() - started, peak_rss


def _distance_stats(adjacency: Sequence[int]) -> tuple[int, int, int]:
    vertex_count = len(adjacency)
    if vertex_count <= 1:
        return 0, 0, 0
    all_vertices = (1 << vertex_count) - 1
    eccentricities: list[int] = []
    distance_sum = 0
    pair_count = 0
    for source in range(vertex_count):
        seen = 1 << source
        frontier = seen
        distance = 0
        eccentricity = 0
        while seen != all_vertices:
            neighbors = 0
            remaining = frontier
            while remaining:
                bit = remaining & -remaining
                neighbors |= adjacency[bit.bit_length() - 1]
                remaining -= bit
            frontier = neighbors & ~seen
            if not frontier:
                raise ValueError("feature graph is disconnected")
            distance += 1
            eccentricity = distance
            if source < vertex_count - 1:
                targets = frontier & ~((1 << (source + 1)) - 1)
                count = targets.bit_count()
                distance_sum += distance * count
                pair_count += count
            seen |= frontier
        eccentricities.append(eccentricity)
    mean_milli = round(1000 * distance_sum / pair_count) if pair_count else 0
    return max(eccentricities), min(eccentricities), mean_milli


def extract_features(code: GraphCode) -> TopologyFeatures:
    nodes, edges = decode_graph(code)
    adjacency = adjacency_bits(nodes, edges)
    degrees = [neighbors.bit_count() for neighbors in adjacency]
    boundary: list[tuple[int, int]] = []
    boundary_neighbors: list[list[int]] = [[] for _ in range(nodes)]
    for left, right in edges:
        if (adjacency[left] & adjacency[right]).bit_count() == 1:
            boundary.append((left, right))
            boundary_neighbors[left].append(right)
            boundary_neighbors[right].append(left)
    boundary_vertices = {vertex for edge in boundary for vertex in edge}

    closable = 0
    for center, neighbors in enumerate(boundary_neighbors):
        if len(neighbors) != 2:
            continue
        left, right = neighbors
        if not (adjacency[left] >> right) & 1 and (
            adjacency[left] & adjacency[right]
        ) == 1 << center:
            closable += 1

    triangles: list[tuple[int, int, int]] = []
    triangle_participation = [0] * nodes
    for left in range(nodes):
        for middle in range(left + 1, nodes):
            if not (adjacency[left] >> middle) & 1:
                continue
            common = adjacency[left] & adjacency[middle]
            common &= ~((1 << (middle + 1)) - 1)
            while common:
                bit = common & -common
                right = bit.bit_length() - 1
                triangles.append((left, middle, right))
                triangle_participation[left] += 1
                triangle_participation[middle] += 1
                triangle_participation[right] += 1
                common -= bit

    edge_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for face_index, (left, middle, right) in enumerate(triangles):
        for edge in ((left, middle), (left, right), (middle, right)):
            edge_faces[tuple(sorted(edge))].append(face_index)
    dual_adjacency = [0] * len(triangles)
    dual_edges = 0
    for faces in edge_faces.values():
        if len(faces) == 2:
            left, right = faces
            dual_adjacency[left] |= 1 << right
            dual_adjacency[right] |= 1 << left
            dual_edges += 1
    dual_degrees = [neighbors.bit_count() for neighbors in dual_adjacency]
    graph_diameter, graph_radius, mean_distance = _distance_stats(adjacency)
    dual_diameter, _, _ = _distance_stats(dual_adjacency)

    return TopologyFeatures(
        nodes=nodes,
        edges=len(edges),
        boundary_vertices=len(boundary_vertices),
        interior_vertices=nodes - len(boundary_vertices),
        closable_wedges=closable,
        minimum_degree=min(degrees),
        maximum_degree=max(degrees),
        degree_two=degrees.count(2),
        degree_three=degrees.count(3),
        degree_four=degrees.count(4),
        degree_five_plus=sum(degree >= 5 for degree in degrees),
        graph_diameter=graph_diameter,
        graph_radius=graph_radius,
        mean_distance_milli=mean_distance,
        max_triangle_participation=max(triangle_participation),
        dual_edges=dual_edges,
        dual_cycle_rank=dual_edges - len(triangles) + 1,
        dual_leaves=dual_degrees.count(1),
        dual_branches=sum(degree >= 3 for degree in dual_degrees),
        dual_diameter=dual_diameter,
        degree_sequence=tuple(sorted(degrees)),
        boundary_degree_sequence=tuple(sorted(degrees[v] for v in boundary_vertices)),
        triangle_participation_sequence=tuple(sorted(triangle_participation)),
        dual_degree_sequence=tuple(sorted(dual_degrees)),
    )


def _feature_batch(
    codes: tuple[GraphCode, ...]
) -> list[tuple[GraphCode, TopologyFeatures]]:
    return [(code, extract_features(code)) for code in codes]


def feature_candidate_pool(
    codes: Iterable[GraphCode], workers: int, batch_size: int
) -> tuple[dict[GraphCode, TopologyFeatures], float]:
    started = time.monotonic()
    ordered = sorted(codes)
    features: dict[GraphCode, TopologyFeatures] = {}
    if workers <= 1:
        for code in ordered:
            features[code] = extract_features(code)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for rows in executor.map(
                _feature_batch, _batched(ordered, batch_size), chunksize=1
            ):
                features.update(rows)
    return features, time.monotonic() - started


def _greedy_cover(
    pool: dict[GraphCode, ParentLinks],
    features: dict[GraphCode, TopologyFeatures],
    selected: set[GraphCode],
    reasons: dict[GraphCode, str],
    uncovered: set[Hashable],
    tokens: Callable[[ParentLinks], set[Hashable]],
    beam_size: int,
    seed: int,
    reason: str,
    stratum_frequency: Counter[tuple[int, ...]],
) -> None:
    heap: list[tuple[int, int, bytes, GraphCode]] = []
    for code, links in pool.items():
        if code in selected:
            continue
        gain = len(tokens(links) & uncovered)
        if gain:
            signature = features[code].coarse_signature()
            heapq.heappush(
                heap,
                (-gain, stratum_frequency[signature], seeded_rank(code, seed), code),
            )
    while uncovered and heap and len(selected) < beam_size:
        negative_gain, frequency, rank, code = heapq.heappop(heap)
        if code in selected:
            continue
        gain_tokens = tokens(pool[code]) & uncovered
        gain = len(gain_tokens)
        if gain != -negative_gain:
            if gain:
                heapq.heappush(heap, (-gain, frequency, rank, code))
            continue
        selected.add(code)
        reasons[code] = reason
        uncovered.difference_update(gain_tokens)


def _structural_order(
    codes: Iterable[GraphCode],
    features: dict[GraphCode, TopologyFeatures],
    seed: int,
) -> list[GraphCode]:
    fine_groups: dict[tuple[object, ...], list[GraphCode]] = defaultdict(list)
    for code in codes:
        fine_groups[features[code].fine_signature()].append(code)
    for values in fine_groups.values():
        values.sort(key=lambda code: seeded_rank(code, seed))
    signatures = sorted(fine_groups, key=lambda value: signature_rank(value, seed))
    ordered: list[GraphCode] = []
    depth = 0
    while signatures:
        remaining: list[tuple[object, ...]] = []
        for signature in signatures:
            values = fine_groups[signature]
            if depth < len(values):
                ordered.append(values[depth])
            if depth + 1 < len(values):
                remaining.append(signature)
        signatures = remaining
        depth += 1
    return ordered


def select_diverse_beam(
    pool: dict[GraphCode, ParentLinks],
    features: dict[GraphCode, TopologyFeatures],
    parent_count: int,
    beam_size: int,
    seed: int,
) -> SelectionResult:
    started = time.monotonic()
    if len(pool) <= beam_size:
        selected = set(pool)
        reasons = {code: "all" for code in selected}
    else:
        selected: set[GraphCode] = set()
        reasons: dict[GraphCode, str] = {}
        stratum_frequency = Counter(
            feature.coarse_signature() for feature in features.values()
        )

        all_parents: set[Hashable] = set(range(parent_count))
        _greedy_cover(
            pool,
            features,
            selected,
            reasons,
            all_parents,
            lambda links: set(links.parents()),
            beam_size,
            seed,
            "parent_coverage",
            stratum_frequency,
        )
        if all_parents:
            raise RuntimeError(f"failed to cover {len(all_parents)} parent branches")

        all_operation_tokens: set[Hashable] = set()
        for links in pool.values():
            all_operation_tokens.update(links.operation_tokens())
        for code in selected:
            all_operation_tokens.difference_update(pool[code].operation_tokens())
        _greedy_cover(
            pool,
            features,
            selected,
            reasons,
            all_operation_tokens,
            lambda links: set(links.operation_tokens()),
            beam_size,
            seed + 1,
            "operation_coverage",
            stratum_frequency,
        )

        groups: dict[tuple[int, ...], list[GraphCode]] = defaultdict(list)
        for code in pool:
            if code not in selected:
                groups[features[code].coarse_signature()].append(code)
        ordered_groups = {
            signature: _structural_order(codes, features, seed + 2)
            for signature, codes in groups.items()
        }
        group_index = {signature: 0 for signature in groups}
        selected_per_group = Counter(
            features[code].coarse_signature() for code in selected
        )
        heap: list[tuple[float, bytes, tuple[int, ...]]] = []
        for signature, codes in groups.items():
            population = stratum_frequency[signature]
            priority = selected_per_group[signature] / math.sqrt(population)
            heapq.heappush(
                heap, (priority, signature_rank(signature, seed + 3), signature)
            )
        while len(selected) < beam_size and heap:
            _, rank, signature = heapq.heappop(heap)
            index = group_index[signature]
            values = ordered_groups[signature]
            while index < len(values) and values[index] in selected:
                index += 1
            if index >= len(values):
                continue
            code = values[index]
            group_index[signature] = index + 1
            selected.add(code)
            reasons[code] = "structural_diversity"
            selected_per_group[signature] += 1
            population = stratum_frequency[signature]
            next_priority = selected_per_group[signature] / math.sqrt(population)
            heapq.heappush(heap, (next_priority, rank, signature))
        if len(selected) != beam_size:
            raise RuntimeError(
                f"selection returned {len(selected)} motifs, expected {beam_size}"
            )

    all_parent_tokens = set(range(parent_count))
    all_operation_tokens: set[tuple[int, str]] = set()
    all_signatures = {feature.coarse_signature() for feature in features.values()}
    covered_parents: set[int] = set()
    covered_operations: set[tuple[int, str]] = set()
    selected_signatures: set[tuple[int, ...]] = set()
    for code in pool:
        all_operation_tokens.update(pool[code].operation_tokens())
    for code in selected:
        covered_parents.update(pool[code].parents())
        covered_operations.update(pool[code].operation_tokens())
        selected_signatures.add(features[code].coarse_signature())
    parent_coverage = (
        100.0 * len(covered_parents) / len(all_parent_tokens)
        if all_parent_tokens
        else 100.0
    )
    operation_coverage = (
        100.0 * len(covered_operations) / len(all_operation_tokens)
        if all_operation_tokens
        else 100.0
    )
    signature_coverage = (
        100.0 * len(selected_signatures) / len(all_signatures)
        if all_signatures
        else 100.0
    )
    return SelectionResult(
        selected,
        reasons,
        parent_coverage,
        operation_coverage,
        signature_coverage,
        len(all_signatures),
        len(selected_signatures),
        time.monotonic() - started,
    )


def select_representatives(
    selected: set[GraphCode],
    features: dict[GraphCode, TopologyFeatures],
    count: int,
    seed: int,
) -> list[GraphCode]:
    if len(selected) <= count:
        return sorted(selected, key=lambda code: seeded_rank(code, seed))
    groups: dict[tuple[int, ...], list[GraphCode]] = defaultdict(list)
    for code in selected:
        groups[features[code].coarse_signature()].append(code)
    representatives = [
        min(codes, key=lambda code: seeded_rank(code, seed))
        for codes in groups.values()
    ]
    representatives.sort(key=lambda code: seeded_rank(code, seed))
    if len(representatives) >= count:
        # Spread deterministically across the complete signature list.
        return [
            representatives[round(index * (len(representatives) - 1) / (count - 1))]
            for index in range(count)
        ]
    remaining = sorted(
        selected - set(representatives), key=lambda code: seeded_rank(code, seed)
    )
    return representatives + remaining[: count - len(representatives)]


def write_level_archive(
    path: Path,
    level: int,
    selected: set[GraphCode],
    parents: Sequence[GraphCode],
    pool: dict[GraphCode, ParentLinks],
    features: dict[GraphCode, TopologyFeatures],
    reasons: dict[GraphCode, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    parent_ids = [topology_id(code) for code in parents]
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        for code in sorted(selected):
            links = pool.get(code, ParentLinks.empty())
            record = {
                "topology_id": topology_id(code),
                "canonical_code_hex": code.hex(),
                "triangle_level": level,
                "selection_reason": reasons[code],
                "attach_parent_ids": [parent_ids[index] for index in sorted(links.attach)],
                "close_parent_ids": [parent_ids[index] for index in sorted(links.close)],
                "features": asdict(features[code]),
            }
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")


def write_csv(path: Path, rows: Sequence[dict[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def persist_tables(
    output_root: Path,
    level_rows: Sequence[dict[str, object]],
    distribution_rows: Sequence[dict[str, object]],
    sample_rows: Sequence[dict[str, object]],
) -> None:
    write_csv(output_root / "level_summary.csv", level_rows, tuple(level_rows[0]))
    diversity_fields = (
        "triangle_level",
        "parent_coverage_pct",
        "operation_coverage_pct",
        "coarse_signatures_pool",
        "coarse_signatures_selected",
        "signature_coverage_pct",
        "node_min",
        "node_max",
    )
    write_csv(
        output_root / "diversity_metrics.csv",
        [{field: row[field] for field in diversity_fields} for row in level_rows],
        diversity_fields,
    )
    write_csv(
        output_root / "node_count_distribution.csv",
        distribution_rows,
        ("triangle_level", "population", "node_count", "topology_count"),
    )
    write_csv(
        output_root / "samples_index.csv",
        sample_rows,
        (
            "triangle_level",
            "sample_index",
            "canonical_id",
            "nodes",
            "edges",
            "gml_path",
            "png_path",
        ),
    )


def write_plots(output_root: Path, rows: Sequence[dict[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    levels = [int(row["triangle_level"]) for row in rows]
    pools = [int(row["candidate_pool"]) for row in rows]
    selected = [int(row["selected"]) for row in rows]
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(levels, pools, marker="o", label="Generated candidate pool")
    axis.plot(levels, selected, marker="o", label="Retained beam")
    axis.axhline(10_000, color="#c53030", linestyle="--", label="10,000 cap")
    axis.set_yscale("log")
    axis.set_xlabel("Triangle level")
    axis.set_ylabel("Topologies (log scale)")
    axis.set_title("Selective BFS candidate pool and retained beam")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_root / "beam_growth.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    parent = [float(row["parent_coverage_pct"]) for row in rows[1:]]
    operation = [float(row["operation_coverage_pct"]) for row in rows[1:]]
    signature = [float(row["signature_coverage_pct"]) for row in rows[1:]]
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(levels[1:], parent, marker="o", label="Parent branches")
    axis.plot(levels[1:], operation, marker="o", label="Attach/close branches")
    axis.plot(levels[1:], signature, marker="o", label="Structural signatures")
    axis.set_ylim(0, 102)
    axis.set_xlabel("Triangle level")
    axis.set_ylabel("Coverage (%)")
    axis.set_title("Selective beam diversity coverage")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_root / "diversity_coverage.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


LEVEL_FIELDS = (
    "triangle_level",
    "mode",
    "parent_beam",
    "raw_relations",
    "candidate_pool",
    "selected",
    "pruned",
    "parent_coverage_pct",
    "operation_coverage_pct",
    "coarse_signatures_pool",
    "coarse_signatures_selected",
    "signature_coverage_pct",
    "node_min",
    "node_max",
    "expand_seconds",
    "feature_seconds",
    "selection_seconds",
    "total_seconds",
    "peak_rss_gib",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run diversity-preserving selective BFS over triangle topologies."
    )
    parser.add_argument("--max-triangles", type=int, default=40)
    parser.add_argument("--beam-size", type=int, default=10_000)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--examples-per-level", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--wall-seconds", type=float, default=3600.0)
    parser.add_argument("--memory-gib", type=float, default=20.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("enumerate_topology/selective_res/results"),
    )
    args = parser.parse_args(argv)
    for name in ("max_triangles", "beam_size", "workers", "batch_size"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.examples_per_level < 0:
        parser.error("--examples-per-level cannot be negative")
    if args.wall_seconds <= 0 or args.memory_gib <= 0:
        parser.error("resource caps must be positive")
    return args


def write_run_summary(
    output_root: Path,
    args: argparse.Namespace,
    rows: Sequence[dict[str, object]],
    started_iso: str,
    started: float,
    termination: str,
) -> None:
    payload = {
        "started_at": started_iso,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "total_seconds": round(time.monotonic() - started, 6),
        "last_completed_level": int(rows[-1]["triangle_level"]) if rows else 0,
        "termination_reason": termination,
        "configuration": vars(args),
        "versions": {"python": platform.python_version()},
    }
    (output_root / "run_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> int:
    output_root: Path = args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    started_iso = datetime.now(timezone.utc).isoformat()
    deadline = started + args.wall_seconds
    peak_rss = process_tree_rss_gib()
    level_rows: list[dict[str, object]] = []
    distribution_rows: list[dict[str, object]] = []
    sample_rows: list[dict[str, object]] = []

    current = {seed_code()}
    current_features = {code: extract_features(code) for code in current}
    previous_pruned = False
    previous_parents: list[GraphCode] = []
    current_links = {code: ParentLinks.empty() for code in current}
    current_reasons = {code: "all" for code in current}

    for level in range(1, args.max_triangles + 1):
        if level > 1:
            parents = sorted(current)
            print(f"level {level}: expanding {len(parents):,} retained parents", flush=True)
            pool, raw_relations, expand_seconds, level_peak = expand_candidate_pool(
                parents, args.workers, args.batch_size
            )
            peak_rss = max(peak_rss, level_peak)
            features, feature_seconds = feature_candidate_pool(
                pool, args.workers, args.batch_size
            )
            selection = select_diverse_beam(
                pool,
                features,
                len(parents),
                args.beam_size,
                args.selection_seed + level,
            )
            selected = selection.selected
            mode = "exhaustive" if not previous_pruned and len(pool) <= args.beam_size else "selective"
            previous_pruned = previous_pruned or len(pool) > args.beam_size
            current = selected
            current_features = {code: features[code] for code in selected}
            previous_parents = parents
            current_links = pool
            current_reasons = selection.reasons
            parent_beam = len(parents)
            candidate_pool = len(pool)
            pruned = len(pool) - len(selected)
        else:
            raw_relations = 0
            expand_seconds = 0.0
            feature_seconds = 0.0
            selection = SelectionResult(current, current_reasons, 100, 100, 100, 1, 1, 0)
            mode = "exhaustive"
            parent_beam = 0
            candidate_pool = 1
            pruned = 0

        for code in current:
            validate_topology(code, level)
        node_distribution_pool = Counter(feature.nodes for feature in current_features.values())
        # For selective levels, reconstruct the pool distribution before discarding features.
        if level > 1:
            pool_distribution = Counter(feature.nodes for feature in features.values())
        else:
            pool_distribution = node_distribution_pool
        distribution_rows.extend(
            {
                "triangle_level": level,
                "population": population,
                "node_count": node_count,
                "topology_count": count,
            }
            for population, distribution in (
                ("candidate_pool", pool_distribution),
                ("selected", node_distribution_pool),
            )
            for node_count, count in sorted(distribution.items())
        )

        level_rows.append(
            {
                "triangle_level": level,
                "mode": mode,
                "parent_beam": parent_beam,
                "raw_relations": raw_relations,
                "candidate_pool": candidate_pool,
                "selected": len(current),
                "pruned": pruned,
                "parent_coverage_pct": round(selection.parent_coverage, 6),
                "operation_coverage_pct": round(selection.operation_coverage, 6),
                "coarse_signatures_pool": selection.pool_signatures,
                "coarse_signatures_selected": selection.selected_signatures,
                "signature_coverage_pct": round(selection.signature_coverage, 6),
                "node_min": min(node_distribution_pool),
                "node_max": max(node_distribution_pool),
                "expand_seconds": round(expand_seconds, 6),
                "feature_seconds": round(feature_seconds, 6),
                "selection_seconds": round(selection.seconds, 6),
                "total_seconds": round(time.monotonic() - started, 6),
                "peak_rss_gib": round(peak_rss, 6),
            }
        )
        write_level_archive(
            output_root / "levels" / f"tri_{level:03d}.jsonl.gz",
            level,
            current,
            previous_parents,
            current_links,
            current_features,
            current_reasons,
        )
        representatives = select_representatives(
            current,
            current_features,
            min(args.examples_per_level, len(current)),
            args.selection_seed + level,
        )
        for index, code in enumerate(representatives, start=1):
            sample_rows.append(write_sample(code, level, index, output_root))
        persist_tables(output_root, level_rows, distribution_rows, sample_rows)
        write_run_summary(
            output_root, args, level_rows, started_iso, started, "running"
        )
        print(
            f"level {level}: pool={candidate_pool:,} selected={len(current):,} "
            f"parent={selection.parent_coverage:.2f}% "
            f"operations={selection.operation_coverage:.2f}% "
            f"signatures={selection.signature_coverage:.2f}%",
            flush=True,
        )
        rss = process_tree_rss_gib()
        peak_rss = max(peak_rss, rss)
        if time.monotonic() >= deadline or rss >= args.memory_gib:
            termination = "wall_time" if time.monotonic() >= deadline else "memory"
            write_plots(output_root, level_rows)
            write_run_summary(
                output_root, args, level_rows, started_iso, started, termination
            )
            return 2

    write_plots(output_root, level_rows)
    write_run_summary(
        output_root, args, level_rows, started_iso, started, "max_triangles"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
