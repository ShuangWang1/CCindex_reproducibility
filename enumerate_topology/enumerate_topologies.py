#!/usr/bin/env python3
"""Enumerate unlabeled topologies grown one triangle at a time.

The two growth operations intentionally mirror ``tree_search()`` in
``tree_search_all_level_threshold_cpp_0806.py``.  Unlike that program, this
module has one vertex label, no target graph, and no subgraph matcher.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Sequence

# igraph imports Matplotlib integration eagerly on some installations. Point
# its cache at writable temporary storage before importing igraph.
_MPL_CACHE = Path("/tmp/enumerate_topology_matplotlib")
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import igraph as ig


GraphCode = bytes
Edge = tuple[int, int]


def encode_graph(vertex_count: int, edges: Iterable[Edge]) -> GraphCode:
    """Encode a canonical graph compactly (valid here because n <= 42 at level 40)."""
    if not 0 < vertex_count < 256:
        raise ValueError("vertex_count must fit in one byte")
    payload = bytearray((vertex_count,))
    for left, right in sorted((min(u, v), max(u, v)) for u, v in edges):
        if left == right or left < 0 or right >= vertex_count:
            raise ValueError(f"invalid edge {(left, right)} for n={vertex_count}")
        payload.extend((left, right))
    return bytes(payload)


def decode_graph(code: GraphCode) -> tuple[int, tuple[Edge, ...]]:
    if not code or len(code) % 2 != 1:
        raise ValueError("invalid compact graph code")
    vertex_count = code[0]
    edges = tuple((code[index], code[index + 1]) for index in range(1, len(code), 2))
    return vertex_count, edges


def canonical_code(vertex_count: int, edges: Iterable[Edge]) -> GraphCode:
    """Return an exact isomorphism-invariant adjacency code using BLISS."""
    normalized = tuple(sorted((min(u, v), max(u, v)) for u, v in edges))
    graph = ig.Graph(n=vertex_count, edges=normalized, directed=False)
    permutation = graph.canonical_permutation()
    canonical_edges = (
        (min(permutation[u], permutation[v]), max(permutation[u], permutation[v]))
        for u, v in normalized
    )
    return encode_graph(vertex_count, canonical_edges)


def adjacency_bits(vertex_count: int, edges: Sequence[Edge]) -> list[int]:
    adjacency = [0] * vertex_count
    for left, right in edges:
        adjacency[left] |= 1 << right
        adjacency[right] |= 1 << left
    return adjacency


def boundary_edges(code: GraphCode) -> tuple[Edge, ...]:
    """Infer the current boundary: its edges belong to exactly one 3-clique."""
    vertex_count, edges = decode_graph(code)
    adjacency = adjacency_bits(vertex_count, edges)
    return tuple(
        edge
        for edge in edges
        if (adjacency[edge[0]] & adjacency[edge[1]]).bit_count() == 1
    )


def expand_code(code: GraphCode) -> tuple[set[GraphCode], int]:
    """Generate unique one-triangle children of a single canonical parent."""
    vertex_count, edge_tuple = decode_graph(code)
    edges = set(edge_tuple)
    adjacency = adjacency_bits(vertex_count, edge_tuple)
    boundary = tuple(
        edge
        for edge in edge_tuple
        if (adjacency[edge[0]] & adjacency[edge[1]]).bit_count() == 1
    )

    children: set[GraphCode] = set()
    raw_children = 0

    # Reference operation 1: attach one new vertex along a boundary edge.
    for left, right in boundary:
        child_edges = edges | {(left, vertex_count), (right, vertex_count)}
        children.add(canonical_code(vertex_count + 1, child_edges))
        raw_children += 1

    # Reference operation 2: close a wedge of two consecutive boundary edges.
    boundary_neighbors: list[list[int]] = [[] for _ in range(vertex_count)]
    for left, right in boundary:
        boundary_neighbors[left].append(right)
        boundary_neighbors[right].append(left)

    for center, neighbors in enumerate(boundary_neighbors):
        if not neighbors:
            # Interior vertices are not incident to the current outer boundary.
            continue
        if len(neighbors) != 2:
            raise ValueError(
                f"boundary invariant failed at vertex {center}: degree={len(neighbors)}"
            )
        left, right = sorted(neighbors)
        closing_edge = (left, right)
        if closing_edge in edges:
            continue
        common = adjacency[left] & adjacency[right]
        if common != 1 << center:
            continue
        children.add(canonical_code(vertex_count, edges | {closing_edge}))
        raw_children += 1

    return children, raw_children


def _expand_batch(batch: tuple[GraphCode, ...]) -> tuple[set[GraphCode], int]:
    children: set[GraphCode] = set()
    raw_children = 0
    for code in batch:
        local_children, local_raw = expand_code(code)
        children.update(local_children)
        raw_children += local_raw
    return children, raw_children


def _batched(values: Iterable[GraphCode], size: int) -> Iterator[tuple[GraphCode, ...]]:
    batch: list[GraphCode] = []
    for value in values:
        batch.append(value)
        if len(batch) == size:
            yield tuple(batch)
            batch.clear()
    if batch:
        yield tuple(batch)


def process_tree_rss_gib() -> float:
    """Return Linux RSS for this process and its recursive worker children."""
    proc = Path("/proc")
    if not proc.is_dir():
        import resource

        # ru_maxrss is KiB on Linux; this fallback monitors the coordinator.
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)

    parent_by_pid: dict[int, int] = {}
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text(encoding="utf-8")
            # Fields after the final ')' start with state and then parent PID.
            tail = stat[stat.rfind(")") + 2 :].split()
            parent_by_pid[int(entry.name)] = int(tail[1])
        except (FileNotFoundError, PermissionError, ValueError, IndexError):
            continue

    descendants = {os.getpid()}
    changed = True
    while changed:
        changed = False
        for pid, parent in parent_by_pid.items():
            if parent in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True

    page_size = os.sysconf("SC_PAGE_SIZE")
    rss_bytes = 0
    for pid in descendants:
        try:
            statm = (proc / str(pid) / "statm").read_text(encoding="utf-8")
            rss_bytes += int(statm.split()[1]) * page_size
        except (FileNotFoundError, PermissionError, ValueError, IndexError):
            continue
    return rss_bytes / (1024**3)


@dataclass(frozen=True)
class ResourceLimits:
    deadline: float | None = None
    memory_gib: float | None = None
    max_states: int | None = None


@dataclass
class ExpansionResult:
    complete: bool
    children: set[GraphCode]
    raw_children: int
    seconds: float
    peak_rss_gib: float
    cap_reason: str = ""


def _cap_reason(
    child_count: int, limits: ResourceLimits, current_rss_gib: float
) -> str:
    if limits.deadline is not None and time.monotonic() >= limits.deadline:
        return "wall_time"
    if limits.memory_gib is not None and current_rss_gib >= limits.memory_gib:
        return "memory"
    if limits.max_states is not None and child_count >= limits.max_states:
        return "state_count"
    return ""


def generate_next_level(
    parents: set[GraphCode],
    *,
    workers: int,
    limits: ResourceLimits,
    batch_size: int = 128,
    progress_interval: float = 10.0,
) -> ExpansionResult:
    """Expand a complete level, stopping cleanly if a resource cap is reached."""
    started = time.monotonic()
    last_progress = started
    peak_rss = process_tree_rss_gib()
    children: set[GraphCode] = set()
    raw_children = 0
    cap_reason = ""

    if workers <= 1:
        for parent_index, parent in enumerate(parents, start=1):
            local, local_raw = expand_code(parent)
            children.update(local)
            raw_children += local_raw
            if parent_index % 128 == 0:
                rss = process_tree_rss_gib()
                peak_rss = max(peak_rss, rss)
                cap_reason = _cap_reason(len(children), limits, rss)
                if cap_reason:
                    break
        if not cap_reason:
            rss = process_tree_rss_gib()
            peak_rss = max(peak_rss, rss)
            cap_reason = _cap_reason(len(children), limits, rss)
        return ExpansionResult(
            not cap_reason,
            children,
            raw_children,
            time.monotonic() - started,
            peak_rss,
            cap_reason,
        )

    batches = iter(_batched(parents, batch_size))
    executor = ProcessPoolExecutor(max_workers=workers)
    pending = set()
    exhausted = False
    try:
        while True:
            while not exhausted and len(pending) < workers * 2:
                try:
                    pending.add(executor.submit(_expand_batch, next(batches)))
                except StopIteration:
                    exhausted = True

            if not pending:
                break

            done, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
            for future in done:
                local, local_raw = future.result()
                children.update(local)
                raw_children += local_raw

            now = time.monotonic()
            if done or now - last_progress >= progress_interval:
                rss = process_tree_rss_gib()
                peak_rss = max(peak_rss, rss)
                cap_reason = _cap_reason(len(children), limits, rss)
                if now - last_progress >= progress_interval:
                    print(
                        f"    parents pending={len(pending):,}  "
                        f"unique children={len(children):,}  rss={rss:.2f} GiB",
                        flush=True,
                    )
                    last_progress = now
                if cap_reason:
                    break
    finally:
        if cap_reason:
            for future in pending:
                future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)

    return ExpansionResult(
        not cap_reason and exhausted and not pending,
        children,
        raw_children,
        time.monotonic() - started,
        peak_rss,
        cap_reason,
    )


def seed_code() -> GraphCode:
    return canonical_code(3, ((0, 1), (0, 2), (1, 2)))


def graph_from_code(code: GraphCode) -> ig.Graph:
    vertex_count, edges = decode_graph(code)
    return ig.Graph(n=vertex_count, edges=edges, directed=False)


def validate_topology(code: GraphCode, triangle_level: int) -> None:
    graph = graph_from_code(code)
    if not graph.is_simple() or not graph.is_connected():
        raise ValueError("topology is not a connected simple graph")
    triangle_count = len(graph.cliques(min=3, max=3))
    if triangle_count != triangle_level:
        raise ValueError(
            f"expected {triangle_level} triangles, found {triangle_count}"
        )
    boundary = boundary_edges(code)
    boundary_degree = Counter(vertex for edge in boundary for vertex in edge)
    if not boundary_degree or set(boundary_degree.values()) != {2}:
        raise ValueError("boundary edges do not form cycles")
    boundary_graph = ig.Graph(n=graph.vcount(), edges=boundary)
    active = [vertex for vertex, degree in boundary_degree.items() if degree]
    if len(boundary_graph.subgraph(active).components()) != 1:
        raise ValueError("boundary is not a single cycle")
    if graph.vcount() - graph.ecount() + triangle_level != 1:
        raise ValueError("triangulated-disk Euler invariant failed")
    import networkx as nx

    if not nx.check_planarity(nx.Graph(graph.get_edgelist()))[0]:
        raise ValueError("topology is not planar")


def stable_rank(code: GraphCode) -> bytes:
    return hashlib.blake2b(code, digest_size=16).digest()


def select_samples(codes: set[GraphCode], sample_count: int) -> list[GraphCode]:
    """Choose deterministic examples spread across the available node counts."""
    if sample_count <= 0 or not codes:
        return []
    per_node: dict[int, list[GraphCode]] = defaultdict(list)
    for code in codes:
        bucket = per_node[code[0]]
        bucket.append(code)
        bucket.sort(key=stable_rank)
        del bucket[sample_count:]

    node_counts = sorted(per_node)
    if len(node_counts) <= sample_count:
        chosen_nodes = node_counts
    else:
        chosen_nodes = sorted(
            {
                node_counts[round(index * (len(node_counts) - 1) / (sample_count - 1))]
                for index in range(sample_count)
            }
        )

    selected = [per_node[node_count].pop(0) for node_count in chosen_nodes]
    remaining = sorted(
        (code for bucket in per_node.values() for code in bucket), key=stable_rank
    )
    selected.extend(remaining[: max(0, sample_count - len(selected))])
    return selected[:sample_count]


def topology_id(code: GraphCode) -> str:
    return hashlib.blake2b(code, digest_size=10).hexdigest()


def boundary_cycle(graph: ig.Graph) -> list[int]:
    """Return the inferred outer boundary in cyclic order."""
    adjacency = [set() for _ in range(graph.vcount())]
    graph_adjacency = [set(graph.neighbors(vertex)) for vertex in range(graph.vcount())]
    for left, right in graph.get_edgelist():
        if len(graph_adjacency[left].intersection(graph_adjacency[right])) == 1:
            adjacency[left].add(right)
            adjacency[right].add(left)

    active = [vertex for vertex, neighbors in enumerate(adjacency) if neighbors]
    if not active or any(len(adjacency[vertex]) != 2 for vertex in active):
        raise ValueError("cannot infer a single degree-two outer boundary")

    start = min(active)
    cycle = [start]
    previous = -1
    current = start
    while True:
        choices = sorted(adjacency[current] - ({previous} if previous >= 0 else set()))
        following = choices[0]
        if following == start:
            break
        if following in cycle:
            raise ValueError("outer boundary contains a premature cycle")
        cycle.append(following)
        previous, current = current, following
    if len(cycle) != len(active):
        raise ValueError("outer boundary is disconnected")
    return cycle


def topology_layout(graph: ig.Graph) -> dict[int, tuple[float, float]]:
    """Create a spacious, organic, crossing-free planar embedding.

    A circular-boundary Tutte embedding provides a safe initial drawing, but
    it is never used as the final layout.  Interior vertices are first moved
    to balance triangular face areas.  A second force refinement then moves
    *all* vertices, including the boundary, so high-level motifs do not look
    like regular polygons. Logarithmic face-area barriers discourage local
    folds, and a final intersection check accepts only a planar deformation.
    """
    import numpy as np
    from scipy.optimize import minimize

    cycle = boundary_cycle(graph)
    boundary = set(cycle)
    positions: dict[int, tuple[float, float]] = {}
    for index, vertex in enumerate(cycle):
        angle = math.pi / 2.0 - 2.0 * math.pi * index / len(cycle)
        positions[vertex] = (math.cos(angle), math.sin(angle))

    interior = [vertex for vertex in range(graph.vcount()) if vertex not in boundary]
    interior_index = {vertex: index for index, vertex in enumerate(interior)}
    if interior:
        matrix = np.zeros((len(interior), len(interior)), dtype=float)
        rhs = np.zeros((len(interior), 2), dtype=float)
        for vertex in interior:
            row = interior_index[vertex]
            neighbors = graph.neighbors(vertex)
            matrix[row, row] = len(neighbors)
            for neighbor in neighbors:
                if neighbor in interior_index:
                    matrix[row, interior_index[neighbor]] -= 1.0
                else:
                    rhs[row] += positions[neighbor]
        coordinates = np.linalg.solve(matrix, rhs)
        for vertex, coordinate in zip(interior, coordinates):
            positions[vertex] = (float(coordinate[0]), float(coordinate[1]))

    # Orient every triangular face consistently with the initial planar
    # embedding. Every 3-clique is a face for these triangulated-disk motifs.
    oriented_faces: list[tuple[int, int, int]] = []
    for left, middle, right in graph.cliques(min=3, max=3):
        a, b, c = positions[left], positions[middle], positions[right]
        doubled_area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (
            c[0] - a[0]
        )
        if abs(doubled_area) < 1e-12:
            return positions
        if doubled_area < 0:
            middle, right = right, middle
        oriented_faces.append((left, middle, right))

    variable_index = {vertex: index for index, vertex in enumerate(interior)}
    initial = np.asarray([positions[vertex] for vertex in interior], dtype=float).ravel()

    def objective(values: np.ndarray) -> tuple[float, np.ndarray]:
        all_coordinates = np.asarray(
            [positions[vertex] for vertex in range(graph.vcount())], dtype=float
        )
        for vertex, index in variable_index.items():
            all_coordinates[vertex] = values[2 * index : 2 * index + 2]

        score = 0.0
        gradient = np.zeros_like(values)
        for left, middle, right in oriented_faces:
            xa, ya = all_coordinates[left]
            xb, yb = all_coordinates[middle]
            xc, yc = all_coordinates[right]
            area = ((xb - xa) * (yc - ya) - (yb - ya) * (xc - xa)) / 2.0
            if area <= 1e-10:
                return 1e12 + 1e9 * (1e-10 - area), gradient
            score -= math.log(area)
            derivatives = {
                left: ((yb - yc) / 2.0, (xc - xb) / 2.0),
                middle: ((yc - ya) / 2.0, (xa - xc) / 2.0),
                right: ((ya - yb) / 2.0, (xb - xa) / 2.0),
            }
            for vertex, (dx, dy) in derivatives.items():
                if vertex in variable_index:
                    index = variable_index[vertex]
                    gradient[2 * index] -= dx / area
                    gradient[2 * index + 1] -= dy / area

        regularization = 1e-3
        delta = values - initial
        score += regularization * float(delta @ delta)
        gradient += 2.0 * regularization * delta
        return score, gradient

    if interior:
        initial_score, _ = objective(initial)
        result = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8},
        )
        refined_score, _ = objective(result.x)
        if math.isfinite(refined_score) and refined_score < initial_score:
            for vertex, index in variable_index.items():
                positions[vertex] = (
                    float(result.x[2 * index]),
                    float(result.x[2 * index + 1]),
                )

    return _organic_planar_refinement(graph, positions, oriented_faces)


def _organic_planar_refinement(
    graph: ig.Graph,
    positions: dict[int, tuple[float, float]],
    oriented_faces: Sequence[tuple[int, int, int]],
) -> dict[int, tuple[float, float]]:
    """Relax every vertex while retaining the initial planar embedding.

    The edge and repulsion terms make the drawing resemble an ordinary force
    layout.  Unlike an unconstrained spring layout, the positive-area barrier
    prevents any triangular face from flipping.  For a triangulated disk, that
    keeps non-incident edges from crossing while allowing an irregular hull.
    """
    import numpy as np
    from scipy.optimize import minimize

    vertex_count = graph.vcount()
    if vertex_count < 3:
        return positions

    initial_coordinates = np.asarray(
        [positions[vertex] for vertex in range(vertex_count)], dtype=float
    )
    edges = tuple(graph.get_edgelist())
    edge_lengths = [
        float(np.linalg.norm(initial_coordinates[left] - initial_coordinates[right]))
        for left, right in edges
    ]
    target_edge_length = float(np.median(edge_lengths))

    # These deliberately modest forces break the circular boundary without
    # overwhelming the face-orientation barrier.
    repulsion_weight = 5e-3
    face_barrier_weight = 1e-1
    centroid_weight = 20.0
    scale_weight = 20.0
    minimum_area = 1e-10

    def objective(values: np.ndarray) -> tuple[float, np.ndarray]:
        coordinates = values.reshape((vertex_count, 2))
        gradient = np.zeros_like(coordinates)
        score = 0.0

        # Equal preferred edge lengths reveal local connectivity more clearly
        # than the fixed regular-polygon hull did.
        for left, right in edges:
            difference = coordinates[left] - coordinates[right]
            distance = max(float(np.linalg.norm(difference)), 1e-12)
            error = distance - target_edge_length
            score += error * error
            derivative = 2.0 * error * difference / distance
            gradient[left] += derivative
            gradient[right] -= derivative

        # A weak all-pairs repulsion separates unrelated vertices and labels.
        for left in range(vertex_count):
            for right in range(left + 1, vertex_count):
                difference = coordinates[left] - coordinates[right]
                squared_distance = float(difference @ difference) + 1e-4
                score += repulsion_weight / squared_distance
                derivative = (
                    -2.0
                    * repulsion_weight
                    * difference
                    / (squared_distance * squared_distance)
                )
                gradient[left] += derivative
                gradient[right] -= derivative

        for left, middle, right in oriented_faces:
            xa, ya = coordinates[left]
            xb, yb = coordinates[middle]
            xc, yc = coordinates[right]
            area = ((xb - xa) * (yc - ya) - (yb - ya) * (xc - xa)) / 2.0
            if area <= minimum_area:
                return 1e12 + 1e9 * (minimum_area - area), gradient.ravel()
            score -= face_barrier_weight * math.log(area)
            derivatives = (
                (left, (yb - yc) / 2.0, (xc - xb) / 2.0),
                (middle, (yc - ya) / 2.0, (xa - xc) / 2.0),
                (right, (ya - yb) / 2.0, (xb - xa) / 2.0),
            )
            for vertex, dx, dy in derivatives:
                gradient[vertex, 0] -= face_barrier_weight * dx / area
                gradient[vertex, 1] -= face_barrier_weight * dy / area

        centroid = coordinates.mean(axis=0)
        score += centroid_weight * float(centroid @ centroid)
        gradient += 2.0 * centroid_weight * centroid / vertex_count

        mean_squared_radius = float(np.mean(coordinates * coordinates)) * 2.0
        scale_error = mean_squared_radius - 1.0
        score += scale_weight * scale_error * scale_error
        gradient += (
            4.0 * scale_weight * scale_error * coordinates / vertex_count
        )
        return score, gradient.ravel()

    initial = initial_coordinates.ravel()
    initial_score, _ = objective(initial)
    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8},
    )
    refined_score, _ = objective(result.x)
    if not math.isfinite(refined_score) or refined_score >= initial_score:
        return positions

    refined = result.x.reshape((vertex_count, 2))

    def acceptable(coordinates: np.ndarray) -> bool:
        candidate_edge_lengths = [
            float(np.linalg.norm(coordinates[left] - coordinates[right]))
            for left, right in edges
        ]
        minimum_vertex_spacing = 0.22 * float(np.median(candidate_edge_lengths))
        for left in range(vertex_count):
            for right in range(left + 1, vertex_count):
                if (
                    float(np.linalg.norm(coordinates[left] - coordinates[right]))
                    < minimum_vertex_spacing
                ):
                    return False
        for left, middle, right in oriented_faces:
            a, b, c = coordinates[left], coordinates[middle], coordinates[right]
            area = (
                (b[0] - a[0]) * (c[1] - a[1])
                - (b[1] - a[1]) * (c[0] - a[0])
            ) / 2.0
            if area <= minimum_area:
                return False
        return _straight_line_layout_is_planar(edges, coordinates)

    if not acceptable(refined):
        # Global overlap is possible even when every individual face retains
        # its orientation. Move only as far toward the force-layout solution
        # as is safe. Halving also leaves visible clearance from the fold.
        displacement = refined - initial_coordinates
        for exponent in range(1, 16):
            candidate = initial_coordinates + (0.5**exponent) * displacement
            if acceptable(candidate):
                refined = candidate
                break
        else:
            return positions
    return {
        vertex: (float(refined[vertex, 0]), float(refined[vertex, 1]))
        for vertex in range(vertex_count)
    }


def _straight_line_layout_is_planar(
    edges: Sequence[Edge], coordinates: object
) -> bool:
    """Return whether non-incident straight edges are mutually disjoint."""

    def orientation(a: object, b: object, c: object) -> float:
        return float(
            (b[0] - a[0]) * (c[1] - a[1])
            - (b[1] - a[1]) * (c[0] - a[0])
        )

    def on_segment(a: object, b: object, point: object, epsilon: float) -> bool:
        return (
            min(a[0], b[0]) - epsilon <= point[0] <= max(a[0], b[0]) + epsilon
            and min(a[1], b[1]) - epsilon
            <= point[1]
            <= max(a[1], b[1]) + epsilon
        )

    epsilon = 1e-10
    for edge_index, (left, right) in enumerate(edges):
        a, b = coordinates[left], coordinates[right]
        for other_left, other_right in edges[edge_index + 1 :]:
            if len({left, right, other_left, other_right}) < 4:
                continue
            c, d = coordinates[other_left], coordinates[other_right]
            abc = orientation(a, b, c)
            abd = orientation(a, b, d)
            cda = orientation(c, d, a)
            cdb = orientation(c, d, b)
            if (
                (abc > epsilon and abd < -epsilon)
                or (abc < -epsilon and abd > epsilon)
            ) and (
                (cda > epsilon and cdb < -epsilon)
                or (cda < -epsilon and cdb > epsilon)
            ):
                return False
            if abs(abc) <= epsilon and on_segment(a, b, c, epsilon):
                return False
            if abs(abd) <= epsilon and on_segment(a, b, d, epsilon):
                return False
            if abs(cda) <= epsilon and on_segment(c, d, a, epsilon):
                return False
            if abs(cdb) <= epsilon and on_segment(c, d, b, epsilon):
                return False
    return True


def write_sample(
    code: GraphCode, triangle_level: int, sample_index: int, output_root: Path
) -> dict[str, object]:
    validate_topology(code, triangle_level)
    graph = graph_from_code(code)
    identifier = topology_id(code)
    graph.vs["id"] = list(range(graph.vcount()))
    graph.vs["label"] = ["1"] * graph.vcount()
    graph["trianglecount"] = triangle_level
    graph["nodecount"] = graph.vcount()
    graph["edgecount"] = graph.ecount()
    graph["canonicalid"] = identifier

    level_dir = output_root / "samples" / f"tri_{triangle_level:03d}"
    level_dir.mkdir(parents=True, exist_ok=True)
    stem = f"sample_{sample_index:02d}_n{graph.vcount():03d}_{identifier}"
    gml_path = level_dir / f"{stem}.gml"
    png_path = level_dir / f"{stem}.png"
    graph.write_gml(str(gml_path))
    render_graph_png(graph, triangle_level, png_path)
    return {
        "triangle_level": triangle_level,
        "sample_index": sample_index,
        "canonical_id": identifier,
        "nodes": graph.vcount(),
        "edges": graph.ecount(),
        "gml_path": str(gml_path.relative_to(output_root)),
        "png_path": str(png_path.relative_to(output_root)),
    }


def render_graph_png(graph: ig.Graph, triangle_level: int, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    topology = nx.Graph()
    topology.add_nodes_from(range(graph.vcount()))
    topology.add_edges_from(graph.get_edgelist())
    positions = topology_layout(graph)
    figure, axis = plt.subplots(figsize=(8, 8))
    nx.draw_networkx_edges(
        topology,
        positions,
        ax=axis,
        edge_color="#718096",
        width=1.9,
        alpha=0.88,
    )
    nx.draw_networkx_nodes(
        topology,
        positions,
        ax=axis,
        node_color="#4299e1",
        edgecolors="#1a365d",
        node_size=540,
        linewidths=1.5,
    )
    nx.draw_networkx_labels(
        topology,
        positions,
        ax=axis,
        font_size=9,
        font_color="white",
        font_weight="bold",
    )
    axis.set_title(
        f"Triangle-grown topology: {triangle_level} triangle"
        f"{'s' if triangle_level != 1 else ''}\n"
        f"{graph.vcount()} nodes, {graph.ecount()} edges, cell type=1",
        fontsize=15,
    )
    axis.margins(0.12)
    axis.set_aspect("equal")
    axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(figure)


LEVEL_FIELDS = (
    "triangle_level",
    "status",
    "unique_topologies",
    "partial_unique_seen",
    "raw_children",
    "node_min",
    "node_max",
    "growth_factor",
    "level_seconds",
    "total_seconds",
    "peak_rss_gib",
    "cap_reason",
)


def write_csv(path: Path, rows: Sequence[dict[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_count_plot(rows: Sequence[dict[str, object]], output_path: Path) -> None:
    exact = [row for row in rows if row["status"] == "exact"]
    if not exact:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    levels = [int(row["triangle_level"]) for row in exact]
    counts = [int(row["unique_topologies"]) for row in exact]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(levels, counts, marker="o", color="#2b6cb0")
    axis.set_yscale("log")
    axis.set_xlabel("Triangle level")
    axis.set_ylabel("Exact non-isomorphic topologies (log scale)")
    axis.set_title("Triangle-grown topology count")
    axis.grid(True, which="both", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def write_run_summary(
    output_root: Path,
    args: argparse.Namespace,
    rows: Sequence[dict[str, object]],
    started_iso: str,
    started_monotonic: float,
) -> None:
    exact_rows = [row for row in rows if row["status"] == "exact"]
    incomplete = next((row for row in rows if row["status"] != "exact"), None)
    summary = {
        "started_at": started_iso,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "total_seconds": round(time.monotonic() - started_monotonic, 6),
        "last_exact_level": max((int(row["triangle_level"]) for row in exact_rows), default=0),
        "termination_reason": incomplete["cap_reason"] if incomplete else "max_triangles",
        "configuration": vars(args),
        "versions": {
            "python": platform.python_version(),
            "igraph": ig.__version__,
        },
    }
    (output_root / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def persist_results(
    output_root: Path,
    level_rows: Sequence[dict[str, object]],
    node_rows: Sequence[dict[str, object]],
    sample_rows: Sequence[dict[str, object]],
) -> None:
    write_csv(output_root / "level_counts.csv", level_rows, LEVEL_FIELDS)
    write_csv(
        output_root / "node_count_distribution.csv",
        node_rows,
        ("triangle_level", "node_count", "topology_count"),
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
    write_count_plot(level_rows, output_root / "topology_counts.png")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enumerate pure unlabeled topologies generated by triangle BFS."
    )
    parser.add_argument("--max-triangles", type=int, default=40)
    parser.add_argument("--workers", type=int, default=min(8, max(1, os.cpu_count() or 1)))
    parser.add_argument("--wall-seconds", type=float, default=3600.0)
    parser.add_argument("--memory-gib", type=float, default=20.0)
    parser.add_argument("--max-states-per-level", type=int, default=5_000_000)
    parser.add_argument("--samples-per-level", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("enumerate_topology/results")
    )
    args = parser.parse_args(argv)
    if args.max_triangles < 1:
        parser.error("--max-triangles must be at least 1")
    if args.workers < 1 or args.batch_size < 1:
        parser.error("--workers and --batch-size must be at least 1")
    if args.wall_seconds <= 0 or args.memory_gib <= 0:
        parser.error("resource caps must be positive")
    if args.max_states_per_level < 1 or args.samples_per_level < 0:
        parser.error("state cap must be positive and sample count non-negative")
    return args


def run(args: argparse.Namespace) -> int:
    output_root: Path = args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)
    started_monotonic = time.monotonic()
    started_iso = datetime.now(timezone.utc).isoformat()
    deadline = started_monotonic + args.wall_seconds
    limits = ResourceLimits(deadline, args.memory_gib, args.max_states_per_level)

    states = {seed_code()}
    level_rows: list[dict[str, object]] = []
    node_rows: list[dict[str, object]] = []
    sample_rows: list[dict[str, object]] = []
    previous_count: int | None = None
    peak_rss = process_tree_rss_gib()

    for triangle_level in range(1, args.max_triangles + 1):
        if triangle_level == 1:
            raw_children = 0
            level_seconds = 0.0
        else:
            print(
                f"level {triangle_level}: expanding {len(states):,} parent topologies",
                flush=True,
            )
            expansion = generate_next_level(
                states,
                workers=args.workers,
                limits=limits,
                batch_size=args.batch_size,
            )
            peak_rss = max(peak_rss, expansion.peak_rss_gib)
            raw_children = expansion.raw_children
            level_seconds = expansion.seconds
            if not expansion.complete:
                level_rows.append(
                    {
                        "triangle_level": triangle_level,
                        "status": "incomplete",
                        "unique_topologies": "",
                        "partial_unique_seen": len(expansion.children),
                        "raw_children": raw_children,
                        "node_min": "",
                        "node_max": "",
                        "growth_factor": "",
                        "level_seconds": round(level_seconds, 6),
                        "total_seconds": round(time.monotonic() - started_monotonic, 6),
                        "peak_rss_gib": round(peak_rss, 6),
                        "cap_reason": expansion.cap_reason,
                    }
                )
                persist_results(output_root, level_rows, node_rows, sample_rows)
                write_run_summary(
                    output_root, args, level_rows, started_iso, started_monotonic
                )
                print(
                    f"level {triangle_level}: incomplete ({expansion.cap_reason}); "
                    f"discarding {len(expansion.children):,} partial states",
                    flush=True,
                )
                return 2
            states = expansion.children

        node_distribution = Counter(code[0] for code in states)
        current_count = len(states)
        growth_factor = (
            "" if previous_count is None else round(current_count / previous_count, 6)
        )
        level_rows.append(
            {
                "triangle_level": triangle_level,
                "status": "exact",
                "unique_topologies": current_count,
                "partial_unique_seen": "",
                "raw_children": raw_children,
                "node_min": min(node_distribution),
                "node_max": max(node_distribution),
                "growth_factor": growth_factor,
                "level_seconds": round(level_seconds, 6),
                "total_seconds": round(time.monotonic() - started_monotonic, 6),
                "peak_rss_gib": round(peak_rss, 6),
                "cap_reason": "",
            }
        )
        node_rows.extend(
            {
                "triangle_level": triangle_level,
                "node_count": node_count,
                "topology_count": count,
            }
            for node_count, count in sorted(node_distribution.items())
        )
        for sample_index, code in enumerate(
            select_samples(states, args.samples_per_level), start=1
        ):
            sample_rows.append(
                write_sample(code, triangle_level, sample_index, output_root)
            )
        previous_count = current_count
        persist_results(output_root, level_rows, node_rows, sample_rows)
        write_run_summary(output_root, args, level_rows, started_iso, started_monotonic)
        print(
            f"level {triangle_level}: exact={current_count:,}  "
            f"nodes={min(node_distribution)}..{max(node_distribution)}  "
            f"seconds={level_seconds:.2f}",
            flush=True,
        )

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
