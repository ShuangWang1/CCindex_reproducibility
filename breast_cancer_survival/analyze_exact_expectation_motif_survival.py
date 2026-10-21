#!/usr/bin/env python3
"""Exact label-permutation expectations and survival for 644 focal motifs.

TRI-1 uses the saved Delaunay-simplex counts.  TRI-2 through TRI-7 are
recounted directly because the original tree search can omit low-frequency
motifs when an ancestor is pruned.  For every core and labeled motif m,

    E[C_m] = N_S * |Aut(S)|/|Aut(m)| * prod_a (n_a)_(k_a) / (N)_k

where N_S is obtained by matching the same topology after replacing all graph
and pattern labels with a single dummy label, then retaining only target vertex
sets whose induced graph is exactly S. This induced-shape condition is required
for the automorphism-ratio probability to apply to every occurrence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/bc-exact-expectation-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import igraph as ig
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay

from analyze_ki67_survival import COLORS, logrank_test, step_function
from analyze_motif_survival import benjamini_hochberg, fit_cox
from analyze_rank_normalized_motif_survival import load_focal_pool
from normalize_significant_motif_frequencies import (
    clinical_core_cohort,
    high_level_counts,
    tri1_counts,
)
from screen_all_rank_normalized_motifs import plot_delaunay_topology
from trace_significant_motifs import recover_target_topologies
from visualize_top_motif_survival import CELL_TYPE_ORDER


PSEUDOCOUNT = 0.5
DEFAULT_SEED = 1729
CORE_COLUMNS = [
    "triangle_level",
    "wl_hash",
    "PID",
    "sample",
    "source_sample",
    "total_cells",
    "shape_id",
    "unlabeled_shape_count",
    "saved_observed_count",
    "observed_count",
    "expected_count",
    "observed_minus_expected",
    "observed_expected_ratio",
    "log2_observed_expected",
]


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pool-csv",
        type=Path,
        default=here / "motif_results/significant_mtf_trace/potential_motif_pool.csv",
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=repo
        / "breast_cancer_classification_workstation/classification_by_level/mtf_occurrence/mrmr_selection/parameter_tuning_bz/artifacts",
    )
    parser.add_argument(
        "--motif-root",
        type=Path,
        default=repo / "breast_cancer_meta/BZ_res_threshold_10_cpp_0811",
    )
    parser.add_argument(
        "--graph-root", type=Path, default=repo / "breast_cancer/BZ_samples_intra"
    )
    parser.add_argument(
        "--metadata", type=Path, default=repo / "breast_cancer/Basel_PatientMetadata.csv"
    )
    parser.add_argument(
        "--label-map",
        type=Path,
        default=repo / "breast_cancer/BZ_samples_intra/node_label_cell_type_mapping.csv",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=here / "motif_results/normalized_by_get_expectation_motif_analysis",
    )
    parser.add_argument("--levels", nargs="+", type=int, default=list(range(1, 8)))
    parser.add_argument(
        "--workers", type=int, default=min(8, max(1, (os.cpu_count() or 2) - 1))
    )
    parser.add_argument("--validation-sample", default="sample1")
    parser.add_argument("--validation-permutations", type=int, default=300)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--rebuild-counts", action="store_true")
    return parser.parse_args()


def falling(x: int, j: int) -> int:
    """Return the falling factorial (x)_j, including the impossible j>x case."""
    x, j = int(x), int(j)
    if x < 0 or j < 0:
        raise ValueError("Falling factorial arguments must be nonnegative")
    if j > x:
        return 0
    return math.prod(x - i for i in range(j))


def expected_count(
    shape_count: int,
    aut_shape: int,
    aut_labeled: int,
    motif_type_counts: dict[int, int],
    sample_type_counts: dict[int, int],
    total_cells: int,
) -> float:
    """Compute the exact expected count under fixed-composition label permutation."""
    if shape_count < 0 or aut_shape < 1 or aut_labeled < 1:
        raise ValueError("Counts and automorphism sizes are invalid")
    if aut_shape % aut_labeled:
        raise ValueError("The label-preserving automorphism group is not a subgroup")
    k = sum(int(value) for value in motif_type_counts.values())
    denominator = falling(total_cells, k)
    if denominator == 0:
        raise ValueError("Motif has more vertices than the sample")
    numerator = math.prod(
        falling(sample_type_counts.get(int(label), 0), int(count))
        for label, count in motif_type_counts.items()
    )
    probability = (aut_shape // aut_labeled) * numerator / denominator
    if probability < 0 or probability > 1 + 1e-12:
        raise ValueError(f"Invalid labeling probability: {probability}")
    return float(shape_count) * min(float(probability), 1.0)


def parse_edges(edge_list: str) -> tuple[tuple[int, int], ...]:
    edges = []
    for encoded in str(edge_list).split(";"):
        left, right = encoded.split("-")
        edge = tuple(sorted((int(left), int(right))))
        edges.append(edge)
    return tuple(sorted(set(edges)))


def canonical_shape(
    node_count: int, edges: Sequence[tuple[int, int]]
) -> tuple[str, str, int]:
    """Return a stable unlabeled shape ID, canonical edge encoding, and |Aut(S)|."""
    graph = ig.Graph(n=int(node_count), edges=list(edges), directed=False)
    graph.simplify()
    canonical = graph.permute_vertices(graph.canonical_permutation())
    canonical_edges = sorted(tuple(sorted(edge.tuple)) for edge in canonical.es)
    encoding = ";".join(f"{left}-{right}" for left, right in canonical_edges)
    signature = f"n={node_count}|{encoding}"
    digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:16]
    return f"shape_{node_count}_{digest}", encoding, int(graph.count_automorphisms())


def labeled_automorphisms(
    node_labels: str, edges: Sequence[tuple[int, int]]
) -> int:
    labels = [int(value) for value in str(node_labels).split("&")]
    graph = ig.Graph(n=len(labels), edges=list(edges), directed=False)
    graph.simplify()
    return int(graph.count_automorphisms(color=labels))


def build_topology_tables(
    pool: pd.DataFrame, artifacts_root: Path, motif_root: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    topology = recover_target_topologies(pool, artifacts_root, motif_root)
    motifs = pool.merge(
        topology,
        on=["triangle_level", "wl_hash"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_topology"),
    )
    rows = []
    motif_rows = []
    for row in motifs.itertuples(index=False):
        edges = parse_edges(str(row.edge_list))
        labels = str(row.node_labels)
        node_count = len(labels.split("&"))
        shape_id, canonical_edges, aut_shape = canonical_shape(node_count, edges)
        aut_labeled = labeled_automorphisms(labels, edges)
        if aut_shape % aut_labeled:
            raise ValueError(f"Invalid automorphism ratio for {row.wl_hash}")
        motif_rows.append(
            {
                "triangle_level": int(row.triangle_level),
                "wl_hash": str(row.wl_hash),
                "shape_id": shape_id,
                "node_count": node_count,
                "node_labels": labels,
                "edge_list": str(row.edge_list),
                "canonical_edge_list": canonical_edges,
                "aut_shape": aut_shape,
                "aut_labeled": aut_labeled,
                "label_orbit_size": aut_shape // aut_labeled,
                "label_multiplicities": json.dumps(
                    dict(sorted(Counter(map(int, labels.split("&"))).items()))
                ),
            }
        )
        rows.append(
            {
                "triangle_level": int(row.triangle_level),
                "shape_id": shape_id,
                "node_count": node_count,
                "canonical_edge_list": canonical_edges,
                "representative_edge_list": str(row.edge_list),
                "aut_shape": aut_shape,
            }
        )
    motif_metadata = pd.DataFrame(motif_rows)
    shapes = pd.DataFrame(rows).drop_duplicates("shape_id")
    consistency = pd.DataFrame(rows).groupby("shape_id").agg(
        levels=("triangle_level", "nunique"),
        canonical_edges=("canonical_edge_list", "nunique"),
        aut_sizes=("aut_shape", "nunique"),
    )
    if not (
        consistency[["levels", "canonical_edges", "aut_sizes"]].eq(1).all().all()
    ):
        raise ValueError("Canonical shape metadata are inconsistent")
    shapes = shapes.sort_values(["triangle_level", "shape_id"]).reset_index(drop=True)
    motif_metadata = motif_metadata.sort_values(
        ["triangle_level", "wl_hash"]
    ).reset_index(drop=True)
    return motif_metadata, shapes


def _custom_graph(
    node_count: int, edges: Sequence[tuple[int, int]], labels: Sequence[str]
):
    import graph_pattern_cpp

    graph = graph_pattern_cpp.Graph()
    for index, label in enumerate(labels):
        graph.addVertex(index, str(label))
    for left, right in edges:
        graph.addEdge(int(left), int(right))
    return graph


def _read_gml(path: Path) -> tuple[int, tuple[tuple[int, int], ...], list[str], np.ndarray]:
    graph = ig.Graph.Read_GML(str(path))
    edges = tuple(sorted(set(tuple(sorted(edge.tuple)) for edge in graph.es)))
    labels = [str(int(float(value))) for value in graph.vs["label"]]
    coordinates = np.column_stack(
        [np.asarray(graph.vs["x"], dtype=float), np.asarray(graph.vs["y"], dtype=float)]
    )
    return graph.vcount(), edges, labels, coordinates


def _pattern_payload(row: pd.Series, dummy: bool = False) -> tuple[int, tuple, tuple]:
    if dummy:
        node_count = int(row.node_count)
        labels = tuple("0" for _ in range(node_count))
        edge_list = str(
            row.representative_edge_list
            if "representative_edge_list" in row.index
            else row.edge_list
        )
    else:
        labels = tuple(str(row.node_labels).split("&"))
        node_count = len(labels)
        edge_list = str(row.edge_list)
    return node_count, parse_edges(edge_list), labels


def _canonical_label_orbit(
    labels: Sequence[str], automorphisms: Sequence[Sequence[int]]
) -> tuple[str, ...]:
    """Canonicalize a labeling under automorphisms of one exact shape."""
    values = tuple(map(str, labels))
    return min(tuple(values[int(mapping[index])] for index in range(len(values)))
               for mapping in automorphisms)


def _exact_isomorphism_mapping(
    node_count: int,
    representative_edges: Sequence[tuple[int, int]],
    induced_edges: Sequence[tuple[int, int]],
) -> tuple[int, ...]:
    """Map representative vertices to an exactly isomorphic graph (k <= 9)."""
    rep_adj = [set() for _ in range(node_count)]
    ind_adj = [set() for _ in range(node_count)]
    for left, right in representative_edges:
        rep_adj[int(left)].add(int(right))
        rep_adj[int(right)].add(int(left))
    for left, right in induced_edges:
        ind_adj[int(left)].add(int(right))
        ind_adj[int(right)].add(int(left))

    def signature(adjacency: list[set[int]], vertex: int) -> tuple[object, ...]:
        neighbors = adjacency[vertex]
        triangles = sum(
            other in adjacency[first]
            for first in neighbors
            for other in neighbors
            if first < other
        )
        return (
            len(neighbors),
            triangles,
            tuple(sorted(len(adjacency[neighbor]) for neighbor in neighbors)),
        )

    rep_signatures = [signature(rep_adj, vertex) for vertex in range(node_count)]
    ind_signatures = [signature(ind_adj, vertex) for vertex in range(node_count)]
    candidates = [
        [target for target in range(node_count) if ind_signatures[target] == rep_signatures[source]]
        for source in range(node_count)
    ]
    order = sorted(
        range(node_count),
        key=lambda vertex: (len(candidates[vertex]), -len(rep_adj[vertex]), vertex),
    )
    mapping = [-1] * node_count
    used: set[int] = set()

    def search(depth: int) -> bool:
        if depth == node_count:
            return True
        source = order[depth]
        for target in candidates[source]:
            if target in used:
                continue
            if any(
                ((mapped_source in rep_adj[source]) != (mapped_target in ind_adj[target]))
                for mapped_source, mapped_target in enumerate(mapping)
                if mapped_target >= 0
            ):
                continue
            mapping[source] = target
            used.add(target)
            if search(depth + 1):
                return True
            used.remove(target)
            mapping[source] = -1
        return False

    if not search(0):
        raise RuntimeError("Exact occurrence failed representative isomorphism")
    return tuple(mapping)


def _shape_counting_payloads(
    motif_metadata: pd.DataFrame, shapes: pd.DataFrame
) -> list[dict[str, object]]:
    """Build picklable exact-shape label classifiers for worker processes."""
    payloads: list[dict[str, object]] = []
    for shape_index, shape in shapes.loc[shapes["triangle_level"].gt(1)].iterrows():
        node_count = int(shape.node_count)
        edges = parse_edges(str(shape.representative_edge_list))
        representative = ig.Graph(n=node_count, edges=list(edges), directed=False)
        automorphisms = [tuple(mapping) for mapping in representative.get_automorphisms_vf2()]
        label_lookup: dict[tuple[str, ...], list[int]] = {}
        members = motif_metadata.index[motif_metadata["shape_id"].eq(shape.shape_id)]
        for motif_index in members:
            motif = motif_metadata.loc[motif_index]
            motif_edges = parse_edges(str(motif.edge_list))
            motif_graph = ig.Graph(n=node_count, edges=list(motif_edges), directed=False)
            isomorphisms = representative.get_isomorphisms_vf2(motif_graph)
            if not isomorphisms:
                raise ValueError(f"Shape mapping failed for {motif.wl_hash}")
            # igraph maps each representative vertex to a motif-graph vertex.
            mapping = isomorphisms[0]
            motif_labels = tuple(str(motif.node_labels).split("&"))
            labels_on_representative = tuple(motif_labels[mapping[index]] for index in range(node_count))
            key = _canonical_label_orbit(labels_on_representative, automorphisms)
            # Distinct motif hashes may retain construction provenance even when
            # they resolve to the same final labeled graph. They share counts.
            label_lookup.setdefault(key, []).append(int(motif_index))
        payloads.append(
            {
                "shape_index": int(shape_index),
                "node_count": node_count,
                "edges": edges,
                "edge_count": len(edges),
                "automorphisms": automorphisms,
                "label_lookup": label_lookup,
            }
        )
    return payloads


def _count_core_worker(
    core_index: int,
    graph_path: str,
    shape_payloads: list[dict[str, object]],
    n_motifs: int,
    n_shapes: int,
    known_labels: list[int],
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, int]:
    import graph_pattern_cpp

    node_count, graph_edges, graph_labels, _ = _read_gml(Path(graph_path))
    dummy_labels = ["0"] * node_count
    shape_graph = _custom_graph(node_count, graph_edges, dummy_labels)
    shape_cache = graph_pattern_cpp.GraphCache(shape_graph)
    observed = np.zeros(n_motifs, dtype=np.int64)
    shape_counts = np.zeros(n_shapes, dtype=np.int64)
    adjacency = [set() for _ in range(node_count)]
    for left, right in graph_edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    for payload in shape_payloads:
        nodes = int(payload["node_count"])
        edges = payload["edges"]
        pattern = _custom_graph(nodes, edges, ["0"] * nodes)
        _, matches = graph_pattern_cpp.find_pattern_matches(
            shape_cache, pattern, True, 0, None
        )
        automorphisms = payload["automorphisms"]
        label_lookup = payload["label_lookup"]
        mapping_cache: dict[tuple[tuple[int, int], ...], tuple[int, ...]] = {}
        exact_count = 0
        for vertices in matches:
            vertices = sorted(map(int, vertices))
            position = {vertex: index for index, vertex in enumerate(vertices)}
            induced_edges = tuple(
                (position[left], position[right])
                for left in vertices
                for right in adjacency[left]
                if left < right and right in position
            )
            # The matcher guarantees all pattern edges. Equal edge counts therefore
            # mean the induced graph is exactly this unlabeled shape, with no chords.
            if len(induced_edges) != int(payload["edge_count"]):
                continue
            exact_count += 1
            induced_edges = tuple(sorted(induced_edges))
            mapping = mapping_cache.get(induced_edges)
            if mapping is None:
                mapping = _exact_isomorphism_mapping(nodes, edges, induced_edges)
                mapping_cache[induced_edges] = mapping
            labels_on_representative = tuple(
                graph_labels[vertices[mapping[index]]] for index in range(nodes)
            )
            key = _canonical_label_orbit(labels_on_representative, automorphisms)
            motif_indices = label_lookup.get(key, ())
            for motif_index in motif_indices:
                observed[int(motif_index)] += 1
        shape_counts[int(payload["shape_index"])] = exact_count
    composition = Counter(map(int, graph_labels))
    cell_counts = np.asarray([composition.get(label, 0) for label in known_labels], dtype=np.int64)
    return core_index, observed, shape_counts, cell_counts, node_count


def _atomic_npz(path: Path, **arrays: object) -> None:
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _count_fingerprint(
    cohort: pd.DataFrame,
    motif_metadata: pd.DataFrame,
    shapes: pd.DataFrame,
    graph_root: Path,
) -> str:
    graph_stats = []
    for sample in cohort["sample"].astype(str):
        path = graph_root / f"{sample}.gml"
        stat = path.stat()
        graph_stats.append((sample, stat.st_size, stat.st_mtime_ns))
    value = {
        "counting_semantics": "exact_induced_shape_v2",
        "motifs": motif_metadata[
            ["triangle_level", "wl_hash", "node_labels", "edge_list"]
        ].to_dict("records"),
        "shapes": shapes[["shape_id", "representative_edge_list"]].to_dict("records"),
        "graphs": graph_stats,
    }
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def exact_graph_counts(
    cohort: pd.DataFrame,
    motif_metadata: pd.DataFrame,
    shapes: pd.DataFrame,
    graph_root: Path,
    motif_root: Path,
    known_labels: list[int],
    checkpoint: Path,
    workers: int,
    rebuild: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return exact observed, N_S, cell-count, and total-cell matrices."""
    n_cores, n_motifs, n_shapes = len(cohort), len(motif_metadata), len(shapes)
    fingerprint = _count_fingerprint(cohort, motif_metadata, shapes, graph_root)
    if checkpoint.is_file() and not rebuild:
        with np.load(checkpoint, allow_pickle=False) as stored:
            saved_fingerprint = str(stored["fingerprint"].item())
            if saved_fingerprint != fingerprint:
                raise ValueError(
                    "Count checkpoint does not match current graphs/topologies; "
                    "rerun with --rebuild-counts"
                )
            observed = stored["observed"]
            shape_counts = stored["shape_counts"]
            cell_counts = stored["cell_counts"]
            total_cells = stored["total_cells"]
            completed = stored["completed"].astype(bool)
    else:
        observed = np.zeros((n_cores, n_motifs), dtype=np.int64)
        shape_counts = np.zeros((n_cores, n_shapes), dtype=np.int64)
        cell_counts = np.zeros((n_cores, len(known_labels)), dtype=np.int64)
        total_cells = np.zeros(n_cores, dtype=np.int64)
        completed = np.zeros(n_cores, dtype=bool)

    expected_shapes = (n_cores, n_shapes)
    if (
        observed.shape != (n_cores, n_motifs)
        or shape_counts.shape != expected_shapes
        or cell_counts.shape != (n_cores, len(known_labels))
    ):
        raise ValueError("Count checkpoint dimensions do not match current inputs")

    high_motifs = motif_metadata.loc[motif_metadata["triangle_level"].gt(1)]
    high_shapes = shapes.loc[shapes["triangle_level"].gt(1)]
    shape_payloads = _shape_counting_payloads(motif_metadata, shapes)

    # TRI-1 is defined by Delaunay faces, not by every three-clique in the graph.
    tri1 = motif_metadata.loc[motif_metadata["triangle_level"].eq(1)].reset_index()
    tri1_hashes = tri1["wl_hash"].astype(str).tolist()
    tri1_shape_index = int(shapes.index[shapes["triangle_level"].eq(1)][0])
    for core_index, core in enumerate(cohort.itertuples(index=False)):
        path = motif_root / str(core.sample) / "ccindex_tri_1.csv"
        frame = pd.read_csv(path, usecols=["wl_hash", "matches"], dtype={"wl_hash": str})
        counts = pd.to_numeric(frame["matches"], errors="raise")
        lookup = frame.assign(matches=counts).set_index("wl_hash")["matches"]
        observed[core_index, tri1["index"].to_numpy(int)] = lookup.reindex(
            tri1_hashes, fill_value=0
        ).to_numpy(np.int64)
        shape_counts[core_index, tri1_shape_index] = int(counts.sum())

    pending = np.flatnonzero(~completed)
    if len(pending):
        print(
            f"Exact recount: {len(pending)} cores pending, {workers} workers, "
            f"{len(shape_payloads)} induced unlabeled shapes/core; labeled counts classified by orbit",
            flush=True,
        )
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _count_core_worker,
                    int(core_index),
                    str(graph_root / f"{cohort.iloc[core_index]['sample']}.gml"),
                    shape_payloads,
                    n_motifs,
                    n_shapes,
                    known_labels,
                ): int(core_index)
                for core_index in pending
            }
            completed_this_run = 0
            for future in as_completed(futures):
                core_index, direct, unlabeled, composition, node_count = future.result()
                observed[core_index, high_motifs.index.to_numpy(int)] = direct[
                    high_motifs.index.to_numpy(int)
                ]
                shape_counts[core_index, high_shapes.index.to_numpy(int)] = unlabeled[
                    high_shapes.index.to_numpy(int)
                ]
                cell_counts[core_index] = composition
                total_cells[core_index] = node_count
                completed[core_index] = True
                completed_this_run += 1
                _atomic_npz(
                    checkpoint,
                    fingerprint=np.asarray(fingerprint),
                    observed=observed,
                    shape_counts=shape_counts,
                    cell_counts=cell_counts,
                    total_cells=total_cells,
                    completed=completed,
                )
                if completed_this_run % 5 == 0 or completed.all():
                    print(
                        f"  exact recount completed {int(completed.sum())}/{n_cores} cores",
                        flush=True,
                    )
    else:
        print("Reused complete exact-count checkpoint", flush=True)

    if not completed.all() or (cell_counts.sum(axis=1) != total_cells).any():
        raise ValueError("Exact count checkpoint is incomplete or cell totals do not reconcile")
    return observed, shape_counts, cell_counts, total_cells


def saved_observed_counts(
    pool: pd.DataFrame,
    cohort: pd.DataFrame,
    artifacts_root: Path,
    motif_root: Path,
) -> np.ndarray:
    saved = np.zeros((len(cohort), len(pool)), dtype=float)
    for level in sorted(pool["triangle_level"].unique()):
        positions = np.flatnonzero(pool["triangle_level"].to_numpy(int) == int(level))
        selected = pool.iloc[positions].reset_index(drop=True)
        if int(level) == 1:
            values, _ = tri1_counts(selected, cohort, motif_root)
        else:
            values, _ = high_level_counts(int(level), selected, cohort, artifacts_root)
        saved[:, positions] = values
    return saved.astype(np.int64)


def calculate_expectations(
    motif_metadata: pd.DataFrame,
    shapes: pd.DataFrame,
    shape_counts: np.ndarray,
    cell_counts: np.ndarray,
    known_labels: list[int],
) -> np.ndarray:
    shape_lookup = {value: index for index, value in enumerate(shapes["shape_id"])}
    expectations = np.zeros((len(shape_counts), len(motif_metadata)), dtype=float)
    for motif_index, motif in motif_metadata.iterrows():
        shape_column = shape_lookup[str(motif.shape_id)]
        multiplicities = Counter(map(int, str(motif.node_labels).split("&")))
        for core_index in range(len(shape_counts)):
            sample_counts = {
                label: int(cell_counts[core_index, position])
                for position, label in enumerate(known_labels)
            }
            expectations[core_index, motif_index] = expected_count(
                int(shape_counts[core_index, shape_column]),
                int(motif.aut_shape),
                int(motif.aut_labeled),
                dict(multiplicities),
                sample_counts,
                int(cell_counts[core_index].sum()),
            )
    if not np.isfinite(expectations).all() or (expectations < 0).any():
        raise ValueError("Expected-count matrix contains invalid values")
    return expectations


def patient_matrices(
    cohort: pd.DataFrame,
    patients: pd.DataFrame,
    observed: np.ndarray,
    expected: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    patient_ids = sorted(cohort["PID"].unique())
    groups = cohort.groupby("PID", sort=False).indices
    observed_patient = np.vstack([observed[groups[pid]].sum(axis=0) for pid in patient_ids])
    expected_patient = np.vstack([expected[groups[pid]].sum(axis=0) for pid in patient_ids])
    metadata = pd.DataFrame(
        {
            "PID": patient_ids,
            "number_of_cores": [len(groups[pid]) for pid in patient_ids],
        }
    ).merge(
        patients[["PID", "time_months", "event"]],
        on="PID",
        how="left",
        validate="one_to_one",
    )
    return metadata, observed_patient, expected_patient


def enrichment_scores(observed: np.ndarray, expected: np.ndarray) -> np.ndarray:
    observed, expected = np.asarray(observed, float), np.asarray(expected, float)
    if observed.shape != expected.shape or (observed < 0).any() or (expected < 0).any():
        raise ValueError("Observed and expected arrays are invalid")
    return np.log2((observed + PSEUDOCOUNT) / (expected + PSEUDOCOUNT))


def survival_statistics(
    score: np.ndarray, observed: np.ndarray, expected: np.ndarray, time: np.ndarray, event: np.ndarray
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    score = np.asarray(score, float)
    enriched = np.asarray(observed, float) > np.asarray(expected, float)
    binary = fit_cox(time, event, enriched)
    if enriched.any() and (~enriched).any():
        try:
            logrank_stat, logrank_p = logrank_test(time, event, enriched)
            logrank_status = "estimated"
        except Exception as exc:
            logrank_stat = logrank_p = np.nan
            logrank_status = f"not_estimable_{type(exc).__name__}"
    else:
        logrank_stat = logrank_p = np.nan
        logrank_status = "not_estimable_no_group_variation"
    sd = float(np.std(score, ddof=1))
    if np.isfinite(sd) and sd > 0:
        score_z = (score - float(np.mean(score))) / sd
        continuous = fit_cox(time, event, score_z)
    else:
        score_z = np.full(len(score), np.nan)
        continuous = {
            "coef": np.nan,
            "hazard_ratio": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "p_value": np.nan,
            "fit_status": "not_estimable_constant_score",
            "fit_warning": "Observed/expected score has zero variance",
        }
    result = {
        "enrichment_group_definition": "summed_patient_observed_count > summed_patient_expected_count",
        "n_survival": int(len(score)),
        "events_survival": int(event.sum()),
        "n_enriched": int(enriched.sum()),
        "events_enriched": int(event[enriched].sum()),
        "n_non_enriched": int((~enriched).sum()),
        "events_non_enriched": int(event[~enriched].sum()),
        "coef_expected_binary": binary["coef"],
        "hazard_ratio_expected_binary": binary["hazard_ratio"],
        "ci95_low_expected_binary": binary["ci95_low"],
        "ci95_high_expected_binary": binary["ci95_high"],
        "p_value_expected_binary": binary["p_value"],
        "expected_binary_fit_status": binary["fit_status"],
        "expected_binary_fit_warning": binary["fit_warning"],
        "logrank_statistic": logrank_stat,
        "logrank_p_value": logrank_p,
        "logrank_status": logrank_status,
        "score_mean": float(np.mean(score)),
        "score_sd": sd,
        "coef_expected_continuous_per_sd": continuous["coef"],
        "hazard_ratio_expected_continuous_per_sd": continuous["hazard_ratio"],
        "ci95_low_expected_continuous_per_sd": continuous["ci95_low"],
        "ci95_high_expected_continuous_per_sd": continuous["ci95_high"],
        "p_value_expected_continuous_per_sd": continuous["p_value"],
        "expected_continuous_fit_status": continuous["fit_status"],
        "expected_continuous_fit_warning": continuous["fit_warning"],
    }
    return result, enriched, score_z


def json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if pd.isna(value):
        return None
    return value


def fmt(value: object) -> str:
    return "NA" if pd.isna(value) else f"{float(value):.3g}"


def motif_directory_name(motif_hash: str, p_value: float) -> str:
    return f"{motif_hash}*" if np.isfinite(p_value) and p_value < 0.05 else motif_hash


def resolve_motif_directory(outdir: Path, level: int, motif_hash: str, p_value: float) -> Path:
    level_dir = outdir / f"tri{level}"
    level_dir.mkdir(parents=True, exist_ok=True)
    destination = level_dir / motif_directory_name(motif_hash, p_value)
    alternate = level_dir / (motif_hash if destination.name.endswith("*") else motif_hash + "*")
    if alternate.is_dir() and not destination.exists():
        alternate.rename(destination)
    elif alternate.exists() and destination.exists():
        shutil.rmtree(alternate)
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def plot_km(data: pd.DataFrame, result: pd.Series, output: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    for group in ["Non-enriched", "Enriched"]:
        subset = data.loc[data["enrichment_group"].eq(group)]
        if subset.empty:
            continue
        km = step_function(
            subset["time_months"].to_numpy(float), subset["event"].to_numpy(int)
        )
        color = COLORS[group]
        ax.step(
            km["time"], km["survival"], where="post", color=color, lw=2.2,
            label=f"{group} (n={len(subset)}, events={int(subset.event.sum())})",
        )
        ax.fill_between(
            km["time"], km["lower"], km["upper"], step="post", color=color, alpha=0.15
        )
        censored = subset.loc[subset["event"].eq(0), "time_months"].to_numpy(float)
        if len(censored):
            positions = np.searchsorted(km["time"].to_numpy(), censored, side="right") - 1
            ax.scatter(
                censored, km["survival"].to_numpy()[positions], marker="|", s=55, color=color
            )
    annotation = (
        f"Binary Cox HR = {fmt(result['hazard_ratio_expected_binary'])}\n"
        f"95% CI {fmt(result['ci95_low_expected_binary'])}–{fmt(result['ci95_high_expected_binary'])}\n"
        f"Cox p = {fmt(result['p_value_expected_binary'])}; "
        f"log-rank p = {fmt(result['logrank_p_value'])}\n"
        f"Continuous HR/SD = {fmt(result['hazard_ratio_expected_continuous_per_sd'])}"
    )
    ax.text(
        0.98, 0.98, annotation, transform=ax.transAxes, ha="right", va="top", fontsize=9.5,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": ".75"},
    )
    ax.set(
        title=f"TRI-{int(result.triangle_level)} {str(result.wl_hash)[:8]}: exact O>E enrichment",
        xlabel="Overall survival (months)", ylabel="Survival probability", ylim=(0, 1.03),
    )
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def plot_enrichment(data: pd.DataFrame, result: pd.Series, output: Path, dpi: int) -> None:
    ordered = data.sort_values(["log2_observed_expected", "PID"], kind="mergesort")
    ordered_colors = np.where(
        ordered["enrichment_group"].eq("Enriched"), COLORS["Enriched"], COLORS["Non-enriched"]
    )
    scatter_colors = np.where(
        data["enrichment_group"].eq("Enriched"), COLORS["Enriched"], COLORS["Non-enriched"]
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4))
    maximum = max(float(data["observed_count"].max()), float(data["expected_count"].max()), 1.0)
    axes[0].scatter(
        data["expected_count"], data["observed_count"], c=scatter_colors, s=18, alpha=0.72
    )
    axes[0].plot([0, maximum], [0, maximum], color="black", ls="--", lw=1)
    axes[0].set(
        xlabel="Exact expected count", ylabel="Exact observed count", title="Patient observed versus expected",
        xlim=(-0.02 * maximum, 1.02 * maximum), ylim=(-0.02 * maximum, 1.02 * maximum),
    )
    axes[1].bar(
        np.arange(len(ordered)), ordered["log2_observed_expected"], color=ordered_colors, width=1
    )
    axes[1].axhline(0, color="black", ls="--", lw=1)
    axes[1].set(
        xlabel="Patients ordered by score", ylabel="log2((observed+0.5)/(expected+0.5))",
        title="Exact label-permutation enrichment score", xlim=(-0.5, len(ordered) - 0.5),
    )
    fig.suptitle(
        f"TRI-{int(result.triangle_level)} {str(result.wl_hash)[:8]} | {result.cell_type_composition}"
    )
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def validation_panel(
    motif_metadata: pd.DataFrame, expectations: np.ndarray, core_index: int
) -> list[int]:
    chosen = []
    for level in sorted(motif_metadata["triangle_level"].unique()):
        indices = np.flatnonzero(motif_metadata["triangle_level"].to_numpy(int) == int(level))
        eligible = indices[expectations[core_index, indices] >= 1.0]
        candidates = eligible if len(eligible) else indices
        chosen.append(int(candidates[np.argmax(expectations[core_index, candidates])]))
    extra_candidates = np.flatnonzero(expectations[core_index] >= 1.0)
    if len(extra_candidates):
        orbit = motif_metadata["label_orbit_size"].to_numpy(int)
        extra = int(extra_candidates[np.argmax(orbit[extra_candidates])])
        if extra not in chosen:
            chosen.append(extra)
    return chosen


def permutation_validation(
    sample: str,
    cohort: pd.DataFrame,
    graph_root: Path,
    motif_metadata: pd.DataFrame,
    shapes: pd.DataFrame,
    shape_counts: np.ndarray,
    cell_counts: np.ndarray,
    expectations: np.ndarray,
    known_labels: list[int],
    permutations: int,
    seed: int,
) -> pd.DataFrame:
    import graph_pattern_cpp

    matches = np.flatnonzero(cohort["sample"].astype(str).to_numpy() == str(sample))
    if len(matches) != 1:
        raise ValueError(f"Validation sample {sample!r} is not one clinical core")
    core_index = int(matches[0])
    panel = validation_panel(motif_metadata, expectations, core_index)
    node_count, graph_edges, labels, coordinates = _read_gml(graph_root / f"{sample}.gml")
    label_array = np.asarray(labels, dtype=object)
    adjacency = [set() for _ in range(node_count)]
    for left, right in graph_edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    patterns = [_pattern_payload(motif_metadata.iloc[index]) for index in panel]
    tri1_simplices = Delaunay(coordinates).simplices
    rng = np.random.default_rng(seed)
    samples = np.zeros((permutations, len(panel)), dtype=float)
    for permutation in range(permutations):
        shuffled = rng.permutation(label_array)
        graph = _custom_graph(node_count, graph_edges, shuffled)
        cache = graph_pattern_cpp.GraphCache(graph)
        for panel_index, (motif_index, payload) in enumerate(zip(panel, patterns)):
            motif = motif_metadata.iloc[motif_index]
            if int(motif.triangle_level) == 1:
                wanted = sorted(str(motif.node_labels).split("&"))
                samples[permutation, panel_index] = sum(
                    sorted(shuffled[vertices].tolist()) == wanted for vertices in tri1_simplices
                )
            else:
                nodes, edges, pattern_labels = payload
                pattern = _custom_graph(nodes, edges, pattern_labels)
                _, motif_matches = graph_pattern_cpp.find_pattern_matches(
                    cache, pattern, True, 0, None
                )
                samples[permutation, panel_index] = sum(
                    sum(
                        right in adjacency[left]
                        for position, left in enumerate(vertices)
                        for right in vertices[position + 1 :]
                    )
                    == len(edges)
                    for vertices in motif_matches
                )
        if (permutation + 1) % 50 == 0:
            print(f"  permutation validation {permutation + 1}/{permutations}", flush=True)
    rows = []
    for panel_index, motif_index in enumerate(panel):
        motif = motif_metadata.iloc[motif_index]
        values = samples[:, panel_index]
        mean = float(values.mean())
        sd = float(values.std(ddof=1)) if permutations > 1 else np.nan
        mcse = sd / math.sqrt(permutations) if np.isfinite(sd) else np.nan
        analytic = float(expectations[core_index, motif_index])
        tolerance = 4 * mcse + 0.05 if np.isfinite(mcse) else np.nan
        rows.append(
            {
                "sample": sample,
                "permutations": permutations,
                "seed": seed,
                "triangle_level": int(motif.triangle_level),
                "wl_hash": str(motif.wl_hash),
                "shape_id": str(motif.shape_id),
                "aut_shape": int(motif.aut_shape),
                "aut_labeled": int(motif.aut_labeled),
                "analytic_expectation": analytic,
                "permutation_mean": mean,
                "permutation_sd": sd,
                "monte_carlo_se": mcse,
                "absolute_difference": abs(mean - analytic),
                "acceptance_tolerance": tolerance,
                "validation_pass": bool(abs(mean - analytic) <= tolerance),
            }
        )
    return pd.DataFrame(rows)


def sync_significant(
    outdir: Path, results: pd.DataFrame, artifact_names: list[str]
) -> dict[str, object]:
    significant = results.loc[
        np.isfinite(results["p_value_expected_binary"])
        & results["p_value_expected_binary"].lt(0.05)
    ].sort_values(["p_value_expected_binary", "triangle_level", "wl_hash"])
    staging = Path(tempfile.mkdtemp(prefix=".expectation-significant-", dir=outdir))
    destination = outdir / "significant"
    backup = outdir / ".significant-backup"
    try:
        for level in range(1, 8):
            (staging / f"tri{level}").mkdir(parents=True)
        for row in significant.itertuples(index=False):
            source = outdir / f"tri{int(row.triangle_level)}" / motif_directory_name(
                str(row.wl_hash), float(row.p_value_expected_binary)
            )
            target = staging / f"tri{int(row.triangle_level)}" / str(row.wl_hash)
            target.mkdir()
            for name in artifact_names:
                shutil.copy2(source / name, target / name)
        significant.to_csv(staging / "significant_expectation_survival_results.csv", index=False)
        manifest = {
            "selection": "finite primary binary Cox p_value_expected_binary < 0.05",
            "total_significant_motifs": int(len(significant)),
            "significant_motifs_by_level": {
                str(level): int(significant["triangle_level"].eq(level).sum())
                for level in range(1, 8)
            },
            "directory_layout": "triN/<bare_wl_hash>/",
        }
        (staging / "significant_summary.json").write_text(
            json.dumps(json_safe(manifest), indent=2) + "\n"
        )
        (staging / "README.md").write_text(
            "# Nominally significant exact-expectation motifs\n\n"
            "These are full copies of motif packages with a finite unadjusted binary "
            "Cox Wald p-value below 0.05 for the literal observed > expected split. "
            "This is not a BH-adjusted selection and is exploratory because the 644 "
            "motifs were selected using survival in the same cohort.\n"
        )
        if backup.exists():
            shutil.rmtree(backup)
        if destination.exists():
            destination.rename(backup)
        staging.rename(destination)
        if backup.exists():
            shutil.rmtree(backup)
        return manifest
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        if backup.exists() and not destination.exists():
            backup.rename(destination)
        raise


def write_report(path: Path, summary: dict[str, object]) -> None:
    validation = summary["permutation_validation"]
    lines = [
        "# Exact label-permutation motif survival analysis",
        "",
        "## Method",
        "",
        "This analysis recounts the 644 focal motifs and normalizes every core by the exact "
        "expectation under permutation of its fixed cell-type labels. TRI-1 retains its "
        "Delaunay-face definition; TRI-2–TRI-7 retain only induced vertex sets whose "
        "unlabeled graph is exactly the motif shape.",
        "",
        "Patient observed and expected counts are summed across cores. Enriched patients "
        "have observed > expected; equality is non-enriched. The primary test is an "
        "unadjusted binary Cox model with Efron ties and a two-sided Wald p-value. A "
        "continuous per-SD log2 observed/expected Cox model is secondary.",
        "",
        "## Results",
        "",
        f"- Cohort: **{summary['survival_patients']} patients**, **{summary['deaths']} deaths**.",
        f"- Motifs: **{summary['focal_motifs']}** across **{summary['unlabeled_shapes']}** unlabeled shapes.",
        f"- Saved versus exact core counts differed in **{summary['core_motif_count_mismatches']:,}** core-motif pairs.",
        f"- Primary Cox p < 0.05: **{summary['binary_p_below_0_05']}**; global BH q < 0.05: **{summary['binary_global_bh_q_below_0_05']}**.",
        f"- Permutation validation: **{validation['passed']}/{validation['tested']}** analytic expectations passed the four-MCSE criterion.",
        "",
        "## Interpretation",
        "",
        "Observed/expected enrichment removes total-cell, geometry, and composition effects "
        "under the specified within-core label-permutation null. It does not adjust for "
        "clinical covariates. Survival results are exploratory because the motifs were "
        "preselected by raw survival association in this cohort. Nominal p-values should "
        "be interpreted with the reported BH q-values and confidence intervals.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "python breast_cancer_survival/analyze_exact_expectation_motif_survival.py",
        "```",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.dpi < 1 or args.validation_permutations < 2:
        raise ValueError("Workers, DPI, and validation permutations must be positive")
    levels = sorted(set(args.levels))
    pool = load_focal_pool(args.pool_csv, levels)
    metadata = pd.read_csv(args.metadata)
    cohort, patients, audit = clinical_core_cohort(args.artifacts_root, metadata)
    label_frame = pd.read_csv(args.label_map, dtype={"node_label": str})
    label_frame["node_label_int"] = label_frame["node_label"].astype(int)
    known_labels = label_frame["node_label_int"].tolist()
    label_map = dict(label_frame[["node_label", "cell_type"]].itertuples(index=False, name=None))

    args.outdir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.outdir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    motif_metadata, shapes = build_topology_tables(pool, args.artifacts_root, args.motif_root)
    # Re-align all matrices to the user-facing pool ordering.
    pool = pool.merge(
        motif_metadata,
        on=["triangle_level", "wl_hash"],
        how="left",
        validate="one_to_one",
    ).sort_values(["triangle_level", "p_value_unadjusted", "wl_hash"], kind="mergesort").reset_index(drop=True)
    motif_metadata = pool[
        [
            "triangle_level", "wl_hash", "shape_id", "node_count", "node_labels", "edge_list",
            "canonical_edge_list", "aut_shape", "aut_labeled", "label_orbit_size", "label_multiplicities",
        ]
    ].copy()
    motif_metadata.to_csv(args.outdir / "motif_topology_automorphisms.csv", index=False)
    shapes.to_csv(args.outdir / "unlabeled_shape_catalog.csv", index=False)

    observed, shape_counts, cell_counts, total_cells = exact_graph_counts(
        cohort,
        motif_metadata,
        shapes,
        args.graph_root,
        args.motif_root,
        known_labels,
        checkpoint_dir / "exact_graph_counts.npz",
        args.workers,
        args.rebuild_counts,
    )
    saved = saved_observed_counts(pool, cohort, args.artifacts_root, args.motif_root)
    expected = calculate_expectations(
        motif_metadata, shapes, shape_counts, cell_counts, known_labels
    )
    shape_frame = pd.DataFrame(shape_counts, columns=shapes["shape_id"])
    shape_frame.insert(0, "source_sample", cohort["source_sample"].to_numpy())
    shape_frame.insert(0, "sample", cohort["sample"].to_numpy())
    shape_frame.to_csv(args.outdir / "unlabeled_shape_counts.csv.gz", index=False, compression="gzip")

    patient_meta, observed_patient, expected_patient = patient_matrices(
        cohort, patients, observed, expected
    )
    patient_score = enrichment_scores(observed_patient, expected_patient)
    survival_mask = patient_meta["time_months"].notna().to_numpy()
    time = patient_meta.loc[survival_mask, "time_months"].to_numpy(float)
    event = patient_meta.loc[survival_mask, "event"].to_numpy(int)

    core_parts = []
    patient_parts = []
    result_rows = []
    motif_patient_frames: dict[tuple[int, str], pd.DataFrame] = {}
    shape_lookup = {value: index for index, value in enumerate(shapes["shape_id"])}
    for motif_index, motif in pool.iterrows():
        level, motif_hash = int(motif.triangle_level), str(motif.wl_hash)
        shape_column = shape_lookup[str(motif.shape_id)]
        core_score = enrichment_scores(observed[:, motif_index], expected[:, motif_index])
        core_parts.append(
            pd.DataFrame(
                {
                    "triangle_level": level,
                    "wl_hash": motif_hash,
                    "PID": cohort["PID"],
                    "sample": cohort["sample"],
                    "source_sample": cohort["source_sample"],
                    "total_cells": total_cells,
                    "shape_id": motif.shape_id,
                    "unlabeled_shape_count": shape_counts[:, shape_column],
                    "saved_observed_count": saved[:, motif_index],
                    "observed_count": observed[:, motif_index],
                    "expected_count": expected[:, motif_index],
                    "observed_minus_expected": observed[:, motif_index] - expected[:, motif_index],
                    "observed_expected_ratio": (observed[:, motif_index] + PSEUDOCOUNT)
                    / (expected[:, motif_index] + PSEUDOCOUNT),
                    "log2_observed_expected": core_score,
                }
            )
        )
        scores = patient_score[:, motif_index]
        stats, enriched, score_z = survival_statistics(
            scores[survival_mask],
            observed_patient[survival_mask, motif_index],
            expected_patient[survival_mask, motif_index],
            time,
            event,
        )
        frame = patient_meta.copy()
        frame.insert(0, "wl_hash", motif_hash)
        frame.insert(0, "triangle_level", level)
        frame["observed_count"] = observed_patient[:, motif_index]
        frame["expected_count"] = expected_patient[:, motif_index]
        frame["observed_minus_expected"] = (
            observed_patient[:, motif_index] - expected_patient[:, motif_index]
        )
        frame["observed_expected_ratio"] = (
            observed_patient[:, motif_index] + PSEUDOCOUNT
        ) / (expected_patient[:, motif_index] + PSEUDOCOUNT)
        frame["log2_observed_expected"] = scores
        frame["enrichment_group"] = np.where(
            observed_patient[:, motif_index] > expected_patient[:, motif_index],
            "Enriched",
            "Non-enriched",
        )
        frame["score_z_survival"] = np.nan
        frame.loc[survival_mask, "score_z_survival"] = score_z
        patient_parts.append(frame)
        motif_patient_frames[(level, motif_hash)] = frame.loc[survival_mask].copy()
        result_rows.append(
            {
                **motif.to_dict(),
                "cohort_observed_count": float(observed_patient[:, motif_index].sum()),
                "cohort_expected_count": float(expected_patient[:, motif_index].sum()),
                "cohort_log2_observed_expected": float(
                    np.log2(
                        (observed_patient[:, motif_index].sum() + PSEUDOCOUNT)
                        / (expected_patient[:, motif_index].sum() + PSEUDOCOUNT)
                    )
                ),
                "core_count_mismatches_saved_vs_exact": int(
                    np.count_nonzero(saved[:, motif_index] != observed[:, motif_index])
                ),
                "saved_count_total": int(saved[:, motif_index].sum()),
                "exact_count_total": int(observed[:, motif_index].sum()),
                **stats,
            }
        )

    core_scores = pd.concat(core_parts, ignore_index=True)[CORE_COLUMNS]
    patient_scores = pd.concat(patient_parts, ignore_index=True)
    results = pd.DataFrame(result_rows)
    results["q_value_bh_global_expected_binary"] = benjamini_hochberg(
        results["p_value_expected_binary"]
    )
    results["q_value_bh_within_level_expected_binary"] = results.groupby(
        "triangle_level"
    )["p_value_expected_binary"].transform(benjamini_hochberg)
    results["rank_global_expected_binary"] = results["p_value_expected_binary"].rank(
        method="min", na_option="keep"
    )
    results["rank_within_level_expected_binary"] = results.groupby("triangle_level")[
        "p_value_expected_binary"
    ].rank(method="min", na_option="keep")
    results = results.sort_values(
        ["p_value_expected_binary", "triangle_level", "wl_hash"],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)
    core_scores.to_csv(args.outdir / "core_observed_expected_scores.csv.gz", index=False, compression="gzip")
    patient_scores.to_csv(
        args.outdir / "patient_observed_expected_scores.csv.gz", index=False, compression="gzip"
    )
    results.to_csv(args.outdir / "exact_expectation_survival_results.csv", index=False)
    reconciliation = results[
        [
            "triangle_level", "wl_hash", "saved_count_total", "exact_count_total",
            "core_count_mismatches_saved_vs_exact",
        ]
    ].copy()
    reconciliation["exact_minus_saved"] = (
        reconciliation["exact_count_total"] - reconciliation["saved_count_total"]
    )
    reconciliation.to_csv(args.outdir / "observed_count_reconciliation.csv", index=False)

    if args.skip_validation:
        validation = pd.DataFrame()
    else:
        print("Running brute-force label-permutation validation", flush=True)
        validation = permutation_validation(
            args.validation_sample,
            cohort,
            args.graph_root,
            motif_metadata,
            shapes,
            shape_counts,
            cell_counts,
            expected,
            known_labels,
            args.validation_permutations,
            args.seed,
        )
        validation.to_csv(args.outdir / "permutation_validation.csv", index=False)
        if not validation["validation_pass"].all():
            failed = validation.loc[~validation["validation_pass"], "wl_hash"].tolist()
            raise RuntimeError(f"Permutation validation failed for {failed}")

    artifact_names = ["patient_expectation_scores.csv", "statistics.json"]
    if not args.skip_plots:
        artifact_names.extend(
            [
                "expectation_survival_km.png",
                "observed_expected_enrichment.png",
                "motif_delaunay_topology.png",
            ]
        )
    result_lookup = results.set_index(["triangle_level", "wl_hash"])
    for motif in pool.itertuples(index=False):
        level, motif_hash = int(motif.triangle_level), str(motif.wl_hash)
        row = result_lookup.loc[(level, motif_hash)].copy()
        row["triangle_level"] = level
        row["wl_hash"] = motif_hash
        directory = resolve_motif_directory(
            args.outdir, level, motif_hash, float(row.p_value_expected_binary)
        )
        data = motif_patient_frames[(level, motif_hash)]
        data.to_csv(directory / "patient_expectation_scores.csv", index=False)
        (directory / "statistics.json").write_text(
            json.dumps(json_safe(row.to_dict()), indent=2) + "\n"
        )
        if not args.skip_plots:
            plot_km(data, row, directory / "expectation_survival_km.png", args.dpi)
            plot_enrichment(data, row, directory / "observed_expected_enrichment.png", args.dpi)
            plot_delaunay_topology(
                level,
                motif_hash,
                str(row.cell_type_composition),
                str(row.node_labels),
                str(row.edge_list),
                label_map,
                directory / "motif_delaunay_topology.png",
                args.dpi,
            )
    significant_manifest = sync_significant(args.outdir, results, artifact_names)

    validation_summary = {
        "sample": args.validation_sample,
        "permutations": 0 if args.skip_validation else args.validation_permutations,
        "tested": int(len(validation)),
        "passed": int(validation.get("validation_pass", pd.Series(dtype=bool)).sum()),
    }
    summary = {
        **audit,
        "pool_csv": str(args.pool_csv.resolve()),
        "graph_root": str(args.graph_root.resolve()),
        "motif_root": str(args.motif_root.resolve()),
        "artifacts_root": str(args.artifacts_root.resolve()),
        "triangle_levels": levels,
        "clinical_tumor_cores": int(len(cohort)),
        "clinical_patients": int(cohort["PID"].nunique()),
        "survival_patients": int(survival_mask.sum()),
        "deaths": int(event.sum()),
        "focal_motifs": int(len(results)),
        "focal_motifs_by_level": {
            str(key): int(value) for key, value in results.groupby("triangle_level").size().items()
        },
        "unlabeled_shapes": int(len(shapes)),
        "unlabeled_shapes_by_level": {
            str(key): int(value) for key, value in shapes.groupby("triangle_level").size().items()
        },
        "tri1_definition": "one occurrence per saved scipy.spatial.Delaunay simplex",
        "tri2_tri7_definition": "induced exact-shape occurrences deduplicated by target vertex set",
        "expectation_definition": "N_S * (Aut(S)/Aut(m)) * product falling(n_a,k_a) / falling(N,k)",
        "patient_aggregation": "sum observed and expected counts across cores before scoring",
        "score_definition": "log2((observed + 0.5)/(expected + 0.5))",
        "primary_group_definition": "summed patient observed count > summed patient expected count",
        "primary_test": "unadjusted binary Cox PH, Efron ties, two-sided Wald p-value",
        "secondary_test": "unadjusted continuous Cox per SD of log2 observed/expected score",
        "core_motif_count_mismatches": int(np.count_nonzero(saved != observed)),
        "binary_estimable": int(results["p_value_expected_binary"].notna().sum()),
        "binary_p_below_0_05": int(results["p_value_expected_binary"].lt(0.05).sum()),
        "binary_global_bh_q_below_0_05": int(
            results["q_value_bh_global_expected_binary"].lt(0.05).sum()
        ),
        "permutation_validation": validation_summary,
        "plots_written": not args.skip_plots,
        "selection_warning": "The 644 motifs were selected by raw survival association in this cohort",
        "significant_collection": significant_manifest,
    }
    (args.outdir / "analysis_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2) + "\n"
    )
    write_report(args.outdir / "README.md", summary)
    print(json.dumps(json_safe(summary), indent=2), flush=True)


if __name__ == "__main__":
    main()
