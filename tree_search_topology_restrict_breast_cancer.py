#!/usr/bin/env python3
"""Topology-restricted motif search across all BZ breast-cancer cores.

The topology growth rule is shared with :mod:`tree_search_topology_restrict`.
Unlike the single-target AD search, candidates are counted exactly in every
BZ core and survive only when their per-core count is at least ``threshold``
in strictly more than ``support_fraction`` of all cores.

The default run uses threshold 1, all 559 BZ cores, a strict-majority cutoff
of 280 cores, and a TRI-40 cap.  Results use the existing breast-cancer
per-sample CSV schema so downstream survival tooling can consume them.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import time
from zipfile import BadZipFile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/breast_cancer_topology_restrict_mpl")

import anndata as ad
import igraph as ig
import numpy as np

import graph_pattern_cpp
from breast_cancer_preprocess import _sample_order, node_type_names
from tree_search_topology_restrict import (
    construct_queue,
    expand_level,
    group_unrooted_states,
    igraph_to_custom_graph,
    motif_hash,
    read_queue,
    rooted_motif_hash,
    visualize_examples,
    write_queue,
)


RESULT_FIELDS = (
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

PREVALENCE_FIELDS = (
    "pattern_id",
    "triangles",
    "nodes",
    "edges",
    "node_labels",
    "edge_list",
    "wl_hash",
    "sample_support",
    "sample_fraction",
    "sample_support_exact",
    "total_matches",
    "match_seconds",
    "retained",
)


@dataclass(frozen=True)
class SampleInfo:
    sample_index: int
    sample: str
    source_sample: str
    graph_path: Path
    nodes: int
    edges: int


@dataclass(frozen=True)
class PatternSpec:
    wl_hash: str
    labels: tuple[str, ...]
    edges: tuple[tuple[int, int], ...]


_WORKER_PATTERNS: list[tuple[str, object]] = []


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", type=Path, default=root / "breast_cancer/BZ.h5ad")
    parser.add_argument(
        "--graph-root",
        type=Path,
        default=root / "breast_cancer/BZ_samples_intra",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root / "breast_cancer_topology_restrict",
    )
    parser.add_argument("--max-depth", type=int, default=40)
    parser.add_argument("--threshold", type=int, default=1)
    parser.add_argument("--support-fraction", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--example-count", type=int, default=5)
    parser.add_argument("--visualization-dpi", type=int, default=150)
    args = parser.parse_args(argv)
    if args.max_depth < 1:
        parser.error("--max-depth must be at least 1")
    if args.threshold < 1:
        parser.error("--threshold must be at least 1")
    if not 0.0 <= args.support_fraction < 1.0:
        parser.error("--support-fraction must be in [0, 1)")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.example_count < 0:
        parser.error("--example-count cannot be negative")
    if args.visualization_dpi < 1:
        parser.error("--visualization-dpi must be positive")
    return args


def strict_support_cutoff(sample_count: int, fraction: float) -> int:
    """Return the smallest integer strictly greater than fraction * count."""
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    if not 0.0 <= fraction < 1.0:
        raise ValueError("fraction must be in [0, 1)")
    return math.floor(sample_count * fraction) + 1


def normalize_label(value: object) -> str:
    number = float(value)
    if not number.is_integer():
        raise ValueError(f"Node label is not integral: {value!r}")
    return str(int(number))


def load_target_graph(path: Path) -> ig.Graph:
    graph = ig.Graph.Read_GML(str(path))
    if "label" not in graph.vs.attributes():
        raise ValueError(f"Graph contains no node labels: {path}")
    graph.vs["label"] = [normalize_label(value) for value in graph.vs["label"]]
    graph.simplify(multiple=True, loops=True)
    return graph


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2)
        output.write("\n")
    os.replace(temporary, path)


def _atomic_csv(path: Path, fields: Sequence[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def validate_bz_sources(h5ad_path: Path, graph_root: Path) -> tuple[list[SampleInfo], dict[str, str]]:
    """Validate the prepared GML inventory against BZ.h5ad."""
    manifest_path = graph_root / "sample_manifest.csv"
    mapping_path = graph_root / "node_label_cell_type_mapping.csv"
    if not h5ad_path.is_file():
        raise FileNotFoundError(f"H5AD file does not exist: {h5ad_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Sample manifest does not exist: {manifest_path}")
    if not mapping_path.is_file():
        raise FileNotFoundError(f"Node-label mapping does not exist: {mapping_path}")

    with manifest_path.open(encoding="utf-8", newline="") as source:
        manifest = list(csv.DictReader(source))
    if not manifest:
        raise ValueError(f"Sample manifest is empty: {manifest_path}")

    adata = ad.read_h5ad(h5ad_path, backed="r")
    try:
        metadata = adata.uns.get("metadata", {})
        image_column = metadata.get("image_col") if hasattr(metadata, "get") else None
        if not image_column or image_column not in adata.obs:
            raise KeyError("BZ.h5ad does not define a valid metadata.image_col")
        image_values = adata.obs[image_column].to_numpy()
        ordered_samples = [str(value) for value in _sample_order(adata, image_values)]
        observed_counts = Counter(str(value) for value in image_values)
        expected_names = [*node_type_names(adata), "unknown"]
    finally:
        adata.file.close()

    if len(manifest) != len(ordered_samples):
        raise ValueError(
            f"Manifest/H5AD sample mismatch: {len(manifest)} != {len(ordered_samples)}"
        )

    samples: list[SampleInfo] = []
    for position, (row, source_sample) in enumerate(zip(manifest, ordered_samples), start=1):
        expected_file = f"sample{position}.gml"
        if int(row["sample_index"]) != position or row["gml_file"] != expected_file:
            raise ValueError(f"Unexpected sample ordering at manifest row {position}")
        if row["source_sample"] != source_sample:
            raise ValueError(
                f"Manifest/H5AD source mismatch at sample{position}: "
                f"{row['source_sample']!r} != {source_sample!r}"
            )
        nodes = int(row["nodes"])
        if nodes != observed_counts[source_sample]:
            raise ValueError(
                f"Node-count mismatch for {source_sample}: "
                f"{nodes} != {observed_counts[source_sample]}"
            )
        graph_path = graph_root / expected_file
        if not graph_path.is_file():
            raise FileNotFoundError(f"Prepared sample graph is missing: {graph_path}")
        samples.append(
            SampleInfo(
                sample_index=position,
                sample=f"sample{position}",
                source_sample=source_sample,
                graph_path=graph_path,
                nodes=nodes,
                edges=int(row["edges"]),
            )
        )

    with mapping_path.open(encoding="utf-8", newline="") as source:
        mapping_rows = list(csv.DictReader(source))
    expected_mapping = {str(index): name for index, name in enumerate(expected_names)}
    actual_mapping = {row["node_label"]: row["cell_type"] for row in mapping_rows}
    if actual_mapping != expected_mapping:
        raise ValueError("The prepared node-label mapping disagrees with BZ.h5ad")
    return samples, expected_mapping


def triangle_frequencies(graph: ig.Graph) -> Counter[str]:
    labels = graph.vs["label"]
    frequencies: Counter[str] = Counter()
    for vertices in graph.cliques(min=3, max=3):
        motif = "&".join(sorted((labels[index] for index in vertices), key=int))
        frequencies[motif] += 1
    return frequencies


def _tri1_worker(sample: SampleInfo) -> dict[str, object]:
    started = time.perf_counter()
    graph = load_target_graph(sample.graph_path)
    if graph.vcount() != sample.nodes:
        raise ValueError(
            f"GML node-count mismatch for {sample.sample}: {graph.vcount()} != {sample.nodes}"
        )
    frequencies = triangle_frequencies(graph)
    return {
        "sample_index": sample.sample_index,
        "counts": dict(frequencies),
        "seconds": time.perf_counter() - started,
        "nodes": graph.vcount(),
        "edges": graph.ecount(),
    }


def _tri1_checkpoint_path(root: Path, sample_index: int) -> Path:
    return root / "checkpoints/tri_1" / f"sample{sample_index}.json"


def count_tri1(
    samples: Sequence[SampleInfo], output_root: Path, workers: int
) -> list[dict[str, object]]:
    results: dict[int, dict[str, object]] = {}
    missing: list[SampleInfo] = []
    for sample in samples:
        path = _tri1_checkpoint_path(output_root, sample.sample_index)
        if path.is_file():
            with path.open(encoding="utf-8") as source:
                row = json.load(source)
            if int(row.get("sample_index", -1)) == sample.sample_index:
                results[sample.sample_index] = row
                continue
        missing.append(sample)

    if missing:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_tri1_worker, sample): sample for sample in missing}
            completed = len(results)
            for future in as_completed(futures):
                row = future.result()
                sample_index = int(row["sample_index"])
                _atomic_json(_tri1_checkpoint_path(output_root, sample_index), row)
                results[sample_index] = row
                completed += 1
                if completed % 25 == 0 or completed == len(samples):
                    print(f"  TRI-1 counted {completed:,}/{len(samples):,} cores", flush=True)
    return [results[sample.sample_index] for sample in samples]


def triangle_graph(motif: str) -> ig.Graph:
    graph = ig.Graph.Full(3)
    graph.vs["label"] = motif.split("&")
    graph["trinum"] = 1
    graph["extendableedge"] = graph.get_edgelist()
    graph["rootnodes"] = "0,1,2"
    graph["newtrianglehop"] = 0
    graph["maxfacehop"] = 0
    graph["hashvalue"] = motif_hash(graph)
    graph["rootedhash"] = rooted_motif_hash(graph)
    return graph


def build_edge_pool(triangles: Iterable[str], labels: Iterable[str]) -> dict[str, list[str]]:
    triangle_list = sorted(triangles, key=lambda value: tuple(map(int, value.split("&"))))
    label_list = sorted(set(labels), key=int)
    pool: dict[str, list[str]] = {}
    for left_index, left in enumerate(label_list):
        for right in label_list[left_index:]:
            edge = f"{left}&{right}"
            required = Counter((left, right))
            matches = [
                triangle
                for triangle in triangle_list
                if not (required - Counter(triangle.split("&")))
            ]
            if matches:
                pool[edge] = matches
    return pool


def pattern_spec(graph: ig.Graph) -> PatternSpec:
    return PatternSpec(
        wl_hash=str(graph["hashvalue"]),
        labels=tuple(str(value) for value in graph.vs["label"]),
        edges=tuple(sorted(tuple(sorted(edge)) for edge in graph.get_edgelist())),
    )


def candidate_digest(specs: Sequence[PatternSpec]) -> str:
    # Representatives with the same label-aware WL hash can receive different
    # vertex numbering when reconstructed from rooted queues.  Exact counts
    # are invariant to that numbering, so checkpoint identity follows the
    # stable, sorted unrooted motif vocabulary rather than serialized edges.
    encoded = json.dumps(
        [spec.wl_hash for spec in specs], separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _init_match_worker(specs: Sequence[PatternSpec]) -> None:
    global _WORKER_PATTERNS
    patterns: list[tuple[str, object]] = []
    for spec in specs:
        graph = graph_pattern_cpp.Graph()
        for vertex, label in enumerate(spec.labels):
            graph.addVertex(vertex, label)
        for left, right in spec.edges:
            graph.addEdge(left, right)
        patterns.append((spec.wl_hash, graph))
    _WORKER_PATTERNS = patterns


def _exact_count_worker(sample: SampleInfo) -> tuple[int, np.ndarray, np.ndarray]:
    target = load_target_graph(sample.graph_path)
    if target.vcount() != sample.nodes:
        raise ValueError(
            f"GML node-count mismatch for {sample.sample}: {target.vcount()} != {sample.nodes}"
        )
    cache = graph_pattern_cpp.GraphCache(igraph_to_custom_graph(target))
    counts = np.zeros(len(_WORKER_PATTERNS), dtype=np.int64)
    seconds = np.zeros(len(_WORKER_PATTERNS), dtype=np.float64)
    for index, (_, pattern) in enumerate(_WORKER_PATTERNS):
        started = time.perf_counter()
        counts[index] = graph_pattern_cpp.count_pattern_matches(cache, pattern)
        seconds[index] = time.perf_counter() - started
    return sample.sample_index, counts, seconds


def _presence_worker(sample: SampleInfo) -> tuple[int, np.ndarray, np.ndarray]:
    """Test the worker's current candidate subset, stopping at one match."""
    target = load_target_graph(sample.graph_path)
    if target.vcount() != sample.nodes:
        raise ValueError(
            f"GML node-count mismatch for {sample.sample}: {target.vcount()} != {sample.nodes}"
        )
    cache = graph_pattern_cpp.GraphCache(igraph_to_custom_graph(target))
    present = np.zeros(len(_WORKER_PATTERNS), dtype=np.bool_)
    seconds = np.zeros(len(_WORKER_PATTERNS), dtype=np.float64)
    for index, (_, pattern) in enumerate(_WORKER_PATTERNS):
        started = time.perf_counter()
        match_count, matches = graph_pattern_cpp.find_pattern_matches(
            cache, pattern, find_all=True, threshold=1, edge_dict=None
        )
        present[index] = int(match_count) >= 1
        seconds[index] = time.perf_counter() - started
        del matches
    return sample.sample_index, present, seconds


def _level_checkpoint_path(root: Path, level: int, sample_index: int) -> Path:
    return root / "checkpoints" / f"tri_{level}" / f"sample{sample_index}.npz"


def _phase_checkpoint_path(
    root: Path, level: int, phase: str, sample_index: int
) -> Path:
    return root / "checkpoints" / f"tri_{level}_{phase}" / f"sample{sample_index}.npz"


def _write_count_checkpoint(
    path: Path, digest: str, counts: np.ndarray, seconds: np.ndarray
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as output:
        np.savez_compressed(
            output,
            digest=np.asarray(digest),
            counts=np.asarray(counts, dtype=np.int64),
            seconds=np.asarray(seconds, dtype=np.float64),
        )
    os.replace(temporary, path)


def _read_count_checkpoint(
    path: Path, digest: str, pattern_count: int
) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if str(data["digest"].item()) != digest:
                return None
            counts = np.asarray(data["counts"], dtype=np.int64)
            seconds = np.asarray(data["seconds"], dtype=np.float64)
    except (OSError, ValueError, KeyError, BadZipFile):
        return None
    if counts.shape != (pattern_count,) or seconds.shape != (pattern_count,):
        return None
    return counts, seconds


def _write_presence_checkpoint(
    path: Path,
    digest: str,
    tested_indices: np.ndarray,
    present: np.ndarray,
    seconds: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as output:
        np.savez_compressed(
            output,
            digest=np.asarray(digest),
            tested_indices=np.asarray(tested_indices, dtype=np.int32),
            present=np.asarray(present, dtype=np.bool_),
            seconds=np.asarray(seconds, dtype=np.float64),
        )
    os.replace(temporary, path)


def _read_presence_checkpoint(
    path: Path, digest: str, pattern_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if str(data["digest"].item()) != digest:
                return None
            indices = np.asarray(data["tested_indices"], dtype=np.int32)
            present = np.asarray(data["present"], dtype=np.bool_)
            seconds = np.asarray(data["seconds"], dtype=np.float64)
    except (OSError, ValueError, KeyError, BadZipFile):
        return None
    if present.shape != indices.shape or seconds.shape != indices.shape:
        return None
    if np.any(indices < 0) or np.any(indices >= pattern_count):
        return None
    return indices, present, seconds


def screen_level_presence(
    samples: Sequence[SampleInfo],
    representatives: Sequence[ig.Graph],
    level: int,
    output_root: Path,
    workers: int,
    cutoff: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Screen by presence with safe early qualification and rejection."""
    specs = [pattern_spec(graph) for graph in representatives]
    digest = candidate_digest(specs)
    pattern_count = len(specs)
    support = np.zeros(pattern_count, dtype=np.int64)
    failures = np.zeros(pattern_count, dtype=np.int64)
    elapsed = np.zeros(pattern_count, dtype=np.float64)
    decided = np.zeros(pattern_count, dtype=np.bool_)
    max_failures = len(samples) - cutoff

    # Replay in sample order so a resumed run reconstructs the exact active set.
    first_missing = 0
    for position, sample in enumerate(samples):
        checkpoint = _read_presence_checkpoint(
            _phase_checkpoint_path(output_root, level, "presence", sample.sample_index),
            digest,
            pattern_count,
        )
        if checkpoint is None:
            first_missing = position
            break
        indices, present, seconds = checkpoint
        support[indices] += present
        failures[indices] += ~present
        elapsed[indices] += seconds
        decided |= (support >= cutoff) | (failures > max_failures)
    else:
        first_missing = len(samples)

    batch_size = max(1, workers * 2)
    for batch_start in range(first_missing, len(samples), batch_size):
        active_indices = np.flatnonzero(~decided).astype(np.int32)
        if not len(active_indices):
            break
        batch = samples[batch_start : batch_start + batch_size]
        active_specs = [specs[int(index)] for index in active_indices]
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_match_worker,
            initargs=(active_specs,),
        ) as executor:
            futures = {
                executor.submit(_presence_worker, sample): sample for sample in batch
            }
            batch_results = {}
            for future in as_completed(futures):
                sample_index, present, seconds = future.result()
                batch_results[sample_index] = (present, seconds)
        for sample in batch:
            present, seconds = batch_results[sample.sample_index]
            _write_presence_checkpoint(
                _phase_checkpoint_path(
                    output_root, level, "presence", sample.sample_index
                ),
                digest,
                active_indices,
                present,
                seconds,
            )
            support[active_indices] += present
            failures[active_indices] += ~present
            elapsed[active_indices] += seconds
            decided |= (support >= cutoff) | (failures > max_failures)
        processed = min(batch_start + len(batch), len(samples))
        print(
            f"  TRI-{level} presence {processed:,}/{len(samples):,} cores; "
            f"qualified={int(np.count_nonzero(support >= cutoff)):,}; "
            f"undecided={int(np.count_nonzero(~decided)):,}",
            flush=True,
        )

    # If every candidate became decided early, no files are needed for later cores.
    if np.any(~decided):
        raise RuntimeError(f"Presence screening ended with undecided TRI-{level} motifs")
    return support, failures, elapsed, digest


def count_retained_exact(
    samples: Sequence[SampleInfo],
    representatives: Sequence[ig.Graph],
    retained_indices: Sequence[int],
    level: int,
    output_root: Path,
    workers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Exactly count only majority-supported candidates in every core."""
    retained_specs = [pattern_spec(representatives[index]) for index in retained_indices]
    digest = candidate_digest(retained_specs)
    totals = np.zeros(len(retained_specs), dtype=np.int64)
    support = np.zeros(len(retained_specs), dtype=np.int64)
    elapsed = np.zeros(len(retained_specs), dtype=np.float64)
    missing: list[SampleInfo] = []
    for sample in samples:
        path = _phase_checkpoint_path(output_root, level, "exact", sample.sample_index)
        checkpoint = _read_count_checkpoint(path, digest, len(retained_specs))
        if checkpoint is None:
            missing.append(sample)
        else:
            counts, seconds = checkpoint
            totals += counts
            support += counts > 0
            elapsed += seconds

    if missing and retained_specs:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_match_worker,
            initargs=(retained_specs,),
        ) as executor:
            futures = {
                executor.submit(_exact_count_worker, sample): sample for sample in missing
            }
            completed = len(samples) - len(missing)
            for future in as_completed(futures):
                sample_index, counts, seconds = future.result()
                _write_count_checkpoint(
                    _phase_checkpoint_path(output_root, level, "exact", sample_index),
                    digest,
                    counts,
                    seconds,
                )
                totals += counts
                support += counts > 0
                elapsed += seconds
                completed += 1
                if completed % 10 == 0 or completed == len(samples):
                    print(
                        f"  TRI-{level} exact {completed:,}/{len(samples):,} cores",
                        flush=True,
                    )
    return support, totals, elapsed, digest


def count_level_exact(
    samples: Sequence[SampleInfo],
    representatives: Sequence[ig.Graph],
    level: int,
    output_root: Path,
    workers: int,
    threshold: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    specs = [pattern_spec(graph) for graph in representatives]
    digest = candidate_digest(specs)
    checkpoint_root = output_root / "checkpoints" / f"tri_{level}"
    manifest_path = checkpoint_root / "candidates.json"
    manifest = {
        "triangle_level": level,
        "candidate_digest": digest,
        "candidate_count": len(specs),
        "wl_hashes": [spec.wl_hash for spec in specs],
    }
    if manifest_path.is_file():
        with manifest_path.open(encoding="utf-8") as source:
            existing_manifest = json.load(source)
            compatible = (
                int(existing_manifest.get("triangle_level", -1)) == level
                and int(existing_manifest.get("candidate_count", -1)) == len(specs)
                and existing_manifest.get("wl_hashes") == manifest["wl_hashes"]
            )
            if not compatible:
                raise ValueError(f"Incompatible checkpoint manifest: {manifest_path}")
            # Preserve the digest already embedded in completed per-sample
            # checkpoints created by an earlier invocation.
            digest = str(existing_manifest["candidate_digest"])
    else:
        _atomic_json(manifest_path, manifest)

    missing: list[SampleInfo] = []
    completed = 0
    for sample in samples:
        path = _level_checkpoint_path(output_root, level, sample.sample_index)
        if _read_count_checkpoint(path, digest, len(specs)) is None:
            missing.append(sample)
        else:
            completed += 1

    if missing:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_match_worker,
            initargs=(specs,),
        ) as executor:
            futures = {
                executor.submit(_exact_count_worker, sample): sample for sample in missing
            }
            for future in as_completed(futures):
                sample_index, counts, seconds = future.result()
                _write_count_checkpoint(
                    _level_checkpoint_path(output_root, level, sample_index),
                    digest,
                    counts,
                    seconds,
                )
                completed += 1
                if completed % 10 == 0 or completed == len(samples):
                    print(
                        f"  TRI-{level} counted {completed:,}/{len(samples):,} cores",
                        flush=True,
                    )

    support = np.zeros(len(specs), dtype=np.int64)
    totals = np.zeros(len(specs), dtype=np.int64)
    elapsed = np.zeros(len(specs), dtype=np.float64)
    for sample in samples:
        checkpoint = _read_count_checkpoint(
            _level_checkpoint_path(output_root, level, sample.sample_index),
            digest,
            len(specs),
        )
        if checkpoint is None:
            raise RuntimeError(f"Missing completed checkpoint for {sample.sample}")
        counts, seconds = checkpoint
        support += counts >= threshold
        totals += counts
        elapsed += seconds
    return support, totals, elapsed, digest


def pattern_metadata(graph: ig.Graph, level: int, index: int) -> dict[str, object]:
    return {
        "pattern_id": f"tri_{level}_{index}",
        "triangles": level,
        "nodes": graph.vcount(),
        "edges": graph.ecount(),
        "node_labels": "&".join(str(value) for value in graph.vs["label"]),
        "edge_list": ";".join(
            f"{left}-{right}" for left, right in sorted(tuple(sorted(edge)) for edge in graph.get_edgelist())
        ),
        "wl_hash": str(graph["hashvalue"]),
    }


def write_prevalence(
    path: Path,
    representatives: Sequence[ig.Graph],
    level: int,
    support: np.ndarray,
    totals: np.ndarray,
    seconds: np.ndarray,
    sample_count: int,
    cutoff: int,
    support_exact: bool | np.ndarray = True,
) -> None:
    rows = []
    for index, graph in enumerate(representatives, start=1):
        row = pattern_metadata(graph, level, index)
        row.update(
            {
                "sample_support": int(support[index - 1]),
                "sample_fraction": float(support[index - 1]) / sample_count,
                "sample_support_exact": bool(support_exact)
                if isinstance(support_exact, bool)
                else bool(support_exact[index - 1]),
                "total_matches": ""
                if int(totals[index - 1]) < 0
                else int(totals[index - 1]),
                "match_seconds": float(seconds[index - 1]),
                "retained": int(support[index - 1]) >= cutoff,
            }
        )
        rows.append(row)
    _atomic_csv(path, PREVALENCE_FIELDS, rows)


def write_sample_level_csvs_two_phase(
    samples: Sequence[SampleInfo],
    representatives: Sequence[ig.Graph],
    retained_indices: Sequence[int],
    level: int,
    output_root: Path,
    exact_digest: str,
) -> None:
    """Write compatible rows from retained-only exact checkpoints."""
    retained_count = len(retained_indices)
    for sample in samples:
        checkpoint = _read_count_checkpoint(
            _phase_checkpoint_path(output_root, level, "exact", sample.sample_index),
            exact_digest,
            retained_count,
        )
        if checkpoint is None:
            if retained_count:
                raise RuntimeError(f"Missing exact checkpoint for {sample.sample} TRI-{level}")
            counts = np.zeros(0, dtype=np.int64)
            seconds = np.zeros(0, dtype=np.float64)
        else:
            counts, seconds = checkpoint
        rows = []
        for retained_position, candidate_index in enumerate(retained_indices):
            row = pattern_metadata(
                representatives[candidate_index], level, candidate_index + 1
            )
            row.update(
                {
                    "matches": int(counts[retained_position]),
                    "seconds": float(seconds[retained_position]),
                }
            )
            rows.append(row)
        _atomic_csv(
            output_root / sample.sample / f"ccindex_tri_{level}.csv",
            RESULT_FIELDS,
            rows,
        )


def write_sample_level_csvs(
    samples: Sequence[SampleInfo],
    representatives: Sequence[ig.Graph],
    retained_indices: Sequence[int],
    level: int,
    output_root: Path,
    digest: str | None,
    tri1_rows: Sequence[dict[str, object]] | None = None,
) -> None:
    for sample_position, sample in enumerate(samples):
        if not representatives:
            counts = np.zeros(0, dtype=np.int64)
            seconds = np.zeros(0, dtype=np.float64)
        elif level == 1:
            if tri1_rows is None:
                raise ValueError("TRI-1 rows are required")
            counts_by_motif = {
                str(key): int(value)
                for key, value in dict(tri1_rows[sample_position]["counts"]).items()
            }
            counts = np.asarray(
                [counts_by_motif.get("&".join(graph.vs["label"]), 0) for graph in representatives],
                dtype=np.int64,
            )
            seconds = np.full(
                len(representatives), float(tri1_rows[sample_position]["seconds"])
            )
        else:
            if digest is None:
                raise ValueError("A candidate digest is required above TRI-1")
            checkpoint = _read_count_checkpoint(
                _level_checkpoint_path(output_root, level, sample.sample_index),
                digest,
                len(representatives),
            )
            if checkpoint is None:
                raise RuntimeError(f"Missing checkpoint for {sample.sample} TRI-{level}")
            counts, seconds = checkpoint

        rows = []
        for candidate_index in retained_indices:
            graph = representatives[candidate_index]
            row = pattern_metadata(graph, level, candidate_index + 1)
            row.update(
                {
                    "matches": int(counts[candidate_index]),
                    "seconds": float(seconds[candidate_index]),
                }
            )
            rows.append(row)
        _atomic_csv(
            output_root / sample.sample / f"ccindex_tri_{level}.csv",
            RESULT_FIELDS,
            rows,
        )


def write_queue_idempotent(states: Sequence[ig.Graph], level: int, queue_root: Path) -> None:
    destination = queue_root / str(level)
    expected = sorted(str(graph["rootedhash"]) for graph in states)
    if destination.is_dir() and any(destination.iterdir()):
        observed = sorted(str(graph["rootedhash"]) for graph in read_queue(queue_root, level))
        if observed != expected:
            raise ValueError(f"Existing queue disagrees with TRI-{level} survivors")
        return
    write_queue(states, level, queue_root)


def write_sample_inventory(
    output_root: Path,
    samples: Sequence[SampleInfo],
    tri1_rows: Sequence[dict[str, object]],
    completed_level: int,
) -> None:
    manifest_rows = []
    for sample, tri1 in zip(samples, tri1_rows):
        counts = dict(tri1["counts"])
        summary = {
            "dataset": "BZ",
            "sample": sample.sample,
            "source_sample": sample.source_sample,
            "graph_file": str(sample.graph_path),
            "nodes": sample.nodes,
            "edges": int(tri1["edges"]),
            "source_triangle_types": len(counts),
            "completed_triangle_level": completed_level,
        }
        _atomic_json(output_root / sample.sample / "run_summary.json", summary)
        manifest_rows.append(summary)
    _atomic_csv(output_root / "run_manifest.csv", tuple(manifest_rows[0]), manifest_rows)


def write_survivor_table(output_root: Path, levels: Sequence[dict[str, object]]) -> None:
    """Write the compact, continuously updated motif-count summary."""
    fields = (
        "triangle_level",
        "candidates_tested",
        "surviving_unrooted_motifs",
        "motifs_with_more_than_one_cell_type",
    )
    rows = []
    for row in levels:
        level = int(row["triangle_level"])
        prevalence_path = output_root / "prevalence" / f"ccindex_tri_{level}.csv"
        multitype_count = 0
        with prevalence_path.open(encoding="utf-8", newline="") as source:
            for motif in csv.DictReader(source):
                if motif["retained"].lower() == "true" and len(
                    set(motif["node_labels"].split("&"))
                ) > 1:
                    multitype_count += 1
        rows.append({
            "triangle_level": int(row["triangle_level"]),
            "candidates_tested": int(row.get("unrooted_patterns_matched", 0)),
            "surviving_unrooted_motifs": int(row["unrooted_survivors"]),
            "motifs_with_more_than_one_cell_type": multitype_count,
        })
    _atomic_csv(output_root / "survived_motifs_by_level.csv", fields, rows)


def ensure_level_visualization(
    output_root: Path,
    level: int,
    label_mapping: dict[str, str],
    example_count: int,
    dpi: int,
    force: bool = False,
) -> None:
    """Create five diverse retained-motif PNGs for one completed level."""
    if example_count == 0:
        return
    destination = output_root / "visualizations" / f"tri_{level}"
    manifest = destination / f"tri{level}_examples.csv"
    if manifest.is_file() and not force:
        return
    states = read_queue(output_root / "queues", level)
    visualize_examples(
        states,
        label_mapping,
        destination,
        sample_count=example_count,
        dpi=dpi,
        queue_paths={},
        show_node_ids=False,
        prefer_multitype=True,
    )


def run_tri1(
    samples: Sequence[SampleInfo],
    output_root: Path,
    cutoff: int,
    threshold: int,
    workers: int,
) -> tuple[list[ig.Graph], list[dict[str, object]], list[str], dict[str, object]]:
    started = time.perf_counter()
    tri1_rows = count_tri1(samples, output_root, workers)
    motifs = sorted(
        {motif for row in tri1_rows for motif in dict(row["counts"])},
        key=lambda value: tuple(map(int, value.split("&"))),
    )
    representatives = [triangle_graph(motif) for motif in motifs]
    support = np.asarray(
        [
            sum(int(dict(row["counts"]).get(motif, 0)) >= threshold for row in tri1_rows)
            for motif in motifs
        ],
        dtype=np.int64,
    )
    totals = np.asarray(
        [sum(int(dict(row["counts"]).get(motif, 0)) for row in tri1_rows) for motif in motifs],
        dtype=np.int64,
    )
    seconds = np.asarray(
        [sum(float(row["seconds"]) for row in tri1_rows)] * len(motifs),
        dtype=np.float64,
    )
    retained_indices = [index for index, value in enumerate(support) if value >= cutoff]
    retained_motifs = [motifs[index] for index in retained_indices]
    retained = [representatives[index] for index in retained_indices]
    for graph, index in zip(retained, retained_indices):
        graph["count"] = int(support[index])
        graph["sample_support"] = int(support[index])
        graph["totalmatches"] = int(totals[index])

    write_prevalence(
        output_root / "prevalence/ccindex_tri_1.csv",
        representatives,
        1,
        support,
        totals,
        seconds,
        len(samples),
        cutoff,
    )
    write_sample_level_csvs(
        samples,
        representatives,
        retained_indices,
        1,
        output_root,
        None,
        tri1_rows,
    )
    write_queue_idempotent(retained, 1, output_root / "queues")
    level_summary = {
        "triangle_level": 1,
        "unrooted_patterns_matched": len(representatives),
        "unrooted_survivors": len(retained),
        "rooted_survivors": len(retained),
        "level_seconds": time.perf_counter() - started,
    }
    return retained, tri1_rows, retained_motifs, level_summary


def load_tri1_rows(samples: Sequence[SampleInfo], output_root: Path) -> list[dict[str, object]]:
    rows = []
    for sample in samples:
        path = _tri1_checkpoint_path(output_root, sample.sample_index)
        if not path.is_file():
            raise FileNotFoundError(f"TRI-1 checkpoint is missing: {path}")
        with path.open(encoding="utf-8") as source:
            rows.append(json.load(source))
    return rows


def config_payload(args: argparse.Namespace, samples: Sequence[SampleInfo], cutoff: int) -> dict[str, object]:
    return {
        "definition": "strict triangle-face hop shells around each TRI-1 root",
        "h5ad_path": str(args.h5ad.resolve()),
        "graph_root": str(args.graph_root.resolve()),
        "sample_count": len(samples),
        "threshold": args.threshold,
        "support_fraction": args.support_fraction,
        "strict_support_cutoff": cutoff,
        "max_depth": args.max_depth,
        "count_definition": "distinct matched node sets",
    }


def prepare_output(
    output_root: Path, graph_root: Path, config: dict[str, object]
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    config_path = output_root / "run_config.json"
    if config_path.is_file():
        with config_path.open(encoding="utf-8") as source:
            if json.load(source) != config:
                raise ValueError(
                    f"Existing output uses incompatible settings: {config_path}"
                )
    else:
        _atomic_json(config_path, config)
    source_mapping = graph_root / "node_label_cell_type_mapping.csv"
    destination_mapping = output_root / "node_label_cell_type_mapping.csv"
    if not destination_mapping.exists():
        shutil.copyfile(source_mapping, destination_mapping)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    total_started = time.perf_counter()
    samples, label_mapping = validate_bz_sources(args.h5ad, args.graph_root)
    cutoff = strict_support_cutoff(len(samples), args.support_fraction)
    config = config_payload(args, samples, cutoff)
    prepare_output(args.output_root, args.graph_root, config)
    print(
        f"Validated {len(samples):,} BZ cores; exact-count threshold={args.threshold}; "
        f"strict support cutoff={cutoff:,}; max depth={args.max_depth}",
        flush=True,
    )

    summary_path = args.output_root / "run_summary.json"
    prior_summary: dict[str, object] = {}
    if summary_path.is_file():
        with summary_path.open(encoding="utf-8") as source:
            prior_summary = json.load(source)
        if prior_summary.get("status") in {"exhausted", "max_depth_complete"}:
            print(f"Run is already complete: {prior_summary.get('status')}")
            return
    for completed in prior_summary.get("levels", []):
        ensure_level_visualization(
            args.output_root,
            int(completed["triangle_level"]),
            label_mapping,
            args.example_count,
            args.visualization_dpi,
        )
    base_seconds = float(prior_summary.get("total_seconds", 0.0))

    levels = list(prior_summary.get("levels", []))
    completed_level = int(prior_summary.get("completed_triangle_level", 0))
    if completed_level:
        queue = read_queue(args.output_root / "queues", completed_level)
        tri1_rows = load_tri1_rows(samples, args.output_root)
        retained_motifs = [str(value) for value in prior_summary["retained_tri1_motifs"]]
        print(f"Resuming from completed TRI-{completed_level}", flush=True)
    else:
        print("Counting TRI-1 motifs", flush=True)
        queue, tri1_rows, retained_motifs, level_summary = run_tri1(
            samples,
            args.output_root,
            cutoff,
            args.threshold,
            args.workers,
        )
        levels = [level_summary]
        completed_level = 1
        write_sample_inventory(args.output_root, samples, tri1_rows, completed_level)
        prior_summary = {
            **config,
            "retained_tri1_motifs": retained_motifs,
            "levels": levels,
            "completed_triangle_level": completed_level,
            "status": "running",
            "total_seconds": base_seconds + time.perf_counter() - total_started,
        }
        _atomic_json(summary_path, prior_summary)
        write_survivor_table(args.output_root, levels)
        print(f"TRI-1 retained {len(queue):,} majority-supported motifs", flush=True)

    edge_pool = build_edge_pool(retained_motifs, label_mapping)
    status = "running"
    for level in range(completed_level + 1, args.max_depth + 1):
        level_started = time.perf_counter()
        print(f"Generating TRI-{level} candidates", flush=True)
        candidates = expand_level(queue, edge_pool)
        representatives, groups = group_unrooted_states(candidates)
        print(
            f"TRI-{level}: {len(candidates):,} rooted candidates, "
            f"{len(representatives):,} unrooted candidates",
            flush=True,
        )
        legacy_manifest = (
            args.output_root / "checkpoints" / f"tri_{level}" / "candidates.json"
        )
        legacy_complete = legacy_manifest.is_file() and all(
            _level_checkpoint_path(args.output_root, level, sample.sample_index).is_file()
            for sample in samples
        )
        two_phase = bool(representatives) and args.threshold == 1 and not legacy_complete
        if representatives and not two_phase:
            support, totals, seconds, digest = count_level_exact(
                samples,
                representatives,
                level,
                args.output_root,
                args.workers,
                args.threshold,
            )
            support_exact: bool | np.ndarray = True
        else:
            support = np.zeros(0, dtype=np.int64)
            totals = np.zeros(0, dtype=np.int64)
            seconds = np.zeros(0, dtype=np.float64)
            digest = candidate_digest([])
            support_exact = True
        if two_phase:
            screened_support, _, presence_seconds, _ = screen_level_presence(
                samples,
                representatives,
                level,
                args.output_root,
                args.workers,
                cutoff,
            )
            retained_indices = [
                index for index, value in enumerate(screened_support) if value >= cutoff
            ]
            exact_support, retained_totals, exact_seconds, digest = count_retained_exact(
                samples,
                representatives,
                retained_indices,
                level,
                args.output_root,
                args.workers,
            )
            if np.any(exact_support < cutoff):
                raise RuntimeError(
                    f"TRI-{level} presence and exact phases disagree on retention"
                )
            support = screened_support.copy()
            totals = np.full(len(representatives), -1, dtype=np.int64)
            seconds = presence_seconds.copy()
            support_exact = np.zeros(len(representatives), dtype=np.bool_)
            for retained_position, candidate_index in enumerate(retained_indices):
                support[candidate_index] = exact_support[retained_position]
                totals[candidate_index] = retained_totals[retained_position]
                seconds[candidate_index] += exact_seconds[retained_position]
                support_exact[candidate_index] = True
        else:
            retained_indices = [
                index for index, value in enumerate(support) if value >= cutoff
            ]
        write_prevalence(
            args.output_root / f"prevalence/ccindex_tri_{level}.csv",
            representatives,
            level,
            support,
            totals,
            seconds,
            len(samples),
            cutoff,
            support_exact,
        )
        if two_phase:
            write_sample_level_csvs_two_phase(
                samples,
                representatives,
                retained_indices,
                level,
                args.output_root,
                digest,
            )
        else:
            write_sample_level_csvs(
                samples,
                representatives,
                retained_indices,
                level,
                args.output_root,
                digest,
            )

        queue = []
        for candidate_index in retained_indices:
            representative = representatives[candidate_index]
            for state in groups[str(representative["hashvalue"])]:
                state["count"] = int(support[candidate_index])
                state["sample_support"] = int(support[candidate_index])
                state["totalmatches"] = int(totals[candidate_index])
                state["time"] = float(seconds[candidate_index])
                queue.append(state)
        write_queue_idempotent(queue, level, args.output_root / "queues")
        ensure_level_visualization(
            args.output_root,
            level,
            label_mapping,
            args.example_count,
            args.visualization_dpi,
        )
        completed_level = level
        level_summary = {
            "triangle_level": level,
            "rooted_candidates": len(candidates),
            "unrooted_patterns_matched": len(representatives),
            "rooted_survivors": len(queue),
            "unrooted_survivors": len(retained_indices),
            "level_seconds": time.perf_counter() - level_started,
        }
        levels.append(level_summary)
        status = "running" if queue else "exhausted"
        write_sample_inventory(args.output_root, samples, tri1_rows, completed_level)
        prior_summary = {
            **config,
            "retained_tri1_motifs": retained_motifs,
            "levels": levels,
            "completed_triangle_level": completed_level,
            "status": status,
            "total_seconds": base_seconds + time.perf_counter() - total_started,
        }
        _atomic_json(summary_path, prior_summary)
        write_survivor_table(args.output_root, levels)
        print(
            f"TRI-{level} retained {len(retained_indices):,} unrooted / "
            f"{len(queue):,} rooted motifs in "
            f"{time.perf_counter() - level_started:.1f}s",
            flush=True,
        )
        if not queue:
            break

    if status == "running" and completed_level >= args.max_depth:
        status = "max_depth_complete"
    final_summary = {
        **config,
        "retained_tri1_motifs": retained_motifs,
        "levels": levels,
        "completed_triangle_level": completed_level,
        "status": status,
        "total_seconds": base_seconds + time.perf_counter() - total_started,
    }
    _atomic_json(summary_path, final_summary)
    write_survivor_table(args.output_root, levels)
    print(
        f"Finished with status={status}, completed TRI-{completed_level}, "
        f"this invocation={time.perf_counter() - total_started:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
