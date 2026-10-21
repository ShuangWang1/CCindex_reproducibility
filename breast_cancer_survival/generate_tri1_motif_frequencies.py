#!/usr/bin/env python3
"""Generate TRI-1 labeled Delaunay-triangle frequencies for completed samples.

TRI-2 and higher levels are graph-search outputs.  TRI-1 is the geometric base
case: each simplex of the sample's Delaunay triangulation is one occurrence,
and occurrences are grouped by the sorted labels of its three cells.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/breast-cancer-tri1-matplotlib")
import igraph as ig
import networkx as nx
import numpy as np
from scipy.spatial import Delaunay, QhullError


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
TRIANGLE_EDGE_LIST = "0-1;0-2;1-2"


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph-root",
        type=Path,
        default=repo / "breast_cancer" / "BZ_samples_intra",
        help="Directory containing sample*.gml and sample_manifest.csv",
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=repo / "breast_cancer_meta" / "BZ_res_threshold_10_cpp_0811",
        help="Existing result directory whose completed samples receive TRI-1 CSVs",
    )
    parser.add_argument(
        "--samples",
        nargs="*",
        default=None,
        metavar="SAMPLE",
        help="Optional sample names/numbers, for example sample1 sample12 or 1 12",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not recalculate an existing ccindex_tri_1.csv",
    )
    return parser.parse_args()


def sample_number(value: str | Path) -> int:
    name = Path(value).stem
    if name.startswith("sample"):
        name = name[6:]
    if not name.isdigit() or int(name) <= 0:
        raise ValueError(f"Expected a positive sample number; found {value!r}")
    return int(name)


def requested_samples(values: list[str] | None) -> set[str] | None:
    if values is None:
        return None
    return {f"sample{sample_number(value)}" for value in values}


def normalize_label(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Node label is not numeric: {value!r}") from error
    if not number.is_integer():
        raise ValueError(f"Node label is not integral: {value!r}")
    return str(int(number))


def load_sample_nodes(path: Path) -> tuple[np.ndarray, np.ndarray]:
    graph = ig.read(path)
    required = {"x", "y", "label"}
    missing = required - set(graph.vs.attributes())
    if missing:
        raise ValueError(f"GML is missing node attributes {sorted(missing)}: {path}")
    coordinates = np.column_stack(
        [
            np.asarray(graph.vs["x"], dtype=float),
            np.asarray(graph.vs["y"], dtype=float),
        ]
    )
    if not np.isfinite(coordinates).all():
        raise ValueError(f"GML contains non-finite coordinates: {path}")
    labels = np.asarray([normalize_label(value) for value in graph.vs["label"]])
    return coordinates, labels


def delaunay_triangle_frequencies(
    coordinates: np.ndarray, labels: np.ndarray
) -> tuple[Counter[tuple[str, str, str]], int]:
    """Count cell-label triples directly from ``Delaunay(...).simplices``."""
    coordinates = np.asarray(coordinates, dtype=float)
    labels = np.asarray(labels, dtype=str)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("Coordinates must have shape (n_cells, 2)")
    if len(coordinates) != len(labels):
        raise ValueError("Coordinates and labels have different row counts")
    if len(coordinates) < 3:
        return Counter(), 0
    try:
        triangulation = Delaunay(coordinates)
    except QhullError:
        # Collinear samples have Delaunay edges but no two-dimensional faces.
        return Counter(), 0

    frequencies: Counter[tuple[str, str, str]] = Counter()
    for simplex in triangulation.simplices:
        motif = tuple(sorted((labels[index] for index in simplex), key=int))
        frequencies[motif] += 1
    return frequencies, len(triangulation.simplices)


def triangle_hash(labels: tuple[str, str, str]) -> str:
    graph = nx.complete_graph(3)
    nx.set_node_attributes(
        graph, {index: label for index, label in enumerate(labels)}, "label"
    )
    return nx.weisfeiler_lehman_graph_hash(graph, node_attr="label")


def result_rows(
    frequencies: Counter[tuple[str, str, str]], elapsed_seconds: float = 0.0
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, (labels, count) in enumerate(
        sorted(frequencies.items(), key=lambda item: tuple(map(int, item[0]))), start=1
    ):
        rows.append(
            {
                "pattern_id": f"tri_1_{index}",
                "triangles": 1,
                "nodes": 3,
                "edges": 3,
                "node_labels": "&".join(labels),
                "edge_list": TRIANGLE_EDGE_LIST,
                "wl_hash": triangle_hash(labels),
                "matches": int(count),
                # TRI-1 uses direct aggregation rather than one timed matcher
                # invocation per pattern. Record total aggregation time per row.
                "seconds": elapsed_seconds,
            }
        )
    return rows


def write_result(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def source_sample_map(graph_root: Path) -> dict[str, str]:
    path = graph_root / "sample_manifest.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Sample manifest does not exist: {path}")
    with path.open(encoding="utf-8", newline="") as source:
        return {
            Path(row["gml_file"]).stem: row["source_sample"]
            for row in csv.DictReader(source)
        }


def completed_result_samples(result_root: Path) -> list[Path]:
    paths = [
        path
        for path in result_root.glob("sample*")
        if path.is_dir() and (path / "run_summary.json").is_file()
    ]
    return sorted(paths, key=lambda path: sample_number(path.name))


def generate_sample(
    sample_dir: Path,
    graph_root: Path,
    source_samples: dict[str, str],
) -> dict[str, object]:
    graph_path = graph_root / f"{sample_dir.name}.gml"
    if not graph_path.is_file():
        raise FileNotFoundError(f"Missing source graph: {graph_path}")
    started = time.perf_counter()
    coordinates, labels = load_sample_nodes(graph_path)
    frequencies, simplex_count = delaunay_triangle_frequencies(coordinates, labels)
    elapsed = time.perf_counter() - started
    rows = result_rows(frequencies, elapsed)
    write_result(sample_dir / "ccindex_tri_1.csv", rows)
    matches_sum = sum(int(row["matches"]) for row in rows)
    if matches_sum != simplex_count:
        raise ValueError(
            f"TRI-1 counts do not reconcile for {sample_dir.name}: "
            f"{matches_sum} != {simplex_count}"
        )
    return {
        "sample": sample_dir.name,
        "source_sample": source_samples.get(sample_dir.name, ""),
        "graph_file": str(graph_path),
        "nodes": len(coordinates),
        "delaunay_triangles": simplex_count,
        "tri_1_motif_types": len(rows),
        "tri_1_matches_sum": matches_sum,
        "seconds": elapsed,
        "output_file": str(sample_dir / "ccindex_tri_1.csv"),
    }


def write_audit(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as output:
        if rows:
            writer = csv.DictWriter(output, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if not args.graph_root.is_dir():
        raise FileNotFoundError(f"Graph root does not exist: {args.graph_root}")
    if not args.result_root.is_dir():
        raise FileNotFoundError(f"Result root does not exist: {args.result_root}")
    wanted = requested_samples(args.samples)
    sample_dirs = completed_result_samples(args.result_root)
    if wanted is not None:
        sample_dirs = [path for path in sample_dirs if path.name in wanted]
        found = {path.name for path in sample_dirs}
        if missing := wanted - found:
            raise FileNotFoundError(
                f"Completed result samples were not found: "
                f"{sorted(missing, key=sample_number)}"
            )
    if not sample_dirs:
        raise ValueError("No completed result samples were selected")

    sources = source_sample_map(args.graph_root)
    generated_rows: list[dict[str, object]] = []
    for position, sample_dir in enumerate(sample_dirs, start=1):
        output_path = sample_dir / "ccindex_tri_1.csv"
        if args.skip_existing and output_path.is_file():
            continue
        audit = generate_sample(sample_dir, args.graph_root, sources)
        generated_rows.append(audit)
        if position % 25 == 0 or position == len(sample_dirs):
            print(f"Generated TRI-1 frequencies for {position:,}/{len(sample_dirs):,} samples")

    existing_path = args.result_root / "tri_1_frequency_manifest.csv"
    existing = []
    if existing_path.is_file() and (args.skip_existing or wanted is not None):
        with existing_path.open(encoding="utf-8", newline="") as source:
            existing = list(csv.DictReader(source))
    updated = {str(row["sample"]): row for row in existing}
    updated.update({str(row["sample"]): row for row in generated_rows})
    audit_rows = [updated[key] for key in sorted(updated, key=sample_number)]
    write_audit(args.result_root / "tri_1_frequency_manifest.csv", audit_rows)

    summary = {
        "graph_root": str(args.graph_root),
        "result_root": str(args.result_root),
        "selected_completed_samples": len(sample_dirs),
        "generated_this_run": len(generated_rows),
        "manifest_samples": len(audit_rows),
        "definition": "one occurrence per scipy.spatial.Delaunay simplex",
        "frequency_filter": None,
        "motif_identity": "sorted numeric node-label triple and labeled WL hash",
    }
    (args.result_root / "tri_1_frequency_audit.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
