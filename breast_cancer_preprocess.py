#!/usr/bin/env python3
"""Convert BZ and METABRIC AnnData samples to aligned GML graphs.

The node type is decoded from ``adata.obsm["node_types"]`` using the column
order in ``adata.uns["node_type_names"]``.  It is then normalized with the
dataset-specific mapping below and encoded with the same integer in both
output graphs.

Each BZ ``core`` and METABRIC ``ImageNumber`` is written to a separate file
named ``sample1.gml``, ``sample2.gml``, etc.  Edges reproduce the convention in
spatial_13months-disease-replicate_1.gml: each Delaunay triangle contributes
three undirected edges.  Consequently, an edge shared by two triangles occurs
twice.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Mapping, Sequence, TextIO

import anndata as ad
import numpy as np
from scipy.spatial import Delaunay, QhullError


# METABRIC has these nine source cell types.
MAPPING_1 = {
    "B cells": "B cells",
    "Endothelial": "endothelial",
    "Fibroblasts": "stromal cells",
    "Macrophages": "macrophages",
    "Myoepithelial": "stromal cells",
    "Myofibroblasts": "stromal cells",
    "T cells": "T cells",
    "Tumor cells": "tumor cells",
    "Vascular SMA+": "stromal cells",
}

# BZ already uses this seven-type vocabulary.
MAPPING_2 = {
    "B cells": "B cells",
    "T and B cells": "T and B cells",
    "T cells": "T cells",
    "endothelial": "endothelial",
    "macrophages": "macrophages",
    "stromal cells": "stromal cells",
    "tumor cells": "tumor cells",
}

# This order is the shared, stable numeric encoding written to both GML files.
COMMON_NODE_TYPES = (
    "B cells",
    "T and B cells",
    "T cells",
    "endothelial",
    "macrophages",
    "stromal cells",
    "tumor cells",
)
COMMON_TYPE_TO_LABEL = {name: i for i, name in enumerate(COMMON_NODE_TYPES)}


def node_type_names(adata: ad.AnnData) -> list[str]:
    """Return one-hot column names from ``uns['node_type_names']``."""
    names = adata.uns["node_type_names"]
    if isinstance(names, Mapping):
        # AnnData preserves the HDF5 group/key order.  In these files the dict
        # keys identify the columns; values hold the display names.
        result = [str(name) for name in names.keys()]
    elif isinstance(names, (Sequence, np.ndarray)) and not isinstance(names, str):
        result = [str(name) for name in names]
    else:
        raise TypeError(
            "adata.uns['node_type_names'] must be a mapping or a sequence, "
            f"not {type(names).__name__}"
        )

    n_columns = adata.obsm["node_types"].shape[1]
    if len(result) != n_columns:
        raise ValueError(
            f"node_type_names contains {len(result)} names, but node_types "
            f"contains {n_columns} columns"
        )
    return result


def aligned_node_labels(
    adata: ad.AnnData, mapping: Mapping[str, str]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Decode one-hot node types and return labels, valid-row mask, and names."""
    source_names = node_type_names(adata)
    missing = sorted(set(source_names) - set(mapping))
    if missing:
        raise ValueError(f"The mapping is missing source cell types: {missing}")

    node_types = np.asarray(adata.obsm["node_types"])
    if node_types.ndim != 2 or node_types.shape[0] != adata.n_obs:
        raise ValueError("adata.obsm['node_types'] has an invalid shape")
    if not np.all(np.isfinite(node_types)):
        raise ValueError("adata.obsm['node_types'] contains non-finite values")

    row_sums = node_types.sum(axis=1)
    valid_rows = row_sums > 0
    if np.any(valid_rows & ~np.isclose(row_sums, 1.0)):
        count = int(np.count_nonzero(valid_rows & ~np.isclose(row_sums, 1.0)))
        raise ValueError(f"node_types has {count} rows that are not one-hot encoded")

    source_index = np.argmax(node_types, axis=1)
    aligned_names = [mapping[name] for name in source_names]
    column_labels = np.asarray(
        [COMMON_TYPE_TO_LABEL[name] for name in aligned_names], dtype=np.int8
    )
    return column_labels[source_index[valid_rows]], valid_rows, source_names


def _number(value: float) -> str:
    """Format a coordinate compactly without discarding fractional values."""
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"Spatial coordinates must be finite; found {value}")
    if value.is_integer():
        return str(int(value))
    return format(value, ".15g")


def _write_node_chunk(
    output: TextIO, start: int, coordinates: np.ndarray, labels: np.ndarray
) -> None:
    blocks = []
    for offset, ((x, y), label) in enumerate(zip(coordinates, labels)):
        blocks.append(
            "  node\n"
            "  [\n"
            f"    id {start + offset}\n"
            f"    x {_number(x)}\n"
            f"    y {_number(y)}\n"
            f"    label {int(label)}\n"
            "  ]\n"
        )
    output.write("".join(blocks))


def _triangle_edges(points: np.ndarray) -> np.ndarray:
    """Return three edges per Delaunay triangle, as in the reference GML."""
    if len(points) < 2:
        return np.empty((0, 2), dtype=np.int64)
    if len(points) == 2:
        return np.asarray([[0, 1]], dtype=np.int64)

    try:
        triangles = Delaunay(points).simplices
    except QhullError:
        # A tissue image can theoretically be collinear.  Connect consecutive
        # cells along its widest coordinate axis so no arbitrary cross-image
        # edges are introduced.
        axis = int(np.argmax(np.ptp(points, axis=0)))
        order = np.argsort(points[:, axis], kind="stable")
        return np.column_stack((order[:-1], order[1:]))

    return np.concatenate(
        (triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]),
        axis=0,
    )


def _write_edges(output: TextIO, coordinates: np.ndarray) -> int:
    edges = _triangle_edges(coordinates)
    blocks = []
    for source, target in edges:
        blocks.append(
            "  edge\n"
            "  [\n"
            f"    source {int(source)}\n"
            f"    target {int(target)}\n"
            "    label 0\n"
            "  ]\n"
        )
    output.write("".join(blocks))
    return len(edges)


def _sample_order(adata: ad.AnnData, image_values: np.ndarray) -> list[object]:
    """Use ``uns['img_keys']`` order, followed by any otherwise unseen keys."""
    observed = list(dict.fromkeys(image_values.tolist()))
    configured = np.asarray(adata.uns.get("img_keys", []), dtype=object).tolist()
    observed_set = set(observed)
    order = [key for key in configured if key in observed_set]
    order.extend(key for key in observed if key not in set(order))
    return order


def _write_sample_gml(
    output_path: Path,
    coordinates: np.ndarray,
    labels: np.ndarray,
    chunk_size: int,
) -> int:
    """Write one sample graph and return its edge count."""
    with output_path.open("w", encoding="utf-8", buffering=1024 * 1024) as output:
        output.write(
            'Creator "breast_cancer_preprocess.py"\n'
            "Version 1\n"
            "graph\n"
            "[\n"
            "  directed 0\n"
        )
        for start in range(0, len(coordinates), chunk_size):
            stop = min(start + chunk_size, len(coordinates))
            _write_node_chunk(output, start, coordinates[start:stop], labels[start:stop])
        edge_count = _write_edges(output, coordinates)
        output.write("]\n")
    return edge_count


def convert_h5ad_to_sample_gmls(
    input_path: Path,
    output_dir: Path,
    mapping: Mapping[str, str],
    chunk_size: int = 100_000,
) -> tuple[int, int, int]:
    """Write one GML per sample and return sample, node, and edge counts."""
    # Backed mode avoids loading the (unused) gene-expression matrix into RAM.
    adata = ad.read_h5ad(input_path, backed="r")
    labels, valid_rows, source_names = aligned_node_labels(adata, mapping)

    all_coordinates = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    if all_coordinates.shape != (adata.n_obs, 2):
        raise ValueError(
            f"Expected obsm['spatial'] shape {(adata.n_obs, 2)}, "
            f"found {all_coordinates.shape}"
        )
    coordinates = all_coordinates[valid_rows]

    metadata = adata.uns.get("metadata", {})
    image_column = metadata.get("image_col") if isinstance(metadata, Mapping) else None
    if not image_column or image_column not in adata.obs:
        raise KeyError("Could not find the image column from uns['metadata']['image_col']")
    all_images = adata.obs[image_column].to_numpy()
    images = all_images[valid_rows]
    node_count = len(coordinates)
    omitted_count = adata.n_obs - node_count

    output_dir.mkdir(parents=True, exist_ok=True)
    sample_keys = _sample_order(adata, all_images)
    edge_count = 0
    manifest_rows = []
    for sample_number, sample_key in enumerate(sample_keys, start=1):
        sample_mask = images == sample_key
        sample_coordinates = coordinates[sample_mask]
        sample_labels = labels[sample_mask]
        output_name = f"sample{sample_number}.gml"
        sample_edges = _write_sample_gml(
            output_dir / output_name, sample_coordinates, sample_labels, chunk_size
        )
        edge_count += sample_edges
        manifest_rows.append(
            {
                "sample_index": sample_number,
                "gml_file": output_name,
                "source_sample": sample_key,
                "nodes": len(sample_coordinates),
                "edges": sample_edges,
                "omitted_unlabeled_rows": int(
                    np.count_nonzero((all_images == sample_key) & ~valid_rows)
                ),
            }
        )

    manifest_path = output_dir / "sample_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as manifest:
        writer = csv.DictWriter(manifest, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Converted {input_path} -> {output_dir}/sample*.gml")
    print(f"  source node types: {source_names}")
    print(f"  common label map: {COMMON_TYPE_TO_LABEL}")
    print(
        f"  samples: {len(sample_keys):,}; nodes: {node_count:,}; "
        f"edges: {edge_count:,}"
    )
    if omitted_count:
        print(f"  omitted unlabeled rows: {omitted_count:,}")
    adata.file.close()
    return len(sample_keys), node_count, edge_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--breast-cancer-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "breast_cancer",
        help="Directory containing BZ.h5ad and metabric.h5ad",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="Number of GML nodes formatted per output chunk",
    )
    parser.add_argument(
        "--dataset",
        choices=("both", "BZ", "metabric"),
        default="both",
        help="Convert both datasets (default) or only one dataset",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")

    jobs = (
        ("BZ", "BZ.h5ad", "BZ_samples", MAPPING_2),
        ("metabric", "metabric.h5ad", "metabric_samples", MAPPING_1),
    )
    for dataset, input_name, output_dir_name, mapping in jobs:
        if args.dataset != "both" and args.dataset != dataset:
            continue
        convert_h5ad_to_sample_gmls(
            args.breast_cancer_dir / input_name,
            args.breast_cancer_dir / output_dir_name,
            mapping,
            args.chunk_size,
        )


if __name__ == "__main__":
    main()
