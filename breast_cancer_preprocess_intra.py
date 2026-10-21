#!/usr/bin/env python3
"""Convert BZ and METABRIC samples to dataset-specific GML graphs.

Unlike :mod:`breast_cancer_preprocess`, this intra-dataset variant does not
align or merge cell types between BZ and METABRIC.  For each dataset, column
``i`` of ``adata.obsm["node_types"]`` is written as integer node label ``i``.
The original column name from ``adata.uns["node_type_names"]`` is recorded in
``node_label_cell_type_mapping.csv`` in the corresponding output directory.
Rows with no assigned node type are retained with an additional ``unknown``
label.

Outputs are written to ``breast_cancer/BZ_samples_intra`` and
``breast_cancer/metabric_samples_intra`` by default.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Mapping

import anndata as ad
import numpy as np

from breast_cancer_preprocess import (
    _sample_order,
    _write_edges,
    _write_node_chunk,
    node_type_names,
)


def original_node_labels(
    adata: ad.AnnData,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Decode one-hot node types, assigning all-zero rows to ``unknown``."""
    source_names = node_type_names(adata)
    node_types = np.asarray(adata.obsm["node_types"])
    if node_types.ndim != 2 or node_types.shape[0] != adata.n_obs:
        raise ValueError("adata.obsm['node_types'] has an invalid shape")
    if not np.all(np.isfinite(node_types)):
        raise ValueError("adata.obsm['node_types'] contains non-finite values")

    row_sums = node_types.sum(axis=1)
    unknown_rows = np.all(np.isclose(node_types, 0.0), axis=1)
    invalid_one_hot = ~unknown_rows & ~np.isclose(row_sums, 1.0)
    if np.any(invalid_one_hot):
        count = int(np.count_nonzero(invalid_one_hot))
        raise ValueError(f"node_types has {count} rows that are not one-hot encoded")

    # The label is the original dataset's node-type column index.  No
    # cross-dataset mapping, renaming, or many-to-one merging is performed.
    unknown_label = len(source_names)
    labels = np.full(adata.n_obs, unknown_label, dtype=np.int16)
    labels[~unknown_rows] = np.argmax(node_types[~unknown_rows], axis=1)
    retained_rows = np.ones(adata.n_obs, dtype=bool)
    return labels, retained_rows, [*source_names, "unknown"]


def write_node_label_mapping(output_dir: Path, source_names: list[str]) -> Path:
    """Write the numeric GML node-label to original cell-type mapping."""
    mapping_path = output_dir / "node_label_cell_type_mapping.csv"
    with mapping_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=("node_label", "cell_type"))
        writer.writeheader()
        writer.writerows(
            {"node_label": label, "cell_type": cell_type}
            for label, cell_type in enumerate(source_names)
        )
    return mapping_path


def _write_sample_gml(
    output_path: Path,
    coordinates: np.ndarray,
    labels: np.ndarray,
    chunk_size: int,
) -> int:
    """Write one intra-dataset sample graph and return its edge count."""
    with output_path.open("w", encoding="utf-8", buffering=1024 * 1024) as output:
        output.write(
            'Creator "breast_cancer_preprocess_intra.py"\n'
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
    chunk_size: int = 100_000,
) -> tuple[int, int, int]:
    """Write one GML per sample with the dataset's original cell-type labels."""
    adata = ad.read_h5ad(input_path, backed="r")
    try:
        labels, valid_rows, source_names = original_node_labels(adata)

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
            raise KeyError(
                "Could not find the image column from uns['metadata']['image_col']"
            )
        all_images = adata.obs[image_column].to_numpy()
        images = all_images[valid_rows]

        output_dir.mkdir(parents=True, exist_ok=True)
        mapping_path = write_node_label_mapping(output_dir, source_names)

        sample_keys = _sample_order(adata, all_images)
        unknown_label = len(source_names) - 1
        edge_count = 0
        manifest_rows: list[dict[str, object]] = []
        for sample_number, sample_key in enumerate(sample_keys, start=1):
            sample_mask = images == sample_key
            sample_coordinates = coordinates[sample_mask]
            sample_labels = labels[sample_mask]
            output_name = f"sample{sample_number}.gml"
            sample_edges = _write_sample_gml(
                output_dir / output_name,
                sample_coordinates,
                sample_labels,
                chunk_size,
            )
            edge_count += sample_edges
            manifest_rows.append(
                {
                    "sample_index": sample_number,
                    "gml_file": output_name,
                    "source_sample": sample_key,
                    "nodes": len(sample_coordinates),
                    "edges": sample_edges,
                    "unknown_rows": int(np.count_nonzero(sample_labels == unknown_label)),
                    "omitted_unlabeled_rows": 0,
                }
            )

        manifest_path = output_dir / "sample_manifest.csv"
        with manifest_path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=manifest_rows[0].keys())
            writer.writeheader()
            writer.writerows(manifest_rows)

        node_count = len(coordinates)
        unknown_count = int(np.count_nonzero(labels == unknown_label))
        label_map = dict(enumerate(source_names))
        print(f"Converted {input_path} -> {output_dir}/sample*.gml")
        print(f"  original label map: {label_map}")
        print(f"  mapping file: {mapping_path}")
        print(
            f"  samples: {len(sample_keys):,}; nodes: {node_count:,}; "
            f"edges: {edge_count:,}"
        )
        if unknown_count:
            print(f"  rows labeled unknown: {unknown_count:,}")
        return len(sample_keys), node_count, edge_count
    finally:
        adata.file.close()


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
        ("BZ", "BZ.h5ad", "BZ_samples_intra"),
        ("metabric", "metabric.h5ad", "metabric_samples_intra"),
    )
    for dataset, input_name, output_dir_name in jobs:
        if args.dataset != "both" and args.dataset != dataset:
            continue
        convert_h5ad_to_sample_gmls(
            args.breast_cancer_dir / input_name,
            args.breast_cancer_dir / output_dir_name,
            args.chunk_size,
        )


if __name__ == "__main__":
    main()
