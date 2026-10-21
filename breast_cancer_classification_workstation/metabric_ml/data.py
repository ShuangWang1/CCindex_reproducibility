"""Read METABRIC motif CSVs and construct a reusable sparse feature matrix."""

from __future__ import annotations

import csv
import gzip
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse


TRIANGLE_LEVELS = tuple(range(2, 8))
MOTIF_COLUMNS = {
    "pattern_id",
    "triangles",
    "nodes",
    "edges",
    "node_labels",
    "edge_list",
    "wl_hash",
    "matches",
    "seconds",
}
SAMPLE_RE = re.compile(r"^sample([1-9][0-9]*)$")


@dataclass(frozen=True)
class PreparedData:
    """Paths and dimensions produced by feature preparation."""

    output_dir: Path
    matrix_path: Path
    samples_path: Path
    feature_levels_path: Path
    feature_index_path: Path
    audit_path: Path
    sample_count: int
    feature_count: int
    nonzero_count: int


def parse_grade(value: str) -> int | None:
    """Decode the one-hot grade strings used by meta_grade.csv."""
    normalized = value.strip()
    if not normalized or "nan" in normalized.lower():
        return None
    if not (normalized.startswith("[") and normalized.endswith("]")):
        raise ValueError(f"Grade is not a bracketed one-hot vector: {value!r}")
    try:
        parts = [float(item) for item in normalized[1:-1].split()]
    except ValueError as exc:
        raise ValueError(f"Grade contains a non-numeric value: {value!r}") from exc
    if len(parts) != 3 or any(item not in (0.0, 1.0) for item in parts):
        raise ValueError(f"Grade must contain three binary values: {value!r}")
    if not math.isclose(sum(parts), 1.0):
        raise ValueError(f"Grade is not one-hot: {value!r}")
    return int(np.argmax(parts))


def read_grades(path: Path) -> dict[str, int | None]:
    if not path.is_file():
        raise FileNotFoundError(f"Grade CSV does not exist: {path}")
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if not {"id", "grade"}.issubset(reader.fieldnames or ()):  # pragma: no cover
            raise ValueError(f"Grade CSV must contain id and grade columns: {path}")
        output: dict[str, int | None] = {}
        for row in reader:
            identifier = row["id"].strip()
            if identifier in output:
                raise ValueError(f"Duplicate grade ID {identifier!r} in {path}")
            output[identifier] = parse_grade(row["grade"])
    return output


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Run manifest does not exist: {path}")
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        required = {
            "sample",
            "source_sample",
            "nodes",
            "edges",
            "source_triangle_types",
            "retained_triangle_types",
            "completed_triangle_level",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Run manifest is missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Run manifest is empty: {path}")
    seen: set[str] = set()
    for row in rows:
        sample = row["sample"]
        if not SAMPLE_RE.fullmatch(sample):
            raise ValueError(f"Invalid sample name in manifest: {sample!r}")
        if sample in seen:
            raise ValueError(f"Duplicate sample in manifest: {sample}")
        seen.add(sample)
    rows.sort(key=lambda row: int(SAMPLE_RE.fullmatch(row["sample"]).group(1)))
    return rows


def read_patient_map(h5ad_path: Path | None) -> dict[str, str]:
    """Map source ImageNumber to patient without loading expression values."""
    if h5ad_path is None:
        return {}
    if not h5ad_path.is_file():
        raise FileNotFoundError(f"H5AD file does not exist: {h5ad_path}")
    try:
        import anndata as ad
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("anndata is required when --h5ad is supplied") from exc

    adata = ad.read_h5ad(h5ad_path, backed="r")
    try:
        required = {"ImageNumber", "metabricId"}
        missing = required - set(adata.obs.columns)
        if missing:
            raise ValueError(f"H5AD obs is missing columns: {sorted(missing)}")
        pairs = adata.obs[["ImageNumber", "metabricId"]].drop_duplicates()
    finally:
        adata.file.close()
    output: dict[str, str] = {}
    for image, patient in pairs.itertuples(index=False, name=None):
        key = str(image)
        value = str(patient)
        if key in output and output[key] != value:
            raise ValueError(f"ImageNumber {key} maps to multiple patients")
        output[key] = value
    return output


def _iter_positive_motifs(path: Path, expected_level: int) -> Iterable[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Motif CSV does not exist: {path}")
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        missing = MOTIF_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        seen: set[str] = set()
        for row_number, row in enumerate(reader, start=2):
            try:
                level = int(row["triangles"])
                matches = int(row["matches"])
            except ValueError as exc:
                raise ValueError(f"Invalid numeric value at {path}:{row_number}") from exc
            if level != expected_level:
                raise ValueError(
                    f"Expected triangle level {expected_level} at {path}:{row_number}, "
                    f"found {level}"
                )
            if matches < 0:
                raise ValueError(f"Negative match count at {path}:{row_number}")
            motif_hash = row["wl_hash"].strip()
            if not motif_hash:
                raise ValueError(f"Empty wl_hash at {path}:{row_number}")
            if motif_hash in seen:
                raise ValueError(f"Duplicate wl_hash {motif_hash} within {path}")
            seen.add(motif_hash)
            if matches > 0:
                yield row


def _sample_table(
    manifest_rows: list[dict[str, str]],
    grades: dict[str, int | None],
    patient_map: dict[str, str],
) -> pd.DataFrame:
    records = []
    for sample_index, row in enumerate(manifest_rows):
        source_sample = row["source_sample"].strip()
        grade = grades.get(source_sample)
        records.append(
            {
                "row_index": sample_index,
                "sample": row["sample"],
                "source_sample": source_sample,
                "patient_id": patient_map.get(source_sample, ""),
                "grade": grade if grade is not None else pd.NA,
                "is_labeled": grade is not None,
                "nodes": int(row["nodes"]),
                "edges": int(row["edges"]),
                "source_triangle_types": int(row["source_triangle_types"]),
                "retained_triangle_types": int(row["retained_triangle_types"]),
            }
        )
    frame = pd.DataFrame.from_records(records)
    frame["grade"] = frame["grade"].astype("Int64")
    return frame


def prepare_metabric_features(
    motif_root: Path,
    grades_path: Path,
    output_dir: Path,
    h5ad_path: Path | None = None,
    triangle_levels: tuple[int, ...] = TRIANGLE_LEVELS,
    manifest_path: Path | None = None,
) -> PreparedData:
    """Build the complete sample-by-motif count matrix in two streaming passes."""
    if tuple(sorted(set(triangle_levels))) != triangle_levels:
        raise ValueError("Triangle levels must be unique and sorted")
    if any(level < 2 for level in triangle_levels):
        raise ValueError("Triangle levels must be at least 2")

    resolved_manifest = manifest_path or (motif_root / "run_manifest.csv")
    manifest_rows = read_manifest(resolved_manifest)
    grades = read_grades(grades_path)
    patient_map = read_patient_map(h5ad_path)
    samples = _sample_table(manifest_rows, grades, patient_map)

    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / "samples.csv"
    matrix_path = output_dir / "motif_counts.npz"
    feature_levels_path = output_dir / "feature_triangle_levels.npy"
    feature_index_path = output_dir / "feature_index.csv.gz"
    audit_path = output_dir / "audit.json"

    feature_to_index: dict[str, int] = {}
    feature_hashes: list[str] = []
    feature_levels: list[int] = []
    feature_nodes: list[int] = []
    feature_edges: list[int] = []
    feature_node_labels: list[str] = []
    document_frequency: list[int] = []
    total_nonzero = 0
    exhausted_level_files: list[str] = []

    def paths_for_sample(row: dict[str, str]) -> Iterable[tuple[int, Path]]:
        completed_level = int(row["completed_triangle_level"])
        sample_dir = motif_root / row["sample"]
        for level in triangle_levels:
            path = sample_dir / f"ccindex_tri_{level}.csv"
            if path.is_file():
                yield level, path
            elif level > completed_level:
                exhausted_level_files.append(str(path.relative_to(motif_root)))
            else:
                raise FileNotFoundError(
                    f"Motif CSV is missing at or below the manifest's completed level: {path}"
                )

    print("Pass 1/2: discovering motif vocabulary and document frequencies")
    for position, row in enumerate(manifest_rows, start=1):
        for level, path in paths_for_sample(row):
            for motif in _iter_positive_motifs(path, level):
                motif_hash = motif["wl_hash"].strip()
                key = f"{level}:{motif_hash}"
                feature_index = feature_to_index.get(key)
                if feature_index is None:
                    feature_index = len(feature_hashes)
                    feature_to_index[key] = feature_index
                    feature_hashes.append(motif_hash)
                    feature_levels.append(level)
                    feature_nodes.append(int(motif["nodes"]))
                    feature_edges.append(int(motif["edges"]))
                    feature_node_labels.append(motif["node_labels"])
                    document_frequency.append(0)
                document_frequency[feature_index] += 1
                total_nonzero += 1
        if position % 50 == 0 or position == len(manifest_rows):
            print(
                f"  scanned {position}/{len(manifest_rows)} samples; "
                f"features={len(feature_hashes):,}, nonzeros={total_nonzero:,}"
            )

    indices = np.empty(total_nonzero, dtype=np.int32)
    values = np.empty(total_nonzero, dtype=np.float32)
    indptr = np.zeros(len(manifest_rows) + 1, dtype=np.int64)
    cursor = 0
    print("Pass 2/2: filling sparse count matrix")
    for sample_index, row in enumerate(manifest_rows):
        entries: list[tuple[int, float]] = []
        for level, path in paths_for_sample(row):
            for motif in _iter_positive_motifs(path, level):
                key = f"{level}:{motif['wl_hash'].strip()}"
                entries.append((feature_to_index[key], float(motif["matches"])))
        entries.sort(key=lambda item: item[0])
        end = cursor + len(entries)
        if entries:
            indices[cursor:end] = [item[0] for item in entries]
            values[cursor:end] = [item[1] for item in entries]
        cursor = end
        indptr[sample_index + 1] = cursor
        if (sample_index + 1) % 50 == 0 or sample_index + 1 == len(manifest_rows):
            print(f"  filled {sample_index + 1}/{len(manifest_rows)} samples")

    matrix = sparse.csr_matrix(
        (values, indices, indptr),
        shape=(len(manifest_rows), len(feature_hashes)),
        dtype=np.float32,
    )
    matrix.sort_indices()
    sparse.save_npz(matrix_path, matrix, compressed=True)
    np.save(feature_levels_path, np.asarray(feature_levels, dtype=np.int8))
    samples.to_csv(samples_path, index=False)

    with gzip.open(feature_index_path, "wt", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "feature_index",
                "triangle_level",
                "wl_hash",
                "document_frequency",
                "nodes",
                "edges",
                "representative_node_labels",
            ]
        )
        writer.writerows(
            zip(
                range(len(feature_hashes)),
                feature_levels,
                feature_hashes,
                document_frequency,
                feature_nodes,
                feature_edges,
                feature_node_labels,
            )
        )

    labeled = samples[samples["is_labeled"]]
    class_counts = {
        str(int(key)): int(value)
        for key, value in labeled["grade"].value_counts().sort_index().items()
    }
    patient_counts = samples.loc[samples["patient_id"] != "", "patient_id"].value_counts()
    audit = {
        "motif_root": str(motif_root.resolve()),
        "manifest_path": str(resolved_manifest.resolve()),
        "grades_path": str(grades_path.resolve()),
        "h5ad_path": str(h5ad_path.resolve()) if h5ad_path else None,
        "triangle_levels": list(triangle_levels),
        "sample_count": len(samples),
        "labeled_sample_count": int(samples["is_labeled"].sum()),
        "unlabeled_sample_count": int((~samples["is_labeled"]).sum()),
        "class_counts": class_counts,
        "patient_count": int(samples.loc[samples["patient_id"] != "", "patient_id"].nunique()),
        "samples_from_repeated_patients": int(patient_counts[patient_counts > 1].sum()),
        "feature_count": matrix.shape[1],
        "nonzero_count": matrix.nnz,
        "search_exhausted_missing_level_files": sorted(set(exhausted_level_files)),
        "matrix_shape": list(matrix.shape),
        "matrix_dtype": str(matrix.dtype),
    }
    with audit_path.open("w", encoding="utf-8") as output:
        json.dump(audit, output, indent=2)
        output.write("\n")

    print(
        f"Prepared {matrix.shape[0]:,} samples x {matrix.shape[1]:,} motifs "
        f"({matrix.nnz:,} nonzeros) in {output_dir}"
    )
    return PreparedData(
        output_dir=output_dir,
        matrix_path=matrix_path,
        samples_path=samples_path,
        feature_levels_path=feature_levels_path,
        feature_index_path=feature_index_path,
        audit_path=audit_path,
        sample_count=matrix.shape[0],
        feature_count=matrix.shape[1],
        nonzero_count=matrix.nnz,
    )
