#!/usr/bin/env python3
"""Export complete sample-by-motif occurrence tables for each TRI level."""

from __future__ import annotations

import argparse
import csv
import gzip
import io
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


OUTPUT_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACTS = OUTPUT_DIR.parent / "artifacts"
DEFAULT_LEVELS = tuple(range(2, 8))


def _csv_fields(values: list[object]) -> bytes:
    """Encode a short sequence as one CSV row without its line ending."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="")
    writer.writerow(values)
    return buffer.getvalue().encode("utf-8")


def _metadata_value(value: object) -> object:
    return "" if pd.isna(value) else value


def _write_occurrence_table(
    path: Path,
    samples: pd.DataFrame,
    features: pd.DataFrame,
    matrix: sparse.csr_matrix,
) -> None:
    """Stream a dense-form CSV while keeping the source matrix sparse in memory."""
    if matrix.shape != (len(samples), len(features)):
        raise ValueError(
            f"Matrix shape {matrix.shape} does not match "
            f"{len(samples)} samples and {len(features)} motifs"
        )
    if not np.all(matrix.data == np.floor(matrix.data)):
        raise ValueError("Occurrence matrix contains non-integer frequencies")

    motif_columns = [f"motif_{value}" for value in features["wl_hash"]]
    header = _csv_fields([*samples.columns, *motif_columns]) + b"\n"
    temporary = path.with_suffix(path.suffix + ".tmp")

    with temporary.open("wb") as raw:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, compresslevel=6, mtime=0
        ) as output:
            output.write(header)
            for row_number, sample_row in enumerate(samples.itertuples(index=False, name=None)):
                metadata = [_metadata_value(value) for value in sample_row]
                output.write(_csv_fields(metadata))

                start, stop = matrix.indptr[row_number : row_number + 2]
                indices = matrix.indices[start:stop]
                values = matrix.data[start:stop]
                previous = -1
                for column, value in zip(indices, values):
                    gap = int(column) - previous - 1
                    if gap:
                        output.write(b",0" * gap)
                    output.write(b"," + str(int(value)).encode("ascii"))
                    previous = int(column)
                remaining = matrix.shape[1] - previous - 1
                if remaining:
                    output.write(b",0" * remaining)
                output.write(b"\n")

    temporary.replace(path)


def export_level(artifacts_root: Path, output_dir: Path, level: int) -> dict[str, int]:
    source = artifacts_root / f"tri_{level}"
    samples = pd.read_csv(source / "samples.csv")
    features = pd.read_csv(source / "feature_index.csv.gz")
    matrix = sparse.load_npz(source / "motif_counts.npz").tocsr()
    matrix.sort_indices()

    occurrence_path = output_dir / f"tri_{level}_mtf_occurrence.csv.gz"
    motif_path = output_dir / f"tri_{level}_motifs.csv.gz"
    _write_occurrence_table(occurrence_path, samples, features, matrix)
    features.to_csv(motif_path, index=False, compression="gzip")

    return {
        "triangle_level": level,
        "samples": matrix.shape[0],
        "motifs": matrix.shape[1],
        "nonzero_frequencies": matrix.nnz,
        "zero_frequencies": matrix.shape[0] * matrix.shape[1] - matrix.nnz,
        "total_occurrences": int(matrix.sum()),
        "samples_with_no_occurrences": int(np.sum(matrix.getnnz(axis=1) == 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--levels", type=int, nargs="+", default=DEFAULT_LEVELS)
    args = parser.parse_args()

    levels = tuple(sorted(set(args.levels)))
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for level in levels:
        print(f"Exporting TRI {level}...", flush=True)
        summary = export_level(args.artifacts, args.output, level)
        summaries.append(summary)
        print(
            f"  {summary['samples']:,} samples x {summary['motifs']:,} motifs; "
            f"{summary['total_occurrences']:,} total occurrences",
            flush=True,
        )

    pd.DataFrame(summaries).to_csv(args.output / "level_summary.csv", index=False)
    print(f"Wrote {len(summaries)} level tables to {args.output}")


if __name__ == "__main__":
    main()
