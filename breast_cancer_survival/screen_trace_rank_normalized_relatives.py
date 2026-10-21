#!/usr/bin/env python3
"""Compute rank-normalized Cox results for every motif in saved lineage traces."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata

import analyze_rank_normalized_motif_survival as rankmod
from analyze_motif_survival_unadjusted import (
    STATUS_LABELS,
    load_inventory,
    vectorized_binary_cox,
)
from normalize_significant_motif_frequencies import clinical_core_cohort
import trace_rank_normalized_significant_motifs as ranktrace


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    here = Path(__file__).resolve().parent
    results = here / "motif_results"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace-root",
        type=Path,
        default=results / "significant_mtf_trace_ranking_normalized",
    )
    parser.add_argument(
        "--base-rank-results",
        type=Path,
        default=(
            results
            / "normalized_by_ranking_motif_analysis"
            / "significant"
            / "all_nonzero_threshold_rank_results.csv.gz"
        ),
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=(
            repo
            / "breast_cancer_classification_workstation"
            / "classification_by_level"
            / "mtf_occurrence"
            / "mrmr_selection"
            / "parameter_tuning_bz"
            / "artifacts"
        ),
    )
    parser.add_argument(
        "--motif-root",
        type=Path,
        default=repo / "breast_cancer_meta" / "BZ_res_threshold_10_cpp_0811",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=repo / "breast_cancer" / "Basel_PatientMetadata.csv",
    )
    parser.add_argument("--levels", type=int, nargs="+", default=list(range(1, 8)))
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--skip-trace-update", action="store_true")
    return parser.parse_args()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".tmp.gz" if str(path).endswith(".gz") else ".tmp"
    temporary = path.with_name(path.name + suffix)
    frame.to_csv(temporary, index=False, compression="gzip" if str(path).endswith(".gz") else None)
    temporary.replace(path)


def structural_union(trace_root: Path, levels: list[int]) -> dict[int, set[str]]:
    wanted = {level: set() for level in levels}
    paths = [
        path
        for level in levels
        for path in sorted((trace_root / f"tri{level}").glob("*/trace_nodes.csv"))
        if not path.parent.name.endswith("_new")
        and not path.parent.name.startswith(".")
    ]
    if not paths:
        raise FileNotFoundError(f"No completed traces found under {trace_root}")
    for position, path in enumerate(paths, start=1):
        nodes = pd.read_csv(
            path,
            usecols=["triangle_level", "wl_hash"],
            dtype={"wl_hash": str},
        )
        for level, subset in nodes.groupby("triangle_level"):
            level = int(level)
            if level in wanted:
                wanted[level].update(subset["wl_hash"].astype(str))
        if position % 100 == 0:
            print(f"Inventoried {position}/{len(paths)} trace tables", flush=True)
    return wanted


def patient_aggregation_matrix(
    cohort: pd.DataFrame, patients: pd.DataFrame
) -> sparse.csr_matrix:
    groups = cohort.groupby("PID", sort=False).indices
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for patient_row, patient in enumerate(patients.itertuples(index=False)):
        indices = np.asarray(groups[patient.PID], int)
        rows.extend([patient_row] * len(indices))
        columns.extend(indices.tolist())
        values.extend([1.0 / len(indices)] * len(indices))
    return sparse.csr_matrix(
        (values, (rows, columns)), shape=(len(patients), len(cohort))
    )


def rank_score_sparse(counts: sparse.spmatrix, vocabulary_size: int) -> sparse.csr_matrix:
    scores = counts.tocsr(copy=True).astype(float)
    scores.sum_duplicates()
    scores.eliminate_zeros()
    if scores.shape[1] != vocabulary_size:
        raise ValueError("Count matrix and vocabulary size differ")
    for row in range(scores.shape[0]):
        start, stop = scores.indptr[row : row + 2]
        values = scores.data[start:stop]
        if len(values):
            if not np.isfinite(values).all() or (values <= 0).any():
                raise ValueError("Sparse motif counts must be finite and positive")
            ranks = rankdata(-values, method="average")
            scores.data[start:stop] = (
                vocabulary_size - ranks
            ) / (vocabulary_size - 1)
    return scores


def tri1_core_scores(
    cohort: pd.DataFrame, motif_root: Path
) -> tuple[sparse.csr_matrix, list[str]]:
    vocabulary = rankmod.tri1_vocabulary(motif_root)
    lookup = {value: index for index, value in enumerate(vocabulary)}
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for row, item in enumerate(cohort.itertuples(index=False)):
        path = motif_root / str(item.sample) / "ccindex_tri_1.csv"
        frame = pd.read_csv(
            path, usecols=["wl_hash", "matches"], dtype={"wl_hash": str}
        )
        counts = pd.to_numeric(frame["matches"], errors="raise").to_numpy(float)
        positive = counts > 0
        rows.extend([row] * int(positive.sum()))
        columns.extend(frame.loc[positive, "wl_hash"].map(lookup).astype(int).tolist())
        values.extend(counts[positive].tolist())
    counts = sparse.csr_matrix(
        (values, (rows, columns)), shape=(len(cohort), len(vocabulary))
    )
    return rank_score_sparse(counts, len(vocabulary)), vocabulary


def high_level_core_scores(
    level: int, cohort: pd.DataFrame, artifacts_root: Path
) -> tuple[sparse.csr_matrix, pd.DataFrame]:
    level_dir = artifacts_root / f"tri_{level}"
    samples = load_inventory(artifacts_root, level)
    observed = samples.iloc[cohort["artifact_row"].to_numpy(int)][
        ["sample", "source_sample"]
    ].reset_index(drop=True)
    expected = cohort[["sample", "source_sample"]].reset_index(drop=True)
    if not observed.equals(expected):
        raise ValueError(f"TRI-{level} sample inventory differs")
    features = pd.read_csv(
        level_dir / "feature_index.csv.gz",
        usecols=["feature_index", "wl_hash"],
        dtype={"wl_hash": str},
    )
    if features["wl_hash"].duplicated().any():
        raise ValueError(f"TRI-{level} feature hashes are duplicated")
    matrix = sparse.load_npz(level_dir / "motif_counts.npz").tocsr()
    if matrix.shape != (len(samples), len(features)):
        raise ValueError(f"TRI-{level} matrix dimensions differ from its indices")
    counts = matrix[cohort["artifact_row"].to_numpy(int)]
    return rank_score_sparse(counts, len(features)), features


def fit_level(
    level: int,
    motif_hashes: list[str],
    cohort: pd.DataFrame,
    patients: pd.DataFrame,
    aggregation: sparse.csr_matrix,
    artifacts_root: Path,
    motif_root: Path,
    chunk_size: int,
) -> pd.DataFrame:
    if level == 1:
        core_scores, vocabulary = tri1_core_scores(cohort, motif_root)
        lookup = pd.Series(np.arange(len(vocabulary), dtype=int), index=vocabulary)
    else:
        core_scores, features = high_level_core_scores(level, cohort, artifacts_root)
        lookup = features.set_index("wl_hash")["feature_index"]
    missing = sorted(set(motif_hashes) - set(lookup.index))
    if missing:
        raise ValueError(f"TRI-{level} structural motifs absent from vocabulary: {missing[:3]}")
    columns = lookup.loc[motif_hashes].to_numpy(int)
    patient_scores = (aggregation @ core_scores).tocsr()
    time = patients["time_months"].to_numpy(float)
    event = patients["event"].to_numpy(int)
    parts: list[pd.DataFrame] = []
    for start in range(0, len(columns), chunk_size):
        stop = min(start + chunk_size, len(columns))
        score = patient_scores[:, columns[start:stop]].toarray()
        medians = np.median(score, axis=0)
        groups = score > medians
        fitted = vectorized_binary_cox(groups, time, event)
        part = pd.DataFrame(
            {
                "triangle_level": level,
                "wl_hash": motif_hashes[start:stop],
                "vocabulary_size": core_scores.shape[1],
                "median_patient_rank_score": medians,
                "rank_group_definition": "patient_rank_score > cohort_median",
                "n_survival": len(time),
                "events_survival": int(event.sum()),
                "n_high_rank": fitted["n_enriched"],
                "events_high_rank": fitted["events_enriched"],
                "n_low_rank": fitted["n_non_enriched"],
                "events_low_rank": fitted["events_non_enriched"],
                "coef_rank_binary": fitted["coef"],
                "hazard_ratio_rank_binary": fitted["hazard_ratio"],
                "ci95_low_rank_binary": fitted["ci95_low"],
                "ci95_high_rank_binary": fitted["ci95_high"],
                "p_value_rank_binary": fitted["p_value"],
                "rank_binary_fit_status": [
                    STATUS_LABELS[int(code)] for code in fitted["status_code"]
                ],
            }
        )
        parts.append(part)
        print(
            f"TRI-{level}: fitted {stop:,}/{len(columns):,} structural relatives",
            flush=True,
        )
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def merge_rank_sources(base: pd.DataFrame, extended: pd.DataFrame) -> pd.DataFrame:
    columns = sorted(set(base.columns) | set(extended.columns))
    combined = pd.concat(
        [base.reindex(columns=columns), extended.reindex(columns=columns)],
        ignore_index=True,
    )
    if combined.duplicated(ranktrace.IDENTITY_COLUMNS).any():
        raise ValueError("Base and extended rank results overlap")
    return combined


def update_trace_tables(
    trace_root: Path, combined: pd.DataFrame, p_threshold: float = 0.05
) -> int:
    lookup = ranktrace.rank_lookup(combined)
    rank_columns = ranktrace.rank_stat_columns(combined)
    removable = set(rank_columns) | {
        "rank_result_available",
        "significant_rank_p",
        "positive_rank_median",
        "selected_focal_rule",
    }
    count = 0
    for nodes_path in sorted(trace_root.glob("tri*/*/trace_nodes.csv")):
        if nodes_path.parent.name.endswith("_new") or nodes_path.parent.name.startswith("."):
            continue
        nodes = pd.read_csv(nodes_path, dtype={"wl_hash": str}, low_memory=False)
        nodes = nodes.drop(columns=[x for x in removable if x in nodes.columns])
        nodes = ranktrace.add_rank_fields(nodes, lookup, p_threshold)
        metadata_path = nodes_path.parent / "trace_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["motifs_with_rank_results"] = int(nodes["rank_result_available"].sum())
        metadata["rank_statistics_coverage"] = (
            "computed for every structural relative; p values may be undefined "
            "when the binary Cox model is not estimable"
        )
        metadata["package_variant"] = "expanded_all_structural_rank_results"
        metadata["source_trace_directory"] = str(nodes_path.parent.resolve())
        level = int(metadata["target_level"])
        target_hash = str(metadata["target_hash"])
        edges = pd.read_csv(
            nodes_path.parent / "trace_edges.csv",
            dtype={"parent_hash": str, "child_hash": str},
            low_memory=False,
        )
        level_directory = nodes_path.parent.parent
        destination = level_directory / f"{target_hash}_new"
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{target_hash}_new.", dir=level_directory)
        )
        try:
            nodes.to_csv(temporary / "trace_nodes.csv", index=False)
            edges.to_csv(temporary / "trace_edges.csv", index=False)
            (temporary / "trace_metadata.json").write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
            )
            ranktrace.plot_rank_p_lineage(
                nodes,
                edges,
                level,
                target_hash,
                temporary / "rank_normalized_cox_p_value_by_level.png",
                p_threshold,
            )
            if destination.exists():
                shutil.rmtree(destination)
            temporary.replace(destination)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        count += 1
        if count % 50 == 0:
            print(f"Updated {count} trace packages", flush=True)
    return count


def main() -> None:
    args = parse_args()
    levels = sorted(set(args.levels))
    if not levels or any(level < 1 or level > 7 for level in levels):
        raise ValueError("--levels must contain values from 1 through 7")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    for level in levels:
        for path in (args.trace_root / f"tri{level}").glob(".*_new.*"):
            if path.is_dir():
                shutil.rmtree(path)
    base = ranktrace.load_rank_results(args.base_rank_results)
    structural = structural_union(args.trace_root, levels)
    base_keys = {
        level: set(base.loc[base["triangle_level"].eq(level), "wl_hash"].astype(str))
        for level in levels
    }
    requested = {
        level: sorted(structural[level] - base_keys[level]) for level in levels
    }
    print(
        "Additional structural relatives by level: "
        + ", ".join(f"TRI-{level}={len(requested[level]):,}" for level in levels),
        flush=True,
    )
    metadata = pd.read_csv(args.metadata)
    cohort, patients, _ = clinical_core_cohort(args.artifacts_root, metadata)
    aggregation = patient_aggregation_matrix(cohort, patients)
    cache_root = args.trace_root / "cache" / "extended_rank_results"
    parts: list[pd.DataFrame] = []
    for level in levels:
        cache_path = cache_root / f"tri{level}_rank_results.csv.gz"
        if cache_path.is_file() and not args.rerun:
            result = pd.read_csv(cache_path, dtype={"wl_hash": str}, low_memory=False)
            if set(result["wl_hash"].astype(str)) != set(requested[level]):
                raise ValueError(f"TRI-{level} cached result identities differ")
            print(f"TRI-{level}: reused {len(result):,} cached results", flush=True)
        elif requested[level]:
            result = fit_level(
                level,
                requested[level],
                cohort,
                patients,
                aggregation,
                args.artifacts_root,
                args.motif_root,
                args.chunk_size,
            )
            atomic_csv(result, cache_path)
        else:
            result = pd.DataFrame()
        if len(result):
            parts.append(result)
    extended = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    output = args.trace_root / "all_structural_relative_rank_results.csv.gz"
    atomic_csv(extended, output)
    summary = {
        "base_rank_results": str(args.base_rank_results.resolve()),
        "selection": "all unique structural relatives in the 740 rank-significant traces",
        "additional_motifs": int(len(extended)),
        "additional_motifs_by_level": {
            str(level): len(requested[level]) for level in levels
        },
        "finite_binary_cox_p_values": int(
            np.isfinite(extended["p_value_rank_binary"]).sum()
        ),
        "undefined_binary_cox_p_values": int(
            (~np.isfinite(extended["p_value_rank_binary"])).sum()
        ),
        "zero_median_motifs": int(extended["median_patient_rank_score"].eq(0).sum()),
        "group_definition": "patient_rank_score > cohort_median",
    }
    (args.trace_root / "all_structural_relative_rank_results_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    if not args.skip_trace_update:
        combined = merge_rank_sources(base, extended)
        updated = update_trace_tables(args.trace_root, combined)
        summary["updated_trace_packages"] = updated
        (args.trace_root / "all_structural_relative_rank_results_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
