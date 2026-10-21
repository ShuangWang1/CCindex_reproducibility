#!/usr/bin/env python3
"""Trace rank-normalized significant motifs backward to TRI-1 and forward to TRI-7.

Focal motifs come from the complete nonzero-median rank-normalized screen and
must have a finite binary Cox p value below the requested threshold. Structural
relatives are retained whether or not rank-normalized survival statistics were
estimable for them.

As in the raw-count trace, the original motif search did not persist parent
identifiers or extendable-edge state. The relationships reported here are
therefore potential labeled-topological relationships rather than unique search
histories.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", "/tmp/breast-cancer-rank-significant-trace-matplotlib"
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd

import trace_significant_motifs as rawtrace


IDENTITY_COLUMNS = ["triangle_level", "wl_hash"]
REQUIRED_RANK_COLUMNS = {
    *IDENTITY_COLUMNS,
    "median_patient_rank_score",
    "p_value_rank_binary",
}


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    survival = Path(__file__).resolve().parent
    results = survival / "motif_results"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rank-results",
        type=Path,
        default=(
            results
            / "normalized_by_ranking_motif_analysis"
            / "significant"
            / "all_nonzero_threshold_rank_results.csv.gz"
        ),
    )
    parser.add_argument(
        "--target-levels", type=int, nargs="+", default=list(range(1, 8))
    )
    parser.add_argument("--target-hashes", nargs="*", default=None)
    parser.add_argument("--p-threshold", type=float, default=0.05)
    parser.add_argument("--results-root", type=Path, default=results)
    parser.add_argument(
        "--motif-root",
        type=Path,
        default=repo / "breast_cancer_meta" / "BZ_res_threshold_10_cpp_0811",
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
        "--output-root",
        type=Path,
        default=results / "significant_mtf_trace_ranking_normalized",
    )
    parser.add_argument(
        "--raw-trace-root",
        type=Path,
        default=results / "significant_mtf_trace",
        help="Reuse matching structural node/edge traces when available",
    )
    parser.add_argument(
        "--max-targets",
        type=int,
        default=None,
        help="Optional lowest-p target limit, primarily for smoke runs",
    )
    parser.add_argument(
        "--rebuild-cache", action="store_true", help="Rebuild the raw Cox cache"
    )
    parser.add_argument(
        "--rerun", action="store_true", help="Replace already completed target folders"
    )
    parser.add_argument(
        "--refresh-plots-only",
        action="store_true",
        help="Regenerate plots from completed trace tables without retracing",
    )
    return parser.parse_args()


def load_rank_results(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing rank-normalized result table: {path}")
    frame = pd.read_csv(path, dtype={"wl_hash": str}, low_memory=False)
    missing = sorted(REQUIRED_RANK_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Rank-normalized result table is missing columns: {missing}")
    frame["triangle_level"] = pd.to_numeric(
        frame["triangle_level"], errors="raise"
    ).astype(int)
    if frame[IDENTITY_COLUMNS].duplicated().any():
        raise ValueError("Rank-normalized result table has duplicate motif identities")
    if not frame["triangle_level"].between(rawtrace.MIN_LEVEL, rawtrace.MAX_LEVEL).all():
        raise ValueError("Rank-normalized result table has invalid triangle levels")
    return frame


def significant_rank_pool(frame: pd.DataFrame, p_threshold: float) -> pd.DataFrame:
    if not 0 < p_threshold < 1:
        raise ValueError("--p-threshold must be between zero and one")
    p_values = pd.to_numeric(frame["p_value_rank_binary"], errors="coerce")
    medians = pd.to_numeric(frame["median_patient_rank_score"], errors="coerce")
    selected = frame.loc[
        np.isfinite(p_values) & p_values.lt(p_threshold) & medians.gt(0)
    ].copy()
    selected["motif_id"] = selected.apply(
        lambda row: f"tri_{int(row.triangle_level)}:{row.wl_hash}", axis=1
    )
    selected["target_directory"] = selected.apply(
        lambda row: f"tri{int(row.triangle_level)}/{row.wl_hash}", axis=1
    )
    return selected.sort_values(
        ["triangle_level", "p_value_rank_binary", "wl_hash"], kind="mergesort"
    ).reset_index(drop=True)


def rank_stat_columns(frame: pd.DataFrame) -> list[str]:
    """Return nonidentity columns not already supplied by the raw Cox cache."""
    raw_columns = set(rawtrace.SQL_COLUMNS)
    return [
        column
        for column in frame.columns
        if column not in IDENTITY_COLUMNS and column not in raw_columns
    ]


def rank_lookup(frame: pd.DataFrame) -> pd.DataFrame:
    columns = rank_stat_columns(frame)
    lookup = frame[IDENTITY_COLUMNS + columns].copy()
    lookup["rank_result_available"] = True
    return lookup.set_index(IDENTITY_COLUMNS, verify_integrity=True)


def rank_lineage_nodes(
    graphs: dict[int, dict[str, object]],
    raw_cox: pd.DataFrame,
    ranked: pd.DataFrame,
    target_level: int,
    target_hash: str,
    p_threshold: float,
) -> pd.DataFrame:
    nodes = rawtrace.lineage_nodes(
        graphs, raw_cox, target_level, target_hash, p_threshold
    ).drop(
        columns=[
            "significant_raw_p",
            "positive_median_threshold",
            "selected_focal_rule",
        ]
    )
    return add_rank_fields(nodes, ranked, p_threshold)


def add_rank_fields(
    nodes: pd.DataFrame, ranked: pd.DataFrame, p_threshold: float
) -> pd.DataFrame:
    """Enrich a structural/raw trace with rank-normalized survival fields."""
    obsolete_flags = [
        column
        for column in [
            "significant_raw_p",
            "positive_median_threshold",
            "selected_focal_rule",
        ]
        if column in nodes.columns
    ]
    if obsolete_flags:
        nodes = nodes.drop(columns=obsolete_flags)
    nodes = nodes.join(ranked, on=IDENTITY_COLUMNS)
    nodes["rank_result_available"] = nodes["rank_result_available"].eq(True)
    p_values = pd.to_numeric(nodes["p_value_rank_binary"], errors="coerce")
    medians = pd.to_numeric(nodes["median_patient_rank_score"], errors="coerce")
    nodes["significant_rank_p"] = np.isfinite(p_values) & p_values.lt(p_threshold)
    nodes["positive_rank_median"] = np.isfinite(medians) & medians.gt(0)
    nodes["selected_focal_rule"] = (
        nodes["significant_rank_p"] & nodes["positive_rank_median"]
    )
    return nodes.sort_values(
        ["triangle_level", "p_value_rank_binary", "wl_hash"],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)


def plot_rank_p_lineage(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    target_level: int,
    target_hash: str,
    output: Path,
    p_threshold: float,
) -> None:
    p_values = pd.to_numeric(nodes["p_value_rank_binary"], errors="coerce")
    finite = nodes.loc[np.isfinite(p_values) & p_values.gt(0)].copy()
    finite["minus_log10_p"] = -np.log10(
        pd.to_numeric(finite["p_value_rank_binary"], errors="raise")
    )
    positions: dict[tuple[int, str], tuple[float, float]] = {}
    for level, subset in finite.groupby("triangle_level", sort=True):
        ordered = subset.sort_values(["p_value_rank_binary", "wl_hash"])
        offsets = (
            np.linspace(-0.16, 0.16, len(ordered))
            if len(ordered) > 1
            else [0.0]
        )
        for offset, row in zip(offsets, ordered.itertuples(index=False)):
            positions[(int(level), row.wl_hash)] = (
                int(level) + float(offset),
                float(row.minus_log10_p),
            )

    fig, ax = plt.subplots(figsize=(15, 8.5))
    segments: list[list[tuple[float, float]]] = []
    for edge in edges.itertuples(index=False):
        parent_key = (int(edge.parent_level), edge.parent_hash)
        child_key = (int(edge.child_level), edge.child_hash)
        if parent_key in positions and child_key in positions:
            segments.append([positions[parent_key], positions[child_key]])
    if segments:
        ax.add_collection(
            LineCollection(
                segments,
                colors="0.72",
                linewidths=1.0,
                zorder=1,
            )
        )

    role_colors = {
        "ancestor": "#377eb8",
        "target": "#d1495b",
        "descendant": "#e69f00",
    }
    for role, subset in finite.groupby("lineage_role"):
        xy = [
            positions[(int(row.triangle_level), row.wl_hash)]
            for row in subset.itertuples(index=False)
        ]
        if xy:
            ax.scatter(
                [value[0] for value in xy],
                [value[1] for value in xy],
                s=95 if role == "target" else 52,
                color=role_colors[role],
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
                label=role.capitalize(),
            )
    if len(finite) <= 50:
        label_rows = finite
    else:
        label_rows = pd.concat(
            [
                finite.nsmallest(20, "p_value_rank_binary"),
                finite.loc[finite["lineage_role"].eq("target")],
            ]
        ).drop_duplicates(IDENTITY_COLUMNS)
    for row in label_rows.itertuples(index=False):
        x, y = positions[(int(row.triangle_level), row.wl_hash)]
        ax.annotate(
            row.wl_hash[:6],
            (x, y),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=7,
        )

    ax.axhline(
        -np.log10(p_threshold),
        color="black",
        linestyle="--",
        linewidth=1.4,
        label=f"Rank-normalized p = {p_threshold:g}",
    )
    levels = sorted(nodes["triangle_level"].unique())
    ax.set_xticks(levels)
    ax.set_xlim(min(levels) - 0.45, max(levels) + 0.45)
    finite_counts = finite.groupby("triangle_level").size()
    structural_counts = nodes.groupby("triangle_level").size()
    for level in levels:
        ax.text(
            level,
            0.012,
            f"{int(finite_counts.get(level, 0))}/{int(structural_counts[level])}\nfinite p / structural",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=7,
            color="0.35",
        )
    ax.set_xlabel("Triangle level", labelpad=31)
    ax.set_ylabel("−log10(rank-normalized binary Cox p value)")
    ax.set_title(
        f"Potential lineage of TRI-{target_level} motif {target_hash[:8]}: "
        "Rank-normalized Cox p\n"
        "Connected points have finite rank p values; all structural relatives remain in the CSV"
    )
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(output, dpi=190)
    plt.close(fig)


def write_target(
    connection: object,
    output_root: Path,
    target_row: pd.Series,
    target_graph: object,
    vocabularies: dict[int, set[str]],
    labels: list[str],
    ranked: pd.DataFrame,
    rank_results_path: Path,
    raw_trace_root: Path,
    p_threshold: float,
    rerun: bool,
) -> dict[str, object]:
    level = int(target_row["triangle_level"])
    target_hash = str(target_row["wl_hash"])
    destination = output_root / f"tri{level}" / target_hash
    metadata_path = destination / "trace_metadata.json"
    if metadata_path.is_file() and not rerun:
        saved = json.loads(metadata_path.read_text(encoding="utf-8"))
        if saved.get("status") == "complete":
            return {**saved, "run_action": "skipped_complete"}

    raw_trace_directory = raw_trace_root / f"tri{level}" / target_hash
    raw_nodes_path = raw_trace_directory / "trace_nodes.csv"
    raw_edges_path = raw_trace_directory / "trace_edges.csv"
    if raw_nodes_path.is_file() and raw_edges_path.is_file():
        raw_nodes = pd.read_csv(
            raw_nodes_path,
            dtype={"wl_hash": str},
            low_memory=False,
        )
        edges = pd.read_csv(
            raw_edges_path,
            dtype={"parent_hash": str, "child_hash": str},
            low_memory=False,
        )
        nodes = add_rank_fields(raw_nodes, ranked, p_threshold)
        structural_source = str(raw_trace_directory.resolve())
    else:
        graphs, edges = rawtrace.trace_lineage(
            target_graph, level, vocabularies, labels
        )
        wanted = {key: set(value.keys()) for key, value in graphs.items()}
        raw_cox = rawtrace.cox_rows(connection, wanted)
        nodes = rank_lineage_nodes(
            graphs, raw_cox, ranked, level, target_hash, p_threshold
        )
        structural_source = "generated from saved motif vocabulary"
    metadata = {
        "status": "complete",
        "motif_id": f"tri_{level}:{target_hash}",
        "target_level": level,
        "target_hash": target_hash,
        "target_p_value_rank_binary": float(target_row["p_value_rank_binary"]),
        "target_median_patient_rank_score": float(
            target_row["median_patient_rank_score"]
        ),
        "selection_rule": (
            f"finite p_value_rank_binary < {p_threshold:g} and "
            "median_patient_rank_score > 0"
        ),
        "rank_results_source": str(rank_results_path.resolve()),
        "structural_trace_source": structural_source,
        "lineage_filter": "all structural relatives in the saved level vocabulary",
        "rank_statistics_coverage": (
            "available only for motifs in the complete nonzero-median rank screen"
        ),
        "minimum_level": int(nodes["triangle_level"].min()),
        "maximum_level": int(nodes["triangle_level"].max()),
        "motifs_by_level": {
            str(int(key)): int(value)
            for key, value in nodes.groupby("triangle_level").size().items()
        },
        "motifs_with_rank_results": int(nodes["rank_result_available"].sum()),
        "potential_parent_child_edges": int(len(edges)),
        "reproduction_command": (
            "python breast_cancer_survival/"
            "trace_rank_normalized_significant_motifs.py "
            f"--target-levels {level} --target-hashes {target_hash} --rerun"
        ),
        "relationship_definition": (
            "potential labeled-topological one-triangle growth relation; original "
            "parent IDs and extendable-edge state were not persisted"
        ),
    }
    level_dir = output_root / f"tri{level}"
    level_dir.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target_hash}.", dir=level_dir))
    try:
        nodes.to_csv(temporary / "trace_nodes.csv", index=False)
        edges.to_csv(temporary / "trace_edges.csv", index=False)
        plot_rank_p_lineage(
            nodes,
            edges,
            level,
            target_hash,
            temporary / "rank_normalized_cox_p_value_by_level.png",
            p_threshold,
        )
        (temporary / "trace_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        if destination.exists():
            shutil.rmtree(destination)
        temporary.replace(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {**metadata, "run_action": "generated"}


def cumulative_manifest(
    output_root: Path, current_records: list[dict[str, object]]
) -> pd.DataFrame:
    records: dict[str, dict[str, object]] = {}
    for level in range(rawtrace.MIN_LEVEL, rawtrace.MAX_LEVEL + 1):
        for path in sorted((output_root / f"tri{level}").glob("*/trace_metadata.json")):
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved.get("status") == "complete":
                records[str(saved["motif_id"])] = {
                    **saved,
                    "run_action": "existing_complete",
                }
    for record in current_records:
        records[str(record["motif_id"])] = record
    frame = pd.DataFrame(records.values())
    return frame.sort_values(
        ["target_level", "target_hash"], kind="mergesort"
    ).reset_index(drop=True)


def refresh_completed_plots(
    output_root: Path,
    levels: list[int],
    target_hashes: list[str] | None,
    p_threshold: float,
) -> int:
    wanted = set(target_hashes) if target_hashes is not None else None
    refreshed: set[str] = set()
    count = 0
    for level in levels:
        for directory in sorted((output_root / f"tri{level}").iterdir()):
            if not directory.is_dir() or directory.name.startswith("."):
                continue
            target_hash = directory.name
            if wanted is not None and target_hash not in wanted:
                continue
            nodes_path = directory / "trace_nodes.csv"
            edges_path = directory / "trace_edges.csv"
            if not nodes_path.is_file() or not edges_path.is_file():
                raise FileNotFoundError(f"Incomplete trace directory: {directory}")
            nodes = pd.read_csv(nodes_path, dtype={"wl_hash": str}, low_memory=False)
            edges = pd.read_csv(
                edges_path,
                dtype={"parent_hash": str, "child_hash": str},
                low_memory=False,
            )
            output = directory / "rank_normalized_cox_p_value_by_level.png"
            temporary = output.with_name(output.name + ".tmp.png")
            plot_rank_p_lineage(
                nodes, edges, level, target_hash, temporary, p_threshold
            )
            temporary.replace(output)
            refreshed.add(target_hash)
            count += 1
            if count % 100 == 0:
                print(f"Refreshed {count} lineage plots", flush=True)
    if wanted is not None and refreshed != wanted:
        raise ValueError(
            f"Requested hashes lack completed traces: {sorted(wanted - refreshed)}"
        )
    return count


def main() -> None:
    args = parse_args()
    levels = rawtrace.validate_levels(args.target_levels)
    if args.max_targets is not None and args.max_targets <= 0:
        raise ValueError("--max-targets must be positive")
    args.output_root.mkdir(parents=True, exist_ok=True)
    for level in range(rawtrace.MIN_LEVEL, rawtrace.MAX_LEVEL + 1):
        (args.output_root / f"tri{level}").mkdir(exist_ok=True)

    if args.refresh_plots_only:
        count = refresh_completed_plots(
            args.output_root,
            levels,
            args.target_hashes,
            args.p_threshold,
        )
        print(f"Refreshed {count} completed lineage plots", flush=True)
        return

    rank_results = load_rank_results(args.rank_results)
    pool = significant_rank_pool(rank_results, args.p_threshold)
    rawtrace.atomic_csv(pool, args.output_root / "potential_motif_pool.csv")
    pool_counts = {
        str(int(level)): int(count)
        for level, count in pool.groupby("triangle_level").size().items()
    }
    rawtrace.atomic_json(
        {
            "selection_rule": (
                f"finite p_value_rank_binary < {args.p_threshold:g} and "
                "median_patient_rank_score > 0"
            ),
            "screening_family": (
                "all motifs with nonzero patient rank-score median; no raw p-value filter"
            ),
            "rank_results_source": str(args.rank_results.resolve()),
            "total_targets": int(len(pool)),
            "targets_by_level": pool_counts,
            "lineage_filter": (
                "all structural relatives regardless of rank p value or median"
            ),
        },
        args.output_root / "potential_motif_pool_summary.json",
    )

    targets = pool.loc[pool["triangle_level"].isin(levels)].copy()
    if args.target_hashes is not None:
        wanted_hashes = set(args.target_hashes)
        targets = targets.loc[targets["wl_hash"].isin(wanted_hashes)]
        missing = wanted_hashes - set(targets["wl_hash"])
        if missing:
            raise ValueError(
                f"Requested hashes are not selected rank-significant motifs: {sorted(missing)}"
            )
    targets = targets.sort_values(
        ["p_value_rank_binary", "triangle_level", "wl_hash"], kind="mergesort"
    )
    if args.max_targets is not None:
        targets = targets.head(args.max_targets)
    if targets.empty:
        raise ValueError("No focal motifs matched the requested target selection")

    connection = rawtrace.build_cox_cache(
        args.output_root / "cache" / "cox_results.sqlite",
        args.results_root,
        args.rebuild_cache,
    )
    topology_cache_path = args.output_root / "cache" / "target_topologies.csv.gz"
    if topology_cache_path.is_file():
        cached_topologies = pd.read_csv(
            topology_cache_path, dtype={"wl_hash": str}
        )
    else:
        cached_topologies = pd.DataFrame(columns=IDENTITY_COLUMNS)
    cached_keys = set(
        zip(
            cached_topologies["triangle_level"].astype(int),
            cached_topologies["wl_hash"].astype(str),
        )
    )
    missing_mask = [
        (int(row.triangle_level), str(row.wl_hash)) not in cached_keys
        for row in targets.itertuples(index=False)
    ]
    missing_targets = targets.loc[missing_mask]
    if len(missing_targets):
        new_topologies = rawtrace.recover_target_topologies(
            missing_targets, args.artifacts_root, args.motif_root
        )
        cached_topologies = rawtrace.merge_topology_cache(
            new_topologies, topology_cache_path
        )
    topology_cache = cached_topologies.set_index(IDENTITY_COLUMNS)
    vocabularies = rawtrace.vocabulary(
        connection, range(rawtrace.MIN_LEVEL, rawtrace.MAX_LEVEL + 1)
    )
    ranked = rank_lookup(rank_results)
    labels = [str(value) for value in range(8)]
    manifest_rows: list[dict[str, object]] = []
    failures: list[str] = []
    for position, target in enumerate(targets.itertuples(index=False), start=1):
        target_row = pd.Series(target._asdict())
        key = (int(target.triangle_level), str(target.wl_hash))
        topology = topology_cache.loc[key]
        graph = rawtrace.graph_from_encoding(
            str(topology.node_labels), str(topology.edge_list), str(target.wl_hash)
        )
        try:
            record = write_target(
                connection,
                args.output_root,
                target_row,
                graph,
                vocabularies,
                labels,
                ranked,
                args.rank_results,
                args.raw_trace_root,
                args.p_threshold,
                args.rerun,
            )
        except Exception as error:
            failures.append(str(target.wl_hash))
            record = {
                "status": "failed",
                "motif_id": (
                    f"tri_{int(target.triangle_level)}:{target.wl_hash}"
                ),
                "target_level": int(target.triangle_level),
                "target_hash": str(target.wl_hash),
                "error": f"{type(error).__name__}: {error}",
                "run_action": "failed",
            }
        manifest_rows.append(record)
        if position % 25 == 0 or position == len(targets):
            print(f"Processed {position}/{len(targets)} focal motifs", flush=True)

    manifest = cumulative_manifest(args.output_root, manifest_rows)
    rawtrace.atomic_csv(manifest, args.output_root / "trace_run_manifest.csv")
    connection.close()
    if failures:
        raise RuntimeError(
            f"Tracing failed for {len(failures)} motifs: {failures[:3]}"
        )
    print(
        f"Completed {len(targets)} targets under {args.output_root}; "
        f"pool contains {len(pool)} motifs",
        flush=True,
    )


if __name__ == "__main__":
    main()
