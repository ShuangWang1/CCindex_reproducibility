#!/usr/bin/env python3
"""Trace significant BZ motifs backward to TRI-1 and forward to TRI-7.

Focal motifs must have a finite raw Cox p value below the requested threshold
and a positive patient-median count.  Significance is only a focal-selection
rule: every structurally valid relative in the saved vocabulary is retained.

The original motif search did not save parent identifiers or extendable-edge
state.  Relationships produced here are therefore potential labeled-
topological relationships, not uniquely recorded search histories.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import tempfile
from itertools import combinations
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/breast-cancer-significant-trace-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse

from trace_back_tri7_motif_survival import (
    graph_encoding,
    graph_from_row,
    motif_hash,
    potential_direct_parents,
    triangle_count,
)


MIN_LEVEL = 1
MAX_LEVEL = 7
COX_COLUMNS = [
    "wl_hash",
    "sample_document_frequency",
    "nodes",
    "edges",
    "representative_node_labels",
    "cell_type_composition",
    "median_patient_raw_count",
    "coef_unadjusted",
    "hazard_ratio_unadjusted",
    "ci95_low_unadjusted",
    "ci95_high_unadjusted",
    "p_value_unadjusted",
    "n_enriched_unadjusted",
    "events_enriched_unadjusted",
    "n_non_enriched_unadjusted",
    "events_non_enriched_unadjusted",
    "fit_status_unadjusted",
    "n_unadjusted",
    "events_unadjusted",
    "q_value_bh_global_unadjusted",
    "q_value_bh_within_level_unadjusted",
    "rank_global_unadjusted",
    "rank_within_level_unadjusted",
]
SQL_COLUMNS = ["triangle_level", *COX_COLUMNS]
NUMERIC_SQL_COLUMNS = set(SQL_COLUMNS) - {
    "wl_hash",
    "representative_node_labels",
    "cell_type_composition",
    "fit_status_unadjusted",
}


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    survival = Path(__file__).resolve().parent
    results = survival / "motif_results"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-levels", type=int, nargs="+", default=[7])
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
        default=repo
        / "breast_cancer_classification_workstation"
        / "classification_by_level"
        / "mtf_occurrence"
        / "mrmr_selection"
        / "parameter_tuning_bz"
        / "artifacts",
    )
    parser.add_argument(
        "--output-root", type=Path, default=results / "significant_mtf_trace"
    )
    parser.add_argument(
        "--max-targets",
        type=int,
        default=None,
        help="Optional lowest-p target limit, primarily for smoke runs",
    )
    parser.add_argument(
        "--rebuild-cache", action="store_true", help="Rebuild the Cox SQLite cache"
    )
    parser.add_argument(
        "--rerun", action="store_true", help="Replace already completed target folders"
    )
    return parser.parse_args()


def validate_levels(levels: Iterable[int]) -> list[int]:
    unique = sorted(set(levels))
    if not unique or any(level < MIN_LEVEL or level > MAX_LEVEL for level in unique):
        raise ValueError("Target levels must be between 1 and 7")
    return unique


def atomic_csv(frame: pd.DataFrame, path: Path, **kwargs: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".tmp.gz" if str(path).endswith(".gz") else ".tmp"
    temporary = path.with_name(path.name + suffix)
    frame.to_csv(temporary, index=False, **kwargs)
    temporary.replace(path)


def atomic_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def cache_is_current(connection: sqlite3.Connection, results_root: Path) -> bool:
    connection.execute(
        "CREATE TABLE IF NOT EXISTS source_files "
        "(triangle_level INTEGER PRIMARY KEY, path TEXT, size INTEGER, mtime_ns INTEGER)"
    )
    saved = {
        int(row[0]): (row[1], int(row[2]), int(row[3]))
        for row in connection.execute(
            "SELECT triangle_level, path, size, mtime_ns FROM source_files"
        )
    }
    expected = {}
    for level in range(MIN_LEVEL, MAX_LEVEL + 1):
        path = results_root / f"tri_{level}_cox_results_sorted.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing Cox result: {path}")
        stat = path.stat()
        expected[level] = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    table_exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='cox'"
    ).fetchone()
    return bool(table_exists and saved == expected)


def build_cox_cache(
    database: Path, results_root: Path, rebuild: bool = False
) -> sqlite3.Connection:
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database)
    if not rebuild and cache_is_current(connection, results_root):
        return connection
    connection.executescript(
        "DROP TABLE IF EXISTS cox; DROP TABLE IF EXISTS source_files;"
    )
    text_columns = {
        "wl_hash",
        "representative_node_labels",
        "cell_type_composition",
        "fit_status_unadjusted",
    }
    definitions = [
        f'"{column}" {"TEXT" if column in text_columns else "REAL"}'
        for column in SQL_COLUMNS
    ]
    definitions[0] = '"triangle_level" INTEGER'
    connection.execute(
        f"CREATE TABLE cox ({', '.join(definitions)}, "
        "PRIMARY KEY (triangle_level, wl_hash)) WITHOUT ROWID"
    )
    connection.execute(
        "CREATE TABLE source_files "
        "(triangle_level INTEGER PRIMARY KEY, path TEXT, size INTEGER, mtime_ns INTEGER)"
    )
    placeholders = ",".join("?" for _ in SQL_COLUMNS)
    insert = f"INSERT INTO cox VALUES ({placeholders})"
    for level in range(MIN_LEVEL, MAX_LEVEL + 1):
        path = results_root / f"tri_{level}_cox_results_sorted.csv"
        print(f"Caching TRI-{level} Cox results", flush=True)
        for chunk in pd.read_csv(
            path, usecols=COX_COLUMNS, dtype={"wl_hash": str}, chunksize=100_000
        ):
            chunk.insert(0, "triangle_level", level)
            records = []
            for row in chunk[SQL_COLUMNS].itertuples(index=False, name=None):
                records.append(
                    tuple(
                        None if pd.isna(value) else value.item() if hasattr(value, "item") else value
                        for value in row
                    )
                )
            connection.executemany(insert, records)
            connection.commit()
        stat = path.stat()
        connection.execute(
            "INSERT INTO source_files VALUES (?, ?, ?, ?)",
            (level, str(path.resolve()), stat.st_size, stat.st_mtime_ns),
        )
        connection.commit()
    return connection


def significant_pool(
    connection: sqlite3.Connection, p_threshold: float
) -> pd.DataFrame:
    if not 0 < p_threshold < 1:
        raise ValueError("--p-threshold must be between zero and one")
    columns = ", ".join(f'"{column}"' for column in SQL_COLUMNS)
    frame = pd.read_sql_query(
        f"SELECT {columns} FROM cox "
        "WHERE p_value_unadjusted IS NOT NULL "
        "AND p_value_unadjusted < ? AND median_patient_raw_count > 0 "
        "ORDER BY triangle_level, p_value_unadjusted, wl_hash",
        connection,
        params=(p_threshold,),
    )
    frame["triangle_level"] = frame["triangle_level"].astype(int)
    frame["motif_id"] = frame.apply(
        lambda row: f"tri_{int(row.triangle_level)}:{row.wl_hash}", axis=1
    )
    frame["target_directory"] = frame.apply(
        lambda row: f"tri{int(row.triangle_level)}/{row.wl_hash}", axis=1
    )
    return frame


def vocabulary(connection: sqlite3.Connection, levels: Iterable[int]) -> dict[int, set[str]]:
    return {
        level: {
            str(row[0])
            for row in connection.execute(
                "SELECT wl_hash FROM cox WHERE triangle_level = ?", (level,)
            )
        }
        for level in levels
    }


def graph_from_encoding(node_labels: str, edge_list: str, expected_hash: str) -> nx.Graph:
    row = pd.Series(
        {"node_labels": node_labels, "edge_list": edge_list, "wl_hash": expected_hash}
    )
    return graph_from_row(row)


def recover_target_topologies(
    targets: pd.DataFrame,
    artifacts_root: Path,
    motif_root: Path,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for level, wanted_frame in targets.groupby("triangle_level", sort=True):
        wanted = set(wanted_frame["wl_hash"].astype(str))
        if int(level) == 1:
            for row in wanted_frame.itertuples(index=False):
                labels = str(row.representative_node_labels)
                edge_list = "0-1;0-2;1-2"
                graph = graph_from_encoding(labels, edge_list, str(row.wl_hash))
                rows.append(
                    {
                        "triangle_level": 1,
                        "wl_hash": str(row.wl_hash),
                        "nodes": graph.number_of_nodes(),
                        "edges": graph.number_of_edges(),
                        "node_labels": labels,
                        "edge_list": edge_list,
                    }
                )
            continue

        level_dir = artifacts_root / f"tri_{int(level)}"
        features = pd.read_csv(
            level_dir / "feature_index.csv.gz",
            usecols=["feature_index", "wl_hash"],
            dtype={"wl_hash": str},
        ).set_index("wl_hash")
        missing_index = wanted - set(features.index)
        if missing_index:
            raise ValueError(f"TRI-{level} targets absent from feature index")
        samples = pd.read_csv(level_dir / "samples.csv", usecols=["sample"])
        matrix = sparse.load_npz(level_dir / "motif_counts.npz").tocsc()
        requested: dict[Path, set[str]] = {}
        for target_hash in wanted:
            column = int(features.loc[target_hash, "feature_index"])
            start, stop = matrix.indptr[column : column + 2]
            if start == stop:
                raise ValueError(f"No source occurrence for TRI-{level} {target_hash}")
            sample_row = int(matrix.indices[start])
            sample = str(samples.iloc[sample_row]["sample"])
            path = motif_root / sample / f"ccindex_tri_{int(level)}.csv"
            requested.setdefault(path, set()).add(target_hash)
        for position, (path, hashes) in enumerate(requested.items(), start=1):
            frame = pd.read_csv(
                path,
                usecols=["wl_hash", "node_labels", "edge_list", "nodes", "edges"],
                dtype={"wl_hash": str, "node_labels": str, "edge_list": str},
            )
            found = frame.loc[frame["wl_hash"].isin(hashes)].drop_duplicates("wl_hash")
            for row in found.itertuples(index=False):
                graph = graph_from_encoding(row.node_labels, row.edge_list, row.wl_hash)
                if triangle_count(graph) != int(level):
                    raise ValueError(f"TRI-{level} topology has wrong triangle count")
                rows.append(
                    {
                        "triangle_level": int(level),
                        "wl_hash": row.wl_hash,
                        "nodes": int(row.nodes),
                        "edges": int(row.edges),
                        "node_labels": row.node_labels,
                        "edge_list": row.edge_list,
                    }
                )
            if position % 25 == 0:
                print(
                    f"Recovered TRI-{level} target topologies from "
                    f"{position}/{len(requested)} source files",
                    flush=True,
                )
    result = pd.DataFrame(rows).drop_duplicates(["triangle_level", "wl_hash"])
    expected = set(zip(targets["triangle_level"].astype(int), targets["wl_hash"]))
    observed = set(zip(result["triangle_level"].astype(int), result["wl_hash"]))
    if expected != observed:
        missing = sorted(expected - observed)
        raise ValueError(f"Could not recover {len(missing)} target topologies: {missing[:3]}")
    return result.sort_values(["triangle_level", "wl_hash"]).reset_index(drop=True)


def merge_topology_cache(new: pd.DataFrame, path: Path) -> pd.DataFrame:
    existing = pd.read_csv(path, dtype={"wl_hash": str}) if path.is_file() else pd.DataFrame()
    combined = pd.concat([existing, new], ignore_index=True)
    combined = combined.drop_duplicates(["triangle_level", "wl_hash"], keep="last")
    combined = combined.sort_values(["triangle_level", "wl_hash"]).reset_index(drop=True)
    atomic_csv(combined, path, compression="gzip")
    return combined


def potential_direct_children(
    graph: nx.Graph, valid_hashes: set[str], labels: Iterable[str]
) -> list[tuple[str, str, nx.Graph]]:
    """Generate unique potential children that add exactly one triangle."""
    expected = triangle_count(graph) + 1
    children: dict[str, tuple[str, str, nx.Graph]] = {}
    new_node = max(graph.nodes, default=-1) + 1
    for left, right in graph.edges:
        for label in labels:
            child = graph.copy()
            child.add_node(new_node, label=str(label))
            child.add_edges_from([(left, new_node), (right, new_node)])
            child_hash = motif_hash(child)
            if child_hash in valid_hashes and triangle_count(child) == expected:
                children.setdefault(
                    child_hash,
                    ("add_vertex", f"{new_node}:{label}@{min(left,right)}-{max(left,right)}", child),
                )
    for left, right in combinations(graph.nodes, 2):
        if graph.has_edge(left, right):
            continue
        if len(set(graph.neighbors(left)).intersection(graph.neighbors(right))) != 1:
            continue
        child = graph.copy()
        child.add_edge(left, right)
        child_hash = motif_hash(child)
        if child_hash in valid_hashes and triangle_count(child) == expected:
            children.setdefault(
                child_hash,
                ("add_edge", f"{min(left,right)}-{max(left,right)}", child),
            )
    return list(children.values())


def trace_lineage(
    target: nx.Graph,
    target_level: int,
    vocabularies: dict[int, set[str]],
    labels: Iterable[str],
) -> tuple[dict[int, dict[str, nx.Graph]], pd.DataFrame]:
    target_hash = motif_hash(target)
    graphs: dict[int, dict[str, nx.Graph]] = {target_level: {target_hash: target}}
    relationships: dict[tuple[int, str, int, str], dict[str, object]] = {}

    for child_level in range(target_level, MIN_LEVEL, -1):
        parent_level = child_level - 1
        parents: dict[str, nx.Graph] = {}
        for child_hash, child in graphs.get(child_level, {}).items():
            for operation, removed, parent in potential_direct_parents(child):
                parent_hash = motif_hash(parent)
                if parent_hash not in vocabularies[parent_level]:
                    continue
                parents.setdefault(parent_hash, parent)
                key = (parent_level, parent_hash, child_level, child_hash)
                relationships.setdefault(
                    key,
                    {
                        "parent_level": parent_level,
                        "parent_hash": parent_hash,
                        "child_level": child_level,
                        "child_hash": child_hash,
                        "growth_operation": operation.replace("remove_", "add_"),
                        "changed_node_or_edge": removed,
                    },
                )
        graphs[parent_level] = parents
        if not parents:
            break

    for parent_level in range(target_level, MAX_LEVEL):
        child_level = parent_level + 1
        children: dict[str, nx.Graph] = {}
        for parent_hash, parent in graphs.get(parent_level, {}).items():
            for operation, changed, child in potential_direct_children(
                parent, vocabularies[child_level], labels
            ):
                child_hash = motif_hash(child)
                children.setdefault(child_hash, child)
                key = (parent_level, parent_hash, child_level, child_hash)
                relationships.setdefault(
                    key,
                    {
                        "parent_level": parent_level,
                        "parent_hash": parent_hash,
                        "child_level": child_level,
                        "child_hash": child_hash,
                        "growth_operation": operation,
                        "changed_node_or_edge": changed,
                    },
                )
        graphs[child_level] = children
        if not children:
            break

    edge_columns = [
        "parent_level",
        "parent_hash",
        "child_level",
        "child_hash",
        "growth_operation",
        "changed_node_or_edge",
    ]
    edges = pd.DataFrame(relationships.values(), columns=edge_columns)
    if len(edges):
        edges = edges.sort_values(edge_columns[:4], kind="mergesort").reset_index(drop=True)
    return graphs, edges


def cox_rows(
    connection: sqlite3.Connection, wanted: dict[int, set[str]]
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    selected_columns = ", ".join(f'"{column}"' for column in SQL_COLUMNS)
    for level, hashes in wanted.items():
        ordered = sorted(hashes)
        for start in range(0, len(ordered), 500):
            values = ordered[start : start + 500]
            placeholders = ",".join("?" for _ in values)
            parts.append(
                pd.read_sql_query(
                    f"SELECT {selected_columns} FROM cox WHERE triangle_level = ? "
                    f"AND wl_hash IN ({placeholders})",
                    connection,
                    params=[level, *values],
                )
            )
    result = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=SQL_COLUMNS)
    expected = {(level, value) for level, hashes in wanted.items() for value in hashes}
    observed = set(zip(result["triangle_level"].astype(int), result["wl_hash"]))
    if expected != observed:
        raise ValueError(f"Missing Cox rows for {len(expected - observed)} lineage motifs")
    return result


def lineage_nodes(
    graphs: dict[int, dict[str, nx.Graph]],
    cox: pd.DataFrame,
    target_level: int,
    target_hash: str,
    p_threshold: float,
) -> pd.DataFrame:
    topology_rows = []
    for level, level_graphs in graphs.items():
        for graph_hash, graph in level_graphs.items():
            labels, edges = graph_encoding(graph)
            topology_rows.append(
                {
                    "triangle_level": level,
                    "wl_hash": graph_hash,
                    "lineage_role": (
                        "target"
                        if level == target_level and graph_hash == target_hash
                        else "ancestor"
                        if level < target_level
                        else "descendant"
                    ),
                    "node_labels": labels,
                    "edge_list": edges,
                    "topology_nodes": graph.number_of_nodes(),
                    "topology_edges": graph.number_of_edges(),
                }
            )
    topology = pd.DataFrame(topology_rows)
    result = topology.merge(
        cox, on=["triangle_level", "wl_hash"], how="left", validate="one_to_one"
    )
    result["significant_raw_p"] = result["p_value_unadjusted"].lt(p_threshold)
    result["positive_median_threshold"] = result["median_patient_raw_count"].gt(0)
    result["selected_focal_rule"] = (
        result["significant_raw_p"] & result["positive_median_threshold"]
    )
    return result.sort_values(
        ["triangle_level", "p_value_unadjusted", "wl_hash"],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)


def plot_p_lineage(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    target_level: int,
    target_hash: str,
    output: Path,
    p_threshold: float,
) -> None:
    finite = nodes.loc[
        np.isfinite(nodes["p_value_unadjusted"]) & nodes["p_value_unadjusted"].gt(0)
    ].copy()
    finite["minus_log10_p"] = -np.log10(finite["p_value_unadjusted"])
    positions: dict[tuple[int, str], tuple[float, float]] = {}
    for level, subset in finite.groupby("triangle_level", sort=True):
        ordered = subset.sort_values(["p_value_unadjusted", "wl_hash"])
        offsets = np.linspace(-0.16, 0.16, len(ordered)) if len(ordered) > 1 else [0.0]
        for offset, row in zip(offsets, ordered.itertuples(index=False)):
            positions[(int(level), row.wl_hash)] = (
                int(level) + float(offset),
                float(row.minus_log10_p),
            )

    fig, ax = plt.subplots(figsize=(15, 8.5))
    for edge in edges.itertuples(index=False):
        parent_key = (int(edge.parent_level), edge.parent_hash)
        child_key = (int(edge.child_level), edge.child_hash)
        if parent_key in positions and child_key in positions:
            x1, y1 = positions[parent_key]
            x2, y2 = positions[child_key]
            ax.plot([x1, x2], [y1, y2], color="0.72", linewidth=1.0, zorder=1)

    role_colors = {"ancestor": "#377eb8", "target": "#d1495b", "descendant": "#e69f00"}
    for role, subset in finite.groupby("lineage_role"):
        xy = [positions[(int(row.triangle_level), row.wl_hash)] for row in subset.itertuples()]
        if not xy:
            continue
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
    label_rows = finite if len(finite) <= 50 else pd.concat(
        [finite.nsmallest(20, "p_value_unadjusted"), finite.loc[finite["lineage_role"].eq("target")]]
    ).drop_duplicates(["triangle_level", "wl_hash"])
    for row in label_rows.itertuples(index=False):
        x, y = positions[(int(row.triangle_level), row.wl_hash)]
        ax.annotate(row.wl_hash[:6], (x, y), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7)

    ax.axhline(-np.log10(p_threshold), color="black", linestyle="--", linewidth=1.4, label=f"Raw p = {p_threshold:g}")
    levels = sorted(nodes["triangle_level"].unique())
    ax.set_xticks(levels)
    ax.set_xlim(min(levels) - 0.45, max(levels) + 0.45)
    ax.set_xlabel("Triangle level")
    ax.set_ylabel("−log10(unadjusted Cox p value)")
    ax.set_title(
        f"Potential lineage of TRI-{target_level} motif {target_hash[:8]}: Raw Cox p\n"
        "Connected points are potential parent–child relationships"
    )
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=190)
    plt.close(fig)


def write_target(
    connection: sqlite3.Connection,
    output_root: Path,
    target_row: pd.Series,
    target_graph: nx.Graph,
    vocabularies: dict[int, set[str]],
    labels: list[str],
    p_threshold: float,
    rerun: bool,
) -> dict[str, object]:
    level = int(target_row["triangle_level"])
    target_hash = str(target_row["wl_hash"])
    destination = output_root / f"tri{level}" / target_hash
    metadata_path = destination / "trace_metadata.json"
    if metadata_path.is_file() and not rerun:
        saved = json.loads(metadata_path.read_text())
        if saved.get("status") == "complete":
            return {**saved, "run_action": "skipped_complete"}

    graphs, edges = trace_lineage(target_graph, level, vocabularies, labels)
    wanted = {key: set(value) for key, value in ((key, graphs[key].keys()) for key in graphs)}
    cox = cox_rows(connection, wanted)
    nodes = lineage_nodes(graphs, cox, level, target_hash, p_threshold)
    metadata = {
        "status": "complete",
        "motif_id": f"tri_{level}:{target_hash}",
        "target_level": level,
        "target_hash": target_hash,
        "target_p_value_unadjusted": float(target_row["p_value_unadjusted"]),
        "target_median_patient_raw_count": float(target_row["median_patient_raw_count"]),
        "selection_rule": f"p_value_unadjusted < {p_threshold:g} and median_patient_raw_count > 0",
        "lineage_filter": "all structural relatives in the saved level vocabulary",
        "minimum_level": int(nodes["triangle_level"].min()),
        "maximum_level": int(nodes["triangle_level"].max()),
        "motifs_by_level": {
            str(int(key)): int(value)
            for key, value in nodes.groupby("triangle_level").size().items()
        },
        "potential_parent_child_edges": int(len(edges)),
        "reproduction_command": (
            "python breast_cancer_survival/trace_significant_motifs.py "
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
        plot_p_lineage(
            nodes,
            edges,
            level,
            target_hash,
            temporary / "cox_p_value_by_level.png",
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
    """Inventory every completed target while retaining current-run failures/actions."""
    records: dict[str, dict[str, object]] = {}
    for level in range(MIN_LEVEL, MAX_LEVEL + 1):
        for path in sorted((output_root / f"tri{level}").glob("*/trace_metadata.json")):
            saved = json.loads(path.read_text())
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


def main() -> None:
    args = parse_args()
    levels = validate_levels(args.target_levels)
    if args.max_targets is not None and args.max_targets <= 0:
        raise ValueError("--max-targets must be positive")
    args.output_root.mkdir(parents=True, exist_ok=True)
    for level in range(MIN_LEVEL, MAX_LEVEL + 1):
        (args.output_root / f"tri{level}").mkdir(exist_ok=True)

    connection = build_cox_cache(
        args.output_root / "cache" / "cox_results.sqlite",
        args.results_root,
        args.rebuild_cache,
    )
    pool = significant_pool(connection, args.p_threshold)
    atomic_csv(pool, args.output_root / "potential_motif_pool.csv")
    pool_counts = {
        str(int(level)): int(count)
        for level, count in pool.groupby("triangle_level").size().items()
    }
    atomic_json(
        {
            "selection_rule": (
                f"p_value_unadjusted < {args.p_threshold:g} and "
                "median_patient_raw_count > 0"
            ),
            "total_targets": int(len(pool)),
            "targets_by_level": pool_counts,
            "lineage_filter": "all structural relatives regardless of p value or median",
        },
        args.output_root / "potential_motif_pool_summary.json",
    )

    targets = pool.loc[pool["triangle_level"].isin(levels)].copy()
    if args.target_hashes is not None:
        wanted_hashes = set(args.target_hashes)
        targets = targets.loc[targets["wl_hash"].isin(wanted_hashes)]
        missing = wanted_hashes - set(targets["wl_hash"])
        if missing:
            raise ValueError(f"Requested hashes are not selected focal motifs: {sorted(missing)}")
    targets = targets.sort_values(["p_value_unadjusted", "triangle_level", "wl_hash"])
    if args.max_targets is not None:
        targets = targets.head(args.max_targets)
    if targets.empty:
        raise ValueError("No focal motifs matched the requested target selection")

    new_topologies = recover_target_topologies(targets, args.artifacts_root, args.motif_root)
    topology_cache = merge_topology_cache(
        new_topologies, args.output_root / "cache" / "target_topologies.csv.gz"
    ).set_index(["triangle_level", "wl_hash"])
    vocabularies = vocabulary(connection, range(MIN_LEVEL, MAX_LEVEL + 1))
    labels = [str(value) for value in range(8)]
    manifest_rows: list[dict[str, object]] = []
    failures: list[str] = []
    for position, target in enumerate(targets.itertuples(index=False), start=1):
        target_row = pd.Series(target._asdict())
        key = (int(target.triangle_level), str(target.wl_hash))
        topology = topology_cache.loc[key]
        graph = graph_from_encoding(
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
                args.p_threshold,
                args.rerun,
            )
        except Exception as error:
            failures.append(str(target.wl_hash))
            record = {
                "status": "failed",
                "motif_id": f"tri_{int(target.triangle_level)}:{target.wl_hash}",
                "target_level": int(target.triangle_level),
                "target_hash": str(target.wl_hash),
                "error": f"{type(error).__name__}: {error}",
                "run_action": "failed",
            }
        manifest_rows.append(record)
        if position % 25 == 0 or position == len(targets):
            print(f"Processed {position}/{len(targets)} focal motifs", flush=True)

    manifest = cumulative_manifest(args.output_root, manifest_rows)
    atomic_csv(manifest, args.output_root / "trace_run_manifest.csv")
    connection.close()
    if failures:
        raise RuntimeError(f"Tracing failed for {len(failures)} motifs: {failures[:3]}")
    print(
        f"Completed {len(targets)} targets under {args.output_root}; "
        f"pool contains {len(pool)} motifs",
        flush=True,
    )


if __name__ == "__main__":
    main()
