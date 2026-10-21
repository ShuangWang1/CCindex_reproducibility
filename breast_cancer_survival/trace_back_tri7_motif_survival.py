#!/usr/bin/env python3
"""Trace a TRI-7 motif to potential lower-level parents and plot survival.

The forward tree search deduplicates motifs by labeled Weisfeiler-Lehman hash
and does not persist parent identifiers.  Consequently, this script reports
*potential structural parents*: lower-level motifs that (1) can be obtained by
reversing one valid triangle-growth operation and (2) occur in the saved
feature index for the corresponding level.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/breast-cancer-trace-back-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse

from analyze_ki67_survival import COLORS, step_function
from analyze_motif_survival_unadjusted import (
    load_inventory,
    prepare_patient_aggregation,
    vectorized_binary_cox,
)
from plot_top30_motif_kaplan_meier import motif_name, wrap_motif_name
from visualize_top_motif_survival import (
    CELL_TYPE_COLORS,
    CELL_TYPE_ORDER,
    CELL_TYPE_SHORT,
    decoded_labels,
)


DEFAULT_TARGET_HASH = "1d7e000e1fd60951fbdd43c9c1680a3d"


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    survival_root = Path(__file__).resolve().parent
    result_root = survival_root / "motif_results_tri4_tri7_unadjusted"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-hash", default=DEFAULT_TARGET_HASH)
    parser.add_argument(
        "--topologies",
        type=Path,
        default=result_root
        / "non-zero_top30_visualizations"
        / "top30_motif_topologies.csv",
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
        "--metadata",
        type=Path,
        default=repo / "breast_cancer" / "Basel_PatientMetadata.csv",
    )
    parser.add_argument(
        "--label-map",
        type=Path,
        default=repo
        / "breast_cancer"
        / "BZ_samples_intra"
        / "node_label_cell_type_mapping.csv",
    )
    parser.add_argument(
        "--outdir", type=Path, default=result_root / "tri-7_trace_back"
    )
    parser.add_argument("--results-root", type=Path, default=result_root)
    parser.add_argument("--minimum-level", type=int, default=2)
    return parser.parse_args()


def triangle_count(graph: nx.Graph) -> int:
    return sum(nx.triangles(graph).values()) // 3


def motif_hash(graph: nx.Graph) -> str:
    return nx.weisfeiler_lehman_graph_hash(graph, node_attr="label")


def graph_from_row(row: pd.Series) -> nx.Graph:
    graph = nx.Graph()
    for index, label in enumerate(str(row["node_labels"]).split("&")):
        graph.add_node(index, label=label)
    for encoded_edge in str(row["edge_list"]).split(";"):
        left, right = (int(value) for value in encoded_edge.split("-"))
        graph.add_edge(left, right)
    expected_hash = str(row["wl_hash"])
    if motif_hash(graph) != expected_hash:
        raise ValueError("Target topology does not reproduce its saved WL hash")
    return graph


def potential_direct_parents(
    graph: nx.Graph,
) -> list[tuple[str, str, nx.Graph]]:
    """Reverse either supported one-triangle extension operation."""
    expected_triangles = triangle_count(graph) - 1
    candidates: list[tuple[str, str, nx.Graph]] = []

    # Reverse adding one new vertex on an existing edge.
    for vertex in list(graph.nodes):
        neighbors = list(graph.neighbors(vertex))
        if graph.degree(vertex) != 2 or not graph.has_edge(*neighbors):
            continue
        parent = graph.copy()
        parent.remove_node(vertex)
        if nx.is_connected(parent) and triangle_count(parent) == expected_triangles:
            candidates.append(("remove_vertex", str(vertex), parent))

    # Reverse adding an edge between two existing boundary vertices.
    for left, right in list(graph.edges):
        parent = graph.copy()
        parent.remove_edge(left, right)
        if nx.is_connected(parent) and triangle_count(parent) == expected_triangles:
            candidates.append(("remove_edge", f"{left}-{right}", parent))
    return candidates


def load_feature_indices(
    artifacts_root: Path, levels: range
) -> dict[int, pd.DataFrame]:
    indices: dict[int, pd.DataFrame] = {}
    for level in levels:
        path = artifacts_root / f"tri_{level}" / "feature_index.csv.gz"
        frame = pd.read_csv(
            path,
            usecols=["feature_index", "wl_hash", "representative_node_labels"],
            dtype={"wl_hash": str, "representative_node_labels": str},
        )
        if frame["wl_hash"].duplicated().any():
            raise ValueError(f"TRI-{level} feature index has duplicate WL hashes")
        indices[level] = frame.set_index("wl_hash", drop=False)
    return indices


def trace_potential_ancestry(
    target: nx.Graph,
    feature_indices: dict[int, pd.DataFrame],
    target_level: int,
    minimum_level: int,
) -> tuple[dict[int, dict[str, nx.Graph]], pd.DataFrame]:
    graphs: dict[int, dict[str, nx.Graph]] = {
        target_level: {motif_hash(target): target}
    }
    relationships: list[dict[str, object]] = []
    for child_level in range(target_level, minimum_level, -1):
        parent_level = child_level - 1
        parents: dict[str, nx.Graph] = {}
        valid_hashes = feature_indices[parent_level].index
        seen_relationships: set[tuple[str, str]] = set()
        for child_hash, child_graph in graphs[child_level].items():
            for operation, removed, parent_graph in potential_direct_parents(child_graph):
                parent_hash = motif_hash(parent_graph)
                relationship = (parent_hash, child_hash)
                if parent_hash not in valid_hashes or relationship in seen_relationships:
                    continue
                seen_relationships.add(relationship)
                parents.setdefault(parent_hash, parent_graph)
                relationships.append(
                    {
                        "parent_level": parent_level,
                        "parent_hash": parent_hash,
                        "child_level": child_level,
                        "child_hash": child_hash,
                        "reverse_operation": operation,
                        "removed_node_or_edge": removed,
                    }
                )
        if not parents:
            raise ValueError(f"No potential TRI-{parent_level} parents were found")
        graphs[parent_level] = parents
    return graphs, pd.DataFrame(relationships)


def graph_encoding(graph: nx.Graph) -> tuple[str, str]:
    nodes = sorted(graph.nodes)
    remap = {node: index for index, node in enumerate(nodes)}
    labels = "&".join(str(graph.nodes[node]["label"]) for node in nodes)
    edges = ";".join(
        f"{min(remap[left], remap[right])}-{max(remap[left], remap[right])}"
        for left, right in sorted(graph.edges)
    )
    return labels, edges


def calculate_level_results(
    level: int,
    graphs: dict[str, nx.Graph],
    features: pd.DataFrame,
    artifacts_root: Path,
    metadata: pd.DataFrame,
    label_map: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    hashes = sorted(graphs)
    selected = features.loc[hashes]
    samples = load_inventory(artifacts_root, level)
    patients, aggregation, _ = prepare_patient_aggregation(samples, metadata)
    matrix = sparse.load_npz(
        artifacts_root / f"tri_{level}" / "motif_counts.npz"
    ).tocsr()
    feature_columns = selected["feature_index"].to_numpy(dtype=int)
    patient_counts = (aggregation @ matrix[:, feature_columns]).toarray()
    medians = np.median(patient_counts, axis=0)
    enriched = patient_counts > medians
    event = patients["event"].to_numpy(dtype=int)
    fitted = vectorized_binary_cox(
        enriched, patients["time_months"].to_numpy(dtype=float), event
    )

    rows: list[dict[str, object]] = []
    status_names = np.asarray(
        ["not_estimable_no_group_variation", "not_estimable_event_separation", "estimated", "fit_failed"]
    )
    for position, motif_id in enumerate(hashes):
        node_labels, edge_list = graph_encoding(graphs[motif_id])
        rows.append(
            {
                "triangle_level": level,
                "wl_hash": motif_id,
                "nodes": graphs[motif_id].number_of_nodes(),
                "edges": graphs[motif_id].number_of_edges(),
                "node_labels": node_labels,
                "edge_list": edge_list,
                "motif_cell_types": motif_name(node_labels, label_map),
                "median_patient_raw_count": medians[position],
                "hazard_ratio_unadjusted": fitted["hazard_ratio"][position],
                "ci95_low_unadjusted": fitted["ci95_low"][position],
                "ci95_high_unadjusted": fitted["ci95_high"][position],
                "p_value_unadjusted": fitted["p_value"][position],
                "n_enriched_unadjusted": fitted["n_enriched"][position],
                "events_enriched_unadjusted": fitted["events_enriched"][position],
                "n_non_enriched_unadjusted": fitted["n_non_enriched"][position],
                "events_non_enriched_unadjusted": fitted["events_non_enriched"][position],
                "fit_status_unadjusted": status_names[fitted["status_code"][position]],
                "significant_raw_p_lt_0_05": bool(fitted["p_value"][position] < 0.05),
            }
        )
    return pd.DataFrame(rows), patients, enriched


def attach_saved_multiple_testing(
    summary: pd.DataFrame, results_root: Path
) -> pd.DataFrame:
    """Attach the screen-wide saved BH q values and validate fitted p values."""
    saved_parts: list[pd.DataFrame] = []
    columns = [
        "wl_hash",
        "p_value_unadjusted",
        "q_value_bh_global_unadjusted",
        "q_value_bh_within_level_unadjusted",
        "rank_global_unadjusted",
        "rank_within_level_unadjusted",
    ]
    for level, subset in summary.groupby("triangle_level", sort=True):
        wanted = set(subset["wl_hash"])
        matches: list[pd.DataFrame] = []
        path = results_root / f"tri_{int(level)}_cox_results_sorted.csv"
        for chunk in pd.read_csv(path, usecols=columns, dtype={"wl_hash": str}, chunksize=200_000):
            found = chunk.loc[chunk["wl_hash"].isin(wanted)]
            if len(found):
                matches.append(found)
        saved = pd.concat(matches, ignore_index=True) if matches else pd.DataFrame(columns=columns)
        if set(saved["wl_hash"]) != wanted or saved["wl_hash"].duplicated().any():
            raise ValueError(f"Could not uniquely recover all saved TRI-{level} results")
        saved["triangle_level"] = int(level)
        saved_parts.append(saved)
    saved = pd.concat(saved_parts, ignore_index=True)
    merged = summary.merge(
        saved,
        on=["triangle_level", "wl_hash"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_saved"),
    )
    if not np.allclose(
        merged["p_value_unadjusted"], merged["p_value_unadjusted_saved"],
        rtol=1e-10, atol=1e-12,
    ):
        raise ValueError("Recalculated Cox p values differ from the saved screen")
    merged = merged.drop(columns="p_value_unadjusted_saved")
    merged["significant_global_bh_q_lt_0_05"] = (
        merged["q_value_bh_global_unadjusted"] < 0.05
    )
    return merged


def plot_survival(
    patients: pd.DataFrame,
    enriched: np.ndarray,
    row: pd.Series,
    output: Path,
) -> None:
    plot_data = patients[["time_months", "event"]].copy()
    plot_data["motif_group"] = np.where(enriched, "Enriched", "Non-enriched")
    fig, ax = plt.subplots(figsize=(8.2, 6.3))
    for group in ["Non-enriched", "Enriched"]:
        data = plot_data.loc[plot_data["motif_group"].eq(group)]
        km = step_function(
            data["time_months"].to_numpy(dtype=float),
            data["event"].to_numpy(dtype=int),
        )
        ax.step(
            km["time"], km["survival"], where="post", color=COLORS[group],
            linewidth=2.2,
            label=f"{group} (n={len(data)}, events={int(data.event.sum())})",
        )
        ax.fill_between(
            km["time"], km["lower"], km["upper"], step="post",
            color=COLORS[group], alpha=0.15,
        )
        censored = data.loc[data["event"].eq(0), "time_months"].to_numpy(dtype=float)
        if len(censored):
            positions = np.searchsorted(km["time"].to_numpy(), censored, side="right") - 1
            ax.scatter(
                censored, km["survival"].to_numpy()[positions], marker="|", s=55,
                color=COLORS[group],
            )

    if np.isfinite(row["p_value_unadjusted"]):
        annotation = (
            f"Unadjusted Cox HR = {row['hazard_ratio_unadjusted']:.3g}\n"
            f"95% CI {row['ci95_low_unadjusted']:.3g}–{row['ci95_high_unadjusted']:.3g}\n"
            f"p = {row['p_value_unadjusted']:.3g}"
        )
    else:
        annotation = f"Unadjusted Cox model\n{row['fit_status_unadjusted']}"
    ax.text(
        0.98, 0.98, annotation, transform=ax.transAxes, ha="right", va="top",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9,
              "edgecolor": "0.75"},
    )
    ax.set_title(
        f"Potential TRI-{int(row['triangle_level'])} ancestor: overall survival\n"
        f"{wrap_motif_name(row['motif_cell_types'])}\n"
        f"WL hash {row['wl_hash'][:8]}",
        fontsize=11, pad=12,
    )
    ax.set_xlabel("Overall survival (months)")
    ax.set_ylabel("Survival probability")
    ax.set_ylim(0, 1.03)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def fixed_target_layout(target: nx.Graph, target_hash: str) -> dict[int, np.ndarray]:
    """Return one deterministic coordinate system shared by every trace plot."""
    return nx.spring_layout(
        target,
        seed=int(target_hash[:8], 16),
        k=1.0 / np.sqrt(len(target)),
    )


def transition_graphs(
    child: nx.Graph,
    relationship: pd.Series | object,
) -> tuple[nx.Graph, set[int], set[tuple[int, int]]]:
    """Reconstruct an edge-specific parent embedding from its child graph."""
    operation = str(getattr(relationship, "reverse_operation"))
    removed = str(getattr(relationship, "removed_node_or_edge"))
    parent_hash = str(getattr(relationship, "parent_hash"))
    parent = child.copy()
    added_nodes: set[int] = set()
    added_edges: set[tuple[int, int]] = set()

    if operation == "remove_vertex":
        node = int(removed)
        if node not in child:
            raise ValueError(f"Transition node {node} is not present in its child")
        added_nodes.add(node)
        added_edges = {
            tuple(sorted((node, neighbor))) for neighbor in child.neighbors(node)
        }
        parent.remove_node(node)
    elif operation == "remove_edge":
        left, right = (int(value) for value in removed.split("-"))
        if not child.has_edge(left, right):
            raise ValueError(
                f"Transition edge {left}-{right} is not present in its child"
            )
        added_edges.add(tuple(sorted((left, right))))
        parent.remove_edge(left, right)
    else:
        raise ValueError(f"Unsupported reverse operation: {operation}")

    if motif_hash(parent) != parent_hash:
        raise ValueError("Reconstructed transition parent does not match its saved hash")
    return parent, added_nodes, added_edges


def draw_aligned_graph(
    ax: plt.Axes,
    target: nx.Graph,
    active: nx.Graph,
    coordinates: dict[int, np.ndarray],
    label_map: dict[str, str],
    added_nodes: set[int] | None = None,
    added_edges: set[tuple[int, int]] | None = None,
) -> None:
    """Draw a motif over a faint, fixed-position TRI-7 scaffold."""
    added_nodes = added_nodes or set()
    added_edges = added_edges or set()
    normalized_added_edges = {tuple(sorted(edge)) for edge in added_edges}
    active_edges = [
        edge
        for edge in active.edges
        if tuple(sorted(edge)) not in normalized_added_edges
    ]

    nx.draw_networkx_edges(
        target,
        coordinates,
        ax=ax,
        width=1.15,
        edge_color="#D9D9D9",
        alpha=0.75,
    )
    nx.draw_networkx_nodes(
        target,
        coordinates,
        ax=ax,
        node_color="#F3F3F3",
        node_size=620,
        edgecolors="#D9D9D9",
        linewidths=1.2,
    )
    if active_edges:
        nx.draw_networkx_edges(
            active,
            coordinates,
            ax=ax,
            edgelist=active_edges,
            width=1.7,
            edge_color="#5F5F5F",
        )
    if normalized_added_edges:
        nx.draw_networkx_edges(
            active,
            coordinates,
            ax=ax,
            edgelist=sorted(normalized_added_edges),
            width=3.2,
            edge_color="#E69F00",
        )

    nodes = list(active.nodes)
    cell_types = [
        label_map.get(str(active.nodes[node]["label"]), f"label {active.nodes[node]['label']}")
        for node in nodes
    ]
    nx.draw_networkx_nodes(
        active,
        coordinates,
        ax=ax,
        nodelist=nodes,
        node_color=[CELL_TYPE_COLORS.get(value, "#BAB0AC") for value in cell_types],
        node_size=620,
        edgecolors=["#E69F00" if node in added_nodes else "white" for node in nodes],
        linewidths=[3.5 if node in added_nodes else 1.5 for node in nodes],
    )
    nx.draw_networkx_labels(
        active,
        coordinates,
        labels={
            node: CELL_TYPE_SHORT.get(cell_type, cell_type[:4])
            for node, cell_type in zip(nodes, cell_types)
        },
        ax=ax,
        font_size=8,
        font_weight="bold",
    )
    ax.set_axis_off()


def plot_motif_gallery(
    topologies: pd.DataFrame,
    graphs: dict[int, dict[str, nx.Graph]],
    target: nx.Graph,
    coordinates: dict[int, np.ndarray],
    label_map: dict[str, str],
    title: str,
    output: Path,
    show_levels: bool,
) -> None:
    """Plot traced motif structures in the style of the top-30 galleries."""
    columns = min(5, len(topologies))
    rows = math.ceil(len(topologies) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4.0 * columns, 4.0 * rows))
    axes = np.asarray(axes, dtype=object).reshape(-1)
    for position, (ax, row) in enumerate(
        zip(axes, topologies.itertuples(index=False)), start=1
    ):
        graph = graphs[int(row.triangle_level)][row.wl_hash]
        draw_aligned_graph(
            ax,
            target,
            graph,
            coordinates,
            label_map,
        )
        identity = (
            f"TRI-{int(row.triangle_level)} #{position} {row.wl_hash[:8]}"
            if show_levels
            else f"#{position} {row.wl_hash[:8]}"
        )
        ax.set_title(
            f"{identity}\n"
            f"p={row.p_value_unadjusted:.2g}; q={row.q_value_bh_global_unadjusted:.2g}; "
            f"HR={row.hazard_ratio_unadjusted:.2g}\n"
            f"E={int(row.n_enriched_unadjusted)}({int(row.events_enriched_unadjusted)})",
            fontsize=9,
        )
    for ax in axes[len(topologies) :]:
        ax.set_axis_off()

    present = {
        cell_type
        for labels in decoded_labels(topologies["node_labels"], label_map)
        for cell_type in labels
    }
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=CELL_TYPE_COLORS[value],
            markeredgecolor="white",
            markersize=10,
            label=f"{CELL_TYPE_SHORT[value]} = {value}",
        )
        for value in CELL_TYPE_ORDER
        if value in present
    ]
    handles.append(
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="#D9D9D9",
            markerfacecolor="#F3F3F3",
            markeredgecolor="#D9D9D9",
            markersize=10,
            label="not present at this level",
        )
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(7, len(handles)),
        frameon=False,
    )
    fig.suptitle(title, fontsize=17, y=0.995)
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_growth_transition_gallery(
    relationships: pd.DataFrame,
    graphs: dict[int, dict[str, nx.Graph]],
    target: nx.Graph,
    coordinates: dict[int, np.ndarray],
    label_map: dict[str, str],
    title: str,
    output: Path,
) -> None:
    """Plot parent-to-child growth steps with the exact addition highlighted."""
    columns = min(5, len(relationships))
    rows = math.ceil(len(relationships) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4.0 * columns, 4.0 * rows))
    axes = np.asarray(axes, dtype=object).reshape(-1)

    for position, (ax, relationship) in enumerate(
        zip(axes, relationships.itertuples(index=False)), start=1
    ):
        child_level = int(relationship.child_level)
        parent_level = int(relationship.parent_level)
        child = graphs[child_level][relationship.child_hash]
        _, added_nodes, added_edges = transition_graphs(child, relationship)
        draw_aligned_graph(
            ax,
            target,
            child,
            coordinates,
            label_map,
            added_nodes=added_nodes,
            added_edges=added_edges,
        )
        if added_nodes:
            node = next(iter(added_nodes))
            cell_type = label_map.get(
                str(child.nodes[node]["label"]), f"label {child.nodes[node]['label']}"
            )
            addition = f"added {CELL_TYPE_SHORT.get(cell_type, cell_type)} node"
        else:
            addition = "added edge"
        ax.set_title(
            f"#{position} TRI-{parent_level} → TRI-{child_level}\n"
            f"{relationship.parent_hash[:8]} → {relationship.child_hash[:8]}\n"
            f"{addition}",
            fontsize=9,
        )

    for ax in axes[len(relationships) :]:
        ax.set_axis_off()

    present = {
        label_map.get(str(data["label"]), f"label {data['label']}")
        for _, data in target.nodes(data=True)
    }
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=CELL_TYPE_COLORS[value],
            markeredgecolor="white",
            markersize=10,
            label=f"{CELL_TYPE_SHORT[value]} = {value}",
        )
        for value in CELL_TYPE_ORDER
        if value in present
    ]
    handles.extend(
        [
            plt.Line2D(
                [0], [0], marker="o", color="#E69F00", markerfacecolor="white",
                markeredgewidth=2.5, markersize=10, label="added node",
            ),
            plt.Line2D([0], [0], color="#E69F00", linewidth=3, label="new edge"),
            plt.Line2D(
                [0], [0], marker="o", color="#D9D9D9",
                markerfacecolor="#F3F3F3", markersize=10,
                label="not present in child",
            ),
        ]
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(7, len(handles)),
        frameon=False,
    )
    fig.suptitle(title, fontsize=17, y=0.995)
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_significance_lineage(
    summary: pd.DataFrame,
    relationships: pd.DataFrame,
    output: Path,
    value_column: str,
    y_label: str,
    title_metric: str,
    target_hash: str,
) -> None:
    finite = summary.loc[np.isfinite(summary[value_column])].copy()
    finite["minus_log10_value"] = -np.log10(finite[value_column])
    positions: dict[str, tuple[float, float]] = {}
    for level, subset in finite.groupby("triangle_level", sort=True):
        ordered = subset.sort_values(["p_value_unadjusted", "wl_hash"])
        offsets = np.linspace(-0.23, 0.23, len(ordered)) if len(ordered) > 1 else [0.0]
        for offset, row in zip(offsets, ordered.itertuples(index=False)):
            positions[row.wl_hash] = (level + float(offset), row.minus_log10_value)

    fig, ax = plt.subplots(figsize=(10, 6.5))
    for edge in relationships.itertuples(index=False):
        if edge.parent_hash in positions and edge.child_hash in positions:
            x1, y1 = positions[edge.parent_hash]
            x2, y2 = positions[edge.child_hash]
            ax.plot([x1, x2], [y1, y2], color="0.72", linewidth=1, zorder=1)
    for row in finite.itertuples(index=False):
        x, y = positions[row.wl_hash]
        color = "#d1495b" if getattr(row, value_column) < 0.05 else "#3973ac"
        ax.scatter(x, y, s=62, color=color, edgecolor="white", linewidth=0.7, zorder=3)
        ax.annotate(
            row.wl_hash[:6], (x, y), xytext=(0, 6), textcoords="offset points",
            ha="center", fontsize=7,
        )
    threshold = -np.log10(0.05)
    ax.axhline(threshold, color="black", linestyle="--", linewidth=1.2,
               label=f"{title_metric} = 0.05")
    ax.set_xticks(sorted(summary["triangle_level"].unique()))
    ax.set_xlabel("Triangle level")
    ax.set_ylabel(y_label)
    ax.set_title(
        f"Potential ancestry of TRI-7 motif {target_hash[:8]}: {title_metric}\n"
        "Connected points are potential parent–child relationships"
    )
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def write_report(
    summary: pd.DataFrame,
    relationships: pd.DataFrame,
    output: Path,
    target_hash: str,
) -> None:
    lines = [
        "# TRI-7 motif trace-back",
        "",
        f"Target WL hash: `{target_hash}`",
        "",
        "The original forward search did not save parent IDs. These are potential "
        "structural parents that reverse one triangle-growth step and are present in "
        "the saved feature index. The trace is therefore a directed acyclic graph, "
        "not a uniquely identified historical search path.",
        "",
        "All motif galleries use the final TRI-7 graph as a fixed, faint scaffold. "
        "The growth-transition galleries highlight each added node and its incident "
        "edges in gold.",
        "",
        "| TRI level | Potential motifs | Raw p < 0.05 | Global BH q < 0.05 | Minimum raw Cox p | Minimum global BH q |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for level, subset in summary.groupby("triangle_level", sort=True):
        finite = subset["p_value_unadjusted"].dropna()
        minimum = f"{finite.min():.4g}" if len(finite) else "not estimable"
        finite_q = subset["q_value_bh_global_unadjusted"].dropna()
        minimum_q = f"{finite_q.min():.4g}" if len(finite_q) else "not estimable"
        lines.append(
            f"| {level} | {len(subset)} | "
            f"{int(subset['significant_raw_p_lt_0_05'].sum())} | "
            f"{int(subset['significant_global_bh_q_lt_0_05'].sum())} | "
            f"{minimum} | {minimum_q} |"
        )
    first_levels = summary.loc[
        summary["significant_raw_p_lt_0_05"], "triangle_level"
    ]
    first_q_levels = summary.loc[
        summary["significant_global_bh_q_lt_0_05"], "triangle_level"
    ]
    lines.extend(
        [
            "",
            (
                f"At least one potential ancestor first has raw p < 0.05 at TRI-{int(first_levels.min())}."
                if len(first_levels)
                else "No traced motif has raw p < 0.05."
            ),
            (
                f"After the saved global Benjamini–Hochberg correction, significance first appears at TRI-{int(first_q_levels.min())}."
                if len(first_q_levels)
                else "No traced motif has global BH q < 0.05."
            ),
            "",
            "This threshold is nominal and does not correct for selecting the TRI-7 "
            "motif or examining multiple potential ancestors. It should be treated as "
            "an exploratory onset of association, not a confirmatory change point.",
            "",
            f"The lineage contains {len(relationships)} potential parent–child edges.",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.minimum_level < 2 or args.minimum_level >= 7:
        raise ValueError("--minimum-level must be between 2 and 6")
    topologies = pd.read_csv(
        args.topologies,
        dtype={"wl_hash": str, "node_labels": str, "edge_list": str},
    )
    selected = topologies.loc[topologies["wl_hash"].eq(args.target_hash)]
    if len(selected) != 1:
        raise ValueError(
            f"Expected exactly one target topology for {args.target_hash}, found {len(selected)}"
        )
    target_row = selected.iloc[0]
    target_level = int(target_row["triangle_level"])
    if target_level != 7:
        raise ValueError(f"Target must be TRI-7, found TRI-{target_level}")
    target = graph_from_row(target_row)
    if triangle_count(target) != target_level:
        raise ValueError("Target topology does not contain exactly seven triangles")

    levels = range(args.minimum_level, target_level + 1)
    features = load_feature_indices(args.artifacts_root, levels)
    graphs, relationships = trace_potential_ancestry(
        target, features, target_level, args.minimum_level
    )
    metadata = pd.read_csv(args.metadata)
    label_frame = pd.read_csv(args.label_map, dtype={"node_label": str})
    label_map = dict(
        label_frame[["node_label", "cell_type"]].itertuples(index=False, name=None)
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    summaries: list[pd.DataFrame] = []
    reference_patients: pd.DataFrame | None = None
    for level in levels:
        level_summary, patients, enriched = calculate_level_results(
            level, graphs[level], features[level], args.artifacts_root, metadata, label_map
        )
        if reference_patients is not None and not patients.equals(reference_patients):
            raise ValueError(f"TRI-{level} patient cohort differs from earlier levels")
        reference_patients = patients
        summaries.append(level_summary)
        level_dir = args.outdir / f"tri{level}"
        for position, row in level_summary.iterrows():
            plot_survival(
                patients, enriched[:, position], row,
                level_dir / f"{row['wl_hash'][:8]}.png",
            )
    summary = pd.concat(summaries, ignore_index=True).sort_values(
        ["triangle_level", "p_value_unadjusted", "wl_hash"],
        kind="mergesort", na_position="last",
    )
    summary = attach_saved_multiple_testing(summary, args.results_root)
    summary.to_csv(args.outdir / "trace_back_summary.csv", index=False)
    relationships.sort_values(
        ["parent_level", "parent_hash", "child_hash"], kind="mergesort"
    ).to_csv(args.outdir / "trace_back_edges.csv", index=False)
    plot_significance_lineage(
        summary, relationships, args.outdir / "cox_p_value_by_level.png",
        "p_value_unadjusted", "−log10(unadjusted Cox p value)", "Raw Cox p", args.target_hash,
    )
    plot_significance_lineage(
        summary, relationships, args.outdir / "cox_global_bh_q_by_level.png",
        "q_value_bh_global_unadjusted", "−log10(global BH q value)",
        "Global BH q", args.target_hash,
    )
    gallery_order = summary.sort_values(
        ["triangle_level", "p_value_unadjusted", "wl_hash"], kind="mergesort"
    )
    target_coordinates = fixed_target_layout(target, args.target_hash)
    plot_motif_gallery(
        gallery_order,
        graphs,
        target,
        target_coordinates,
        label_map,
        f"TRI-7 motif {args.target_hash[:8]}: all potential trace-back structures",
        args.outdir / "all_trace_back_motif_graphs.png",
        show_levels=True,
    )
    for level, level_motifs in gallery_order.groupby("triangle_level", sort=True):
        plot_motif_gallery(
            level_motifs,
            graphs,
            target,
            target_coordinates,
            label_map,
            f"TRI-{int(level)}: potential ancestors of TRI-7 motif {args.target_hash[:8]}",
            args.outdir / f"tri_{int(level)}_trace_back_motif_graphs.png",
            show_levels=False,
        )
    transition_order = relationships.sort_values(
        ["child_level", "parent_hash", "child_hash", "reverse_operation", "removed_node_or_edge"],
        kind="mergesort",
    )
    plot_growth_transition_gallery(
        transition_order,
        graphs,
        target,
        target_coordinates,
        label_map,
        f"TRI-7 motif {args.target_hash[:8]}: all potential growth transitions",
        args.outdir / "all_trace_back_growth_transitions.png",
    )
    for child_level, transitions in transition_order.groupby("child_level", sort=True):
        parent_level = int(child_level) - 1
        plot_growth_transition_gallery(
            transitions,
            graphs,
            target,
            target_coordinates,
            label_map,
            f"TRI-{parent_level} to TRI-{int(child_level)} growth transitions "
            f"toward TRI-7 motif {args.target_hash[:8]}",
            args.outdir
            / f"tri_{parent_level}_to_tri_{int(child_level)}_growth_transitions.png",
        )
    write_report(summary, relationships, args.outdir / "REPORT.md", args.target_hash)
    with (args.outdir / "trace_back_metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "target_hash": args.target_hash,
                "target_level": target_level,
                "minimum_level": args.minimum_level,
                "potential_motifs_by_level": {
                    str(level): len(graphs[level]) for level in levels
                },
                "potential_parent_child_edges": len(relationships),
                "visualization_layout": "fixed final TRI-7 scaffold",
                "growth_operations": {
                    str(operation): int(count)
                    for operation, count in relationships["reverse_operation"]
                    .value_counts()
                    .sort_index()
                    .items()
                },
                "parent_definition": (
                    "reverse one vertex-on-edge or one edge-closing-triangle growth; "
                    "require parent WL hash in saved level feature index"
                ),
            },
            stream,
            indent=2,
        )
        stream.write("\n")
    print(f"Wrote {len(summary)} motif results under {args.outdir}")


if __name__ == "__main__":
    main()
