#!/usr/bin/env python3
"""Compare significant and nonsignificant TRI-5 motifs sharing a TRI-4 parent."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/breast-cancer-same-parent-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import Patch
import networkx as nx
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.stats import levene, mannwhitneyu


PROJECT_DIR = Path(__file__).resolve().parent
TRACE_DIR = (
    PROJECT_DIR.parent
    / "significant_mtf_trace"
    / "tri1"
    / "1fded5000748985170d30bdca433a111"
)
DEFAULT_FEATURES = PROJECT_DIR / "results" / "tables" / "tri5_motif_criteria.csv.gz"
DEFAULT_OUTDIR = PROJECT_DIR / "results" / "same_parent_comparsion"

SPEC = importlib.util.spec_from_file_location("sibling_topology_new", PROJECT_DIR / "run_analysis.py")
if SPEC is None or SPEC.loader is None:
    raise ImportError("Cannot load sibling_topology_analysis_new/run_analysis.py")
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, default=TRACE_DIR)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--min-per-class", type=int, default=3)
    parser.add_argument("--top-criteria", type=int, default=24)
    parser.add_argument("--panels-per-page", type=int, default=12)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--motif-galleries-only",
        action="store_true",
        help="Generate child-motif galleries from existing parent folders without recomputing tests.",
    )
    mode.add_argument(
        "--clustered-heatmap-only",
        action="store_true",
        help="Regenerate the clustered location-BH-q heatmap from existing combined results.",
    )
    return parser.parse_args()


def intrinsic_metrics(features: pd.DataFrame) -> list[str]:
    """Return numeric intrinsic criteria, excluding metadata and label-derived leakage."""
    excluded = set(analysis.METADATA)
    return [
        column
        for column in features.columns
        if column not in excluded
        and not column.startswith("dispersion_")
        and pd.api.types.is_numeric_dtype(features[column])
    ]


def prepare_relationships(edges: pd.DataFrame) -> pd.DataFrame:
    """Return one row per immediate TRI-4 parent/TRI-5 child relationship."""
    selected = edges.loc[
        edges["parent_level"].eq(4) & edges["child_level"].eq(5),
        ["parent_hash", "child_hash", "growth_operation", "changed_node_or_edge"],
    ].copy()
    selected[["parent_hash", "child_hash"]] = selected[["parent_hash", "child_hash"]].astype(str)
    if selected.duplicated(["parent_hash", "child_hash"]).any():
        selected = (
            selected.groupby(["parent_hash", "child_hash"], as_index=False)
            .agg({
                "growth_operation": lambda values: ";".join(sorted(set(map(str, values)))),
                "changed_node_or_edge": lambda values: ";".join(sorted(set(map(str, values)))),
            })
        )
    return selected.sort_values(["parent_hash", "child_hash"]).reset_index(drop=True)


def build_parent_index(
    relationships: pd.DataFrame,
    features: pd.DataFrame,
    minimum: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the complete parent audit and analyzable relationship table."""
    labels = features[["wl_hash", "significance_group"]].rename(columns={"wl_hash": "child_hash"})
    analyzable = relationships.merge(labels, on="child_hash", how="inner", validate="many_to_one")
    trace_counts = relationships.groupby("parent_hash")["child_hash"].nunique()
    class_counts = (
        analyzable.groupby(["parent_hash", "significance_group"])["child_hash"]
        .nunique()
        .unstack(fill_value=0)
    )
    index = pd.DataFrame(index=sorted(relationships["parent_hash"].unique()))
    index.index.name = "parent_hash"
    index["trace_tri5_children"] = trace_counts.reindex(index.index, fill_value=0).astype(int)
    for group in ["nonsignificant", "significant"]:
        values = class_counts[group] if group in class_counts else pd.Series(dtype=int)
        index[f"analyzable_{group}_children"] = values.reindex(index.index, fill_value=0).astype(int)
    index["analyzable_total_children"] = (
        index["analyzable_nonsignificant_children"] + index["analyzable_significant_children"]
    )
    index["mixed_significance_group"] = (
        index["analyzable_nonsignificant_children"].gt(0)
        & index["analyzable_significant_children"].gt(0)
    )
    index["eligible"] = (
        index["analyzable_nonsignificant_children"].ge(minimum)
        & index["analyzable_significant_children"].ge(minimum)
    )

    def exclusion_reason(row: pd.Series) -> str:
        if row.eligible:
            return ""
        if row.analyzable_total_children == 0:
            return "no_analyzable_tri5_children"
        if not row.mixed_significance_group:
            return "missing_significance_class"
        return f"fewer_than_{minimum}_children_in_one_or_both_classes"

    index["exclusion_reason"] = index.apply(exclusion_reason, axis=1)
    return index.reset_index(), analyzable


def describe(values: pd.Series) -> dict[str, float | int]:
    return analysis.describe_metric(values)


def compare_parent(
    group: pd.DataFrame,
    metrics: list[str],
    minimum: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create descriptive and inferential criterion tables for one parent."""
    distribution_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    for metric in metrics:
        for significance in ["nonsignificant", "significant"]:
            values = group.loc[group["significance_group"].eq(significance), metric]
            distribution_rows.append({
                "metric": metric,
                "display_name": analysis.display_name(metric),
                "family": analysis.family_for(metric),
                "group": significance,
                **describe(values),
            })
        control = (
            pd.to_numeric(
                group.loc[group["significance_group"].eq("nonsignificant"), metric],
                errors="coerce",
            )
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .to_numpy(float)
        )
        case = (
            pd.to_numeric(
                group.loc[group["significance_group"].eq("significant"), metric],
                errors="coerce",
            )
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .to_numpy(float)
        )
        if (
            len(control) < minimum
            or len(case) < minimum
            or len(np.unique(np.concatenate([control, case]))) < 2
        ):
            continue
        test = mannwhitneyu(case, control, alternative="two-sided", method="asymptotic")
        auc = float(test.statistic / (len(case) * len(control)))
        pooled_sd = float(np.std(np.concatenate([control, case]), ddof=0))
        row: dict[str, object] = {
            "metric": metric,
            "display_name": analysis.display_name(metric),
            "family": analysis.family_for(metric),
            "n_nonsignificant": len(control),
            "n_significant": len(case),
            "mann_whitney_u": float(test.statistic),
            "location_p_value": float(test.pvalue),
            "cliffs_delta": 2 * auc - 1,
            "median_difference": float(np.median(case) - np.median(control)),
            "mean_difference": float(np.mean(case) - np.mean(control)),
            "standardized_median_difference": (
                float((np.median(case) - np.median(control)) / pooled_sd)
                if pooled_sd > 0 else np.nan
            ),
        }
        if len(control) > 1 and len(case) > 1:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                dispersion = levene(case, control, center="median")
            control_variance = float(np.var(control, ddof=1))
            row.update({
                "brown_forsythe_statistic": (
                    float(dispersion.statistic) if np.isfinite(dispersion.statistic) else np.nan
                ),
                "dispersion_p_value": (
                    float(dispersion.pvalue) if np.isfinite(dispersion.pvalue) else np.nan
                ),
                "variance_ratio_significant_over_nonsignificant": (
                    float(np.var(case, ddof=1) / control_variance)
                    if control_variance > 0 else np.nan
                ),
            })
        comparison_rows.append(row)

    distributions = pd.DataFrame(distribution_rows)
    comparisons = pd.DataFrame(comparison_rows)
    if comparisons.empty:
        return distributions, comparisons
    comparisons["location_q_bh"] = analysis.base.benjamini_hochberg(
        comparisons["location_p_value"]
    )
    comparisons["dispersion_q_bh"] = analysis.base.benjamini_hochberg(
        comparisons.get("dispersion_p_value", pd.Series(np.nan, index=comparisons.index))
    )
    comparisons["absolute_cliffs_delta"] = comparisons["cliffs_delta"].abs()
    comparisons = comparisons.sort_values(
        ["location_q_bh", "absolute_cliffs_delta", "metric"],
        ascending=[True, False, True],
        na_position="last",
    ).reset_index(drop=True)
    comparisons["rank_within_parent"] = np.arange(1, len(comparisons) + 1)
    return distributions, comparisons


def plot_top_criteria(
    group: pd.DataFrame,
    top: pd.DataFrame,
    parent_hash: str,
    parent_description: str,
    outdir: Path,
    panels_per_page: int,
) -> list[dict[str, object]]:
    """Render ranked parent-specific violin/scatter pages."""
    index_rows: list[dict[str, object]] = []
    columns = 2
    rows = math.ceil(panels_per_page / columns)
    for page, start in enumerate(range(0, len(top), panels_per_page), 1):
        selected = top.iloc[start:start + panels_per_page]
        fig, axes = plt.subplots(rows, columns, figsize=(13, 3.1 * rows), squeeze=False)
        for panel, row in enumerate(selected.itertuples(index=False), 1):
            ax = axes.flat[panel - 1]
            control = group.loc[group.significance_group.eq("nonsignificant"), row.metric]
            case = group.loc[group.significance_group.eq("significant"), row.metric]
            analysis.draw_violin(ax, control, case)
            ax.set_title(
                f"{row.display_name}\nBH q={row.location_q_bh:.3g}  |  "
                f"Cliff δ={row.cliffs_delta:+.2f}  |  "
                f"n={row.n_nonsignificant}/{row.n_significant}",
                fontsize=8,
            )
            index_rows.append({
                "parent_hash": parent_hash,
                "metric": row.metric,
                "rank_within_parent": int(row.rank_within_parent),
                "figure_file": f"top_criteria_distributions_page_{page:02d}.png",
                "panel_on_page": panel,
            })
        for ax in axes.flat[len(selected):]:
            ax.axis("off")
        fig.suptitle(
            f"TRI-5 children sharing TRI-4 parent {parent_hash}\n"
            f"{parent_description} — ranked by within-parent BH q (page {page})",
            fontsize=12,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.965))
        fig.savefig(outdir / f"top_criteria_distributions_page_{page:02d}.png", dpi=180)
        plt.close(fig)
    return index_rows


def plot_child_motif_gallery(
    children: pd.DataFrame,
    parent_graph: nx.Graph,
    parent_hash: str,
    parent_description: str,
    path: Path,
) -> list[dict[str, object]]:
    """Draw children on one fixed parent layout, adding only their growth element."""
    ordered = children.copy()
    ordered["group_order"] = ordered["significance_group"].map(
        {"nonsignificant": 0, "significant": 1}
    )
    ordered = ordered.sort_values(
        ["group_order", "p_value_unadjusted", "child_hash"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    parent_positions = analysis.triangulation_planar_layout(parent_graph)

    def normalized_edge(left: int, right: int) -> tuple[int, int]:
        return tuple(sorted((int(left), int(right))))

    def parse_changed(operation: str, changed: str) -> tuple[str | None, set[int]]:
        if operation == "add_vertex" and "@" in changed:
            prefix, edge = changed.split("@", 1)
            label = prefix.split(":", 1)[1] if ":" in prefix else None
            return label, {int(value) for value in edge.split("-")}
        if operation == "add_edge":
            return None, {int(value) for value in changed.split("-")}
        return None, set()

    def parent_mapping(
        child_graph: nx.Graph, operation: str, changed: str
    ) -> tuple[dict[int, int], set[int], set[tuple[int, int]]]:
        node_match = lambda left, right: str(left.get("label")) == str(right.get("label"))
        matcher = nx.algorithms.isomorphism.GraphMatcher(
            child_graph, parent_graph, node_match=node_match
        )
        iterator = (
            matcher.subgraph_isomorphisms_iter()
            if operation == "add_vertex"
            else matcher.subgraph_monomorphisms_iter()
        )
        changed_label, expected_parent_nodes = parse_changed(operation, changed)
        child_coordinates = getattr(analysis, "TRANSITION_COORDINATES", "parent") == "child"
        expected_child_node = int(changed) if child_coordinates and operation == "add_vertex" else None
        candidates: list[tuple[tuple[int, ...], dict[int, int], set[int], set[tuple[int, int]]]] = []
        parent_nodes = sorted(parent_graph.nodes)
        for child_to_parent in iterator:
            if len(child_to_parent) != len(parent_graph):
                continue
            mapping = {int(parent): int(child) for child, parent in child_to_parent.items()}
            if set(mapping) != set(parent_graph.nodes):
                continue
            mapped_parent_edges = {
                normalized_edge(mapping[left], mapping[right])
                for left, right in parent_graph.edges
            }
            child_edges = {normalized_edge(left, right) for left, right in child_graph.edges}
            growth_edges = child_edges - mapped_parent_edges
            added_nodes = set(child_graph.nodes) - set(mapping.values())
            if operation == "add_vertex":
                if len(added_nodes) != 1 or any(
                    left not in added_nodes and right not in added_nodes
                    for left, right in growth_edges
                ):
                    continue
                added = next(iter(added_nodes))
                if expected_child_node is not None and added != expected_child_node:
                    continue
                if changed_label is not None and str(child_graph.nodes[added]["label"]) != changed_label:
                    continue
                actual_parent_neighbors = {
                    int(child_to_parent[neighbor])
                    for neighbor in child_graph.neighbors(added)
                    if neighbor in child_to_parent
                }
                if expected_parent_nodes and actual_parent_neighbors != expected_parent_nodes:
                    continue
            elif operation == "add_edge":
                if added_nodes or len(growth_edges) != 1:
                    continue
                growth_child_edge = next(iter(growth_edges))
                actual_parent_edge = {
                    int(child_to_parent[growth_child_edge[0]]),
                    int(child_to_parent[growth_child_edge[1]]),
                }
                actual_edge = set(growth_child_edge) if child_coordinates else actual_parent_edge
                if expected_parent_nodes and actual_edge != expected_parent_nodes:
                    continue
            else:
                continue
            key = tuple(mapping[node] for node in parent_nodes)
            candidates.append((key, mapping, added_nodes, growth_edges))
        if not candidates:
            raise ValueError(
                f"Cannot align child growth operation {operation} {changed} to parent {parent_hash}"
            )
        _, mapping, added_nodes, growth_edges = min(candidates, key=lambda value: value[0])
        return mapping, added_nodes, growth_edges

    def child_positions(
        child_graph: nx.Graph,
        mapping: dict[int, int],
        added_nodes: set[int],
    ) -> dict[int, np.ndarray]:
        positions = {
            child_node: np.asarray(parent_positions[parent_node], dtype=float).copy()
            for parent_node, child_node in mapping.items()
        }
        if not added_nodes:
            return positions
        added = next(iter(added_nodes))
        anchors = [positions[node] for node in child_graph.neighbors(added) if node in positions]
        if len(anchors) < 2:
            raise ValueError(f"Added node in {parent_hash} child has fewer than two parent anchors")
        left, right = anchors[:2]
        midpoint = (left + right) / 2
        edge_vector = right - left
        edge_length = float(np.linalg.norm(edge_vector))
        normal = np.array([-edge_vector[1], edge_vector[0]], dtype=float)
        normal /= max(float(np.linalg.norm(normal)), 1e-12)
        parent_centroid = np.mean(np.asarray(list(parent_positions.values())), axis=0)
        if float(np.dot(parent_centroid - midpoint, normal)) > 0:
            normal *= -1
        distance = max(0.55 * edge_length, 0.42)
        candidate = midpoint + normal * distance
        parent_coordinates = np.asarray(list(positions.values()))
        while np.min(np.linalg.norm(parent_coordinates - candidate, axis=1)) < 0.34:
            distance *= 1.18
            candidate = midpoint + normal * distance
        positions[added] = candidate
        return positions

    prepared: list[dict[str, object]] = []
    for child in ordered.itertuples(index=False):
        graph = analysis.base.graph_from_encoding(
            str(child.node_labels), str(child.edge_list), str(child.child_hash)
        )
        mapping, added_nodes, growth_edges = parent_mapping(
            graph, str(child.growth_operation), str(child.changed_node_or_edge)
        )
        prepared.append({
            "row": child,
            "graph": graph,
            "positions": child_positions(graph, mapping, added_nodes),
            "added_nodes": added_nodes,
            "growth_edges": growth_edges,
            "parent_to_child_mapping": mapping,
        })

    all_coordinates = np.vstack([
        np.asarray(list(item["positions"].values()), dtype=float) for item in prepared
    ])
    lower = all_coordinates.min(axis=0)
    upper = all_coordinates.max(axis=0)
    padding = np.maximum((upper - lower) * 0.12, 0.18)
    x_limits = (lower[0] - padding[0], upper[0] + padding[0])
    y_limits = (lower[1] - padding[1], upper[1] + padding[1])

    columns = min(5, max(1, len(ordered)))
    rows = math.ceil(len(ordered) / columns)
    fig, axes = plt.subplots(
        rows, columns, figsize=(3.05 * columns, 3.0 * rows + 0.8), squeeze=False
    )
    index_rows: list[dict[str, object]] = []
    abbreviations = analysis.LABEL_ABBREVIATIONS
    for panel, item in enumerate(prepared, 1):
        ax = axes.flat[panel - 1]
        child = item["row"]
        graph = item["graph"]
        positions = item["positions"]
        added_nodes = item["added_nodes"]
        growth_edges = item["growth_edges"]
        group = str(child.significance_group)
        abbreviation = "NONSIG" if group == "nonsignificant" else "SIGNIFICANT"
        operation = str(child.growth_operation).replace("add_vertex", "+node").replace(
            "add_edge", "+edge"
        )
        parent_edges = [
            edge for edge in graph.edges if normalized_edge(*edge) not in growth_edges
        ]
        nx.draw_networkx_edges(
            graph, positions, ax=ax, edgelist=parent_edges,
            edge_color="#666666", width=1.4, alpha=0.9,
        )
        nx.draw_networkx_edges(
            graph, positions, ax=ax, edgelist=list(growth_edges),
            edge_color="#E69F00", width=2.5,
        )
        nodes = sorted(graph.nodes)
        group_color = "#D1495B" if group == "significant" else "#6F8499"
        node_edge_colors = ["#E69F00" if node in added_nodes else group_color for node in nodes]
        node_widths = [2.8 if node in added_nodes else 1.8 for node in nodes]
        nx.draw_networkx_nodes(
            graph, positions, ax=ax, nodelist=nodes,
            node_color=[analysis.base.LABEL_COLORS[str(graph.nodes[node]["label"])] for node in nodes],
            node_size=410, edgecolors=node_edge_colors, linewidths=node_widths,
        )
        nx.draw_networkx_labels(
            graph, positions, ax=ax,
            labels={node: abbreviations[str(graph.nodes[node]["label"])] for node in nodes},
            font_size=6.1, font_weight="bold",
        )
        ax.set_title(
            f"{abbreviation}  |  {str(child.child_hash)[:8]}\n"
            f"p={float(child.p_value_unadjusted):.2g}  "
            f"HR={float(child.hazard_ratio_unadjusted):.2g}  |  {operation}",
            fontsize=8.0, color=group_color, pad=7, linespacing=1.18,
        )
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        ax.set_aspect("equal")
        ax.set_axis_off()
        index_rows.append({
            "parent_hash": parent_hash,
            "child_hash": str(child.child_hash),
            "significance_group": group,
            "panel": panel,
            "figure_file": path.name,
            "parent_to_child_mapping": json.dumps(item["parent_to_child_mapping"], sort_keys=True),
            "added_nodes": json.dumps(sorted(item["added_nodes"])),
            "added_edges": json.dumps(sorted(item["growth_edges"])),
        })
    for ax in axes.flat[len(ordered):]:
        ax.axis("off")

    label_handles = [
        plt.Line2D(
            [0], [0], marker="o", color="none",
            markerfacecolor=analysis.base.LABEL_COLORS[label], markeredgecolor="white",
            markersize=7, label=f"{label}={analysis.LABEL_NAMES[label]}",
        )
        for label in analysis.LABELS
    ]
    group_handles = [
        plt.Line2D(
            [0], [0], marker="o", color="none", markerfacecolor="white",
            markeredgecolor=color, markeredgewidth=2, markersize=8, label=label,
        )
        for label, color in [
            ("nonsignificant outline", "#6F8499"),
            ("significant outline", "#D1495B"),
        ]
    ]
    counts = ordered.significance_group.value_counts()
    fig.suptitle(
        f"TRI-5 children sharing TRI-4 parent {parent_hash}\n"
        f"{parent_description}  |  nonsignificant n={counts.get('nonsignificant', 0)}, "
        f"significant n={counts.get('significant', 0)}",
        fontsize=13,
        y=0.995,
    )
    fig.legend(
        handles=label_handles + group_handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=7.5,
    )
    fig.tight_layout(rect=(0.015, 0.045, 0.985, 0.965), h_pad=1.35, w_pad=0.9)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return index_rows


def generate_existing_motif_galleries(args: argparse.Namespace) -> dict[str, int]:
    """Generate galleries using the already materialized parent membership tables."""
    index_path = args.outdir / "parent_group_index.csv"
    if not index_path.is_file():
        raise FileNotFoundError("--motif-galleries-only requires parent_group_index.csv")
    parent_index = pd.read_csv(index_path, dtype={"parent_hash": str})
    nodes = pd.read_csv(
        args.trace_dir / "trace_nodes.csv",
        dtype={"wl_hash": str, "node_labels": str, "edge_list": str},
    )
    parent_nodes = nodes.loc[nodes.triangle_level.eq(4)].drop_duplicates("wl_hash").set_index("wl_hash")
    gallery_rows: list[dict[str, object]] = []
    eligible = parent_index.loc[parent_index.eligible].reset_index(drop=True)
    for position, parent in enumerate(eligible.itertuples(index=False), 1):
        parent_dir = args.outdir / str(parent.parent_hash)
        children = pd.read_csv(
            parent_dir / "child_motifs.csv",
            dtype={"parent_hash": str, "child_hash": str, "node_labels": str, "edge_list": str},
        )
        parent_source = parent_nodes.loc[str(parent.parent_hash)]
        parent_graph = analysis.base.graph_from_encoding(
            str(parent_source.node_labels), str(parent_source.edge_list), str(parent.parent_hash)
        )
        gallery_rows.extend(
            plot_child_motif_gallery(
                children,
                parent_graph,
                str(parent.parent_hash),
                str(parent.parent_cell_type_composition),
                parent_dir / "tri5_child_motif_gallery.png",
            )
        )
        if position % 10 == 0 or position == len(eligible):
            print(f"completed motif galleries {position}/{len(eligible)}", flush=True)
    pd.DataFrame(gallery_rows).to_csv(args.outdir / "motif_gallery_index.csv", index=False)
    summary_path = args.outdir / "analysis_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    summary.update({
        "motif_galleries": int(len(eligible)),
        "motif_gallery_panels": int(len(gallery_rows)),
    })
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_readme(args.outdir / "README.md", summary, args)
    return {
        "motif_galleries": int(len(eligible)),
        "motif_gallery_panels": int(len(gallery_rows)),
    }


def build_recurrence(
    all_comparisons: pd.DataFrame,
    all_top: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if all_comparisons.empty:
        return pd.DataFrame()
    for metric, tested in all_comparisons.groupby("metric"):
        selected = all_top.loc[all_top.metric.eq(metric)]
        rows.append({
            "metric": metric,
            "display_name": tested.display_name.iloc[0],
            "family": tested.family.iloc[0],
            "parents_tested": tested.parent_hash.nunique(),
            "parents_bh_q_lt_0_05": int((tested.location_q_bh < 0.05).sum()),
            "parents_in_top_criteria": selected.parent_hash.nunique(),
            "top_criteria_positive_delta": int((selected.cliffs_delta > 0).sum()),
            "top_criteria_negative_delta": int((selected.cliffs_delta < 0).sum()),
            "median_cliffs_delta_when_top": (
                float(selected.cliffs_delta.median()) if len(selected) else np.nan
            ),
            "median_absolute_cliffs_delta_when_top": (
                float(selected.absolute_cliffs_delta.median()) if len(selected) else np.nan
            ),
            "median_bh_q_when_top": (
                float(selected.location_q_bh.median()) if len(selected) else np.nan
            ),
        })
    return pd.DataFrame(rows).sort_values(
        ["parents_in_top_criteria", "parents_bh_q_lt_0_05", "metric"],
        ascending=[False, False, True],
    )


HEATMAP_Q_FLOOR = 1e-12


def build_location_q_matrix(
    comparisons: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the complete testable-criterion by eligible-parent raw-q matrix."""
    required = {
        "parent_hash", "metric", "display_name", "family", "location_q_bh",
        "n_nonsignificant", "n_significant",
    }
    missing = required.difference(comparisons.columns)
    if missing:
        raise ValueError(f"Combined comparisons lack required columns: {sorted(missing)}")
    if comparisons.duplicated(["metric", "parent_hash"]).any():
        raise ValueError("Expected one location BH q-value per criterion and parent")

    matrix = (
        comparisons.pivot(index="metric", columns="parent_hash", values="location_q_bh")
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    metric_metadata = (
        comparisons[["metric", "display_name", "family"]]
        .drop_duplicates()
        .sort_values("metric")
        .set_index("metric")
    )
    if metric_metadata.index.duplicated().any():
        raise ValueError("A criterion has inconsistent display-name or family metadata")
    parent_metadata = (
        comparisons.groupby("parent_hash", as_index=False)[
            ["n_nonsignificant", "n_significant"]
        ]
        .max()
        .sort_values("parent_hash")
        .set_index("parent_hash")
    )
    return matrix, metric_metadata, parent_metadata


def clustered_matrix_order(
    matrix: pd.DataFrame,
) -> tuple[list[str], list[str], np.ndarray | None, np.ndarray | None]:
    """Return deterministic two-way clustering orders for a raw-q matrix."""
    evidence = -np.log10(matrix.clip(lower=HEATMAP_Q_FLOOR, upper=1.0))
    evidence = evidence.fillna(0.0)

    row_linkage = None
    if len(evidence.index) > 1:
        row_linkage = linkage(
            evidence.to_numpy(), method="average", metric="euclidean", optimal_ordering=True
        )
        row_order = evidence.index[leaves_list(row_linkage)].tolist()
    else:
        row_order = evidence.index.tolist()

    column_linkage = None
    if len(evidence.columns) > 1:
        column_linkage = linkage(
            evidence.to_numpy().T, method="average", metric="euclidean", optimal_ordering=True
        )
        column_order = evidence.columns[leaves_list(column_linkage)].tolist()
    else:
        column_order = evidence.columns.tolist()
    return row_order, column_order, row_linkage, column_linkage


def plot_clustered_location_q_heatmap(
    matrix: pd.DataFrame,
    metric_metadata: pd.DataFrame,
    parent_metadata: pd.DataFrame,
    outdir: Path,
) -> dict[str, int]:
    """Write a large two-way clustered location-BH-q heatmap and audit tables."""
    if matrix.empty:
        raise ValueError("Cannot cluster an empty location BH q-value matrix")
    row_order, column_order, row_linkage, column_linkage = clustered_matrix_order(matrix)
    ordered = matrix.loc[row_order, column_order]
    ordered.to_csv(
        outdir / "clustered_location_bh_q_matrix.csv.gz",
        index_label="metric",
        compression="gzip",
    )

    order_rows: list[dict[str, object]] = []
    for position, metric in enumerate(row_order, 1):
        metadata = metric_metadata.loc[metric]
        order_rows.append({
            "axis": "criterion",
            "position": position,
            "id": metric,
            "display_name": metadata.display_name,
            "family": metadata.family,
            "nonmissing_q_values": int(ordered.loc[metric].notna().sum()),
            "n_nonsignificant": np.nan,
            "n_significant": np.nan,
        })
    for position, parent_hash in enumerate(column_order, 1):
        metadata = parent_metadata.loc[parent_hash]
        order_rows.append({
            "axis": "parent",
            "position": position,
            "id": parent_hash,
            "display_name": parent_hash[:8],
            "family": np.nan,
            "nonmissing_q_values": int(ordered[parent_hash].notna().sum()),
            "n_nonsignificant": int(metadata.n_nonsignificant),
            "n_significant": int(metadata.n_significant),
        })
    pd.DataFrame(order_rows).to_csv(
        outdir / "clustered_location_bh_q_order.csv", index=False
    )

    n_rows, n_columns = ordered.shape
    figure_width = max(24.0, 0.22 * n_columns + 8.0)
    figure_height = max(32.0, 0.205 * n_rows + 7.0)
    fig = plt.figure(figsize=(figure_width, figure_height))
    grid = fig.add_gridspec(
        2, 3,
        width_ratios=[2.3, 0.28, 15.0],
        height_ratios=[2.4, 18.0],
        left=0.035,
        right=0.93,
        bottom=0.055,
        top=0.955,
        wspace=0.025,
        hspace=0.025,
    )
    top_axis = fig.add_subplot(grid[0, 2])
    row_axis = fig.add_subplot(grid[1, 0])
    family_axis = fig.add_subplot(grid[1, 1])
    heat_axis = fig.add_subplot(grid[1, 2])

    if column_linkage is not None:
        dendrogram(
            column_linkage,
            ax=top_axis,
            no_labels=True,
            color_threshold=0,
            above_threshold_color="#555555",
            link_color_func=lambda _: "#555555",
        )
    top_axis.set_axis_off()
    if row_linkage is not None:
        dendrogram(
            row_linkage,
            ax=row_axis,
            orientation="left",
            no_labels=True,
            color_threshold=0,
            above_threshold_color="#555555",
            link_color_func=lambda _: "#555555",
        )
        row_axis.invert_yaxis()
    row_axis.set_axis_off()

    families = sorted(metric_metadata.loc[row_order, "family"].astype(str).unique())
    family_codes = {family: index for index, family in enumerate(families)}
    palette = plt.get_cmap("tab20", max(1, len(families)))
    family_values = np.asarray(
        [family_codes[str(metric_metadata.loc[metric, "family"])] for metric in row_order]
    ).reshape(-1, 1)
    family_axis.imshow(
        family_values,
        aspect="auto",
        interpolation="nearest",
        cmap=ListedColormap([palette(index) for index in range(len(families))]),
        vmin=-0.5,
        vmax=max(0.5, len(families) - 0.5),
    )
    family_axis.set_xticks([0], ["family"], rotation=90, fontsize=6)
    family_axis.set_yticks([])
    family_axis.tick_params(length=0)
    for spine in family_axis.spines.values():
        spine.set_visible(False)

    plotted = ordered.clip(lower=0.0, upper=1.0).to_numpy(dtype=float)
    plotted = np.ma.masked_invalid(plotted)
    colormap = plt.get_cmap("magma").copy()
    colormap.set_bad("#d9d9d9")
    image = heat_axis.imshow(
        plotted,
        aspect="auto",
        interpolation="nearest",
        cmap=colormap,
        norm=Normalize(vmin=0.0, vmax=1.0),
    )
    ylabels = [str(metric_metadata.loc[metric, "display_name"]) for metric in row_order]
    heat_axis.set_yticks(np.arange(n_rows), ylabels, fontsize=4.4)
    heat_axis.yaxis.tick_right()
    heat_axis.tick_params(axis="y", length=0, pad=2)
    heat_axis.set_xticks(
        np.arange(n_columns), [parent_hash[:8] for parent_hash in column_order],
        rotation=90, fontsize=5.2,
    )
    heat_axis.tick_params(axis="x", length=0, pad=2)
    heat_axis.set_xlabel("TRI-4 parent motif (first 8 characters of WL hash)", fontsize=10)
    heat_axis.set_ylabel("Criterion", fontsize=10, labelpad=8)
    heat_axis.yaxis.set_label_position("right")

    # Keep the heatmap and its top dendrogram at exactly the same width. Attaching
    # a colorbar directly to heat_axis would shrink only the heatmap and visually
    # push the dendrogram's rightmost branches beyond the matrix.
    colorbar_axis = fig.add_axes([0.972, 0.385, 0.010, 0.22])
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    ticks = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels([f"{tick:g}" for tick in ticks])
    colorbar.set_label("Within-parent location BH q-value (linear 0–1 scale)")

    legend_handles = [
        Patch(facecolor=palette(index), edgecolor="none", label=family)
        for index, family in enumerate(families)
    ]
    legend_handles.append(Patch(facecolor="#d9d9d9", edgecolor="#888888", label="not testable"))
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.008),
        ncol=min(6, max(1, len(legend_handles))),
        fontsize=7,
        frameon=False,
    )
    fig.suptitle(
        "Same-parent TRI-5 comparison: clustered location BH q-values\n"
        f"{n_rows} testable criteria × {n_columns} eligible TRI-4 parents",
        fontsize=16,
        y=0.992,
    )
    png_path = outdir / "clustered_location_bh_q_heatmap.png"
    pdf_path = outdir / "clustered_location_bh_q_heatmap.pdf"
    fig.savefig(png_path, dpi=180, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)
    return {
        "clustered_heatmap_criteria": int(n_rows),
        "clustered_heatmap_parents": int(n_columns),
        "clustered_heatmap_observed_q_values": int(ordered.notna().sum().sum()),
        "clustered_heatmap_missing_cells": int(ordered.isna().sum().sum()),
    }


def plot_clustered_tri4_parent_gallery(
    parent_order: list[str],
    trace_dir: Path,
    outdir: Path,
) -> dict[str, int]:
    """Draw all eligible TRI-4 parents in the heatmap's clustered column order."""
    parent_index = pd.read_csv(
        outdir / "parent_group_index.csv", dtype={"parent_hash": str}
    ).set_index("parent_hash")
    nodes = pd.read_csv(
        trace_dir / "trace_nodes.csv",
        dtype={"wl_hash": str, "node_labels": str, "edge_list": str},
    )
    parent_nodes = (
        nodes.loc[nodes.triangle_level.eq(4), ["wl_hash", "node_labels", "edge_list"]]
        .drop_duplicates("wl_hash")
        .set_index("wl_hash")
    )
    eligible = set(parent_index.index[parent_index.eligible])
    if set(parent_order) != eligible:
        raise ValueError("Clustered parent order does not match the eligible TRI-4 parents")

    abbreviations = analysis.LABEL_ABBREVIATIONS
    columns = 8
    rows = math.ceil(len(parent_order) / columns)
    fig, axes = plt.subplots(
        rows, columns, figsize=(3.0 * columns, 2.65 * rows + 1.0), squeeze=False
    )
    index_rows: list[dict[str, object]] = []
    for position, parent_hash in enumerate(parent_order, 1):
        ax = axes.flat[position - 1]
        source = parent_nodes.loc[parent_hash]
        graph = analysis.base.graph_from_encoding(
            str(source.node_labels), str(source.edge_list), parent_hash
        )
        coordinates = analysis.triangulation_planar_layout(graph)
        nx.draw_networkx_edges(
            graph, coordinates, ax=ax, edge_color="#666666", width=1.35, alpha=0.95
        )
        ordered_nodes = sorted(graph.nodes)
        nx.draw_networkx_nodes(
            graph,
            coordinates,
            ax=ax,
            nodelist=ordered_nodes,
            node_color=[
                analysis.base.LABEL_COLORS[str(graph.nodes[node]["label"])]
                for node in ordered_nodes
            ],
            node_size=390,
            edgecolors="#4d4d4d",
            linewidths=1.35,
        )
        nx.draw_networkx_labels(
            graph,
            coordinates,
            ax=ax,
            labels={
                node: abbreviations[str(graph.nodes[node]["label"])]
                for node in ordered_nodes
            },
            font_size=5.8,
            font_weight="bold",
        )
        metadata = parent_index.loc[parent_hash]
        nonsignificant = int(metadata.analyzable_nonsignificant_children)
        significant = int(metadata.analyzable_significant_children)
        ax.set_title(
            f"#{position:02d}  {parent_hash[:8]}\n"
            f"TRI-5 children: non={nonsignificant}, sig={significant}",
            fontsize=7.2,
            pad=5,
        )
        ax.margins(0.22)
        ax.set_aspect("equal")
        ax.set_axis_off()
        index_rows.append({
            "cluster_position": position,
            "parent_hash": parent_hash,
            "analyzable_nonsignificant_children": nonsignificant,
            "analyzable_significant_children": significant,
            "analyzable_total_children": int(metadata.analyzable_total_children),
            "parent_cell_type_composition": metadata.parent_cell_type_composition,
        })
    for ax in axes.flat[len(parent_order):]:
        ax.axis("off")

    label_handles = [
        plt.Line2D(
            [0], [0], marker="o", color="none",
            markerfacecolor=analysis.base.LABEL_COLORS[label], markeredgecolor="#4d4d4d",
            markersize=7, label=f"{label}={analysis.LABEL_NAMES[label]}",
        )
        for label in analysis.LABELS
    ]
    fig.suptitle(
        "Eligible TRI-4 parent motifs in clustered heatmap order\n"
        "Panel order follows the heatmap columns from left to right",
        fontsize=15,
        y=0.997,
    )
    fig.legend(
        handles=label_handles,
        loc="lower center",
        ncol=8,
        frameon=False,
        fontsize=8,
    )
    fig.tight_layout(rect=(0.012, 0.035, 0.988, 0.972), h_pad=1.15, w_pad=0.7)
    png_path = outdir / "clustered_tri4_parent_motif_gallery.png"
    pdf_path = outdir / "clustered_tri4_parent_motif_gallery.pdf"
    fig.savefig(png_path, dpi=200, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)
    pd.DataFrame(index_rows).to_csv(
        outdir / "clustered_tri4_parent_motif_gallery_index.csv", index=False
    )
    return {"clustered_parent_gallery_motifs": int(len(parent_order))}


def generate_clustered_heatmap_from_existing(args: argparse.Namespace) -> dict[str, object]:
    """Regenerate only the clustered heatmap and update its output audit."""
    comparisons_path = args.outdir / "all_parent_criteria_comparisons.csv.gz"
    if not comparisons_path.is_file():
        raise FileNotFoundError(f"Missing combined comparison table: {comparisons_path}")
    comparisons = pd.read_csv(comparisons_path, dtype={"parent_hash": str})
    matrix, metric_metadata, parent_metadata = build_location_q_matrix(comparisons)
    heatmap_summary = plot_clustered_location_q_heatmap(
        matrix, metric_metadata, parent_metadata, args.outdir
    )
    _, parent_order, _, _ = clustered_matrix_order(matrix)
    heatmap_summary.update(
        plot_clustered_tri4_parent_gallery(
            parent_order, args.trace_dir, args.outdir
        )
    )
    summary_path = args.outdir / "analysis_summary.json"
    summary: dict[str, object] = (
        json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    )
    summary.update(heatmap_summary)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_readme(args.outdir / "README.md", summary, args)
    return heatmap_summary


def write_readme(path: Path, summary: dict[str, object], args: argparse.Namespace) -> None:
    path.write_text(
        f"""# Same-parent TRI-5 comparison

This study compares significant and nonsignificant TRI-5 motifs that share the
same immediate TRI-4 parent in the lineage. Both `add_vertex` and `add_edge`
relationships are included. A TRI-5 motif with multiple valid TRI-4 parents is
included independently in every applicable parent folder.

## Eligibility and ranking

- TRI-4 parents in the lineage: {summary['trace_parents']:,}
- Parents with both significance classes: {summary['mixed_parents']:,}
- Eligible parents with at least {args.min_per_class} motifs per class: {summary['eligible_parents']:,}
- Candidate intrinsic criteria: {summary['candidate_criteria']:,}
- Top criteria plotted per parent: {args.top_criteria}

Significance is the established raw Cox threshold (`p < 0.05`) in the TRI-5
criterion table. Criteria are tested within each parent using a two-sided
Mann–Whitney test. BH correction is applied separately over all testable
criteria within that parent. Criteria are ordered by BH q-value, then absolute
Cliff's delta, then metric name. Thus q-values from differently sized parent
groups should not be interpreted as directly comparable evidence strengths.

Outcome-derived `dispersion_*` criteria, metadata, constant criteria, and
criteria with fewer than {args.min_per_class} finite values in either class are
excluded from testing. The black vertical bar in each plot is the IQR, and the
orange point is the median; jittered points are individual TRI-5 motifs.

## Root files

- `parent_group_index.csv`: every TRI-4 parent, class counts, eligibility, and exclusion reason.
- `top_criteria_recurrence.csv`: recurrence and direction of top-ranked criteria across eligible parents.
- `all_parent_criteria_comparisons.csv.gz`: complete inferential results across eligible parents.
- `clustered_location_bh_q_heatmap.png` and `.pdf`: two-way clustered map of within-parent location BH q-values.
- `clustered_location_bh_q_matrix.csv.gz`: raw q-values in the plotted row and column order; blank cells were not testable.
- `clustered_location_bh_q_order.csv`: complete criterion and parent identifiers, annotations, and plotted positions.
- `clustered_tri4_parent_motif_gallery.png` and `.pdf`: all eligible TRI-4 parent graphs in heatmap column order.
- `clustered_tri4_parent_motif_gallery_index.csv`: complete hashes, cluster positions, compositions, and child counts.
- `figure_index.csv`: parent, criterion, rank, figure page, and panel lookup.
- `motif_gallery_index.csv`: parent, child, significance class, and gallery panel lookup.
- `analysis_summary.json`: machine-readable run counts and settings.

## Clustered BH-q heatmap

The heatmap contains the union of criteria testable for at least one eligible
parent. Rows and columns use average-linkage hierarchical clustering with
Euclidean distance on `-log10(q)`; missing tests contribute zero evidence only
while deriving the cluster order. The plotted and exported cells retain their
raw within-parent `location_q_bh` values and use a linear 0-to-1 color scale:
0 is darkest, 1 is lightest, and 0.05 is labeled explicitly. Unavailable tests
are shown in gray. Each parent's BH correction was performed
separately, so the heatmap is an exploratory pattern display rather than a new
global multiple-testing analysis.

## Parent folders

Each eligible folder is named by the complete TRI-4 WL hash and contains:

- `child_motifs.csv`: analyzed TRI-5 children and their parent transition metadata.
- `criteria_distributions.csv.gz`: descriptive summaries for both groups.
- `criteria_comparisons.csv`: all testable intrinsic criteria.
- `top_criteria.csv`: the first {args.top_criteria} criteria under the ranking rule.
- `top_criteria_distributions_page_01.png` and `page_02.png`: 12 panels per page when at least 24 criteria are testable.
- `tri5_child_motif_gallery.png`: all analyzed children on one fixed TRI-4 parent layout. Parent nodes and edges retain identical coordinates in every panel; only the orange added node/edges change. Blue outlines denote nonsignificant motifs and red outlines denote significant motifs.

The parent-to-child alignment is a cell-label-preserving graph embedding checked
against the recorded `changed_node_or_edge` transition. The galleries preserve
motif adjacency and triangular faces. Original spatial coordinates are absent,
so the fixed parent is a compact planar embedding rather than a reconstruction
of the physical tissue Delaunay coordinates.

This is a descriptive conditional comparison, not a causal analysis. Children
within and across parent groups are not statistically independent because the
same TRI-5 motif may have multiple parents and motifs belong to the same lineage.
""",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.min_per_class < 1 or args.top_criteria < 1 or args.panels_per_page < 1:
        raise ValueError("Minimum group size and plot counts must be positive")
    args.outdir.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(args.features, dtype={"wl_hash": str})
    edges = pd.read_csv(
        args.trace_dir / "trace_edges.csv",
        dtype={"parent_hash": str, "child_hash": str},
    )
    nodes = pd.read_csv(args.trace_dir / "trace_nodes.csv", dtype={"wl_hash": str})
    relationships = prepare_relationships(edges)
    parent_index, analyzable = build_parent_index(relationships, features, args.min_per_class)
    parent_nodes = nodes.loc[nodes.triangle_level.eq(4)].drop_duplicates("wl_hash").set_index("wl_hash")
    parent_index["parent_node_labels"] = parent_index.parent_hash.map(parent_nodes.node_labels)
    parent_index["parent_cell_type_composition"] = parent_index.parent_hash.map(
        parent_nodes.cell_type_composition
    )
    parent_index.to_csv(args.outdir / "parent_group_index.csv", index=False)

    metrics = intrinsic_metrics(features)
    analyzable = analyzable.merge(
        features.drop(columns="significance_group"),
        left_on="child_hash",
        right_on="wl_hash",
        how="left",
        validate="many_to_one",
    )
    child_encodings = (
        nodes.loc[nodes.triangle_level.eq(5), ["wl_hash", "node_labels", "edge_list"]]
        .drop_duplicates("wl_hash")
    )
    analyzable = analyzable.merge(
        child_encodings,
        on="wl_hash",
        how="left",
        validate="many_to_one",
    )
    eligible = parent_index.loc[parent_index.eligible, "parent_hash"].tolist()
    all_comparisons: list[pd.DataFrame] = []
    all_top: list[pd.DataFrame] = []
    figure_index: list[dict[str, object]] = []
    gallery_index: list[dict[str, object]] = []

    for position, parent_hash in enumerate(eligible, 1):
        parent_dir = args.outdir / parent_hash
        parent_dir.mkdir(parents=True, exist_ok=True)
        group = analyzable.loc[analyzable.parent_hash.eq(parent_hash)].copy()
        parent_memberships = relationships.groupby("child_hash")["parent_hash"].nunique()
        group["tri4_parent_count"] = group.child_hash.map(parent_memberships).astype(int)
        group = group.sort_values(["significance_group", "child_hash"])
        child_columns = [
            "parent_hash", "child_hash", "growth_operation", "changed_node_or_edge",
            "tri4_parent_count", "significance_group", "p_value_unadjusted",
            "hazard_ratio_unadjusted", "sample_document_frequency",
            "median_patient_raw_count", "node_labels", "edge_list",
        ]
        group[[column for column in child_columns if column in group]].to_csv(
            parent_dir / "child_motifs.csv", index=False
        )
        parent_row = parent_index.set_index("parent_hash").loc[parent_hash]
        gallery_index.extend(
            plot_child_motif_gallery(
                group,
                analysis.base.graph_from_encoding(
                    str(parent_nodes.loc[parent_hash].node_labels),
                    str(parent_nodes.loc[parent_hash].edge_list),
                    parent_hash,
                ),
                parent_hash,
                str(parent_row.parent_cell_type_composition),
                parent_dir / "tri5_child_motif_gallery.png",
            )
        )
        distributions, comparisons = compare_parent(group, metrics, args.min_per_class)
        distributions.to_csv(
            parent_dir / "criteria_distributions.csv.gz", index=False, compression="gzip"
        )
        comparisons.to_csv(parent_dir / "criteria_comparisons.csv", index=False)
        top = comparisons.head(args.top_criteria).copy()
        top.to_csv(parent_dir / "top_criteria.csv", index=False)
        if not comparisons.empty:
            comparisons.insert(0, "parent_hash", parent_hash)
            all_comparisons.append(comparisons)
        if not top.empty:
            top.insert(0, "parent_hash", parent_hash)
            all_top.append(top)
            description = str(parent_row.parent_cell_type_composition)
            figure_index.extend(
                plot_top_criteria(
                    group, top, parent_hash, description, parent_dir, args.panels_per_page
                )
            )
        if position % 10 == 0 or position == len(eligible):
            print(f"completed {position}/{len(eligible)} eligible parents", flush=True)

    combined = pd.concat(all_comparisons, ignore_index=True) if all_comparisons else pd.DataFrame()
    selected = pd.concat(all_top, ignore_index=True) if all_top else pd.DataFrame()
    combined.to_csv(
        args.outdir / "all_parent_criteria_comparisons.csv.gz",
        index=False,
        compression="gzip",
    )
    q_matrix, metric_metadata, parent_metadata = build_location_q_matrix(combined)
    heatmap_summary = plot_clustered_location_q_heatmap(
        q_matrix, metric_metadata, parent_metadata, args.outdir
    )
    _, parent_order, _, _ = clustered_matrix_order(q_matrix)
    heatmap_summary.update(
        plot_clustered_tri4_parent_gallery(
            parent_order, args.trace_dir, args.outdir
        )
    )
    recurrence = build_recurrence(combined, selected)
    recurrence.to_csv(args.outdir / "top_criteria_recurrence.csv", index=False)
    pd.DataFrame(figure_index).to_csv(args.outdir / "figure_index.csv", index=False)
    pd.DataFrame(gallery_index).to_csv(args.outdir / "motif_gallery_index.csv", index=False)
    summary: dict[str, object] = {
        "status": "complete",
        "trace_parents": int(len(parent_index)),
        "analyzable_parents": int((parent_index.analyzable_total_children > 0).sum()),
        "mixed_parents": int(parent_index.mixed_significance_group.sum()),
        "eligible_parents": int(parent_index.eligible.sum()),
        "minimum_per_class": args.min_per_class,
        "candidate_criteria": len(metrics),
        "top_criteria_per_parent": args.top_criteria,
        "panels_per_page": args.panels_per_page,
        "parent_criterion_tests": int(len(combined)),
        "figure_panels": int(len(figure_index)),
        "motif_galleries": int(len(eligible)),
        "motif_gallery_panels": int(len(gallery_index)),
    }
    summary.update(heatmap_summary)
    (args.outdir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    write_readme(args.outdir / "README.md", summary, args)
    return summary


def main() -> None:
    args = parse_args()
    if args.motif_galleries_only:
        result = generate_existing_motif_galleries(args)
    elif args.clustered_heatmap_only:
        result = generate_clustered_heatmap_from_existing(args)
    else:
        result = run(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
