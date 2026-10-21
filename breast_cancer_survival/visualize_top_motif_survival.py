#!/usr/bin/env python3
"""Visualize and summarize the top unadjusted Cox motifs for TRI-2–TRI-7."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/breast-cancer-motif-visualization-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy import sparse


RISK_COLOR = "#C44E52"
PROTECTIVE_COLOR = "#4C72B0"
CELL_TYPE_ORDER = [
    "B cells",
    "T and B cells",
    "T cells",
    "endothelial",
    "macrophages",
    "stromal cells",
    "tumor cells",
    "unknown",
]
CELL_TYPE_COLORS = {
    "B cells": "#4C78A8",
    "T and B cells": "#72B7B2",
    "T cells": "#54A24B",
    "endothelial": "#F2B134",
    "macrophages": "#B279A2",
    "stromal cells": "#9D755D",
    "tumor cells": "#E45756",
    "unknown": "#BAB0AC",
}
CELL_TYPE_SHORT = {
    "B cells": "B",
    "T and B cells": "T+B",
    "T cells": "T",
    "endothelial": "Endo",
    "macrophages": "Mφ",
    "stromal cells": "Str",
    "tumor cells": "Tum",
    "unknown": "?",
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=root / "motif_results_tri4_tri7_unadjusted",
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=root.parent
        / "breast_cancer_classification_workstation"
        / "classification_by_level"
        / "mtf_occurrence"
        / "mrmr_selection"
        / "parameter_tuning_bz"
        / "artifacts",
    )
    parser.add_argument(
        "--motif-root",
        type=Path,
        default=root.parent / "breast_cancer_meta" / "BZ_res_threshold_10_cpp_0811",
    )
    parser.add_argument(
        "--label-map",
        type=Path,
        default=root.parent
        / "breast_cancer"
        / "BZ_samples_intra"
        / "node_label_cell_type_mapping.csv",
    )
    parser.add_argument("--levels", type=int, nargs="+", default=list(range(2, 8)))
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument(
        "--nonzero-threshold-only",
        action="store_true",
        help="Select only motifs whose median patient raw-count threshold is nonzero.",
    )
    parser.add_argument("--outdir", type=Path, default=None)
    return parser.parse_args()


def compact_composition(value: str) -> str:
    replacements = {
        "tumor cells": "Tumor",
        "stromal cells": "Stromal",
        "macrophages": "Macrophage",
        "endothelial": "Endothelial",
        "T and B cells": "T+B",
        "T cells": "T",
        "B cells": "B",
        "unknown": "Unknown",
        " x": "×",
    }
    result = value
    for source, target in replacements.items():
        result = result.replace(source, target)
    return result


def decoded_labels(series: pd.Series, label_map: dict[str, str]) -> list[list[str]]:
    return [
        [label_map.get(label, f"label {label}") for label in str(value).split("&")]
        for value in series
    ]


def plot_top_level(frame: pd.DataFrame, level: int, top_n: int) -> plt.Figure:
    top = frame.loc[frame["p_value_unadjusted"].notna()].head(top_n).copy()
    top = top.iloc[::-1].reset_index(drop=True)
    scores = -np.log10(top["p_value_unadjusted"].clip(lower=np.finfo(float).tiny))
    colors = np.where(top["hazard_ratio_unadjusted"].ge(1), RISK_COLOR, PROTECTIVE_COLOR)
    labels = [
        f"{int(row.rank_within_level_unadjusted):02d} | {row.wl_hash[:9]} | "
        f"{compact_composition(row.cell_type_composition)}"
        for row in top.itertuples(index=False)
    ]

    fig, ax = plt.subplots(figsize=(18, 13.5))
    positions = np.arange(len(top))
    ax.barh(positions, scores, color=colors, alpha=0.88, height=0.72)
    ax.set_yticks(positions, labels=labels, fontsize=8)
    ax.axvline(-np.log10(0.05), color="0.25", linestyle="--", linewidth=1)
    ax.set_xlabel("−log10(unadjusted Cox p-value)")
    ax.set_title(
        f"TRI-{level}: top {len(top)} motif survival associations",
        fontsize=16,
        pad=18,
    )
    ax.text(
        0,
        1.012,
        "Red: enriched HR > 1 (higher hazard)   Blue: enriched HR < 1 (lower hazard)   "
        "E n(events) shown at bar end",
        transform=ax.transAxes,
        fontsize=10,
    )
    offset = max(float(scores.max()) * 0.012, 0.05)
    for position, (score, row) in enumerate(zip(scores, top.itertuples(index=False))):
        ax.text(
            score + offset,
            position,
            f"HR {row.hazard_ratio_unadjusted:.2g}; "
            f"E {row.n_enriched_unadjusted}({row.events_enriched_unadjusted})",
            va="center",
            fontsize=7.5,
        )
    ax.set_xlim(0, float(scores.max()) * 1.28)
    ax.grid(axis="x", alpha=0.18)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.subplots_adjust(left=0.40, right=0.97, top=0.91, bottom=0.07)
    return fig


def plot_cell_type_heatmap(summary: pd.DataFrame, levels: list[int]) -> plt.Figure:
    available_types = [
        value for value in CELL_TYPE_ORDER if value in set(summary["cell_type"])
    ]
    pivot = summary.pivot(index="triangle_level", columns="cell_type", values="top30_node_percent")
    values = pivot.reindex(index=levels, columns=available_types).fillna(0).to_numpy()
    fig, ax = plt.subplots(figsize=(12.5, 5.5))
    image = ax.imshow(values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=max(50, values.max()))
    ax.set_xticks(np.arange(len(available_types)), labels=available_types, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(levels)), labels=[f"TRI-{level}" for level in levels])
    ax.set_title("Cell-type share among node positions in each level's top 30 motifs", pad=15)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            color = "white" if value > 32 else "black"
            ax.text(column, row, f"{value:.1f}%", ha="center", va="center", color=color, fontsize=9)
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("Node positions (%)")
    fig.tight_layout()
    return fig


def load_topologies(
    top: pd.DataFrame,
    level: int,
    artifacts_root: Path,
    motif_root: Path,
) -> pd.DataFrame:
    """Recover one raw edge-list representative for each selected motif."""
    level_dir = artifacts_root / f"tri_{level}"
    samples = pd.read_csv(level_dir / "samples.csv", usecols=["sample"])
    features = pd.read_csv(
        level_dir / "feature_index.csv.gz", usecols=["feature_index", "wl_hash"]
    ).set_index("wl_hash")
    matrix = sparse.load_npz(level_dir / "motif_counts.npz").tocsc()
    requested: dict[Path, set[str]] = {}
    for motif_hash in top["wl_hash"]:
        feature_index = int(features.loc[motif_hash, "feature_index"])
        start, stop = matrix.indptr[feature_index : feature_index + 2]
        if start == stop:
            raise ValueError(f"No source occurrence for TRI-{level} {motif_hash}")
        sample_row = int(matrix.indices[start])
        path = motif_root / str(samples.iloc[sample_row]["sample"]) / f"ccindex_tri_{level}.csv"
        requested.setdefault(path, set()).add(motif_hash)
    rows: list[pd.DataFrame] = []
    for path, hashes in requested.items():
        frame = pd.read_csv(
            path,
            usecols=["wl_hash", "node_labels", "edge_list", "nodes", "edges"],
            dtype={"wl_hash": str, "node_labels": str, "edge_list": str},
        )
        rows.append(frame.loc[frame["wl_hash"].isin(hashes)])
    topology = pd.concat(rows, ignore_index=True).drop_duplicates("wl_hash").set_index("wl_hash")
    missing = sorted(set(top["wl_hash"]) - set(topology.index))
    if missing:
        raise ValueError(f"Missing raw topology for TRI-{level}: {missing[:3]}")
    result = top.merge(topology.reset_index(), on="wl_hash", how="left", validate="one_to_one")
    mismatch = result["representative_node_labels"].astype(str).ne(result["node_labels"].astype(str))
    if mismatch.any():
        raise ValueError(f"Representative node-label mismatch in TRI-{level}")
    return result


def motif_graph(node_labels: str, edge_list: str) -> nx.Graph:
    graph = nx.Graph()
    labels = str(node_labels).split("&")
    graph.add_nodes_from(range(len(labels)))
    for value in str(edge_list).split(";"):
        left, right = value.split("-")
        graph.add_edge(int(left), int(right))
    return graph


def plot_motif_gallery(
    topologies: pd.DataFrame,
    level: int,
    label_map: dict[str, str],
) -> plt.Figure:
    columns = 5
    rows = math.ceil(len(topologies) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(20, 4.0 * rows))
    axes = np.asarray(axes).reshape(-1)
    for ax, row in zip(axes, topologies.itertuples(index=False)):
        graph = motif_graph(row.node_labels, row.edge_list)
        numeric_labels = str(row.node_labels).split("&")
        cell_types = [label_map.get(value, f"label {value}") for value in numeric_labels]
        seed = int(row.wl_hash[:8], 16)
        positions = nx.spring_layout(graph, seed=seed, k=1.0 / np.sqrt(len(graph)))
        nx.draw_networkx_edges(graph, positions, ax=ax, width=1.5, edge_color="#666666")
        nx.draw_networkx_nodes(
            graph,
            positions,
            ax=ax,
            node_color=[CELL_TYPE_COLORS.get(value, "#BAB0AC") for value in cell_types],
            node_size=620,
            edgecolors="white",
            linewidths=1.5,
        )
        nx.draw_networkx_labels(
            graph,
            positions,
            labels={i: CELL_TYPE_SHORT.get(value, value[:4]) for i, value in enumerate(cell_types)},
            ax=ax,
            font_size=8,
            font_weight="bold",
        )
        ax.set_title(
            f"#{int(row.top_rank)} {row.wl_hash[:8]}\n"
            f"p={row.p_value_unadjusted:.2g}; HR={row.hazard_ratio_unadjusted:.2g}; "
            f"E={row.n_enriched_unadjusted}({row.events_enriched_unadjusted})",
            fontsize=9,
        )
        ax.set_axis_off()
    for ax in axes[len(topologies) :]:
        ax.set_axis_off()
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
        if value in {item for labels in decoded_labels(topologies["node_labels"], label_map) for item in labels}
    ]
    fig.legend(handles=handles, loc="lower center", ncol=min(7, len(handles)), frameon=False)
    fig.suptitle(
        f"TRI-{level}: network structures of the top {len(topologies)} Cox motifs",
        fontsize=17,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    return fig


def summarize_level(
    frame: pd.DataFrame,
    level: int,
    top_n: int,
    label_map: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    estimable = frame.loc[frame["p_value_unadjusted"].notna()].copy()
    top = estimable.head(top_n).copy()
    top["top_rank"] = np.arange(1, len(top) + 1)
    top_labels = decoded_labels(top["representative_node_labels"], label_map)
    background_labels = decoded_labels(estimable["representative_node_labels"], label_map)
    top_nodes = Counter(value for labels in top_labels for value in labels)
    background_nodes = Counter(value for labels in background_labels for value in labels)
    top_presence = Counter(value for labels in top_labels for value in set(labels))
    background_presence = Counter(value for labels in background_labels for value in set(labels))
    cell_types = [value for value in CELL_TYPE_ORDER if value in top_nodes or value in background_nodes]
    top_total_nodes = sum(top_nodes.values())
    background_total_nodes = sum(background_nodes.values())
    rows = []
    for cell_type in cell_types:
        top_share = 100 * top_nodes[cell_type] / top_total_nodes
        background_share = 100 * background_nodes[cell_type] / background_total_nodes
        rows.append(
            {
                "triangle_level": level,
                "cell_type": cell_type,
                "top30_motifs_containing": top_presence[cell_type],
                "top30_motifs_containing_percent": 100 * top_presence[cell_type] / len(top),
                "top30_node_count": top_nodes[cell_type],
                "top30_node_percent": top_share,
                "estimable_motifs_containing": background_presence[cell_type],
                "estimable_motifs_containing_percent": 100
                * background_presence[cell_type]
                / len(estimable),
                "estimable_node_percent": background_share,
                "top30_vs_estimable_node_enrichment": top_share / background_share
                if background_share
                else np.nan,
            }
        )
    pattern = {
        "triangle_level": level,
        "top_motifs": len(top),
        "higher_hazard_motifs": int(top["hazard_ratio_unadjusted"].gt(1).sum()),
        "lower_hazard_motifs": int(top["hazard_ratio_unadjusted"].lt(1).sum()),
        "top_motifs_with_enriched_n_le_5": int(top["n_enriched_unadjusted"].le(5).sum()),
        "median_enriched_n": float(top["n_enriched_unadjusted"].median()),
        "minimum_p_value": float(top["p_value_unadjusted"].min()),
    }
    return top, pd.DataFrame(rows), pattern


def write_summary_report(
    path: Path,
    cell_summary: pd.DataFrame,
    patterns: list[dict[str, object]],
    top_all: pd.DataFrame,
) -> None:
    total_risk = int(top_all["hazard_ratio_unadjusted"].gt(1).sum())
    total_small = int(top_all["n_enriched_unadjusted"].le(5).sum())
    lines = [
        "# Cell-type patterns in the top 30 motifs per TRI level",
        "",
        "## Cross-level overview",
        "",
        f"Across the {len(top_all)} displayed motifs, {total_risk} have HR > 1 and "
        f"{len(top_all)-total_risk} have HR < 1. {total_small} have five or fewer "
        "enriched patients, so the strongest raw associations are often rare and unstable.",
        "",
    ]
    for pattern in patterns:
        level = int(pattern["triangle_level"])
        subset = cell_summary.loc[cell_summary["triangle_level"].eq(level)].nlargest(
            3, "top30_node_percent"
        )
        types = ", ".join(
            f"{row.cell_type} ({row.top30_node_percent:.1f}%; "
            f"{row.top30_vs_estimable_node_enrichment:.2f}× background)"
            for row in subset.itertuples(index=False)
        )
        lines.extend(
            [
                f"## TRI-{level}",
                "",
                f"Leading cell types by node share: {types}. "
                f"Risk-direction motifs: {pattern['higher_hazard_motifs']}/30; "
                f"enriched n ≤ 5: {pattern['top_motifs_with_enriched_n_le_5']}/30; "
                f"median enriched n: {pattern['median_enriched_n']:.1f}.",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            "Cell-type percentages count node positions in the representative motif, not "
            "patients or cells in the cohort. The enrichment ratio compares the top 30 "
            "with all finite Cox motifs at the same TRI level. Motifs are strongly "
            "correlated and many top motifs share the same few enriched patients; therefore "
            "repeated cell-type compositions are descriptive signals, not independent "
            "biological discoveries.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    levels = sorted(set(args.levels))
    if args.top_n <= 0:
        raise ValueError("--top-n must be positive")
    outdir = args.outdir or args.results_dir / "top30_visualizations"
    outdir.mkdir(parents=True, exist_ok=True)
    label_frame = pd.read_csv(args.label_map, dtype={"node_label": str})
    label_map = dict(label_frame[["node_label", "cell_type"]].itertuples(index=False, name=None))
    required = [
        "triangle_level",
        "wl_hash",
        "representative_node_labels",
        "cell_type_composition",
        "rank_within_level_unadjusted",
        "median_patient_raw_count",
        "n_enriched_unadjusted",
        "events_enriched_unadjusted",
        "n_non_enriched_unadjusted",
        "events_non_enriched_unadjusted",
        "hazard_ratio_unadjusted",
        "ci95_low_unadjusted",
        "ci95_high_unadjusted",
        "p_value_unadjusted",
        "q_value_bh_global_unadjusted",
        "fit_status_unadjusted",
    ]
    all_top: list[pd.DataFrame] = []
    all_cell_summaries: list[pd.DataFrame] = []
    patterns: list[dict[str, object]] = []
    pdf_path = outdir / "top30_motifs_all_levels.pdf"
    graph_pdf_path = outdir / "top30_motif_graphs_all_levels.pdf"
    topology_tables: list[pd.DataFrame] = []
    with PdfPages(pdf_path) as pdf, PdfPages(graph_pdf_path) as graph_pdf:
        for level in levels:
            path = args.results_dir / f"tri_{level}_cox_results_sorted.csv"
            print(f"Reading TRI-{level}", flush=True)
            frame = pd.read_csv(path, usecols=required)
            if args.nonzero_threshold_only:
                frame = frame.loc[
                    frame["median_patient_raw_count"].notna()
                    & frame["median_patient_raw_count"].ne(0)
                ].copy()
                if len(frame.loc[frame["p_value_unadjusted"].notna()]) < args.top_n:
                    raise ValueError(
                        f"TRI-{level} has fewer than {args.top_n} estimable motifs "
                        "with nonzero thresholds"
                    )
            top, cell_summary, pattern = summarize_level(frame, level, args.top_n, label_map)
            all_top.append(top)
            all_cell_summaries.append(cell_summary)
            patterns.append(pattern)
            fig = plot_top_level(frame, level, args.top_n)
            fig.savefig(outdir / f"tri_{level}_top30_motifs.png", dpi=200)
            pdf.savefig(fig)
            plt.close(fig)
            topology = load_topologies(
                top, level, args.artifacts_root, args.motif_root
            )
            topology_tables.append(topology)
            gallery = plot_motif_gallery(topology, level, label_map)
            gallery.savefig(outdir / f"tri_{level}_top30_motif_graphs.png", dpi=200)
            graph_pdf.savefig(gallery)
            plt.close(gallery)

    top_all = pd.concat(all_top, ignore_index=True)
    cell_summary = pd.concat(all_cell_summaries, ignore_index=True)
    top_all.to_csv(outdir / "top30_motifs_by_level.csv", index=False)
    pd.concat(topology_tables, ignore_index=True).to_csv(
        outdir / "top30_motif_topologies.csv", index=False
    )
    cell_summary.to_csv(outdir / "top30_cell_type_summary.csv", index=False)
    pd.DataFrame(patterns).to_csv(outdir / "top30_level_summary.csv", index=False)
    heatmap = plot_cell_type_heatmap(cell_summary, levels)
    heatmap.savefig(outdir / "top30_cell_type_heatmap.png", dpi=220)
    heatmap.savefig(outdir / "top30_cell_type_heatmap.pdf")
    plt.close(heatmap)
    write_summary_report(outdir / "REPORT.md", cell_summary, patterns, top_all)
    with (outdir / "visualization_summary.json").open("w") as handle:
        json.dump(
            {
                "levels": levels,
                "top_n": args.top_n,
                "nonzero_threshold_only": args.nonzero_threshold_only,
                "displayed_motifs": int(len(top_all)),
                "higher_hazard_motifs": int(top_all["hazard_ratio_unadjusted"].gt(1).sum()),
                "lower_hazard_motifs": int(top_all["hazard_ratio_unadjusted"].lt(1).sum()),
                "top_motifs_with_enriched_n_le_5": int(
                    top_all["n_enriched_unadjusted"].le(5).sum()
                ),
            },
            handle,
            indent=2,
        )
    print(f"Wrote visualizations to {outdir}", flush=True)


if __name__ == "__main__":
    main()
