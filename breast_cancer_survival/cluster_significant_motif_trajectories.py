#!/usr/bin/env python3
"""Cluster complete significant-motif lineages by branch-aware p-value shape."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/breast-cancer-trajectory-clustering-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from trace_significant_motifs import graph_from_encoding


LEVELS = list(range(1, 8))
CELL_TYPE_NAMES = {
    "0": "B cells",
    "1": "T and B cells",
    "2": "T cells",
    "3": "endothelial",
    "4": "macrophages",
    "5": "stromal cells",
    "6": "tumor cells",
    "7": "unknown",
}


def parse_args() -> argparse.Namespace:
    trace_root = Path(__file__).resolve().parent / "motif_results" / "significant_mtf_trace"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", type=Path, default=trace_root)
    parser.add_argument(
        "--outdir", type=Path, default=trace_root / "trajectory_clustering"
    )
    parser.add_argument("--maximum-clusters", type=int, default=10)
    parser.add_argument("--trend-tolerance", type=float, default=0.1)
    return parser.parse_args()


def target_descriptors(row: pd.Series) -> dict[str, object]:
    graph = graph_from_encoding(row.node_labels, row.edge_list, row.wl_hash)
    degrees = np.asarray([degree for _, degree in graph.degree()], dtype=float)
    triangle_participation = np.asarray(list(nx.triangles(graph).values()), dtype=float)
    labels = nx.get_node_attributes(graph, "label")
    result: dict[str, object] = {
        "target_nodes": graph.number_of_nodes(),
        "target_edges": graph.number_of_edges(),
        "target_density": nx.density(graph),
        "target_mean_degree": float(degrees.mean()),
        "target_max_degree": float(degrees.max()),
        "target_degree_sd": float(degrees.std()),
        "target_articulation_points": len(list(nx.articulation_points(graph))),
        "target_mean_triangle_participation": float(triangle_participation.mean()),
        "target_node_labels": row.node_labels,
        "target_cell_type_composition": row.cell_type_composition,
    }
    for label in map(str, range(8)):
        nodes = [node for node, value in labels.items() if value == label]
        result[f"label_{label}_nodes"] = len(nodes)
        result[f"label_{label}_mean_degree"] = (
            float(np.mean([graph.degree(node) for node in nodes])) if nodes else np.nan
        )
    return result


def summarize_target(nodes: pd.DataFrame, metadata: dict[str, object]) -> dict[str, object]:
    result: dict[str, object] = {
        "motif_id": metadata["motif_id"],
        "target_level": int(metadata["target_level"]),
        "target_hash": metadata["target_hash"],
        "target_p_value_unadjusted": metadata["target_p_value_unadjusted"],
    }
    present_levels: list[int] = []
    for level in LEVELS:
        subset = nodes.loc[
            nodes["triangle_level"].eq(level)
            & np.isfinite(nodes["p_value_unadjusted"])
            & nodes["p_value_unadjusted"].gt(0)
        ]
        if subset.empty:
            for name in ("median", "q25", "q75", "iqr", "significant_fraction"):
                result[f"level_{level}_{name}_minus_log10_p"] = np.nan
            result[f"level_{level}_relative_motifs"] = 0
            continue
        present_levels.append(level)
        values = -np.log10(subset["p_value_unadjusted"].to_numpy(float))
        q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75])
        result[f"level_{level}_median_minus_log10_p"] = median
        result[f"level_{level}_q25_minus_log10_p"] = q25
        result[f"level_{level}_q75_minus_log10_p"] = q75
        result[f"level_{level}_iqr_minus_log10_p"] = q75 - q25
        result[f"level_{level}_significant_fraction_minus_log10_p"] = float(
            subset["p_value_unadjusted"].lt(0.05).mean()
        )
        result[f"level_{level}_relative_motifs"] = len(subset)
    result["lineage_status"] = (
        "complete_tri1_tri7" if present_levels == LEVELS else "incomplete_lineage"
    )
    target = nodes.loc[
        nodes["triangle_level"].eq(metadata["target_level"])
        & nodes["wl_hash"].eq(metadata["target_hash"])
    ]
    if len(target) != 1:
        raise ValueError(f"Could not uniquely recover target {metadata['motif_id']}")
    result.update(target_descriptors(target.iloc[0]))
    return result


def load_trajectory_features(trace_root: Path) -> pd.DataFrame:
    rows = []
    for level in LEVELS:
        for metadata_path in sorted((trace_root / f"tri{level}").glob("*/trace_metadata.json")):
            metadata = json.loads(metadata_path.read_text())
            if metadata.get("status") != "complete":
                continue
            nodes = pd.read_csv(
                metadata_path.parent / "trace_nodes.csv",
                dtype={"wl_hash": str, "node_labels": str, "edge_list": str},
            )
            rows.append(summarize_target(nodes, metadata))
    if not rows:
        raise ValueError(f"No completed traces found under {trace_root}")
    return pd.DataFrame(rows).sort_values(["target_level", "target_hash"]).reset_index(drop=True)


def clustering_matrix(features: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    median_columns = [f"level_{level}_median_minus_log10_p" for level in LEVELS]
    medians = features[median_columns].to_numpy(float)
    centered = medians - medians.mean(axis=1, keepdims=True)
    iqr_columns = [f"level_{level}_iqr_minus_log10_p" for level in LEVELS]
    fraction_columns = [
        f"level_{level}_significant_fraction_minus_log10_p" for level in LEVELS
    ]
    count_columns = [f"level_{level}_relative_motifs" for level in LEVELS]
    raw = np.column_stack(
        [
            centered,
            features[iqr_columns].to_numpy(float),
            features[fraction_columns].to_numpy(float),
            np.log1p(features[count_columns].to_numpy(float)),
        ]
    )
    names = (
        [f"centered_median_level_{level}" for level in LEVELS]
        + iqr_columns
        + fraction_columns
        + [f"log1p_relative_motifs_level_{level}" for level in LEVELS]
    )
    return StandardScaler().fit_transform(raw), names


def choose_clustering(matrix: np.ndarray, maximum_clusters: int) -> tuple[np.ndarray, int, float]:
    if len(matrix) < 3:
        raise ValueError("At least three complete lineages are required for clustering")
    maximum = min(maximum_clusters, len(matrix) - 1)
    if maximum < 2:
        raise ValueError("--maximum-clusters must permit at least two clusters")
    candidates = []
    for count in range(2, maximum + 1):
        labels = AgglomerativeClustering(n_clusters=count, linkage="ward").fit_predict(matrix)
        score = silhouette_score(matrix, labels)
        candidates.append((score, -count, labels))
    score, negative_count, labels = max(candidates, key=lambda value: (value[0], value[1]))
    return labels.astype(int) + 1, -negative_count, float(score)


def trend_label(values: np.ndarray, tolerance: float = 0.1) -> str:
    differences = np.diff(np.asarray(values, dtype=float))
    signs = [1 if value > tolerance else -1 if value < -tolerance else 0 for value in differences]
    nonzero = [value for value in signs if value]
    if not nonzero:
        return "flat"
    runs = [nonzero[0]]
    for value in nonzero[1:]:
        if value != runs[-1]:
            runs.append(value)
    if runs == [1]:
        return "increasing_significance"
    if runs == [-1]:
        return "decreasing_significance"
    if runs == [1, -1]:
        return "increasing_then_decreasing"
    if runs == [-1, 1]:
        return "decreasing_then_increasing"
    return "mixed"


def plot_cluster_centroids(complete: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))
    columns = [f"level_{level}_median_minus_log10_p" for level in LEVELS]
    for row in summary.itertuples(index=False):
        subset = complete.loc[complete["cluster_id"].eq(row.cluster_id), columns]
        center = subset.median(axis=0).to_numpy(float)
        q25 = subset.quantile(0.25).to_numpy(float)
        q75 = subset.quantile(0.75).to_numpy(float)
        label = f"Cluster {row.cluster_id}: {row.trend_label} (n={row.motifs})"
        ax.plot(LEVELS, center, marker="o", linewidth=2, label=label)
        ax.fill_between(LEVELS, q25, q75, alpha=0.12)
    ax.axhline(-np.log10(0.05), color="black", linestyle="--", linewidth=1.2, label="Raw p = 0.05")
    ax.set_xticks(LEVELS)
    ax.set_xlabel("Triangle level")
    ax.set_ylabel("Median lineage −log10(raw Cox p)")
    ax.set_title("Branch-aware significant-motif trajectory clusters")
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_trajectory_heatmap(complete: pd.DataFrame, path: Path) -> None:
    columns = [f"level_{level}_median_minus_log10_p" for level in LEVELS]
    ordered = complete.sort_values(["cluster_id", "target_level", "target_hash"])
    values = ordered[columns].to_numpy(float)
    fig, ax = plt.subplots(figsize=(10, max(6, min(30, len(ordered) * 0.025))))
    image = ax.imshow(values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(LEVELS)), [f"TRI-{level}" for level in LEVELS])
    ax.set_xlabel("Triangle level")
    ax.set_ylabel("Focal motifs ordered by cluster")
    ax.set_title("Median branch-aware Cox significance trajectories")
    fig.colorbar(image, ax=ax, label="−log10(raw Cox p)")
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.maximum_clusters < 2 or args.trend_tolerance < 0:
        raise ValueError("Invalid clustering parameters")
    features = load_trajectory_features(args.trace_root)
    complete = features.loc[features["lineage_status"].eq("complete_tri1_tri7")].copy()
    incomplete = features.loc[~features.index.isin(complete.index)].copy()
    matrix, matrix_columns = clustering_matrix(complete)
    labels, cluster_count, silhouette = choose_clustering(matrix, args.maximum_clusters)
    complete["cluster_id"] = labels

    summary_rows = []
    median_columns = [f"level_{level}_median_minus_log10_p" for level in LEVELS]
    for cluster_id, subset in complete.groupby("cluster_id", sort=True):
        centroid = subset[median_columns].median(axis=0).to_numpy(float)
        compositions = Counter(subset["target_node_labels"])
        cell_type_compositions = Counter(subset["target_cell_type_composition"])
        summary_rows.append(
            {
                "cluster_id": int(cluster_id),
                "motifs": int(len(subset)),
                "trend_label": trend_label(centroid, args.trend_tolerance),
                "most_common_target_label_sequence": compositions.most_common(1)[0][0],
                "most_common_cell_type_composition": cell_type_compositions.most_common(1)[0][0],
                "mean_target_nodes": subset["target_nodes"].mean(),
                "mean_target_edges": subset["target_edges"].mean(),
                "mean_target_density": subset["target_density"].mean(),
                "mean_target_degree": subset["target_mean_degree"].mean(),
                "mean_target_articulation_points": subset["target_articulation_points"].mean(),
                **{f"centroid_level_{level}_minus_log10_p": centroid[level - 1] for level in LEVELS},
            }
        )
    summary = pd.DataFrame(summary_rows)
    cell_type_rows = []
    for cluster_id, subset in complete.groupby("cluster_id", sort=True):
        for label, cell_type in CELL_TYPE_NAMES.items():
            counts = subset[f"label_{label}_nodes"]
            cell_type_rows.append(
                {
                    "cluster_id": int(cluster_id),
                    "node_label": label,
                    "cell_type": cell_type,
                    "mean_nodes_per_focal_motif": float(counts.mean()),
                    "motifs_containing_cell_type": int(counts.gt(0).sum()),
                    "fraction_motifs_containing_cell_type": float(counts.gt(0).mean()),
                }
            )
    cell_type_summary = pd.DataFrame(cell_type_rows)
    assignments = features.merge(
        complete[["motif_id", "cluster_id"]], on="motif_id", how="left", validate="one_to_one"
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.outdir / "trajectory_features.csv", index=False)
    assignments.to_csv(args.outdir / "cluster_assignments.csv", index=False)
    summary.to_csv(args.outdir / "cluster_summary.csv", index=False)
    cell_type_summary.to_csv(args.outdir / "cluster_cell_type_summary.csv", index=False)
    incomplete.to_csv(args.outdir / "incomplete_lineages.csv", index=False)
    pd.DataFrame({"clustering_feature": matrix_columns}).to_csv(
        args.outdir / "clustering_feature_definition.csv", index=False
    )
    plot_cluster_centroids(complete, summary, args.outdir / "cluster_centroid_curves.png")
    plot_trajectory_heatmap(complete, args.outdir / "trajectory_heatmap.png")
    audit = {
        "completed_traces": int(len(features)),
        "complete_tri1_tri7_lineages": int(len(complete)),
        "incomplete_lineages": int(len(incomplete)),
        "clusters": cluster_count,
        "silhouette_score": silhouette,
        "linkage": "ward",
        "candidate_cluster_counts": [2, min(args.maximum_clusters, len(complete) - 1)],
        "trajectory_scale": "minus_log10 raw Cox p; median trajectory centered per motif",
        "increasing_definition": "increasing minus_log10(p), equivalent to decreasing p",
    }
    (args.outdir / "clustering_summary.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
