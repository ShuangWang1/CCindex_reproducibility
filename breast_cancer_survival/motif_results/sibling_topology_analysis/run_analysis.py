#!/usr/bin/env python3
"""Compare labeled motif topology for significant and nonsignificant lineage peers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
SURVIVAL_DIR = PROJECT_DIR.parents[1]
if str(SURVIVAL_DIR) not in sys.path:
    sys.path.insert(0, str(SURVIVAL_DIR))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/breast-cancer-sibling-topology-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy
import statsmodels
from scipy.stats import fisher_exact, mannwhitneyu, spearmanr

from trace_significant_motifs import graph_from_encoding, motif_hash, triangle_count


TARGET_HASH = "1fded5000748985170d30bdca433a111"
FEATURE_CACHE_VERSION = 1
LABEL_NAMES = {
    "0": "B cells",
    "1": "T and B cells",
    "2": "T cells",
    "3": "endothelial",
    "4": "macrophages",
    "5": "stromal cells",
    "6": "tumor cells",
    "7": "unknown",
}
LABEL_COLORS = {
    "0": "#4C78A8",
    "1": "#72B7B2",
    "2": "#54A24B",
    "3": "#ECA82C",
    "4": "#E45756",
    "5": "#B279A2",
    "6": "#7A5195",
    "7": "#9D9D9D",
}
REQUIRED_COLUMNS = {
    "triangle_level",
    "wl_hash",
    "node_labels",
    "edge_list",
    "topology_nodes",
    "topology_edges",
    "sample_document_frequency",
    "median_patient_raw_count",
    "p_value_unadjusted",
    "coef_unadjusted",
    "hazard_ratio_unadjusted",
    "fit_status_unadjusted",
    "n_enriched_unadjusted",
    "n_non_enriched_unadjusted",
}


def parse_args() -> argparse.Namespace:
    trace = (
        PROJECT_DIR.parent
        / "significant_mtf_trace"
        / "tri1"
        / TARGET_HASH
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, default=trace)
    parser.add_argument("--outdir", type=Path, default=PROJECT_DIR / "results")
    parser.add_argument("--primary-level", type=int, default=5)
    parser.add_argument("--p-threshold", type=float, default=0.05)
    parser.add_argument("--min-class-size", type=int, default=20)
    parser.add_argument("--min-feature-prevalence", type=float, default=0.01)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--layout-seed", type=int, default=20260914)
    parser.add_argument("--rebuild-features", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def degree_gini(values: list[float]) -> float:
    array = np.sort(np.asarray(values, dtype=float))
    if not len(array) or array.sum() == 0:
        return 0.0
    indices = np.arange(1, len(array) + 1)
    return float(
        2 * np.sum(indices * array) / (len(array) * array.sum())
        - (len(array) + 1) / len(array)
    )


def entropy(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    array = array[array > 0]
    if not len(array):
        return 0.0
    probabilities = array / array.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))


def safe_stat(values: list[float], operation: str) -> float:
    if not values:
        return 0.0
    array = np.asarray(values, dtype=float)
    if operation == "mean":
        return float(array.mean())
    if operation == "max":
        return float(array.max())
    if operation == "sd":
        return float(array.std(ddof=0))
    if operation == "gini":
        return degree_gini(array.tolist())
    raise ValueError(operation)


def all_triangles(graph: nx.Graph) -> list[tuple[int, int, int]]:
    result: list[tuple[int, int, int]] = []
    for left in graph.nodes:
        for middle, right in combinations(sorted(graph.neighbors(left)), 2):
            if graph.has_edge(middle, right) and left < middle < right:
                result.append((left, middle, right))
    return result


def validate_graph(row: pd.Series, graph: nx.Graph) -> None:
    level = int(row["triangle_level"])
    if motif_hash(graph) != str(row["wl_hash"]):
        raise ValueError(f"WL hash mismatch for {row['wl_hash']}")
    if not nx.is_connected(graph):
        raise ValueError(f"Disconnected topology for {row['wl_hash']}")
    if graph.number_of_nodes() != int(row["topology_nodes"]):
        raise ValueError(f"Node-count mismatch for {row['wl_hash']}")
    if graph.number_of_edges() != int(row["topology_edges"]):
        raise ValueError(f"Edge-count mismatch for {row['wl_hash']}")
    if triangle_count(graph) != level:
        raise ValueError(f"Triangle-level mismatch for {row['wl_hash']}")
    labels = {str(data.get("label")) for _, data in graph.nodes(data=True)}
    if not labels.issubset(LABEL_NAMES):
        raise ValueError(f"Unknown node label for {row['wl_hash']}: {sorted(labels)}")


def triangle_overlap_features(triangles: list[tuple[int, int, int]]) -> dict[str, float]:
    overlap = nx.Graph()
    overlap.add_nodes_from(range(len(triangles)))
    share_edge = share_node = disjoint = 0
    for left, right in combinations(range(len(triangles)), 2):
        common = len(set(triangles[left]).intersection(triangles[right]))
        if common == 2:
            share_edge += 1
            overlap.add_edge(left, right)
        elif common == 1:
            share_node += 1
        else:
            disjoint += 1
    components = list(nx.connected_components(overlap)) if len(overlap) else []
    return {
        "triangle_pairs_share_edge": float(share_edge),
        "triangle_pairs_share_node_only": float(share_node),
        "triangle_pairs_disjoint": float(disjoint),
        "triangle_edge_overlap_components": float(len(components)),
        "triangle_largest_edge_overlap_component": float(
            max((len(value) for value in components), default=0)
        ),
    }


def graph_features(graph: nx.Graph) -> dict[str, float]:
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    degrees = dict(graph.degree())
    degree_values = list(degrees.values())
    maximum_degree = max(degree_values)
    degree_denominator = (n - 1) * (n - 2)
    triangles_by_node = nx.triangles(graph)
    triangle_values = list(triangles_by_node.values())
    triangles = all_triangles(graph)
    clustering = nx.clustering(graph)
    betweenness = nx.betweenness_centrality(graph, normalized=True)
    edge_support = [
        len(set(graph.neighbors(left)).intersection(graph.neighbors(right)))
        for left, right in graph.edges
    ]
    triangle_nodes = {node for value in triangles for node in value}
    triangle_edges = {
        tuple(sorted(edge))
        for value in triangles
        for edge in combinations(value, 2)
    }
    adjacency = nx.to_numpy_array(graph, nodelist=sorted(graph.nodes))
    adjacency_eigenvalues = np.linalg.eigvalsh(adjacency)
    laplacian_eigenvalues = np.sort(nx.laplacian_spectrum(graph))
    cliques = list(nx.find_cliques(graph))
    core = nx.core_number(graph)
    result: dict[str, float] = {
        "graph_nodes": float(n),
        "graph_edges": float(m),
        "graph_density": float(nx.density(graph)),
        "graph_cycle_rank": float(m - n + 1),
        "degree_mean": safe_stat(degree_values, "mean"),
        "degree_max": safe_stat(degree_values, "max"),
        "degree_sd": safe_stat(degree_values, "sd"),
        "degree_gini": safe_stat(degree_values, "gini"),
        "degree_centralization": float(
            sum(maximum_degree - value for value in degree_values) / degree_denominator
        ) if degree_denominator else 0.0,
        "degree_fraction_two_or_less": float(np.mean(np.asarray(degree_values) <= 2)),
        "degree_fraction_four_or_more": float(np.mean(np.asarray(degree_values) >= 4)),
        "connectivity_node": float(nx.node_connectivity(graph)),
        "connectivity_edge": float(nx.edge_connectivity(graph)),
        "connectivity_articulation_points": float(len(list(nx.articulation_points(graph)))),
        "connectivity_bridges": float(len(list(nx.bridges(graph)))),
        "connectivity_max_core": float(max(core.values())),
        "distance_diameter": float(nx.diameter(graph)),
        "distance_radius": float(nx.radius(graph)),
        "distance_average_shortest_path": float(nx.average_shortest_path_length(graph)),
        "distance_global_efficiency": float(nx.global_efficiency(graph)),
        "clustering_average": float(nx.average_clustering(graph)),
        "clustering_transitivity": float(nx.transitivity(graph)),
        "clique_max_size": float(max(map(len, cliques))),
        "clique_k4_count": float(sum(1 for value in combinations(graph.nodes, 4) if graph.subgraph(value).number_of_edges() == 6)),
        "spectral_radius": float(np.max(adjacency_eigenvalues)),
        "spectral_algebraic_connectivity": float(laplacian_eigenvalues[1]),
        "spectral_adjacency_gap": float(adjacency_eigenvalues[-1] - adjacency_eigenvalues[-2]),
        "triangle_node_participation_mean": safe_stat(triangle_values, "mean"),
        "triangle_node_participation_max": safe_stat(triangle_values, "max"),
        "triangle_node_participation_sd": safe_stat(triangle_values, "sd"),
        "triangle_node_participation_gini": safe_stat(triangle_values, "gini"),
        "triangle_edge_support_mean": safe_stat(edge_support, "mean"),
        "triangle_edge_support_max": safe_stat(edge_support, "max"),
        "triangle_edge_support_sd": safe_stat(edge_support, "sd"),
        "triangle_fraction_nodes_participating": float(len(triangle_nodes) / n),
        "triangle_fraction_edges_participating": float(len(triangle_edges) / m),
    }
    result.update(triangle_overlap_features(triangles))

    label_counts = Counter(str(graph.nodes[node]["label"]) for node in graph.nodes)
    result["label_entropy"] = entropy(list(label_counts.values()))
    result["label_distinct_count"] = float(len(label_counts))
    total_degree = sum(degree_values)
    total_triangle_participation = sum(triangle_values)
    for label in LABEL_NAMES:
        nodes = [node for node in graph if str(graph.nodes[node]["label"]) == label]
        label_degrees = [degrees[node] for node in nodes]
        label_triangles = [triangles_by_node[node] for node in nodes]
        label_betweenness = [betweenness[node] for node in nodes]
        label_clustering = [clustering[node] for node in nodes]
        prefix = f"label_{label}"
        result[f"{prefix}_present"] = float(bool(nodes))
        result[f"{prefix}_node_count"] = float(len(nodes))
        result[f"{prefix}_node_fraction"] = float(len(nodes) / n)
        result[f"{prefix}_degree_mean"] = safe_stat(label_degrees, "mean")
        result[f"{prefix}_degree_max"] = safe_stat(label_degrees, "max")
        result[f"{prefix}_degree_share"] = float(sum(label_degrees) / total_degree)
        result[f"{prefix}_triangle_mean"] = safe_stat(label_triangles, "mean")
        result[f"{prefix}_triangle_max"] = safe_stat(label_triangles, "max")
        result[f"{prefix}_triangle_share"] = float(
            sum(label_triangles) / total_triangle_participation
        )
        result[f"{prefix}_betweenness_mean"] = safe_stat(label_betweenness, "mean")
        result[f"{prefix}_betweenness_max"] = safe_stat(label_betweenness, "max")
        result[f"{prefix}_clustering_mean"] = safe_stat(label_clustering, "mean")

    edge_counts: Counter[tuple[str, str]] = Counter()
    homotypic = 0
    for left, right in graph.edges:
        pair = tuple(sorted((str(graph.nodes[left]["label"]), str(graph.nodes[right]["label"]))))
        edge_counts[pair] += 1
        homotypic += int(pair[0] == pair[1])
    for left_index, left in enumerate(LABEL_NAMES):
        for right in list(LABEL_NAMES)[left_index:]:
            count = edge_counts[(left, right)]
            result[f"edge_{left}_{right}_count"] = float(count)
            result[f"edge_{left}_{right}_fraction"] = float(count / m)
    result["edge_homotypic_fraction"] = float(homotypic / m)
    result["edge_heterotypic_fraction"] = float(1 - homotypic / m)
    with np.errstate(divide="ignore", invalid="ignore"):
        assortativity = nx.attribute_assortativity_coefficient(graph, "label")
    result["edge_label_assortativity"] = float(assortativity)

    triangle_label_counts: Counter[tuple[str, str, str]] = Counter()
    homogeneous = 0
    distinct_label_counts = []
    for nodes in triangles:
        labels = tuple(sorted(str(graph.nodes[node]["label"]) for node in nodes))
        triangle_label_counts[labels] += 1
        homogeneous += int(len(set(labels)) == 1)
        distinct_label_counts.append(len(set(labels)))
    label_values = list(LABEL_NAMES)
    # combinations-with-replacement without materializing an extra dependency.
    for i in range(len(label_values)):
        for j in range(i, len(label_values)):
            for k in range(j, len(label_values)):
                labels = (label_values[i], label_values[j], label_values[k])
                count = triangle_label_counts[labels]
                result[f"labeled_triangle_{labels[0]}_{labels[1]}_{labels[2]}_count"] = float(count)
                result[f"labeled_triangle_{labels[0]}_{labels[1]}_{labels[2]}_fraction"] = float(count / len(triangles))
    result["labeled_triangle_homogeneous_fraction"] = float(homogeneous / len(triangles))
    result["labeled_triangle_mixed_fraction"] = float(1 - homogeneous / len(triangles))
    result["labeled_triangle_mean_distinct_labels"] = safe_stat(distinct_label_counts, "mean")
    return result


def feature_family(name: str) -> str:
    if name.startswith("label_"):
        return "cell_type_role"
    if name.startswith("edge_"):
        return "labeled_edges"
    if name.startswith("labeled_triangle_"):
        return "labeled_triangles"
    if name.startswith("triangle_"):
        return "triangle_organization"
    if name.startswith("degree_"):
        return "degree_organization"
    if name.startswith("connectivity_") or name.startswith("distance_"):
        return "connectivity_distance"
    if name.startswith("clustering_") or name.startswith("clique_"):
        return "clustering_cliques"
    if name.startswith("spectral_"):
        return "spectral"
    return "graph_organization"


def metric_display_name(name: str) -> str:
    if name.startswith("label_"):
        parts = name.split("_")
        if len(parts) >= 3 and parts[1] in LABEL_NAMES:
            return f"{LABEL_NAMES[parts[1]]}: {' '.join(parts[2:])}"
    if name.startswith("edge_"):
        parts = name.split("_")
        if len(parts) >= 4 and parts[1] in LABEL_NAMES and parts[2] in LABEL_NAMES:
            return (
                f"{LABEL_NAMES[parts[1]]}–{LABEL_NAMES[parts[2]]} edge "
                f"{' '.join(parts[3:])}"
            )
    if name.startswith("labeled_triangle_"):
        parts = name.split("_")
        if len(parts) >= 6 and all(value in LABEL_NAMES for value in parts[2:5]):
            return (
                f"{LABEL_NAMES[parts[2]]}–{LABEL_NAMES[parts[3]]}–"
                f"{LABEL_NAMES[parts[4]]} triangle {' '.join(parts[5:])}"
            )
    return name.replace("_", " ")


def composition_display(value: str) -> str:
    counts = Counter(str(value).split("&"))
    return "; ".join(
        f"{LABEL_NAMES[label]}{' x' + str(count) if count > 1 else ''}"
        for label, count in sorted(counts.items(), key=lambda item: int(item[0]))
    )


def build_metric_dictionary(columns: list[str]) -> pd.DataFrame:
    rows = []
    for name in columns:
        sparse = name.startswith("edge_") and name.endswith(("_count", "_fraction"))
        sparse |= name.startswith("labeled_triangle_") and name.endswith(("_count", "_fraction"))
        rows.append(
            {
                "metric": name,
                "display_name": metric_display_name(name),
                "family": feature_family(name),
                "cell_type_labels": ";".join(
                    f"{label}={cell_type}" for label, cell_type in LABEL_NAMES.items()
                ),
                "sparse_pattern_filter": bool(sparse),
                "higher_value_meaning": name.replace("_", " "),
            }
        )
    return pd.DataFrame(rows)


def benjamini_hochberg(values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.notna()
    if not valid.any():
        return result
    p = values.loc[valid].to_numpy(float)
    order = np.argsort(p, kind="mergesort")
    ranked = p[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.clip(adjusted, 0, 1)
    result.loc[valid] = restored
    return result


def association_for_metric(
    frame: pd.DataFrame,
    metric: str,
    family: str,
    sparse_pattern: bool,
    min_class_size: int,
    min_prevalence: float,
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    subset = frame[[metric, "significant_raw_p"]].replace([np.inf, -np.inf], np.nan).dropna()
    cases = subset.loc[subset["significant_raw_p"], metric].to_numpy(float)
    controls = subset.loc[~subset["significant_raw_p"], metric].to_numpy(float)
    base = {
        "metric": metric,
        "family": family,
        "n_nonmissing": len(subset),
        "n_significant": len(cases),
        "n_nonsignificant": len(controls),
    }
    if min(len(cases), len(controls)) < min_class_size:
        return None, {**base, "reason": "insufficient_class_size"}
    unique = np.unique(subset[metric])
    if len(unique) < 2:
        return None, {**base, "reason": "constant"}
    nonzero_prevalence = float(np.mean(subset[metric].to_numpy(float) != 0))
    if sparse_pattern and not min_prevalence <= nonzero_prevalence <= 1 - min_prevalence:
        return None, {**base, "reason": "sparse_pattern", "nonzero_prevalence": nonzero_prevalence}
    row: dict[str, object] = {
        **base,
        "nonzero_prevalence": nonzero_prevalence,
        "significant_median": float(np.median(cases)),
        "significant_q25": float(np.quantile(cases, 0.25)),
        "significant_q75": float(np.quantile(cases, 0.75)),
        "nonsignificant_median": float(np.median(controls)),
        "nonsignificant_q25": float(np.quantile(controls, 0.25)),
        "nonsignificant_q75": float(np.quantile(controls, 0.75)),
    }
    if set(unique).issubset({0.0, 1.0}):
        table = np.array(
            [
                [(cases == 1).sum(), (cases == 0).sum()],
                [(controls == 1).sum(), (controls == 0).sum()],
            ],
            dtype=int,
        )
        if table.min() < 5:
            return None, {**base, "reason": "binary_cell_below_5", "minimum_cell": int(table.min())}
        odds_ratio, p_value = fisher_exact(table, alternative="two-sided")
        prevalence_difference = float(np.mean(cases) - np.mean(controls))
        row.update(
            {
                "test": "fisher_exact",
                "test_statistic": float(odds_ratio),
                "topology_assoc_p_value": float(p_value),
                "effect_size": prevalence_difference,
                "effect_name": "prevalence_difference",
                "auc_probability_superiority": float(0.5 + prevalence_difference / 2),
                "odds_ratio": float(odds_ratio),
                "significant_prevalence": float(np.mean(cases)),
                "nonsignificant_prevalence": float(np.mean(controls)),
            }
        )
    else:
        test = mannwhitneyu(cases, controls, alternative="two-sided", method="asymptotic")
        auc = float(test.statistic / (len(cases) * len(controls)))
        row.update(
            {
                "test": "mann_whitney_u",
                "test_statistic": float(test.statistic),
                "topology_assoc_p_value": float(test.pvalue),
                "effect_size": float(2 * auc - 1),
                "effect_name": "cliffs_delta",
                "auc_probability_superiority": auc,
                "odds_ratio": np.nan,
                "significant_prevalence": np.nan,
                "nonsignificant_prevalence": np.nan,
            }
        )
    effect = float(row["effect_size"])
    row["effect_direction"] = "higher_in_significant" if effect > 0 else "lower_in_significant" if effect < 0 else "no_difference"
    return row, None


def analyze_levels(
    features: pd.DataFrame,
    dictionary: pd.DataFrame,
    min_class_size: int,
    min_prevalence: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_rows: list[dict[str, object]] = []
    excluded_rows: list[dict[str, object]] = []
    for level, level_frame in features.loc[features["eligible_for_comparison"]].groupby("triangle_level"):
        for definition in dictionary.itertuples(index=False):
            result, excluded = association_for_metric(
                level_frame,
                definition.metric,
                definition.family,
                bool(definition.sparse_pattern_filter),
                min_class_size,
                min_prevalence,
            )
            if result is not None:
                result_rows.append({"triangle_level": int(level), **result})
            if excluded is not None:
                excluded_rows.append({"triangle_level": int(level), **excluded})
    results = pd.DataFrame(result_rows)
    if len(results):
        results["topology_assoc_q_bh"] = results.groupby("triangle_level", group_keys=False)[
            "topology_assoc_p_value"
        ].apply(benjamini_hochberg)
        results = results.sort_values(
            ["triangle_level", "topology_assoc_q_bh", "metric"], kind="mergesort"
        ).reset_index(drop=True)
    return results, pd.DataFrame(excluded_rows)


def composition_enrichment(
    features: pd.DataFrame, primary_level: int, min_prevalence: float
) -> pd.DataFrame:
    frame = features.loc[
        features["eligible_for_comparison"] & features["triangle_level"].eq(primary_level)
    ].copy()
    total = len(frame)
    rows = []
    for composition, group in frame.groupby("label_composition", sort=True):
        present = frame["label_composition"].eq(composition)
        prevalence = float(present.mean())
        if not min_prevalence <= prevalence <= 1 - min_prevalence:
            continue
        significant = frame["significant_raw_p"]
        table = np.array(
            [
                [(present & significant).sum(), (~present & significant).sum()],
                [(present & ~significant).sum(), (~present & ~significant).sum()],
            ]
        )
        odds_ratio, p_value = fisher_exact(table)
        rows.append(
            {
                "triangle_level": primary_level,
                "label_composition": composition,
                "composition_display": composition_display(composition),
                "motifs": len(group),
                "prevalence": prevalence,
                "significant_motifs": int(group["significant_raw_p"].sum()),
                "significant_fraction": float(group["significant_raw_p"].mean()),
                "odds_ratio": float(odds_ratio),
                "topology_assoc_p_value": float(p_value),
                "total_eligible": total,
            }
        )
    result = pd.DataFrame(rows)
    if len(result):
        result["topology_assoc_q_bh"] = benjamini_hochberg(result["topology_assoc_p_value"])
        result = result.sort_values(["topology_assoc_q_bh", "label_composition"]).reset_index(drop=True)
    return result


def redundancy_groups(
    features: pd.DataFrame,
    associations: pd.DataFrame,
    primary_level: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = associations.loc[associations["triangle_level"].eq(primary_level)].copy()
    if primary.empty:
        return pd.DataFrame(), primary
    metrics = primary["metric"].tolist()
    frame = features.loc[
        features["eligible_for_comparison"] & features["triangle_level"].eq(primary_level), metrics
    ]
    correlation = frame.corr(method="spearman").abs()
    graph = nx.Graph()
    graph.add_nodes_from(metrics)
    for left_index, left in enumerate(metrics):
        for right in metrics[left_index + 1 :]:
            value = correlation.loc[left, right]
            if np.isfinite(value) and value >= 0.90:
                graph.add_edge(left, right)
    rows = []
    group_for_metric: dict[str, int] = {}
    for group_id, component in enumerate(sorted(nx.connected_components(graph), key=lambda x: sorted(x)[0]), 1):
        subset = primary.loc[primary["metric"].isin(component)].sort_values(
            ["topology_assoc_q_bh", "effect_size"],
            ascending=[True, False],
            key=lambda column: column.abs() if column.name == "effect_size" else column,
        )
        representative = str(subset.iloc[0]["metric"])
        for metric in sorted(component):
            group_for_metric[metric] = group_id
            rows.append(
                {
                    "redundancy_group": group_id,
                    "metric": metric,
                    "representative_metric": representative,
                    "is_representative": metric == representative,
                    "group_size": len(component),
                }
            )
    primary["redundancy_group"] = primary["metric"].map(group_for_metric)
    group_frame = pd.DataFrame(rows)
    representatives = set(group_frame.loc[group_frame["is_representative"], "metric"])
    primary["is_representative"] = primary["metric"].isin(representatives)
    primary["exploratory_candidate"] = primary["topology_assoc_q_bh"].lt(0.05)
    primary = primary.sort_values(
        ["topology_assoc_q_bh", "effect_size"],
        ascending=[True, False],
        key=lambda column: column.abs() if column.name == "effect_size" else column,
    ).reset_index(drop=True)
    return group_frame, primary


def cross_level_consistency(
    primary: pd.DataFrame, associations: pd.DataFrame, primary_level: int
) -> pd.DataFrame:
    candidates = primary.loc[primary["exploratory_candidate"] & primary["is_representative"]]
    rows = []
    for metric in candidates["metric"]:
        subset = associations.loc[associations["metric"].eq(metric)].set_index("triangle_level")
        primary_effect = float(subset.loc[primary_level, "effect_size"])
        signs = []
        significant_reverse = False
        row: dict[str, object] = {"metric": metric, "primary_effect_size": primary_effect}
        row["display_name"] = str(
            candidates.loc[candidates["metric"].eq(metric), "display_name"].iloc[0]
        )
        for level in [4, 5, 6, 7]:
            if level in subset.index:
                level_row = subset.loc[level]
                effect = float(level_row["effect_size"])
                q_value = float(level_row["topology_assoc_q_bh"])
                row[f"tri{level}_effect_size"] = effect
                row[f"tri{level}_q_bh"] = q_value
                if level != primary_level:
                    signs.append(np.sign(effect) == np.sign(primary_effect))
                    significant_reverse |= q_value < 0.05 and np.sign(effect) != np.sign(primary_effect)
            else:
                row[f"tri{level}_effect_size"] = np.nan
                row[f"tri{level}_q_bh"] = np.nan
        row["secondary_levels_same_direction"] = int(sum(signs))
        row["significant_reverse_secondary"] = bool(significant_reverse)
        row["cross_level_consistent"] = bool(sum(signs) >= 2 and not significant_reverse)
        rows.append(row)
    return pd.DataFrame(rows)


def level_summary(features: pd.DataFrame, p_threshold: float) -> pd.DataFrame:
    rows = []
    for level, frame in features.groupby("triangle_level", sort=True):
        eligible = frame.loc[frame["eligible_for_comparison"]]
        rows.append(
            {
                "triangle_level": int(level),
                "total_lineage_motifs": len(frame),
                "estimable_motifs": len(eligible),
                "excluded_nonestimable": int((~frame["eligible_for_comparison"]).sum()),
                "significant_raw_p": int(eligible["significant_raw_p"].sum()),
                "nonsignificant_raw_p": int((~eligible["significant_raw_p"]).sum()),
                "significant_fraction_estimable": float(eligible["significant_raw_p"].mean()),
                "raw_p_threshold": p_threshold,
                "hazardous_significant": int(
                    (eligible["significant_raw_p"] & eligible["coef_unadjusted"].gt(0)).sum()
                ),
                "protective_significant": int(
                    (eligible["significant_raw_p"] & eligible["coef_unadjusted"].lt(0)).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_level_overview(summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(summary))
    ax.bar(x, summary["estimable_motifs"], color="#B9C6D3", label="Estimable")
    ax.bar(x, summary["significant_raw_p"], color="#D1495B", label="Raw p < 0.05")
    ax.set_yscale("log")
    ax.set_xticks(x, [f"TRI-{level}" for level in summary["triangle_level"]])
    ax.set_ylabel("Motifs (log scale)")
    ax.set_title("Eligible and raw-significant motifs in the example lineage")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=210)
    plt.close(fig)


def plot_forest(primary: pd.DataFrame, path: Path, top_n: int) -> None:
    selected = primary.loc[primary["is_representative"]].head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, max(5, 0.38 * len(selected))))
    colors = np.where(selected["topology_assoc_q_bh"].lt(0.05), "#D1495B", "#7F8C8D")
    ax.barh(range(len(selected)), selected["effect_size"], color=colors)
    ax.set_yticks(range(len(selected)), selected["display_name"], fontsize=8)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Effect size (positive = higher in significant motifs)")
    ax.set_title("TRI-5 labeled-topology associations")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_distributions(
    features: pd.DataFrame, primary: pd.DataFrame, primary_level: int, path: Path
) -> None:
    metrics = primary.loc[primary["is_representative"], "metric"].head(8).tolist()
    frame = features.loc[
        features["eligible_for_comparison"] & features["triangle_level"].eq(primary_level)
    ]
    rows = math.ceil(len(metrics) / 2)
    fig, axes = plt.subplots(rows, 2, figsize=(12, 3.2 * rows), squeeze=False)
    for ax, metric in zip(axes.flat, metrics):
        control = frame.loc[~frame["significant_raw_p"], metric].replace([np.inf, -np.inf], np.nan).dropna()
        case = frame.loc[frame["significant_raw_p"], metric].replace([np.inf, -np.inf], np.nan).dropna()
        ax.boxplot([control, case], tick_labels=["nonsignificant", "significant"], showfliers=False)
        display = str(primary.loc[primary["metric"].eq(metric), "display_name"].iloc[0])
        ax.set_title(display, fontsize=9)
    for ax in axes.flat[len(metrics):]:
        ax.axis("off")
    fig.suptitle("TRI-5 distributions for leading nonredundant metrics")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_correlation(
    features: pd.DataFrame, primary: pd.DataFrame, primary_level: int, path: Path
) -> None:
    metrics = primary.loc[primary["is_representative"], "metric"].head(20).tolist()
    frame = features.loc[
        features["eligible_for_comparison"] & features["triangle_level"].eq(primary_level), metrics
    ]
    values = frame.corr(method="spearman").to_numpy(float)
    fig, ax = plt.subplots(figsize=(11, 9))
    image = ax.imshow(values, vmin=-1, vmax=1, cmap="coolwarm")
    display_map = primary.set_index("metric")["display_name"]
    labels = [display_map[value] for value in metrics]
    ax.set_xticks(range(len(metrics)), labels, rotation=75, ha="right", fontsize=7)
    ax.set_yticks(range(len(metrics)), labels, fontsize=7)
    fig.colorbar(image, ax=ax, label="Spearman correlation")
    ax.set_title("Correlation among leading TRI-5 topology metrics")
    fig.tight_layout()
    fig.savefig(path, dpi=210)
    plt.close(fig)


def plot_cross_level(consistency: pd.DataFrame, path: Path) -> None:
    if consistency.empty:
        return
    columns = [f"tri{level}_effect_size" for level in [4, 5, 6, 7]]
    frame = consistency.head(20).set_index("metric")[columns]
    values = frame.to_numpy(float)
    limit = max(0.1, float(np.nanmax(np.abs(values))))
    fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(frame))))
    image = ax.imshow(values, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(4), ["TRI-4", "TRI-5", "TRI-6", "TRI-7"])
    display = consistency.set_index("metric")["display_name"].reindex(frame.index)
    ax.set_yticks(range(len(frame)), display, fontsize=8)
    fig.colorbar(image, ax=ax, label="Effect size")
    ax.set_title("Cross-level direction of TRI-5 candidate metrics")
    fig.tight_layout()
    fig.savefig(path, dpi=210)
    plt.close(fig)


def plot_family_summary(primary: pd.DataFrame, path: Path) -> None:
    tested = primary.groupby("family").size().rename("tested")
    candidates = primary.loc[primary["topology_assoc_q_bh"].lt(0.05)].groupby("family").size().rename("q_below_0_05")
    frame = pd.concat([tested, candidates], axis=1).fillna(0).sort_values("tested")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    y = np.arange(len(frame))
    ax.barh(y, frame["tested"], color="#B9C6D3", label="Tested")
    ax.barh(y, frame["q_below_0_05"], color="#D1495B", label="q < 0.05")
    ax.set_yticks(y, frame.index.str.replace("_", " "))
    ax.set_xlabel("Metrics")
    ax.set_title("TRI-5 topology candidates by metric family")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=210)
    plt.close(fig)


def draw_graph(ax: plt.Axes, graph: nx.Graph, title: str, seed: int) -> None:
    positions = nx.spring_layout(graph, seed=seed)
    labels = {node: LABEL_NAMES[str(graph.nodes[node]["label"])] for node in graph}
    colors = [LABEL_COLORS[str(graph.nodes[node]["label"])] for node in graph]
    nx.draw_networkx(
        graph,
        pos=positions,
        ax=ax,
        node_color=colors,
        node_size=620,
        font_size=0,
        edge_color="#555555",
        width=1.8,
    )
    for node, (x, y) in positions.items():
        ax.text(x, y, str(graph.nodes[node]["label"]), ha="center", va="center", color="white", fontsize=9)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def plot_topology_examples(
    features: pd.DataFrame,
    source: pd.DataFrame,
    primary: pd.DataFrame,
    primary_level: int,
    path: Path,
    seed: int,
) -> None:
    metrics = primary.loc[primary["is_representative"], "metric"].head(3).tolist()
    eligible = features.loc[
        features["eligible_for_comparison"] & features["triangle_level"].eq(primary_level)
    ]
    source_index = source.set_index(["triangle_level", "wl_hash"])
    fig, axes = plt.subplots(len(metrics), 2, figsize=(9, 4 * len(metrics)), squeeze=False)
    for row_index, metric in enumerate(metrics):
        display = str(primary.loc[primary["metric"].eq(metric), "display_name"].iloc[0])
        for column_index, significant in enumerate([False, True]):
            group = eligible.loc[eligible["significant_raw_p"].eq(significant), ["wl_hash", metric]].dropna()
            median = float(group[metric].median())
            selected = group.assign(distance=(group[metric] - median).abs()).sort_values(
                ["distance", "wl_hash"]
            ).iloc[0]
            raw = source_index.loc[(primary_level, selected.wl_hash)]
            graph = graph_from_encoding(str(raw.node_labels), str(raw.edge_list), str(selected.wl_hash))
            draw_graph(
                axes[row_index, column_index],
                graph,
                f"{'significant' if significant else 'nonsignificant'} {str(selected.wl_hash)[:8]}\n"
                f"{display}={selected[metric]:.3g}",
                seed,
            )
    fig.suptitle("Representative TRI-5 labeled topologies nearest each class median")
    fig.text(
        0.5,
        0.005,
        "Node labels: " + "; ".join(f"{key}={value}" for key, value in LABEL_NAMES.items()),
        ha="center",
        fontsize=7,
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.98))
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_report(
    path: Path,
    summary: pd.DataFrame,
    primary: pd.DataFrame,
    consistency: pd.DataFrame,
    composition: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    primary_summary = summary.set_index("triangle_level").loc[args.primary_level]
    candidates = primary.loc[primary["exploratory_candidate"] & primary["is_representative"]]
    lines = [
        "# Sibling topology analysis",
        "",
        "## Result overview",
        "",
        f"The primary TRI-{args.primary_level} comparison contains "
        f"{int(primary_summary.estimable_motifs):,} estimable motifs: "
        f"{int(primary_summary.significant_raw_p):,} with raw Cox p < {args.p_threshold:g} "
        f"and {int(primary_summary.nonsignificant_raw_p):,} controls.",
        "",
        f"After correcting the topology-association tests within TRI-{args.primary_level}, "
        f"{int(primary.exploratory_candidate.sum()):,} metrics had BH q < 0.05; "
        f"{len(candidates):,} are representatives after grouping metrics connected at "
        "|Spearman rho| >= 0.90.",
        "",
        "## Leading nonredundant criteria",
        "",
        "| Metric | Family | Direction | Effect | Topology q | Cross-level consistent |",
        "|---|---|---|---:|---:|---|",
    ]
    consistency_index = consistency.set_index("metric") if len(consistency) else pd.DataFrame()
    for row in primary.loc[primary["is_representative"]].head(args.top_n).itertuples(index=False):
        cross = (
            bool(consistency_index.loc[row.metric, "cross_level_consistent"])
            if len(consistency_index) and row.metric in consistency_index.index
            else False
        )
        lines.append(
            f"| {row.display_name} (`{row.metric}`) | {row.family} | {row.effect_direction} | "
            f"{row.effect_size:.3g} | {row.topology_assoc_q_bh:.3g} | {'yes' if cross else 'no'} |"
        )
    lines.extend(
        [
            "",
            "Positive effects mean the metric tends to be larger among raw-significant motifs. "
            "For nonbinary metrics the effect is Cliff's delta; for binary metrics it is the "
            "difference in prevalence.",
            "",
            "## Cell-type composition",
            "",
        ]
    )
    if len(composition):
        enriched = composition.loc[composition["topology_assoc_q_bh"].lt(0.05)].head(10)
        if len(enriched):
            lines.extend(
                [
                    "The following exact cell-type compositions pass the exploratory within-table BH threshold:",
                    "",
                    "| Label composition | Motifs | Significant fraction | OR | q |",
                    "|---|---:|---:|---:|---:|",
                ]
            )
            for row in enriched.itertuples(index=False):
                lines.append(
                    f"| {row.composition_display} (`{row.label_composition}`) | {row.motifs} | "
                    f"{row.significant_fraction:.3f} | "
                    f"{row.odds_ratio:.3g} | {row.topology_assoc_q_bh:.3g} |"
                )
        else:
            lines.append("No exact cell-type composition passes BH q < 0.05.")
    lines.extend(
        [
            "",
            "## Structural context",
            "",
        ]
    )
    structural_families = {
        "graph_organization",
        "degree_organization",
        "connectivity_distance",
        "clustering_cliques",
        "triangle_organization",
        "spectral",
    }
    structural = primary.loc[
        primary["is_representative"] & primary["family"].isin(structural_families)
    ].head(8)
    if len(structural):
        lines.extend(
            [
                "The strongest results overall are cell-type composition and labeled-contact "
                "criteria. The leading shape-oriented criteria are smaller:",
                "",
                "| Shape criterion | Effect | q |",
                "|---|---:|---:|",
            ]
        )
        for row in structural.itertuples(index=False):
            lines.append(
                f"| {row.display_name} | {row.effect_size:.3g} | "
                f"{row.topology_assoc_q_bh:.3g} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "This is an exploratory association screen, not a causal test. Same-level motifs are "
            "called siblings here even though they need not share an immediate parent. The lineage "
            "was selected around a raw-significant TRI-1 motif, and all Cox tests reuse the same "
            "278 patients. Nonestimable Cox fits were excluded rather than treated as controls.",
            "",
            "The requested analysis does not adjust for motif abundance, document frequency, or "
            "enriched-group balance and does not use patient or motif resampling. Consequently, "
            "nominal topology association p/q values can overstate certainty and cannot show that "
            "topology itself caused the survival association. Cell-type composition and graph shape "
            "are intentionally combined as labeled topology.",
            "",
            "Original Cox significance is defined only as raw p < 0.05. The BH q-values in the "
            "tables correct the separate topology-metric screen and are not alternate Cox endpoints.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.primary_level not in range(2, 8):
        raise ValueError("--primary-level must be between 2 and 7")
    if not 0 < args.p_threshold < 1:
        raise ValueError("--p-threshold must be between zero and one")
    if args.min_class_size < 2:
        raise ValueError("--min-class-size must be at least two")
    if not 0 < args.min_feature_prevalence < 0.5:
        raise ValueError("--min-feature-prevalence must be between zero and 0.5")

    nodes_path = args.trace_dir / "trace_nodes.csv"
    metadata_path = args.trace_dir / "trace_metadata.json"
    if not nodes_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Trace inputs are missing under {args.trace_dir}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("target_hash") != TARGET_HASH:
        raise ValueError(f"Expected example lineage {TARGET_HASH}")
    source = pd.read_csv(
        nodes_path, dtype={"wl_hash": str, "node_labels": str, "edge_list": str}
    )
    missing = REQUIRED_COLUMNS - set(source.columns)
    if missing:
        raise ValueError(f"trace_nodes.csv is missing columns: {sorted(missing)}")
    if source.duplicated(["triangle_level", "wl_hash"]).any():
        raise ValueError("The selected trace contains duplicate level/hash rows")

    tables = args.outdir / "tables"
    figures = args.outdir / "figures"
    feature_cache = tables / "motif_topology_features.csv.gz"
    prior_audit_path = args.outdir / "analysis_summary.json"
    current_source_hash = sha256(nodes_path)
    use_cache = False
    if feature_cache.is_file() and prior_audit_path.is_file() and not args.rebuild_features:
        prior_audit = json.loads(prior_audit_path.read_text(encoding="utf-8"))
        use_cache = (
            prior_audit.get("input_trace_nodes_sha256") == current_source_hash
            and int(prior_audit.get("motifs", -1)) == len(source)
            and int(prior_audit.get("feature_cache_version", FEATURE_CACHE_VERSION))
            == FEATURE_CACHE_VERSION
        )
    if use_cache:
        features = pd.read_csv(feature_cache, dtype={"wl_hash": str})
        expected_keys = set(zip(source["triangle_level"].astype(int), source["wl_hash"]))
        cached_keys = set(zip(features["triangle_level"].astype(int), features["wl_hash"]))
        if expected_keys != cached_keys:
            raise ValueError("Cached feature rows do not match the selected trace")
        print(f"Reused validated topology cache for {len(features):,} motifs", flush=True)
    else:
        feature_rows = []
        for position, (_, row) in enumerate(source.iterrows(), 1):
            graph = graph_from_encoding(str(row.node_labels), str(row.edge_list), str(row.wl_hash))
            validate_graph(row, graph)
            values = graph_features(graph)
            labels = sorted(str(graph.nodes[node]["label"]) for node in graph)
            feature_rows.append(
                {
                    "triangle_level": int(row.triangle_level),
                    "wl_hash": str(row.wl_hash),
                    "label_composition": "&".join(labels),
                    "p_value_unadjusted": row.p_value_unadjusted,
                    "coef_unadjusted": row.coef_unadjusted,
                    "hazard_ratio_unadjusted": row.hazard_ratio_unadjusted,
                    "fit_status_unadjusted": row.fit_status_unadjusted,
                    "sample_document_frequency": row.sample_document_frequency,
                    "median_patient_raw_count": row.median_patient_raw_count,
                    "n_enriched_unadjusted": row.n_enriched_unadjusted,
                    "n_non_enriched_unadjusted": row.n_non_enriched_unadjusted,
                    **values,
                }
            )
            if position % 10_000 == 0 or position == len(source):
                print(
                    f"Computed labeled topology for {position:,}/{len(source):,} motifs",
                    flush=True,
                )
        features = pd.DataFrame(feature_rows)
    features["eligible_for_comparison"] = (
        features["fit_status_unadjusted"].eq("estimated")
        & np.isfinite(features["p_value_unadjusted"])
        & features["p_value_unadjusted"].gt(0)
        & features["p_value_unadjusted"].le(1)
    )
    features["significant_raw_p"] = (
        features["eligible_for_comparison"]
        & features["p_value_unadjusted"].lt(args.p_threshold)
    )
    metadata_columns = {
        "triangle_level", "wl_hash", "label_composition", "p_value_unadjusted",
        "coef_unadjusted", "hazard_ratio_unadjusted", "fit_status_unadjusted",
        "sample_document_frequency", "median_patient_raw_count", "n_enriched_unadjusted",
        "n_non_enriched_unadjusted", "eligible_for_comparison", "significant_raw_p",
    }
    metric_columns = [column for column in features.columns if column not in metadata_columns]
    dictionary = build_metric_dictionary(metric_columns)
    associations, exclusions = analyze_levels(
        features,
        dictionary,
        args.min_class_size,
        args.min_feature_prevalence,
    )
    associations = associations.merge(
        dictionary[["metric", "display_name"]],
        on="metric",
        how="left",
        validate="many_to_one",
    )
    summary = level_summary(features, args.p_threshold)
    composition = composition_enrichment(features, args.primary_level, args.min_feature_prevalence)
    groups, primary = redundancy_groups(features, associations, args.primary_level)
    consistency = cross_level_consistency(primary, associations, args.primary_level)

    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    features.to_csv(tables / "motif_topology_features.csv.gz", index=False, compression="gzip")
    dictionary.to_csv(tables / "metric_dictionary.csv", index=False)
    summary.to_csv(tables / "level_summary.csv", index=False)
    associations.to_csv(tables / "topology_associations_all_levels.csv", index=False)
    exclusions.to_csv(tables / "excluded_metrics.csv", index=False)
    groups.to_csv(tables / "metric_redundancy_groups.csv", index=False)
    primary.to_csv(tables / "tri5_ranked_metrics.csv", index=False)
    consistency.to_csv(tables / "cross_level_consistency.csv", index=False)
    composition.to_csv(tables / "cell_composition_enrichment.csv", index=False)

    plot_level_overview(summary, figures / "level_overview.png")
    plot_forest(primary, figures / "tri5_metric_forest.png", args.top_n)
    plot_distributions(features, primary, args.primary_level, figures / "tri5_metric_distributions.png")
    plot_correlation(features, primary, args.primary_level, figures / "tri5_metric_correlation.png")
    plot_cross_level(consistency, figures / "cross_level_consistency.png")
    plot_family_summary(primary, figures / "metric_family_summary.png")
    plot_topology_examples(
        features, source, primary, args.primary_level, figures / "tri5_topology_examples.png", args.layout_seed
    )
    write_report(args.outdir / "REPORT.md", summary, primary, consistency, composition, args)

    command = (
        "python breast_cancer_survival/motif_results/sibling_topology_analysis/run_analysis.py"
    )
    audit = {
        "status": "complete",
        "analysis_scope": f"single lineage tri_1:{TARGET_HASH}; same-level peers",
        "primary_level": args.primary_level,
        "cox_significance_rule": f"finite estimated raw p < {args.p_threshold:g}",
        "topology_definition": "internal labeled motif graph",
        "adjustment": "none",
        "resampling": "none",
        "input_trace_nodes": str(nodes_path.resolve()),
        "input_trace_nodes_sha256": current_source_hash,
        "input_trace_metadata_sha256": sha256(metadata_path),
        "motifs": len(features),
        "metrics_computed": len(metric_columns),
        "feature_cache_version": FEATURE_CACHE_VERSION,
        "reproduction_command": command,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "networkx": nx.__version__,
            "statsmodels": statsmodels.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    (args.outdir / "analysis_summary.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
