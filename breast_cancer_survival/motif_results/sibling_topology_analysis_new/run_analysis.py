#!/usr/bin/env python3
"""TRI-5 distributions for the triangulation-motif topology criteria."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import textwrap
import warnings
from collections import Counter, defaultdict
from itertools import combinations, combinations_with_replacement
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
PARENT_DIR = PROJECT_DIR.parent
TARGET_HASH = "1fded5000748985170d30bdca433a111"
TRACE_DIR = PARENT_DIR / "significant_mtf_trace" / "tri1" / TARGET_HASH
BASE_DIR = PARENT_DIR / "sibling_topology_analysis"

os.environ.setdefault("MPLCONFIGDIR", "/tmp/breast-cancer-topology-criteria-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import levene, mannwhitneyu
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

spec = importlib.util.spec_from_file_location("base_sibling_topology", BASE_DIR / "run_analysis.py")
if spec is None or spec.loader is None:
    raise ImportError("Cannot load the established sibling-topology implementation")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

LABEL_NAMES = base.LABEL_NAMES
LABELS = tuple(sorted(LABEL_NAMES, key=int))
EDGE_TYPES = tuple(combinations_with_replacement(LABELS, 2))
TRIANGLE_TYPES = tuple(combinations_with_replacement(LABELS, 3))
LABEL_ABBREVIATIONS = {
    "0": "B", "1": "T+B", "2": "T", "3": "End",
    "4": "Mac", "5": "Str", "6": "Tum", "7": "Unk",
}


def configure_labels(names, colors, abbreviations):
    """Configure this isolated engine instance for a cohort's label vocabulary."""
    global LABEL_NAMES, LABELS, EDGE_TYPES, TRIANGLE_TYPES, LABEL_ABBREVIATIONS
    if set(names) != set(colors) or set(names) != set(abbreviations):
        raise ValueError("Names, colors and abbreviations must cover identical labels")
    LABEL_NAMES = dict(names)
    LABELS = tuple(sorted(names, key=int))
    EDGE_TYPES = tuple(combinations_with_replacement(LABELS, 2))
    TRIANGLE_TYPES = tuple(combinations_with_replacement(LABELS, 3))
    LABEL_ABBREVIATIONS = dict(abbreviations)
    base.LABEL_NAMES = dict(names)
    base.LABEL_COLORS = dict(colors)
METADATA = [
    "triangle_level", "wl_hash", "significance_group", "significant_raw_p",
    "p_value_unadjusted", "coef_unadjusted", "hazard_ratio_unadjusted",
    "sample_document_frequency", "median_patient_raw_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, default=TRACE_DIR)
    parser.add_argument("--base-cache", type=Path, default=BASE_DIR / "results/tables/motif_topology_features.csv.gz")
    parser.add_argument("--outdir", type=Path, default=PROJECT_DIR / "results")
    parser.add_argument("--primary-level", type=int, default=5)
    parser.add_argument("--p-threshold", type=float, default=0.05)
    parser.add_argument("--permutations", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--panels-per-page", type=int, default=12)
    parser.add_argument(
        "--representative-pairs",
        type=int,
        default=12,
        help="Number of significant/nonsignificant representative motif pairs.",
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Regenerate figures from existing result tables without recomputing motif criteria.",
    )
    return parser.parse_args()


def entropy_from_counts(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[values > 0]
    if not len(values):
        return 0.0
    p = values / values.sum()
    return float(-(p * np.log(p)).sum())


def cv(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    mean = float(values.mean()) if len(values) else 0.0
    return float(values.std(ddof=0) / mean) if mean else 0.0


def stable_seed(text: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{text}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def zscores(observed: np.ndarray, permutations: np.ndarray) -> np.ndarray:
    means = permutations.mean(axis=0)
    sd = permutations.std(axis=0, ddof=1)
    return np.divide(observed - means, sd, out=np.full_like(means, np.nan), where=sd > 0)


def js_distance(vector: np.ndarray, centroid: np.ndarray) -> float:
    p = np.asarray(vector, dtype=float)
    q = np.asarray(centroid, dtype=float)
    if p.sum() > 0:
        p = p / p.sum()
    if q.sum() > 0:
        q = q / q.sum()
    midpoint = (p + q) / 2
    p_mask = p > 0
    q_mask = q > 0
    left = np.sum(p[p_mask] * np.log(p[p_mask] / midpoint[p_mask]))
    right = np.sum(q[q_mask] * np.log(q[q_mask] / midpoint[q_mask]))
    return float(np.sqrt(max(0.0, (left + right) / 2)))


def describe_metric(values: pd.Series) -> dict[str, float | int]:
    clean = values.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if clean.empty:
        return {key: np.nan for key in ["mean", "sd", "variance", "minimum", "q05", "q25", "median", "q75", "q95", "maximum", "iqr", "zero_fraction"]} | {"n": 0, "n_missing": int(len(values))}
    quantiles = clean.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "n": int(len(clean)), "n_missing": int(len(values) - len(clean)),
        "mean": float(clean.mean()), "sd": float(clean.std(ddof=1)),
        "variance": float(clean.var(ddof=1)), "minimum": float(clean.min()),
        "q05": float(quantiles.loc[0.05]), "q25": float(quantiles.loc[0.25]),
        "median": float(quantiles.loc[0.5]), "q75": float(quantiles.loc[0.75]),
        "q95": float(quantiles.loc[0.95]), "maximum": float(clean.max()),
        "iqr": float(quantiles.loc[0.75] - quantiles.loc[0.25]),
        "zero_fraction": float((clean == 0).mean()),
    }


def extra_features(graph: nx.Graph, motif_hash: str, permutations: int, seed: int) -> tuple[dict[str, float], dict[str, list[float]]]:
    nodes = sorted(graph.nodes)
    n = len(nodes)
    edges = np.asarray([(nodes.index(u), nodes.index(v)) for u, v in graph.edges], dtype=int)
    triangles_nodes = base.all_triangles(graph)
    triangles = np.asarray([[nodes.index(v) for v in tri] for tri in triangles_nodes], dtype=int)
    labels = np.asarray([str(graph.nodes[node]["label"]) for node in nodes])
    degrees = np.asarray([graph.degree(node) for node in nodes], dtype=float)
    triangle_participation = np.asarray([nx.triangles(graph, node) for node in nodes], dtype=float)
    edge_support = np.asarray([
        len(set(graph.neighbors(u)).intersection(graph.neighbors(v))) for u, v in graph.edges
    ])
    boundary_edges = [edge for edge, support in zip(graph.edges, edge_support) if support == 1]
    boundary_nodes = {node for edge in boundary_edges for node in edge}
    components = nx.number_connected_components(graph)
    triangle_count = len(triangles_nodes)
    chi = n - graph.number_of_edges() + triangle_count
    holes = max(0, components - chi)

    distance_values: list[float] = []
    for component in nx.connected_components(graph):
        subgraph = graph.subgraph(component)
        lengths = dict(nx.all_pairs_shortest_path_length(subgraph))
        distance_values.extend(float(lengths[u][v]) for u, v in combinations(sorted(component), 2))
    diameter = max(distance_values, default=0.0)
    mean_path = float(np.mean(distance_values)) if distance_values else 0.0

    adjacency = nx.to_numpy_array(graph, nodelist=nodes)
    invsqrt = np.divide(1.0, np.sqrt(degrees), out=np.zeros_like(degrees), where=degrees > 0)
    normalized_laplacian = np.eye(n) - invsqrt[:, None] * adjacency * invsqrt[None, :]
    eigenvalues = np.clip(np.linalg.eigvalsh(normalized_laplacian), 0, None)
    positive_eigenvalues = eigenvalues[eigenvalues > 1e-12]
    spectral_entropy = entropy_from_counts(positive_eigenvalues)
    spectral_entropy_normalized = spectral_entropy / np.log(len(positive_eigenvalues)) if len(positive_eigenvalues) > 1 else 0.0

    label_counts = np.asarray([(labels == label).sum() for label in LABELS], dtype=float)
    proportions = label_counts / n
    node_entropy = entropy_from_counts(label_counts)
    richness = int((label_counts > 0).sum())
    edge_counts = Counter(tuple(sorted((labels[u], labels[v]))) for u, v in edges)
    triangle_counts = Counter(tuple(sorted(labels[index] for index in tri)) for tri in triangles)
    observed_edges = np.asarray([edge_counts[pair] for pair in EDGE_TYPES], dtype=float)
    observed_triangles = np.asarray([triangle_counts[triple] for triple in TRIANGLE_TYPES], dtype=float)

    permutation_edges = np.zeros((permutations, len(EDGE_TYPES)), dtype=float)
    permutation_triangles = np.zeros((permutations, len(TRIANGLE_TYPES)), dtype=float)
    edge_index = {pair: index for index, pair in enumerate(EDGE_TYPES)}
    triangle_index = {triple: index for index, triple in enumerate(TRIANGLE_TYPES)}
    rng = np.random.default_rng(stable_seed(motif_hash, seed))
    for permutation in range(permutations):
        shuffled = rng.permutation(labels)
        for u, v in edges:
            permutation_edges[permutation, edge_index[tuple(sorted((shuffled[u], shuffled[v])))]] += 1
        for tri in triangles:
            key = tuple(sorted(shuffled[index] for index in tri))
            permutation_triangles[permutation, triangle_index[key]] += 1
    edge_z = zscores(observed_edges, permutation_edges)
    triangle_z = zscores(observed_triangles, permutation_triangles)

    local_entropy = []
    egophily = []
    for node in nodes:
        neighbor_labels = [str(graph.nodes[value]["label"]) for value in graph.neighbors(node)]
        local_entropy.append(entropy_from_counts(np.asarray(list(Counter(neighbor_labels).values()))))
        egophily.append(float(np.mean(np.asarray(neighbor_labels) == str(graph.nodes[node]["label"]))))
    local_entropy_array = np.asarray(local_entropy)
    egophily_array = np.asarray(egophily)

    result: dict[str, float] = {
        "size_nodes": float(n), "size_edges": float(graph.number_of_edges()),
        "size_triangles": float(triangle_count), "topology_triangles_per_node": float(triangle_count / n),
        "boundary_edge_count": float(len(boundary_edges)), "boundary_node_count": float(len(boundary_nodes)),
        "boundary_node_fraction": float(len(boundary_nodes) / n), "interior_node_fraction": float(1 - len(boundary_nodes) / n),
        "boundary_edges_per_node": float(len(boundary_edges) / n),
        "boundary_edges_per_triangle": float(len(boundary_edges) / triangle_count) if triangle_count else np.nan,
        "topology_component_count": float(components), "topology_euler_characteristic": float(chi),
        "topology_hole_count": float(holes),
        "degree_mean": float(degrees.mean()), "degree_variance": float(degrees.var(ddof=0)),
        "degree_sd": float(degrees.std(ddof=0)), "degree_cv": cv(degrees), "degree_maximum": float(degrees.max()),
        "degree_q25": float(np.quantile(degrees, .25)), "degree_median": float(np.median(degrees)), "degree_q75": float(np.quantile(degrees, .75)),
        "triangle_participation_mean": float(triangle_participation.mean()),
        "triangle_participation_variance": float(triangle_participation.var(ddof=0)),
        "triangle_participation_sd": float(triangle_participation.std(ddof=0)),
        "triangle_participation_cv": cv(triangle_participation),
        "triangle_participation_q25": float(np.quantile(triangle_participation, .25)),
        "triangle_participation_median": float(np.median(triangle_participation)),
        "triangle_participation_q75": float(np.quantile(triangle_participation, .75)),
        "triangle_participation_maximum": float(triangle_participation.max()),
        "path_diameter": float(diameter), "path_mean_shortest": mean_path,
        "path_diameter_normalized_sqrt_n": float(diameter / np.sqrt(n)),
        "path_mean_normalized_sqrt_n": float(mean_path / np.sqrt(n)),
        "connectivity_normalized_lambda2": float(eigenvalues[1]) if n > 1 else 0.0,
        "spectral_normalized_lambda3": float(eigenvalues[2]) if n > 2 else np.nan,
        "spectral_normalized_lambda4": float(eigenvalues[3]) if n > 3 else np.nan,
        "spectral_normalized_laplacian_entropy": spectral_entropy,
        "spectral_normalized_laplacian_entropy_scaled": spectral_entropy_normalized,
        "composition_type_richness": float(richness), "composition_shannon_entropy": node_entropy,
        "composition_shannon_entropy_normalized": float(node_entropy / np.log(min(len(LABELS), n))) if min(len(LABELS), n) > 1 else 0.0,
        "composition_simpson_diversity": float(1 - np.square(proportions).sum()),
        "composition_hill_d0": float(richness), "composition_hill_d1": float(np.exp(node_entropy)),
        "composition_hill_d2": float(1 / np.square(proportions).sum()),
        "composition_evenness": float(node_entropy / np.log(richness)) if richness > 1 else 0.0,
        "composition_dominant_type_fraction": float(proportions.max()),
        "composition_top2_type_fraction": float(np.sort(proportions)[-2:].sum()),
        "contact_homotypic_fraction": float(sum(count for pair, count in edge_counts.items() if pair[0] == pair[1]) / len(edges)),
        "contact_heterotypic_fraction": float(sum(count for pair, count in edge_counts.items() if pair[0] != pair[1]) / len(edges)),
        "contact_edge_type_entropy": entropy_from_counts(observed_edges),
        "contact_edge_type_entropy_normalized": float(entropy_from_counts(observed_edges) / np.log((observed_edges > 0).sum())) if (observed_edges > 0).sum() > 1 else 0.0,
        "triangle_AAA_fraction": float(sum(count for key, count in triangle_counts.items() if len(set(key)) == 1) / triangle_count),
        "triangle_AAB_fraction": float(sum(count for key, count in triangle_counts.items() if len(set(key)) == 2) / triangle_count),
        "triangle_ABC_fraction": float(sum(count for key, count in triangle_counts.items() if len(set(key)) == 3) / triangle_count),
        "triangle_type_entropy": entropy_from_counts(observed_triangles),
        "triangle_type_entropy_normalized": float(entropy_from_counts(observed_triangles) / np.log((observed_triangles > 0).sum())) if (observed_triangles > 0).sum() > 1 else 0.0,
        "local_neighbor_entropy_mean": float(local_entropy_array.mean()),
        "local_neighbor_entropy_median": float(np.median(local_entropy_array)),
        "local_neighbor_entropy_variance": float(local_entropy_array.var(ddof=0)),
        "local_neighbor_entropy_iqr": float(np.quantile(local_entropy_array, .75) - np.quantile(local_entropy_array, .25)),
        "local_egophily_mean": float(egophily_array.mean()),
        "local_egophily_variance": float(egophily_array.var(ddof=0)),
    }
    for index, label in enumerate(LABELS):
        name = f"type_{label}_{LABEL_NAMES[label].lower().replace(' ', '_')}"
        selected = labels == label
        result[f"{name}_count"] = float(selected.sum())
        result[f"{name}_fraction"] = float(selected.mean())
        result[f"{name}_degree_mean"] = float(degrees[selected].mean()) if selected.any() else np.nan
        result[f"{name}_triangle_participation_mean"] = float(triangle_participation[selected].mean()) if selected.any() else np.nan
        result[f"{name}_boundary_enrichment_difference"] = float(np.mean([nodes[i] in boundary_nodes for i in np.where(selected)[0]]) - len(boundary_nodes) / n) if selected.any() else np.nan
        result[f"{name}_egophily_mean"] = float(egophily_array[selected].mean()) if selected.any() else np.nan
    for index, pair in enumerate(EDGE_TYPES):
        stem = f"contact_{pair[0]}_{pair[1]}"
        result[f"{stem}_count"] = float(observed_edges[index])
        result[f"{stem}_fraction"] = float(observed_edges[index] / len(edges))
        result[f"{stem}_permutation_z"] = float(edge_z[index])
    for index, triple in enumerate(TRIANGLE_TYPES):
        stem = f"typed_triangle_{triple[0]}_{triple[1]}_{triple[2]}"
        result[f"{stem}_count"] = float(observed_triangles[index])
        result[f"{stem}_fraction"] = float(observed_triangles[index] / triangle_count)
        result[f"{stem}_permutation_z"] = float(triangle_z[index])
    raw = {
        "degree": degrees.tolist(), "triangle_participation": triangle_participation.tolist(),
        "path_length": distance_values, "normalized_laplacian_eigenvalue": eigenvalues.tolist(),
    }
    return result, raw


def family_for(metric: str) -> str:
    prefixes = [
        ("size_", "size"), ("boundary_", "boundary_interior"), ("topology_", "pure_topology"),
        ("interior_", "boundary_interior"),
        ("degree_", "degree_structure"), ("triangle_participation_", "triangle_participation"),
        ("path_", "path_topology"), ("connectivity_", "connectivity"), ("spectral_", "spectral"),
        ("composition_", "node_composition"), ("contact_", "pairwise_contacts"),
        ("triangle_", "triangle_classes"), ("typed_triangle_", "typed_triangles"),
        ("local_", "local_organization"), ("type_", "type_topology_coupling"),
        ("dispersion_", "between_motif_heterogeneity"),
    ]
    return next((family for prefix, family in prefixes if metric.startswith(prefix)), "other")


def display_name(metric: str) -> str:
    value = metric
    for label, name in LABEL_NAMES.items():
        value = value.replace(f"type_{label}_{name.lower().replace(' ', '_')}", name)
    if metric.startswith("contact_"):
        parts = metric.split("_")
        if len(parts) >= 4 and parts[1] in LABEL_NAMES and parts[2] in LABEL_NAMES:
            value = f"{LABEL_NAMES[parts[1]]}–{LABEL_NAMES[parts[2]]} contact " + " ".join(parts[3:])
    if metric.startswith("typed_triangle_"):
        parts = metric.split("_")
        if len(parts) >= 6:
            value = "–".join(LABEL_NAMES[p] for p in parts[2:5]) + " triangle " + " ".join(parts[5:])
    return value.replace("_", " ")


def compact_metric_name(metric: str, width: int = 32) -> str:
    """Make long typed-contact/triangle names readable in gallery titles."""
    value = display_name(metric)
    replacements = {
        "T and B cells": "T+B", "stromal cells": "Str", "tumor cells": "Tum",
        "macrophages": "Mac", "endothelial": "End", "T cells": "T",
        "B cells": "B", "unknown": "Unk", "permutation z": "Z",
        "triangle participation": "tri participation", "fraction": "frac",
    }
    for original, replacement in replacements.items():
        value = value.replace(original, replacement)
    return textwrap.shorten(value, width=width, placeholder="…")


def add_dispersion_metrics(features: pd.DataFrame) -> None:
    compositions = [c for c in features if c.startswith("type_") and c.endswith("_fraction")]
    contacts = [f"contact_{a}_{b}_fraction" for a, b in EDGE_TYPES]
    contact_z = [f"contact_{a}_{b}_permutation_z" for a, b in EDGE_TYPES]
    triangles = [f"typed_triangle_{a}_{b}_{c}_fraction" for a, b, c in TRIANGLE_TYPES]
    for group, index in features.groupby("significance_group").groups.items():
        selection = features.loc[index]
        composition_centroid = selection[compositions].mean().to_numpy(float)
        contact_centroid = selection[contacts].mean().to_numpy(float)
        contact_z_centroid = selection[contact_z].mean(skipna=True).fillna(0).to_numpy(float)
        triangle_centroid = selection[triangles].mean().to_numpy(float)
        features.loc[index, "dispersion_composition_js_to_group_centroid"] = [js_distance(row, composition_centroid) for row in selection[compositions].to_numpy(float)]
        features.loc[index, "dispersion_mixing_frobenius_to_group_centroid"] = np.linalg.norm(selection[contacts].to_numpy(float) - contact_centroid, axis=1)
        features.loc[index, "dispersion_null_corrected_mixing_frobenius_to_group_centroid"] = np.linalg.norm(np.nan_to_num(selection[contact_z].to_numpy(float)) - contact_z_centroid, axis=1)
        features.loc[index, "dispersion_triangle_js_to_group_centroid"] = [js_distance(row, triangle_centroid) for row in selection[triangles].to_numpy(float)]


def build_registry(metrics: list[str], permutations: int) -> pd.DataFrame:
    rows = [{
        "criterion": metric, "display_name": display_name(metric), "family": family_for(metric),
        "availability": "computed", "reason_or_method": "Computed from TRI-5 graph adjacency and node labels.",
    } for metric in metrics]
    unavailable = {
        "geometry_physical_area": "Node coordinates are absent from trace_nodes.csv.",
        "geometry_perimeter": "Node coordinates are absent; graph boundary-edge count is provided separately.",
        "geometry_cell_density": "Physical area is unavailable.",
        "geometry_compactness": "Physical area and perimeter are unavailable.",
        "geometry_solidity": "Coordinates and convex-hull area are unavailable.",
        "geometry_elongation_anisotropy": "Coordinates are unavailable.",
        "geometry_edge_length_distribution": "Coordinates are unavailable.",
        "geometry_triangle_area_distribution": "Coordinates are unavailable.",
        "geometry_triangle_shape": "Coordinates are unavailable.",
        "phenotype_label_null": "This lineage contains motif-level Cox results, not healthy/disease sample labels.",
        "patient_level_dependence_model": "No motif-to-patient or region membership is present in the trace.",
    }
    rows.extend({"criterion": key, "display_name": key.replace("_", " "), "family": "unavailable", "availability": "not_available", "reason_or_method": reason} for key, reason in unavailable.items())
    result = pd.DataFrame(rows)
    result.loc[result["criterion"].str.endswith("permutation_z"), "reason_or_method"] = f"Within-motif fixed-composition node-label null using {permutations} deterministic Monte Carlo permutations."
    return result


def distribution_tables(features: pd.DataFrame, metrics: list[str], dictionary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    long_rows = []
    comparison_rows = []
    dictionary_index = dictionary.set_index("criterion")
    for metric in metrics:
        for group in ["nonsignificant", "significant"]:
            row = describe_metric(features.loc[features["significance_group"].eq(group), metric])
            long_rows.append({"metric": metric, "display_name": dictionary_index.loc[metric, "display_name"], "family": dictionary_index.loc[metric, "family"], "group": group, **row})
        control = features.loc[features["significance_group"].eq("nonsignificant"), metric].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
        case = features.loc[features["significance_group"].eq("significant"), metric].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
        row = {"metric": metric, "display_name": dictionary_index.loc[metric, "display_name"], "family": dictionary_index.loc[metric, "family"], "n_nonsignificant": len(control), "n_significant": len(case)}
        if len(control) and len(case) and len(np.unique(np.concatenate([control, case]))) > 1:
            mw = mannwhitneyu(case, control, alternative="two-sided", method="asymptotic")
            auc = float(mw.statistic / (len(case) * len(control)))
            row |= {"mann_whitney_u": float(mw.statistic), "location_p_value": float(mw.pvalue), "cliffs_delta": 2 * auc - 1, "median_difference": float(np.median(case) - np.median(control)), "mean_difference": float(np.mean(case) - np.mean(control))}
            if len(control) > 1 and len(case) > 1:
                variance_test = levene(case, control, center="median")
                row |= {"brown_forsythe_statistic": float(variance_test.statistic), "dispersion_p_value": float(variance_test.pvalue), "variance_ratio_significant_over_nonsignificant": float(np.var(case, ddof=1) / np.var(control, ddof=1)) if np.var(control, ddof=1) > 0 else np.nan}
        comparison_rows.append(row)
    long = pd.DataFrame(long_rows)
    comparison = pd.DataFrame(comparison_rows)
    comparison["location_q_bh"] = base.benjamini_hochberg(comparison["location_p_value"])
    comparison["dispersion_q_bh"] = base.benjamini_hochberg(comparison["dispersion_p_value"])
    comparison = comparison.sort_values(["location_q_bh", "metric"], na_position="last").reset_index(drop=True)
    return long, comparison


def draw_violin(ax: plt.Axes, control: pd.Series, case: pd.Series) -> None:
    """Draw robust two-group violins with jittered points and median/IQR overlays.

    Matplotlib's KDE cannot represent a constant sample. Such a group is shown
    as a short thick horizontal segment at its observed value instead.
    """
    colors = {1: "#7A8FA6", 2: "#D1495B"}
    for position, values in [(1, control), (2, case)]:
        clean = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
        if not len(clean):
            continue
        if len(clean) >= 2 and np.unique(clean).size >= 2:
            parts = ax.violinplot(
                [clean], positions=[position], widths=0.72,
                showmeans=False, showmedians=False, showextrema=False,
                bw_method="scott",
            )
            for body in parts["bodies"]:
                body.set_facecolor(colors[position])
                body.set_edgecolor("#333333")
                body.set_alpha(0.48)
                body.set_linewidth(0.8)
        else:
            value = float(clean[0])
            ax.hlines(value, position - 0.18, position + 0.18, color=colors[position], linewidth=4, zorder=3)
        # Deterministic horizontal jitter exposes discrete values and sample density.
        rng = np.random.default_rng(20260917 + position * 1_000_003 + len(clean))
        jitter = rng.uniform(-0.16, 0.16, size=len(clean))
        ax.scatter(
            position + jitter,
            clean,
            s=4,
            color=colors[position],
            alpha=0.13 if position == 1 else 0.20,
            edgecolors="none",
            rasterized=True,
            zorder=2,
        )
        q25, median, q75 = np.quantile(clean, [0.25, 0.5, 0.75])
        ax.vlines(position, q25, q75, color="#222222", linewidth=3.2, zorder=3)
        ax.scatter(position, median, color="#F28E2B", edgecolor="white", linewidth=0.45, s=25, zorder=4)
    ax.set_xticks([1, 2], ["nonsignificant", "significant"])
    ax.set_xlim(0.5, 2.5)


def plot_pages(features: pd.DataFrame, comparison: pd.DataFrame, outdir: Path, panels_per_page: int) -> tuple[list[str], pd.DataFrame]:
    testable = comparison.loc[comparison["location_p_value"].notna()].copy()
    metrics = testable["metric"].tolist()
    pages = []
    index_rows = []
    columns = 2
    rows = math.ceil(panels_per_page / columns)
    for page_index, start in enumerate(range(0, len(metrics), panels_per_page), 1):
        selected = metrics[start:start + panels_per_page]
        fig, axes = plt.subplots(rows, columns, figsize=(13, 3.1 * rows), squeeze=False)
        for ax, metric in zip(axes.flat, selected):
            control = features.loc[features["significance_group"].eq("nonsignificant"), metric].dropna()
            case = features.loc[features["significance_group"].eq("significant"), metric].dropna()
            draw_violin(ax, control, case)
            title = str(testable.loc[testable["metric"].eq(metric), "display_name"].iloc[0])
            q = float(testable.loc[testable["metric"].eq(metric), "location_q_bh"].iloc[0])
            ax.set_title(f"{title}\nlocation BH q={q:.3g}", fontsize=8)
        for ax in axes.flat[len(selected):]:
            ax.axis("off")
        fig.suptitle(f"TRI-5 criterion distributions — page {page_index}")
        fig.tight_layout(rect=(0, 0, 1, .98))
        filename = f"tri5_all_criteria_distributions_page_{page_index:02d}.png"
        fig.savefig(outdir / filename, dpi=180)
        plt.close(fig)
        pages.append(filename)
        index_rows.extend(
            {
                "metric": metric,
                "display_name": str(testable.loc[testable["metric"].eq(metric), "display_name"].iloc[0]),
                "family": str(testable.loc[testable["metric"].eq(metric), "family"].iloc[0]),
                "figure_file": filename,
                "panel_on_page": offset + 1,
            }
            for offset, metric in enumerate(selected)
        )
    return pages, pd.DataFrame(index_rows)


def plot_leading_distributions(
    features: pd.DataFrame, comparison: pd.DataFrame, path: Path
) -> None:
    leading = comparison.loc[
        ~comparison.metric.str.contains(r"^contact_\d|^typed_triangle_\d", regex=True)
    ].head(8)
    fig, axes = plt.subplots(4, 2, figsize=(12, 12.8), squeeze=False)
    for ax, row in zip(axes.flat, leading.itertuples(index=False)):
        control = features.loc[features.significance_group.eq("nonsignificant"), row.metric].dropna()
        case = features.loc[features.significance_group.eq("significant"), row.metric].dropna()
        draw_violin(ax, control, case)
        ax.set_title(row.display_name, fontsize=9)
    for ax in axes.flat[len(leading):]:
        ax.axis("off")
    fig.suptitle("TRI-5 distributions for leading triangulation-motif criteria")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def add_covariance_ellipse(
    ax: plt.Axes, points: np.ndarray, color: str, label: str
) -> None:
    """Add a 95% covariance ellipse for a two-dimensional point cloud."""
    if len(points) < 3:
        return
    covariance = np.cov(points, rowvar=False)
    if not np.all(np.isfinite(covariance)):
        return
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order], 0, None)
    eigenvectors = eigenvectors[:, order]
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    # sqrt(chi-square quantile with 2 df at 0.95) = sqrt(5.991).
    scale = np.sqrt(5.991)
    ellipse = Ellipse(
        xy=points.mean(axis=0),
        width=2 * scale * np.sqrt(eigenvalues[0]),
        height=2 * scale * np.sqrt(eigenvalues[1]),
        angle=angle,
        facecolor="none",
        edgecolor=color,
        linewidth=2,
        linestyle="--",
        label=f"{label} 95% covariance ellipse",
        zorder=4,
    )
    ax.add_patch(ellipse)


def plot_all_criteria_pca(
    features: pd.DataFrame, tables: Path, figures: Path, seed: int
) -> dict[str, float | int]:
    """Standardize intrinsic criteria and visualize motifs using two-component PCA."""
    candidate_metrics = [column for column in features.columns if column not in METADATA]
    leakage_metrics = {
        column for column in candidate_metrics if column.startswith("dispersion_")
    }
    included: list[str] = []
    preprocessing_rows: list[dict[str, object]] = []
    columns: list[np.ndarray] = []
    for metric in candidate_metrics:
        numeric = pd.to_numeric(features[metric], errors="coerce").to_numpy(float)
        finite = np.isfinite(numeric)
        row: dict[str, object] = {
            "metric": metric,
            "finite_values": int(finite.sum()),
            "missing_values": int((~finite).sum()),
        }
        if metric in leakage_metrics:
            row |= {"included_in_pca": False, "exclusion_reason": "group_label_derived_heterogeneity_metric"}
            preprocessing_rows.append(row)
            continue
        if finite.sum() < 2:
            row |= {"included_in_pca": False, "exclusion_reason": "fewer_than_two_finite_values"}
            preprocessing_rows.append(row)
            continue
        median = float(np.median(numeric[finite]))
        imputed = np.where(finite, numeric, median)
        mean = float(imputed.mean())
        sd = float(imputed.std(ddof=0))
        if not np.isfinite(sd) or sd <= 0:
            row |= {"included_in_pca": False, "exclusion_reason": "constant_after_imputation", "imputation_median": median}
            preprocessing_rows.append(row)
            continue
        standardized = (imputed - mean) / sd
        included.append(metric)
        columns.append(standardized)
        row |= {
            "included_in_pca": True,
            "exclusion_reason": "",
            "imputation_median": median,
            "post_imputation_mean": mean,
            "post_imputation_sd": sd,
        }
        preprocessing_rows.append(row)
    if len(included) < 2:
        raise ValueError("Fewer than two usable intrinsic criteria are available for PCA")
    matrix = np.column_stack(columns)
    pca = PCA(n_components=2, svd_solver="randomized", random_state=seed)
    scores = pca.fit_transform(matrix)
    explained = pca.explained_variance_ratio_

    score_frame = features[["wl_hash", "significance_group", "significant_raw_p", "p_value_unadjusted"]].copy()
    score_frame["PC1"] = scores[:, 0]
    score_frame["PC2"] = scores[:, 1]
    score_frame.to_csv(tables / "tri5_pca_scores.csv.gz", index=False, compression="gzip")
    loading_frame = pd.DataFrame({
        "metric": included,
        "family": [family_for(metric) for metric in included],
        "PC1_loading": pca.components_[0],
        "PC2_loading": pca.components_[1],
        "loading_magnitude_2d": np.linalg.norm(pca.components_.T, axis=1),
    }).sort_values("loading_magnitude_2d", ascending=False)
    loading_frame.to_csv(tables / "tri5_pca_loadings.csv", index=False)
    pd.DataFrame(preprocessing_rows).to_csv(tables / "tri5_pca_preprocessing.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    styles = [
        ("nonsignificant", "#6F8499", 8, 0.20),
        ("significant", "#D1495B", 13, 0.55),
    ]
    for group, color, size, alpha in styles:
        mask = features["significance_group"].eq(group).to_numpy()
        points = scores[mask]
        ax.scatter(
            points[:, 0], points[:, 1], s=size, alpha=alpha, color=color,
            edgecolors="none", rasterized=True, label=f"{group} (n={mask.sum():,})",
        )
        centroid = points.mean(axis=0)
        ax.scatter(
            centroid[0], centroid[1], marker="X", s=150, color=color,
            edgecolor="white", linewidth=1.2, zorder=5,
        )
        add_covariance_ellipse(ax, points, color, group)
    ax.axhline(0, color="#BBBBBB", linewidth=0.7, zorder=0)
    ax.axvline(0, color="#BBBBBB", linewidth=0.7, zorder=0)
    ax.set_xlabel(f"PC1 ({100 * explained[0]:.2f}% variance explained)")
    ax.set_ylabel(f"PC2 ({100 * explained[1]:.2f}% variance explained)")
    ax.set_title(
        "TRI-5 PCA of standardized intrinsic motif criteria\n"
        "Outcome-derived group-dispersion criteria excluded"
    )
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(figures / "tri5_all_criteria_pca.png", dpi=220)
    plt.close(fig)
    return {
        "pca_candidate_criteria": len(candidate_metrics),
        "pca_included_criteria": len(included),
        "pca_excluded_label_derived": len(leakage_metrics),
        "pca_pc1_variance_fraction": float(explained[0]),
        "pca_pc2_variance_fraction": float(explained[1]),
    }


def pca_family_metric_sets(features: pd.DataFrame) -> dict[str, list[str]]:
    """Map criteria to four interpretable PCA blocks without outcome-derived fields."""
    metrics = [column for column in features.columns if column not in METADATA]
    topology_families = {
        "size", "pure_topology", "boundary_interior", "degree_structure",
        "triangle_participation", "path_topology", "connectivity", "spectral",
    }
    result = {
        "topology": [metric for metric in metrics if family_for(metric) in topology_families],
        "composition": [
            metric for metric in metrics
            if family_for(metric) == "node_composition"
            or (metric.startswith("type_") and metric.endswith(("_count", "_fraction")))
        ],
        "contacts": [
            metric for metric in metrics
            if family_for(metric) in {"pairwise_contacts", "local_organization"}
            or (
                metric.startswith("type_")
                and metric.endswith(("_degree_mean", "_boundary_enrichment_difference", "_egophily_mean"))
            )
        ],
        "triangles": [
            metric for metric in metrics
            if family_for(metric) in {"triangle_classes", "typed_triangles"}
            or (metric.startswith("type_") and metric.endswith("_triangle_participation_mean"))
        ],
    }
    return {name: sorted(set(values)) for name, values in result.items()}


def fit_pca_block(
    features: pd.DataFrame, candidate_metrics: list[str], seed: int
) -> tuple[np.ndarray, PCA, list[str], pd.DataFrame]:
    included: list[str] = []
    columns: list[np.ndarray] = []
    audit_rows: list[dict[str, object]] = []
    for metric in candidate_metrics:
        numeric = pd.to_numeric(features[metric], errors="coerce").to_numpy(float)
        finite = np.isfinite(numeric)
        row: dict[str, object] = {
            "metric": metric,
            "finite_values": int(finite.sum()),
            "missing_values": int((~finite).sum()),
        }
        if finite.sum() < 2:
            row |= {"included_in_pca": False, "exclusion_reason": "fewer_than_two_finite_values"}
            audit_rows.append(row)
            continue
        median = float(np.median(numeric[finite]))
        imputed = np.where(finite, numeric, median)
        mean = float(imputed.mean())
        sd = float(imputed.std(ddof=0))
        if not np.isfinite(sd) or sd <= 0:
            row |= {"included_in_pca": False, "exclusion_reason": "constant_after_imputation", "imputation_median": median}
            audit_rows.append(row)
            continue
        included.append(metric)
        columns.append((imputed - mean) / sd)
        row |= {
            "included_in_pca": True, "exclusion_reason": "",
            "imputation_median": median, "post_imputation_mean": mean,
            "post_imputation_sd": sd,
        }
        audit_rows.append(row)
    if len(included) < 2:
        raise ValueError("PCA block has fewer than two usable criteria")
    matrix = np.column_stack(columns)
    pca = PCA(n_components=2, svd_solver="randomized", random_state=seed)
    return pca.fit_transform(matrix), pca, included, pd.DataFrame(audit_rows)


def jitter_coincident_scores(scores: np.ndarray, seed: int) -> tuple[np.ndarray, int, int]:
    """Jitter only exactly coincident PCA coordinates for scatter visibility."""
    displayed = np.asarray(scores, dtype=float).copy()
    _, inverse, counts = np.unique(displayed, axis=0, return_inverse=True, return_counts=True)
    duplicate_groups = np.flatnonzero(counts > 1)
    if not len(duplicate_groups):
        return displayed, 0, 0
    axis_ranges = np.ptp(displayed, axis=0)
    fallback = np.std(displayed, axis=0, ddof=0)
    jitter_scale = np.where(axis_ranges > 0, axis_ranges * 0.006, np.maximum(fallback * 0.03, 0.03))
    rng = np.random.default_rng(seed)
    jittered_points = 0
    for group in duplicate_groups:
        indices = np.flatnonzero(inverse == group)
        offsets = rng.normal(0, jitter_scale, size=(len(indices), 2))
        # Center each stack so jitter does not move its visual centroid.
        offsets -= offsets.mean(axis=0)
        displayed[indices] += offsets
        jittered_points += len(indices)
    return displayed, int(jittered_points), int(len(duplicate_groups))


def draw_pca_axes(
    ax: plt.Axes,
    scores: np.ndarray,
    groups: pd.Series,
    explained: np.ndarray,
    title: str,
    show_legend: bool = True,
    display_seed: int = 20260917,
) -> None:
    displayed_scores, jittered_points, duplicate_stacks = jitter_coincident_scores(
        scores, display_seed
    )
    styles = [
        ("nonsignificant", "#6F8499", 8, 0.20),
        ("significant", "#D1495B", 13, 0.55),
    ]
    for group, color, size, alpha in styles:
        mask = groups.eq(group).to_numpy()
        points = scores[mask]
        displayed_points = displayed_scores[mask]
        ax.scatter(
            displayed_points[:, 0], displayed_points[:, 1], s=size, alpha=alpha, color=color,
            edgecolors="none", rasterized=True, label=f"{group} (n={mask.sum():,})",
        )
        centroid = points.mean(axis=0)
        ax.scatter(centroid[0], centroid[1], marker="X", s=120, color=color, edgecolor="white", linewidth=1, zorder=5)
        add_covariance_ellipse(ax, points, color, group)
    ax.axhline(0, color="#BBBBBB", linewidth=0.7, zorder=0)
    ax.axvline(0, color="#BBBBBB", linewidth=0.7, zorder=0)
    ax.set_xlabel(f"PC1 ({100 * explained[0]:.2f}%)")
    ax.set_ylabel(f"PC2 ({100 * explained[1]:.2f}%)")
    ax.set_title(title)
    if jittered_points:
        ax.text(
            0.01, 0.01,
            f"Display jitter applied to {jittered_points:,} points in {duplicate_stacks:,} coincident stacks",
            transform=ax.transAxes, fontsize=7, color="#555555", ha="left", va="bottom",
        )
    if show_legend:
        ax.legend(frameon=False, fontsize=7, loc="best")


def plot_family_pcas(
    features: pd.DataFrame, tables: Path, figures: Path, seed: int
) -> dict[str, float | int]:
    blocks = pca_family_metric_sets(features)
    fitted: dict[str, tuple[np.ndarray, PCA, list[str]]] = {}
    summary: dict[str, float | int] = {}
    score_metadata = features[["wl_hash", "significance_group", "significant_raw_p", "p_value_unadjusted"]]
    titles = {
        "topology": "Graph topology PCA",
        "composition": "Cell-composition PCA",
        "contacts": "Pairwise and local-contact PCA",
        "triangles": "Triangle-organization PCA",
    }
    for offset, (name, candidates) in enumerate(blocks.items()):
        scores, pca, included, audit = fit_pca_block(features, candidates, seed + offset)
        fitted[name] = (scores, pca, included)
        score_frame = score_metadata.copy()
        score_frame["PC1"] = scores[:, 0]
        score_frame["PC2"] = scores[:, 1]
        score_frame.to_csv(tables / f"tri5_pca_{name}_scores.csv.gz", index=False, compression="gzip")
        pd.DataFrame({
            "metric": included,
            "family": [family_for(metric) for metric in included],
            "PC1_loading": pca.components_[0],
            "PC2_loading": pca.components_[1],
            "loading_magnitude_2d": np.linalg.norm(pca.components_.T, axis=1),
        }).sort_values("loading_magnitude_2d", ascending=False).to_csv(
            tables / f"tri5_pca_{name}_loadings.csv", index=False
        )
        audit.to_csv(tables / f"tri5_pca_{name}_preprocessing.csv", index=False)
        fig, ax = plt.subplots(figsize=(9, 7))
        draw_pca_axes(
            ax, scores, features["significance_group"], pca.explained_variance_ratio_,
            titles[name], display_seed=seed + offset,
        )
        fig.tight_layout()
        fig.savefig(figures / f"tri5_pca_{name}.png", dpi=220)
        plt.close(fig)
        summary[f"pca_{name}_candidate_criteria"] = len(candidates)
        summary[f"pca_{name}_included_criteria"] = len(included)
        summary[f"pca_{name}_pc1_variance_fraction"] = float(pca.explained_variance_ratio_[0])
        summary[f"pca_{name}_pc2_variance_fraction"] = float(pca.explained_variance_ratio_[1])

    fig, axes = plt.subplots(2, 2, figsize=(15, 12), squeeze=False)
    for ax, name in zip(axes.flat, ["topology", "composition", "contacts", "triangles"]):
        scores, pca, included = fitted[name]
        draw_pca_axes(
            ax, scores, features["significance_group"], pca.explained_variance_ratio_,
            f"{titles[name]} ({len(included)} criteria)", show_legend=name == "topology",
            display_seed=seed + ["topology", "composition", "contacts", "triangles"].index(name),
        )
    fig.suptitle("TRI-5 PCA separated by criterion family")
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(figures / "tri5_pca_by_family.png", dpi=220)
    plt.close(fig)
    return summary


def family_balanced_selection_matrix(
    features: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Return standardized intrinsic criteria with equal weight per macro family."""
    blocks = pca_family_metric_sets(features)
    standardized_columns: dict[str, np.ndarray] = {}
    audit_rows: list[dict[str, object]] = []
    retained_by_block: dict[str, list[str]] = {}
    for block, candidates in blocks.items():
        retained: list[str] = []
        for metric in candidates:
            numeric = pd.to_numeric(features[metric], errors="coerce").to_numpy(float)
            finite = np.isfinite(numeric)
            row: dict[str, object] = {
                "metric": metric, "macro_family": block,
                "finite_values": int(finite.sum()), "missing_values": int((~finite).sum()),
            }
            if finite.sum() < 2:
                row |= {"included": False, "exclusion_reason": "fewer_than_two_finite_values"}
                audit_rows.append(row)
                continue
            median = float(np.median(numeric[finite]))
            imputed = np.where(finite, numeric, median)
            mean = float(imputed.mean())
            sd = float(imputed.std(ddof=0))
            if not np.isfinite(sd) or sd <= 0:
                row |= {
                    "included": False, "exclusion_reason": "constant_after_imputation",
                    "imputation_median": median,
                }
                audit_rows.append(row)
                continue
            standardized_columns[metric] = (imputed - mean) / sd
            retained.append(metric)
            row |= {
                "included": True, "exclusion_reason": "", "imputation_median": median,
                "post_imputation_mean": mean, "post_imputation_sd": sd,
            }
            audit_rows.append(row)
        retained_by_block[block] = retained
    ordered_metrics = [
        metric for block in ["topology", "composition", "contacts", "triangles"]
        for metric in retained_by_block[block]
    ]
    if not ordered_metrics:
        raise ValueError("No usable intrinsic criteria for representative selection")
    weighted_columns = []
    for block in ["topology", "composition", "contacts", "triangles"]:
        block_metrics = retained_by_block[block]
        if not block_metrics:
            raise ValueError(f"No usable criteria in representative-selection block {block}")
        weight = 1 / np.sqrt(len(block_metrics))
        weighted_columns.extend(standardized_columns[metric] * weight for metric in block_metrics)
        for row in audit_rows:
            if row["metric"] in block_metrics:
                row["family_balance_weight"] = weight
    matrix = np.column_stack(weighted_columns)
    standardized = pd.DataFrame(
        {metric: standardized_columns[metric] for metric in ordered_metrics},
        index=features.index,
    )
    return matrix, standardized, pd.DataFrame(audit_rows)


def unlabeled_topology_hash(graph: nx.Graph) -> str:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return nx.weisfeiler_lehman_graph_hash(graph)


def allocate_topology_quotas(
    topology_counts: pd.DataFrame, total_pairs: int
) -> dict[str, int]:
    """Give every shared topology one pair, then apportion by sqrt(case count)."""
    if total_pairs < len(topology_counts):
        raise ValueError(
            f"--representative-pairs must be at least {len(topology_counts)} "
            "to cover every shared unlabeled topology"
        )
    quotas = {str(index): 1 for index in topology_counts.index}
    capacities = {
        str(index): int(min(row.significant, row.nonsignificant))
        for index, row in topology_counts.iterrows()
    }
    weights = {
        str(index): np.sqrt(float(row.significant))
        for index, row in topology_counts.iterrows()
    }
    remaining = total_pairs - len(quotas)
    while remaining:
        eligible = [key for key in quotas if quotas[key] < capacities[key]]
        if not eligible:
            raise ValueError("Requested more representative pairs than shared topology capacity")
        selected = max(eligible, key=lambda key: (weights[key] / (quotas[key] + 1), key))
        quotas[selected] += 1
        remaining -= 1
    return quotas


def cluster_representatives(
    row_indices: np.ndarray, matrix: np.ndarray, clusters: int, seed: int
) -> list[dict[str, object]]:
    subset = matrix[row_indices]
    unique = np.unique(subset, axis=0)
    if clusters > len(unique):
        raise ValueError(
            f"Cannot select {clusters} representatives from only {len(unique)} unique vectors"
        )
    model = KMeans(n_clusters=clusters, random_state=seed, n_init=20)
    labels = model.fit_predict(subset)
    rows = []
    for cluster_id in range(clusters):
        members_local = np.flatnonzero(labels == cluster_id)
        distances = np.linalg.norm(subset[members_local] - model.cluster_centers_[cluster_id], axis=1)
        best_local = members_local[np.argmin(distances)]
        rows.append({
            "row_index": int(row_indices[best_local]),
            "cluster_id": int(cluster_id + 1),
            "cluster_support": int(len(members_local)),
            "distance_to_cluster_center": float(distances.min()),
        })
    return rows


def best_label_preserving_isomorphism(
    reference: nx.Graph, comparison: nx.Graph
) -> dict[int, int]:
    matcher = nx.algorithms.isomorphism.GraphMatcher(reference, comparison)
    best_mapping: dict[int, int] | None = None
    best_key: tuple[int, tuple[int, ...]] | None = None
    reference_nodes = sorted(reference.nodes)
    for mapping in matcher.isomorphisms_iter():
        agreements = sum(
            str(reference.nodes[node].get("label"))
            == str(comparison.nodes[mapping[node]].get("label"))
            for node in reference_nodes
        )
        mapped = tuple(int(mapping[node]) for node in reference_nodes)
        key = (agreements, tuple(-value for value in mapped))
        if best_key is None or key > best_key:
            best_key = key
            best_mapping = {int(left): int(right) for left, right in mapping.items()}
    if best_mapping is None:
        raise ValueError("Representative pair is not graph-isomorphic")
    return best_mapping


def triangulation_planar_layout(graph: nx.Graph) -> dict[int, np.ndarray]:
    """Return a compact crossing-free embedding with the largest face outside.

    TRI motifs are usually triangulated disks.  NetworkX's default planar drawing
    may put one of the small triangular faces on the outside, which turns the
    remaining graph into a long fan.  Here the longest facial walk is used as the
    outer boundary.  Boundary vertices are placed on a mildly flattened convex
    polygon and any interior vertices receive Tutte (barycentric) coordinates.
    """
    planar, embedding = nx.check_planarity(graph, counterexample=True)
    if not planar:
        raise ValueError("TRI motif graph is not planar and cannot use a triangulation layout")

    marked_half_edges: set[tuple[int, int]] = set()
    faces: list[list[int]] = []
    for left, right in embedding.edges():
        if (left, right) not in marked_half_edges:
            faces.append(embedding.traverse_face(left, right, marked_half_edges))
    outer = max(
        faces,
        key=lambda face: (len(face), tuple(-int(node) for node in face)),
    )
    boundary = list(dict.fromkeys(outer))
    positions: dict[int, np.ndarray] = {}
    angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, len(boundary), endpoint=False)
    for node, angle in zip(boundary, angles):
        positions[node] = np.array([np.cos(angle), 0.78 * np.sin(angle)], dtype=float)

    interior = [node for node in graph.nodes if node not in positions]
    if interior:
        interior_index = {node: index for index, node in enumerate(interior)}
        system = np.zeros((len(interior), len(interior)), dtype=float)
        rhs = np.zeros((len(interior), 2), dtype=float)
        for node, row in interior_index.items():
            neighbors = list(graph.neighbors(node))
            system[row, row] = len(neighbors)
            for neighbor in neighbors:
                if neighbor in interior_index:
                    system[row, interior_index[neighbor]] -= 1.0
                else:
                    rhs[row] += positions[neighbor]
        solved = np.linalg.solve(system, rhs)
        positions.update({node: solved[index] for node, index in interior_index.items()})

        def orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
            ab = np.asarray(b) - np.asarray(a)
            ac = np.asarray(c) - np.asarray(a)
            return float(ab[0] * ac[1] - ab[1] * ac[0])

        def has_crossing(candidate: dict[int, np.ndarray]) -> bool:
            edges = list(graph.edges)
            for edge_index, (a, b) in enumerate(edges):
                for c, d in edges[edge_index + 1:]:
                    if len({a, b, c, d}) < 4:
                        continue
                    if (
                        orientation(candidate[a], candidate[b], candidate[c])
                        * orientation(candidate[a], candidate[b], candidate[d]) < -1e-10
                        and orientation(candidate[c], candidate[d], candidate[a])
                        * orientation(candidate[c], candidate[d], candidate[b]) < -1e-10
                    ):
                        return True
            return False

        def minimum_spacing(candidate: dict[int, np.ndarray]) -> float:
            coordinates = np.asarray([candidate[node] for node in graph.nodes], dtype=float)
            differences = coordinates[:, None, :] - coordinates[None, :, :]
            distances = np.linalg.norm(differences, axis=2)
            distances[np.eye(len(coordinates), dtype=bool)] = np.inf
            return float(distances.min())

        # Tutte coordinates can coincide for motifs that are planar but not
        # 3-connected.  Relax only the interior nodes and retain a candidate
        # solely when its straight-line edges remain crossing-free.
        best_positions = positions
        best_spacing = minimum_spacing(positions)
        for spring_length in (0.22, 0.32, 0.45, 0.60):
            candidate = nx.spring_layout(
                graph, pos=positions, fixed=boundary, k=spring_length,
                iterations=300, threshold=1e-6, seed=17,
            )
            if not has_crossing(candidate):
                spacing = minimum_spacing(candidate)
                if spacing > best_spacing:
                    best_positions = {
                        node: np.asarray(position, dtype=float)
                        for node, position in candidate.items()
                    }
                    best_spacing = spacing
        positions = best_positions

    array = np.asarray([positions[node] for node in graph.nodes], dtype=float)
    center = array.mean(axis=0)
    array -= center
    spans = np.ptp(array, axis=0)
    if spans[0] > 0:
        array[:, 0] /= spans[0] / 2
    if spans[1] > 0:
        array[:, 1] /= spans[1] / 2
    return {node: array[index] for index, node in enumerate(graph.nodes)}


def draw_representative_motif(
    ax: plt.Axes,
    graph: nx.Graph,
    positions: dict[int, np.ndarray],
    group: str,
    title: str,
) -> None:
    nx.draw_networkx_edges(
        graph, positions, ax=ax, edge_color="#666666", width=1.45, alpha=0.9,
    )
    nodes = sorted(graph.nodes)
    colors = [base.LABEL_COLORS[str(graph.nodes[node]["label"])] for node in nodes]
    edge_color = "#D1495B" if group == "significant" else "#6F8499"
    nx.draw_networkx_nodes(
        graph, positions, ax=ax, nodelist=nodes, node_color=colors,
        node_size=430, edgecolors=edge_color, linewidths=1.8,
    )
    abbreviations = LABEL_ABBREVIATIONS
    nx.draw_networkx_labels(
        graph, positions, ax=ax,
        labels={node: abbreviations[str(graph.nodes[node]["label"])] for node in nodes},
        font_size=6.2, font_weight="bold",
    )
    ax.set_title(title, fontsize=8.1, pad=7, linespacing=1.18)
    ax.margins(0.14)
    ax.set_axis_off()


def generate_representative_gallery(
    features: pd.DataFrame,
    source: pd.DataFrame,
    tables: Path,
    figures: Path,
    seed: int,
    requested_pairs: int,
) -> dict[str, int]:
    """Select, match, tabulate, and draw actual representative TRI-5 motifs."""
    if requested_pairs < 1:
        raise ValueError("--representative-pairs must be positive")
    matrix, standardized, preprocessing = family_balanced_selection_matrix(features)
    preprocessing.to_csv(tables / "tri5_representative_selection_preprocessing.csv", index=False)
    source_index = source.set_index("wl_hash")
    graphs: dict[str, nx.Graph] = {}
    topology_hashes = []
    for motif_hash in features["wl_hash"].astype(str):
        row = source_index.loc[motif_hash]
        graph = base.graph_from_encoding(str(row.node_labels), str(row.edge_list), motif_hash)
        graphs[motif_hash] = graph
        topology_hashes.append(unlabeled_topology_hash(graph))
    working = features.reset_index(drop=True).copy()
    working["unlabeled_topology_hash"] = topology_hashes
    counts = working.groupby(
        ["unlabeled_topology_hash", "significance_group"]
    ).size().unstack(fill_value=0)
    counts = counts.loc[
        counts.get("significant", 0).gt(0) & counts.get("nonsignificant", 0).gt(0)
    ].sort_values(["significant", "nonsignificant"], ascending=False)
    quotas = allocate_topology_quotas(counts, requested_pairs)
    topology_ids = {str(value): index + 1 for index, value in enumerate(counts.index)}

    representatives: list[dict[str, object]] = []
    for topology_hash in counts.index:
        topology_hash = str(topology_hash)
        quota = quotas[topology_hash]
        for group in ["nonsignificant", "significant"]:
            indices = working.index[
                working["unlabeled_topology_hash"].eq(topology_hash)
                & working["significance_group"].eq(group)
            ].to_numpy(int)
            selected = cluster_representatives(
                indices, matrix, quota,
                stable_seed(f"representatives:{topology_hash}:{group}", seed) % (2**32 - 1),
            )
            for value in selected:
                representatives.append({
                    **value, "significance_group": group,
                    "unlabeled_topology_hash": topology_hash,
                    "topology_id": topology_ids[topology_hash],
                })

    representative_frame = pd.DataFrame(representatives)
    pair_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    difference_rows: list[dict[str, object]] = []
    pair_id = 0
    metric_to_block = preprocessing.loc[preprocessing["included"]].set_index("metric")["macro_family"]
    for topology_hash in counts.index:
        topology_hash = str(topology_hash)
        control = representative_frame.loc[
            representative_frame["unlabeled_topology_hash"].eq(topology_hash)
            & representative_frame["significance_group"].eq("nonsignificant")
        ].reset_index(drop=True)
        case = representative_frame.loc[
            representative_frame["unlabeled_topology_hash"].eq(topology_hash)
            & representative_frame["significance_group"].eq("significant")
        ].reset_index(drop=True)
        distances = np.linalg.norm(
            matrix[control["row_index"].to_numpy(int)][:, None, :]
            - matrix[case["row_index"].to_numpy(int)][None, :, :], axis=2,
        )
        control_assignment, case_assignment = linear_sum_assignment(distances)
        for control_position, case_position in zip(control_assignment, case_assignment):
            pair_id += 1
            control_rep = control.iloc[control_position]
            case_rep = case.iloc[case_position]
            control_index = int(control_rep.row_index)
            case_index = int(case_rep.row_index)
            control_row = working.iloc[control_index]
            case_row = working.iloc[case_index]
            pair_distance = float(distances[control_position, case_position])
            deltas = standardized.iloc[case_index] - standardized.iloc[control_index]
            ordered = deltas.abs().sort_values(ascending=False)
            top_metrics = ordered.head(3).index.tolist()
            top_text = "; ".join(
                f"{compact_metric_name(metric, width=31)} "
                f"{'↑' if deltas[metric] > 0 else '↓'}"
                for metric in top_metrics
            )
            control_graph = graphs[str(control_row.wl_hash)]
            case_graph = graphs[str(case_row.wl_hash)]
            pair_rows.append({
                "pair_id": pair_id, "topology_id": int(control_rep.topology_id),
                "unlabeled_topology_hash": topology_hash,
                "unlabeled_graph_isomorphic": bool(nx.is_isomorphic(control_graph, case_graph)),
                "nonsignificant_wl_hash": str(control_row.wl_hash),
                "significant_wl_hash": str(case_row.wl_hash),
                "family_balanced_pair_distance": pair_distance,
                "top_difference_1": top_metrics[0], "top_difference_2": top_metrics[1],
                "top_difference_3": top_metrics[2], "top_differences_display": top_text,
            })
            for side, rep, row in [
                ("left", control_rep, control_row), ("right", case_rep, case_row)
            ]:
                selected_rows.append({
                    "pair_id": pair_id, "panel_side": side,
                    "significance_group": str(rep.significance_group),
                    "topology_id": int(rep.topology_id),
                    "unlabeled_topology_hash": topology_hash, "wl_hash": str(row.wl_hash),
                    "p_value_unadjusted": float(row.p_value_unadjusted),
                    "hazard_ratio_unadjusted": float(row.hazard_ratio_unadjusted),
                    "sample_document_frequency": float(row.sample_document_frequency),
                    "median_patient_raw_count": float(row.median_patient_raw_count),
                    "cluster_id_within_group_topology": int(rep.cluster_id),
                    "cluster_support": int(rep.cluster_support),
                    "distance_to_cluster_center": float(rep.distance_to_cluster_center),
                    "family_balanced_pair_distance": pair_distance,
                    "size_nodes": float(row.size_nodes), "size_edges": float(row.size_edges),
                    "size_triangles": float(row.size_triangles),
                })
            for rank, metric in enumerate(ordered.index, 1):
                control_raw = pd.to_numeric(pd.Series([control_row[metric]]), errors="coerce").iloc[0]
                case_raw = pd.to_numeric(pd.Series([case_row[metric]]), errors="coerce").iloc[0]
                difference_rows.append({
                    "pair_id": pair_id, "topology_id": int(control_rep.topology_id),
                    "metric_rank_within_pair": rank, "metric": metric,
                    "display_name": display_name(metric),
                    "macro_family": str(metric_to_block.loc[metric]),
                    "nonsignificant_value_raw": control_raw,
                    "significant_value_raw": case_raw,
                    "raw_difference_significant_minus_nonsignificant": (
                        float(case_raw - control_raw)
                        if np.isfinite(control_raw) and np.isfinite(case_raw) else np.nan
                    ),
                    "standardized_difference_significant_minus_nonsignificant": float(deltas[metric]),
                    "absolute_standardized_difference": float(abs(deltas[metric])),
                })

    pair_frame = pd.DataFrame(pair_rows).sort_values("pair_id")
    selected_frame = pd.DataFrame(selected_rows).sort_values(["pair_id", "panel_side"])
    differences = pd.DataFrame(difference_rows).sort_values(["pair_id", "metric_rank_within_pair"])
    selected_frame.to_csv(tables / "tri5_representative_motifs.csv", index=False)
    pair_frame.to_csv(tables / "tri5_representative_pair_summary.csv", index=False)
    differences.to_csv(tables / "tri5_representative_pair_criteria_differences.csv", index=False)

    pair_columns = 2
    axes_columns = pair_columns * 2
    figure_rows = math.ceil(len(pair_frame) / pair_columns)
    fig, axes = plt.subplots(
        figure_rows, axes_columns, figsize=(3.55 * axes_columns, 3.45 * figure_rows), squeeze=False,
    )
    for pair_position, pair in enumerate(pair_frame.itertuples(index=False)):
        row_position = pair_position // pair_columns
        pair_column = pair_position % pair_columns
        control_ax = axes[row_position, pair_column * 2]
        case_ax = axes[row_position, pair_column * 2 + 1]
        control_data = selected_frame.loc[
            selected_frame.pair_id.eq(pair.pair_id)
            & selected_frame.significance_group.eq("nonsignificant")
        ].iloc[0]
        case_data = selected_frame.loc[
            selected_frame.pair_id.eq(pair.pair_id)
            & selected_frame.significance_group.eq("significant")
        ].iloc[0]
        control_graph = graphs[str(control_data.wl_hash)]
        case_graph = graphs[str(case_data.wl_hash)]
        mapping = best_label_preserving_isomorphism(control_graph, case_graph)
        control_positions = triangulation_planar_layout(control_graph)
        case_positions = {mapping[node]: position for node, position in control_positions.items()}
        draw_representative_motif(
            control_ax, control_graph, control_positions, "nonsignificant",
            f"Pair {pair.pair_id:02d} | topology {pair.topology_id} | NONSIG\n"
            f"{str(control_data.wl_hash)[:8]}  p={control_data.p_value_unadjusted:.2g}  "
            f"HR={control_data.hazard_ratio_unadjusted:.2g}\n"
            f"cluster n={int(control_data.cluster_support)}  •  center d={control_data.distance_to_cluster_center:.2f}",
        )
        draw_representative_motif(
            case_ax, case_graph, case_positions, "significant",
            f"Pair {pair.pair_id:02d} | topology {pair.topology_id} | SIGNIFICANT\n"
            f"{str(case_data.wl_hash)[:8]}  p={case_data.p_value_unadjusted:.2g}  "
            f"HR={case_data.hazard_ratio_unadjusted:.2g}\n"
            f"cluster n={int(case_data.cluster_support)}  •  pair d={pair.family_balanced_pair_distance:.2f}",
        )
    used_axes = len(pair_frame) * 2
    for ax in axes.flat[used_axes:]:
        ax.set_axis_off()
    handles = [
        plt.Line2D(
            [0], [0], marker="o", color="none", markerfacecolor=base.LABEL_COLORS[label],
            markeredgecolor="white", markersize=8, label=f"{label}={LABEL_NAMES[label]}",
        )
        for label in LABELS
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8.5)
    fig.suptitle(
        "TRI-5 representative motif head-to-head comparison\n"
        "Exact unlabeled topology matched within each pair; crossing-free planar embeddings",
        fontsize=16, y=0.996,
    )
    fig.tight_layout(rect=(0.025, 0.04, 0.975, 0.965), h_pad=1.7, w_pad=1.25)
    fig.savefig(figures / "tri5_representative_motif_head_to_head.png", dpi=220)
    plt.close(fig)
    return {
        "representative_pairs": int(len(pair_frame)),
        "representative_motifs": int(len(selected_frame)),
        "representative_topologies": int(pair_frame.unlabeled_topology_hash.nunique()),
        "representative_selection_criteria": int(len(standardized.columns)),
    }


def regenerate_plots(args: argparse.Namespace) -> dict[str, float | int]:
    tables = args.outdir / "tables"
    figures = args.outdir / "figures"
    features_path = tables / "tri5_motif_criteria.csv.gz"
    comparison_path = tables / "tri5_criteria_comparisons.csv"
    if not features_path.is_file() or not comparison_path.is_file():
        raise FileNotFoundError("--plots-only requires existing motif criteria and comparison tables")
    features = pd.read_csv(features_path, dtype={"wl_hash": str})
    comparison = pd.read_csv(comparison_path)
    figures.mkdir(parents=True, exist_ok=True)
    pages, figure_index = plot_pages(features, comparison, figures, args.panels_per_page)
    figure_index.to_csv(tables / "tri5_distribution_figure_index.csv", index=False)
    plot_leading_distributions(features, comparison, figures / "tri5_metric_distributions.png")
    pca_summary = plot_all_criteria_pca(features, tables, figures, args.seed)
    family_pca_summary = plot_family_pcas(features, tables, figures, args.seed)
    source = pd.read_csv(
        args.trace_dir / "trace_nodes.csv",
        dtype={"wl_hash": str, "node_labels": str, "edge_list": str},
    )
    source = source.loc[source["triangle_level"].eq(args.primary_level)].copy()
    representative_summary = generate_representative_gallery(
        features, source, tables, figures, args.seed, args.representative_pairs
    )
    return {
        "distribution_figure_pages": len(pages), "plotted_criteria": len(figure_index),
        **pca_summary, **family_pca_summary, **representative_summary,
    }


def write_readme(path: Path, features: pd.DataFrame, metrics: list[str], registry: pd.DataFrame, comparison: pd.DataFrame, pages: list[str], args: argparse.Namespace) -> None:
    family_counts = registry.loc[registry.availability.eq("computed")].groupby("family").size().sort_values(ascending=False)
    lines = [
        "# TRI-5 criterion distributions", "",
        "This analysis implements every criterion in `triangulation_motif_topology_criteria.md` that can be derived from the lineage's graph adjacency and cell labels. It compares estimable TRI-5 motifs with raw Cox p < 0.05 against estimable nonsignificant TRI-5 motifs.", "",
        f"- Significant motifs: {(features.significance_group == 'significant').sum():,}",
        f"- Nonsignificant motifs: {(features.significance_group == 'nonsignificant').sum():,}",
        f"- Computed scalar criteria: {len(metrics):,}",
        f"- Distribution figure pages: {len(pages):,}",
        f"- Label-null permutations per motif: {args.permutations}", "",
        "## Files", "",
        "- `tri5_criteria_distribution_long.csv`: one row per metric and significance group with n, missingness, mean, SD, variance, quantiles, range, IQR, and zero fraction.",
        "- `tri5_criteria_distribution_wide.csv`: the same summaries with significant and nonsignificant values side by side.",
        "- `tri5_criteria_comparisons.csv`: location and dispersion tests, Cliff's delta, differences, variance ratios, and BH corrections.",
        "- `criteria_registry.csv`: complete computed/unavailable criterion audit.",
        "- `metric_dictionary.csv`: names and families for every computed scalar criterion.",
        "- `tri5_motif_criteria.csv.gz`: motif-level values used by the tables and plots.",
        "- `distributions_by_family/*.csv`: distribution summaries split by criterion family.",
        "- `tri5_distribution_figure_index.csv`: maps every plotted criterion to its PNG page and panel.",
        "- `tri5_within_motif_distributions.csv`: pooled degree, triangle participation, and path-length histograms.",
        "- `tri5_normalized_laplacian_eigenvalues.csv.gz`: full normalized-Laplacian spectra.", "",
        "- `tri5_pca_scores.csv.gz`: two-dimensional PCA coordinates for every motif.",
        "- `tri5_pca_loadings.csv`: PC1/PC2 loadings for every criterion retained by PCA.",
        "- `tri5_pca_preprocessing.csv`: inclusion, imputation, scaling, and exclusion audit.", "",
        "- `tri5_pca_{topology,composition,contacts,triangles}.png`: separate PCA views by criterion block.",
        "- `tri5_pca_by_family.png`: combined 2x2 view of the four separate PCA fits.",
        "- `tri5_pca_*_{scores,loadings,preprocessing}.csv*`: block-specific PCA audit tables.", "",
        "- `tri5_representative_motif_head_to_head.png`: 12 topology-matched representative motif pairs.",
        "- `tri5_representative_motifs.csv`: selected motifs and cluster-support metadata.",
        "- `tri5_representative_pair_summary.csv`: pair identities, distances, and leading differences.",
        "- `tri5_representative_pair_criteria_differences.csv`: all standardized criterion differences by pair.", "",
        "## Computed families", "",
    ]
    lines.extend(f"- {family.replace('_', ' ')}: {count}" for family, count in family_counts.items())
    guide_path = PROJECT_DIR / "criteria_family_guide.md"
    lines += ["", guide_path.read_text(encoding="utf-8").strip(), ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.plots_only:
        plot_summary = regenerate_plots(args)
        audit_path = args.outdir / "analysis_summary.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.is_file() else {}
        audit.update(plot_summary)
        audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": "complete", "mode": "plots_only", **plot_summary}, indent=2))
        return
    if args.permutations < 2:
        raise ValueError("--permutations must be at least 2")
    source_path = args.trace_dir / "trace_nodes.csv"
    source = pd.read_csv(source_path, dtype={"wl_hash": str, "node_labels": str, "edge_list": str})
    cache = pd.read_csv(args.base_cache, dtype={"wl_hash": str})
    eligible = cache.loc[
        cache["triangle_level"].eq(args.primary_level)
        & cache["eligible_for_comparison"]
    ].copy()
    source_level = source.loc[source["triangle_level"].eq(args.primary_level)].set_index("wl_hash")
    rows = []
    raw_distributions: dict[tuple[str, str, float], int] = defaultdict(int)
    spectra_rows = []
    for position, cached in enumerate(eligible.itertuples(index=False), 1):
        source_row = source_level.loc[str(cached.wl_hash)]
        graph = base.graph_from_encoding(str(source_row.node_labels), str(source_row.edge_list), str(cached.wl_hash))
        values, raw = extra_features(graph, str(cached.wl_hash), args.permutations, args.seed)
        significant = float(cached.p_value_unadjusted) < args.p_threshold
        group = "significant" if significant else "nonsignificant"
        rows.append({
            "triangle_level": args.primary_level, "wl_hash": str(cached.wl_hash),
            "significance_group": group, "significant_raw_p": significant,
            "p_value_unadjusted": float(cached.p_value_unadjusted), "coef_unadjusted": float(cached.coef_unadjusted),
            "hazard_ratio_unadjusted": float(cached.hazard_ratio_unadjusted),
            "sample_document_frequency": float(cached.sample_document_frequency),
            "median_patient_raw_count": float(cached.median_patient_raw_count), **values,
        })
        for distribution, entries in raw.items():
            if distribution == "normalized_laplacian_eigenvalue":
                spectra_rows.extend({"wl_hash": str(cached.wl_hash), "significance_group": group, "eigenvalue_index": index, "eigenvalue": value} for index, value in enumerate(entries))
            else:
                for value, count in Counter(entries).items():
                    raw_distributions[(distribution, group, value)] += count
        if position % 500 == 0 or position == len(eligible):
            print(f"Computed criteria for {position:,}/{len(eligible):,} TRI-{args.primary_level} motifs", flush=True)
    features = pd.DataFrame(rows)
    add_dispersion_metrics(features)
    metrics = [column for column in features.columns if column not in METADATA]
    registry = build_registry(metrics, args.permutations)
    dictionary = registry.loc[registry.availability.eq("computed"), ["criterion", "display_name", "family", "reason_or_method"]].rename(columns={"criterion": "metric"})
    distribution, comparison = distribution_tables(features, metrics, registry)
    wide = distribution.pivot(index=["metric", "display_name", "family"], columns="group").reset_index()
    wide.columns = ["_".join(str(value) for value in column if value).rstrip("_") if isinstance(column, tuple) else column for column in wide.columns]

    tables = args.outdir / "tables"
    figures = args.outdir / "figures"
    families = tables / "distributions_by_family"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    families.mkdir(parents=True, exist_ok=True)
    features.to_csv(tables / "tri5_motif_criteria.csv.gz", index=False, compression="gzip")
    registry.to_csv(tables / "criteria_registry.csv", index=False)
    dictionary.to_csv(tables / "metric_dictionary.csv", index=False)
    distribution.to_csv(tables / "tri5_criteria_distribution_long.csv", index=False)
    wide.to_csv(tables / "tri5_criteria_distribution_wide.csv", index=False)
    comparison.to_csv(tables / "tri5_criteria_comparisons.csv", index=False)
    for family, frame in distribution.groupby("family"):
        frame.to_csv(families / f"{family}.csv", index=False)
    pd.DataFrame([
        {"distribution": key[0], "significance_group": key[1], "value": key[2], "count": count}
        for key, count in sorted(raw_distributions.items())
    ]).to_csv(tables / "tri5_within_motif_distributions.csv", index=False)
    pd.DataFrame(spectra_rows).to_csv(tables / "tri5_normalized_laplacian_eigenvalues.csv.gz", index=False, compression="gzip")
    pages, figure_index = plot_pages(features, comparison, figures, args.panels_per_page)
    figure_index.to_csv(tables / "tri5_distribution_figure_index.csv", index=False)
    # Preserve the familiar filename as a compact view of the eight strongest aggregate metrics.
    plot_leading_distributions(features, comparison, figures / "tri5_metric_distributions.png")
    pca_summary = plot_all_criteria_pca(features, tables, figures, args.seed)
    family_pca_summary = plot_family_pcas(features, tables, figures, args.seed)
    representative_summary = generate_representative_gallery(
        features,
        source.loc[source["triangle_level"].eq(args.primary_level)].copy(),
        tables,
        figures,
        args.seed,
        args.representative_pairs,
    )
    write_readme(tables / "README.md", features, metrics, registry, comparison, pages, args)
    audit = {
        "status": "complete", "primary_level": args.primary_level,
        "significant_motifs": int(features.significant_raw_p.sum()),
        "nonsignificant_motifs": int((~features.significant_raw_p).sum()),
        "computed_scalar_criteria": len(metrics), "unavailable_criteria": int(registry.availability.eq("not_available").sum()),
        "label_null_permutations": args.permutations, "distribution_figure_pages": len(pages),
        **pca_summary,
        **family_pca_summary,
        **representative_summary,
        "input_trace": str(source_path.resolve()), "base_cache": str(args.base_cache.resolve()),
        "reproduction_command": "python breast_cancer_survival/motif_results/sibling_topology_analysis_new/run_analysis.py",
    }
    (args.outdir / "analysis_summary.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
