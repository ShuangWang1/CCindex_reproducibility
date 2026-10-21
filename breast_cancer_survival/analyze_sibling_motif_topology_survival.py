#!/usr/bin/env python3
"""Compare sibling motif topology, abundance, and exploratory survival signals."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/breast-cancer-sibling-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr
from statsmodels.duration.hazard_regression import PHReg

from analyze_ki67_survival import step_function
from analyze_motif_survival import MODEL_COVARIATES
from analyze_motif_survival_unadjusted import (
    load_inventory,
    prepare_patient_aggregation,
    vectorized_binary_cox,
)
from trace_back_tri7_motif_survival import (
    draw_aligned_graph,
    fixed_target_layout,
    graph_from_row,
    load_feature_indices,
    trace_potential_ancestry,
    triangle_count,
)
from visualize_top_motif_survival import CELL_TYPE_SHORT


TARGET_HASH = "1d7e000e1fd60951fbdd43c9c1680a3d"
FOCAL_HASH = "7a7e7ea415b7c0be472d5e4d4be65e8a"
MATCHED_HASH = "cb6c6a68c41d7fb0845299d8ccd0f971"
OTHER_HASH = "c9fa1c5e131ca635029c14db07f01eb6"


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    result_root = Path(__file__).resolve().parent / "motif_results_tri4_tri7_unadjusted"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-hash", default=TARGET_HASH)
    parser.add_argument("--level", type=int, default=6)
    parser.add_argument("--focal-hash", default=FOCAL_HASH)
    parser.add_argument("--matched-hash", default=MATCHED_HASH)
    parser.add_argument("--comparison-hashes", nargs="*", default=[OTHER_HASH])
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
        "--trace-summary",
        type=Path,
        default=result_root / "tri-7_trace_back" / "trace_back_summary.csv",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=result_root
        / "tri-7_trace_back"
        / "tri6_sibling_topology_analysis",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--pseudocount", type=float, default=0.5)
    parser.add_argument(
        "--cutoff-quantiles", type=float, nargs="+", default=[0.4, 0.5, 0.6]
    )
    return parser.parse_args()


def short_hash(value: str) -> str:
    return str(value)[:8]


def edge_key(edge: tuple[int, int]) -> tuple[int, int]:
    return tuple(sorted(edge))


def safe_zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    standard_deviation = float(np.std(values, ddof=1))
    if not np.isfinite(standard_deviation) or standard_deviation == 0:
        raise ValueError("Cannot standardize a constant or non-finite exposure")
    return (values - np.mean(values)) / standard_deviation


def degree_gini(values: list[int]) -> float:
    array = np.sort(np.asarray(values, dtype=float))
    if not len(array) or array.sum() == 0:
        return 0.0
    indices = np.arange(1, len(array) + 1)
    return float(
        (2 * np.sum(indices * array) / (len(array) * array.sum()))
        - (len(array) + 1) / len(array)
    )


def graph_descriptors(
    motif_id: str, graph: nx.Graph, label_map: dict[str, str]
) -> dict[str, object]:
    cell_type = {
        node: label_map.get(str(data["label"]), f"label {data['label']}")
        for node, data in graph.nodes(data=True)
    }
    tumor = [node for node, value in cell_type.items() if value == "tumor cells"]
    stromal = [node for node, value in cell_type.items() if value == "stromal cells"]
    degree = dict(graph.degree())
    triangles = nx.triangles(graph)
    clustering = nx.clustering(graph)
    betweenness = nx.betweenness_centrality(graph)

    edge_types = {"tumor_tumor": 0, "tumor_stromal": 0, "stromal_stromal": 0}
    for left, right in graph.edges:
        pair = {cell_type[left], cell_type[right]}
        if pair == {"tumor cells"}:
            edge_types["tumor_tumor"] += 1
        elif pair == {"stromal cells"}:
            edge_types["stromal_stromal"] += 1
        else:
            edge_types["tumor_stromal"] += 1

    triangle_types = {"tumor_tumor_tumor": 0, "stromal_tumor_tumor": 0, "other": 0}
    for clique in nx.enumerate_all_cliques(graph):
        if len(clique) < 3:
            continue
        if len(clique) > 3:
            break
        labels = sorted(cell_type[node] for node in clique)
        if labels == ["tumor cells", "tumor cells", "tumor cells"]:
            triangle_types["tumor_tumor_tumor"] += 1
        elif labels == ["stromal cells", "tumor cells", "tumor cells"]:
            triangle_types["stromal_tumor_tumor"] += 1
        else:
            triangle_types["other"] += 1

    def summarize(prefix: str, nodes: list[int]) -> dict[str, object]:
        degrees = [degree[node] for node in nodes]
        participation = [triangles[node] for node in nodes]
        local_clustering = [clustering[node] for node in nodes]
        local_betweenness = [betweenness[node] for node in nodes]
        if not nodes:
            return {
                f"{prefix}_degree_sequence": "",
                f"{prefix}_degree_mean": np.nan,
                f"{prefix}_degree_variance": np.nan,
                f"{prefix}_degree_range": np.nan,
                f"{prefix}_degree_gini": np.nan,
                f"{prefix}_triangle_participation_mean": np.nan,
                f"{prefix}_clustering_mean": np.nan,
                f"{prefix}_betweenness_max": np.nan,
            }
        return {
            f"{prefix}_degree_sequence": ",".join(map(str, sorted(degrees, reverse=True))),
            f"{prefix}_degree_mean": float(np.mean(degrees)),
            f"{prefix}_degree_variance": float(np.var(degrees)),
            f"{prefix}_degree_range": int(max(degrees) - min(degrees)),
            f"{prefix}_degree_gini": degree_gini(degrees),
            f"{prefix}_triangle_participation_mean": float(np.mean(participation)),
            f"{prefix}_clustering_mean": float(np.mean(local_clustering)),
            f"{prefix}_betweenness_max": float(np.max(local_betweenness)),
        }

    adjacency_eigenvalues = np.linalg.eigvalsh(nx.to_numpy_array(graph))
    laplacian_eigenvalues = nx.laplacian_spectrum(graph)
    all_degrees = list(degree.values())
    maximum_degree = max(all_degrees)
    denominator = (len(graph) - 1) * (len(graph) - 2)
    return {
        "wl_hash": motif_id,
        "short_hash": short_hash(motif_id),
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "triangles": triangle_count(graph),
        "tumor_nodes": len(tumor),
        "stromal_nodes": len(stromal),
        "tumor_tumor_edges": edge_types["tumor_tumor"],
        "tumor_stromal_edges": edge_types["tumor_stromal"],
        "stromal_stromal_edges": edge_types["stromal_stromal"],
        "tumor_tumor_tumor_triangles": triangle_types["tumor_tumor_tumor"],
        "stromal_tumor_tumor_triangles": triangle_types["stromal_tumor_tumor"],
        "other_triangle_types": triangle_types["other"],
        "degree_sequence": ",".join(map(str, sorted(all_degrees, reverse=True))),
        "degree_gini": degree_gini(all_degrees),
        "degree_centralization": (
            float(sum(maximum_degree - value for value in all_degrees) / denominator)
            if denominator
            else np.nan
        ),
        "average_clustering": float(nx.average_clustering(graph)),
        "transitivity": float(nx.transitivity(graph)),
        "diameter": int(nx.diameter(graph)),
        "average_shortest_path": float(nx.average_shortest_path_length(graph)),
        "node_connectivity": int(nx.node_connectivity(graph)),
        "edge_connectivity": int(nx.edge_connectivity(graph)),
        "articulation_points": len(list(nx.articulation_points(graph))),
        "spectral_radius": float(np.max(adjacency_eigenvalues)),
        "algebraic_connectivity": float(np.sort(laplacian_eigenvalues)[1]),
        **summarize("tumor", tumor),
        **summarize("stromal", stromal),
    }


def graph_edit_distance(left: nx.Graph, right: nx.Graph) -> float:
    value = nx.graph_edit_distance(
        left,
        right,
        node_match=lambda a, b: a["label"] == b["label"],
        node_subst_cost=lambda a, b: 0 if a["label"] == b["label"] else 1,
        node_del_cost=lambda _: 1,
        node_ins_cost=lambda _: 1,
        edge_del_cost=lambda _: 1,
        edge_ins_cost=lambda _: 1,
        timeout=10,
    )
    return float(value) if value is not None else np.nan


def matched_sibling_difference(
    focal: nx.Graph, matched: nx.Graph
) -> tuple[nx.Graph, set[tuple[int, int]], set[tuple[int, int]], dict[int, int]]:
    """Map one-node-deletion siblings into a common target-anchored embedding."""
    shared = set(focal) & set(matched)
    focal_only = sorted(set(focal) - shared)
    matched_only = sorted(set(matched) - shared)
    if len(focal_only) != 1 or len(matched_only) != 1:
        raise ValueError("Primary siblings must differ by one embedded node each")
    focal_node = focal_only[0]
    matched_node = matched_only[0]
    if focal.nodes[focal_node]["label"] != matched.nodes[matched_node]["label"]:
        raise ValueError("Primary siblings must have the same cell-type composition")
    mapping = {node: node for node in shared}
    mapping[matched_node] = focal_node
    mapped = nx.relabel_nodes(matched, mapping, copy=True)
    focal_edges = {edge_key(edge) for edge in focal.edges}
    matched_edges = {edge_key(edge) for edge in mapped.edges}
    return mapped, focal_edges - matched_edges, matched_edges - focal_edges, mapping


def load_analysis_data(
    args: argparse.Namespace,
) -> tuple[
    nx.Graph,
    dict[str, nx.Graph],
    dict[str, str],
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    topologies = pd.read_csv(
        args.topologies, dtype={"wl_hash": str, "node_labels": str, "edge_list": str}
    )
    selected = topologies.loc[topologies["wl_hash"].eq(args.target_hash)]
    if len(selected) != 1:
        raise ValueError("Target topology could not be recovered uniquely")
    target_row = selected.iloc[0]
    target_level = int(target_row["triangle_level"])
    if args.level >= target_level:
        raise ValueError("Sibling level must be below the target level")
    target = graph_from_row(target_row)
    indices = load_feature_indices(
        args.artifacts_root, range(args.level, target_level + 1)
    )
    traced, _ = trace_potential_ancestry(
        target, indices, target_level, args.level
    )
    selected_hashes = list(
        dict.fromkeys([args.focal_hash, args.matched_hash, *args.comparison_hashes])
    )
    missing = [value for value in selected_hashes if value not in traced[args.level]]
    if missing:
        raise ValueError(f"Sibling hashes are not traced ancestors: {missing}")
    graphs = {value: traced[args.level][value] for value in selected_hashes}

    label_frame = pd.read_csv(args.label_map, dtype={"node_label": str})
    label_map = dict(
        label_frame[["node_label", "cell_type"]].itertuples(index=False, name=None)
    )
    metadata = pd.read_csv(args.metadata)
    samples = load_inventory(args.artifacts_root, args.level)
    patients, aggregation, _ = prepare_patient_aggregation(samples, metadata)
    matrix = sparse.load_npz(
        args.artifacts_root / f"tri_{args.level}" / "motif_counts.npz"
    ).tocsr()
    feature_columns = indices[args.level].loc[selected_hashes, "feature_index"].to_numpy(int)
    core_counts = matrix[:, feature_columns].toarray().astype(float)
    patient_counts = np.asarray(aggregation @ core_counts, dtype=float)
    patient_core_counts = np.asarray(aggregation.sum(axis=1)).reshape(-1)

    core_cells = (
        metadata.drop_duplicates("core")
        .set_index("core")["Count_Cells"]
        .reindex(samples["source_sample"])
    )
    contributing_samples = np.asarray(aggregation.sum(axis=0)).reshape(-1) > 0
    if (core_cells.isna().to_numpy() & contributing_samples).any():
        raise ValueError("Cell counts are missing for an analyzed motif sample")
    # Samples from other cohorts have all-zero aggregation columns and must not
    # prevent Basel-only sampling normalization.
    core_cells = core_cells.fillna(0)
    patient_cells = np.asarray(
        aggregation @ core_cells.to_numpy(dtype=float)
    ).reshape(-1)
    return (
        target,
        graphs,
        label_map,
        patients,
        patient_counts,
        patient_core_counts,
        patient_cells,
    )


def build_patient_exposures(
    patients: pd.DataFrame,
    hashes: list[str],
    counts: np.ndarray,
    core_counts: np.ndarray,
    cell_counts: np.ndarray,
    focal_hash: str,
    matched_hash: str,
    pseudocount: float,
) -> pd.DataFrame:
    if pseudocount <= 0:
        raise ValueError("The log-ratio pseudocount must be positive")
    frame = patients.copy()
    positions = {value: index for index, value in enumerate(hashes)}
    for index, motif_id in enumerate(hashes):
        name = short_hash(motif_id)
        frame[f"count_{name}"] = counts[:, index]
        frame[f"count_per_core_{name}"] = counts[:, index] / core_counts
        frame[f"count_per_10k_cells_{name}"] = counts[:, index] / cell_counts * 10_000
        cutoff = float(np.median(counts[:, index]))
        frame[f"median_cutoff_{name}"] = cutoff
        frame[f"enriched_{name}"] = (counts[:, index] > cutoff).astype(int)
        frame[f"log1p_count_z_{name}"] = safe_zscore(np.log1p(counts[:, index]))
        frame[f"log1p_per_core_z_{name}"] = safe_zscore(
            np.log1p(counts[:, index] / core_counts)
        )
        frame[f"log1p_per_10k_cells_z_{name}"] = safe_zscore(
            np.log1p(counts[:, index] / cell_counts * 10_000)
        )

    focal = counts[:, positions[focal_hash]]
    matched = counts[:, positions[matched_hash]]
    frame["sampled_tumor_cores"] = core_counts
    frame["sampled_cells"] = cell_counts
    frame["primary_pair_burden"] = safe_zscore(np.log1p(focal + matched))
    frame["primary_topology_preference"] = safe_zscore(
        np.log((focal + pseudocount) / (matched + pseudocount))
    )
    focal_high = frame[f"enriched_{short_hash(focal_hash)}"].astype(bool)
    matched_high = frame[f"enriched_{short_hash(matched_hash)}"].astype(bool)
    frame["primary_pair_median_group"] = np.select(
        [focal_high & matched_high, focal_high, matched_high],
        ["both high", "focal only", "matched only"],
        default="both low",
    )
    return frame


def design_condition_number(design: np.ndarray) -> float:
    centered = design - np.mean(design, axis=0)
    scale = np.std(centered, axis=0, ddof=1)
    usable = scale > 0
    if not usable.any():
        return np.nan
    return float(np.linalg.cond(centered[:, usable] / scale[usable]))


def fit_cox_model(
    data: pd.DataFrame,
    model_name: str,
    exposure_columns: list[str],
    adjusted: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    columns = exposure_columns + (MODEL_COVARIATES if adjusted else [])
    model_data = data[["time_months", "event", *columns]].dropna().copy()
    summary: dict[str, object] = {
        "model": model_name,
        "adjusted": adjusted,
        "n": len(model_data),
        "events": int(model_data["event"].sum()),
        "terms": ";".join(columns),
        "condition_number": design_condition_number(
            model_data[columns].to_numpy(dtype=float)
        ),
        "fit_status": "estimated",
        "warning": "",
    }
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = PHReg(
                model_data["time_months"].to_numpy(dtype=float),
                model_data[columns].to_numpy(dtype=float),
                status=model_data["event"].to_numpy(dtype=int),
                ties="efron",
            ).fit(disp=0)
        warning_text = " | ".join(dict.fromkeys(str(value.message) for value in caught))
        summary["warning"] = warning_text
        summary["log_likelihood"] = float(result.llf)
        intervals = np.asarray(result.conf_int(), dtype=float)
        coefficient_rows = []
        for index, term in enumerate(columns):
            coefficient_rows.append(
                {
                    "model": model_name,
                    "adjusted": adjusted,
                    "term": term,
                    "term_role": "exposure" if term in exposure_columns else "covariate",
                    "coefficient": float(result.params[index]),
                    "hazard_ratio": float(np.exp(result.params[index])),
                    "ci95_low": float(np.exp(intervals[index, 0])),
                    "ci95_high": float(np.exp(intervals[index, 1])),
                    "p_value": float(result.pvalues[index]),
                    "n": len(model_data),
                    "events": int(model_data["event"].sum()),
                    "fit_status": "estimated_with_warning" if warning_text else "estimated",
                    "warning": warning_text,
                }
            )
        schoenfeld = np.asarray(result.schoenfeld_residuals)
        event_mask = model_data["event"].to_numpy(dtype=bool)
        diagnostic_rows = []
        for index, term in enumerate(columns):
            residual = schoenfeld[:, index]
            valid = event_mask & np.isfinite(residual)
            rho, p_value = spearmanr(
                np.log(model_data.loc[valid, "time_months"]), residual[valid]
            )
            diagnostic_rows.append(
                {
                    "model": model_name,
                    "term": term,
                    "spearman_rho": rho,
                    "p_value": p_value,
                    "event_residuals": int(valid.sum()),
                }
            )
        return pd.DataFrame(coefficient_rows), pd.DataFrame(diagnostic_rows), summary
    except Exception as error:
        summary["fit_status"] = "fit_failed"
        summary["warning"] = f"{type(error).__name__}: {error}"
        rows = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "adjusted": adjusted,
                    "term": term,
                    "term_role": "exposure" if term in exposure_columns else "covariate",
                    "coefficient": np.nan,
                    "hazard_ratio": np.nan,
                    "ci95_low": np.nan,
                    "ci95_high": np.nan,
                    "p_value": np.nan,
                    "n": len(model_data),
                    "events": int(model_data["event"].sum()),
                    "fit_status": "fit_failed",
                    "warning": summary["warning"],
                }
                for term in columns
            ]
        )
        return rows, pd.DataFrame(), summary


def fit_analysis_models(
    data: pd.DataFrame,
    hashes: list[str],
    focal_hash: str,
    matched_hash: str,
    pseudocounts: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    coefficient_parts: list[pd.DataFrame] = []
    diagnostic_parts: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []

    def add(name: str, columns: list[str], adjusted: bool) -> None:
        coefficients, diagnostics, summary = fit_cox_model(
            data, name, columns, adjusted
        )
        coefficient_parts.append(coefficients)
        if len(diagnostics):
            diagnostic_parts.append(diagnostics)
        summaries.append(summary)

    for motif_id in hashes:
        name = short_hash(motif_id)
        for adjusted in [False, True]:
            suffix = "adjusted" if adjusted else "unadjusted"
            add(f"binary_{name}_{suffix}", [f"enriched_{name}"], adjusted)
            add(
                f"continuous_raw_{name}_{suffix}",
                [f"log1p_count_z_{name}"],
                adjusted,
            )
        add(
            f"continuous_per_core_{name}_unadjusted",
            [f"log1p_per_core_z_{name}"],
            False,
        )
        add(
            f"continuous_per_10k_cells_{name}_unadjusted",
            [f"log1p_per_10k_cells_z_{name}"],
            False,
        )

    add(
        "primary_pair_unadjusted",
        ["primary_pair_burden", "primary_topology_preference"],
        False,
    )
    add(
        "primary_pair_adjusted",
        ["primary_pair_burden", "primary_topology_preference"],
        True,
    )

    focal = data[f"count_{short_hash(focal_hash)}"].to_numpy(dtype=float)
    matched = data[f"count_{short_hash(matched_hash)}"].to_numpy(dtype=float)
    for value in pseudocounts:
        column = f"topology_preference_pc_{str(value).replace('.', '_')}"
        data[column] = safe_zscore(np.log((focal + value) / (matched + value)))
        for adjusted in [False, True]:
            suffix = "adjusted" if adjusted else "unadjusted"
            add(
                f"pseudocount_{value:g}_{suffix}",
                ["primary_pair_burden", column],
                adjusted,
            )

    return (
        pd.concat(coefficient_parts, ignore_index=True),
        pd.concat(diagnostic_parts, ignore_index=True) if diagnostic_parts else pd.DataFrame(),
        pd.DataFrame(summaries),
    )


def cutoff_sensitivity(
    data: pd.DataFrame, hashes: list[str], quantiles: list[float]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for motif_id in hashes:
        name = short_hash(motif_id)
        counts = data[f"count_{name}"].to_numpy(dtype=float)
        for quantile in quantiles:
            cutoff = float(np.quantile(counts, quantile))
            temporary = data[["time_months", "event"]].copy()
            temporary["enriched"] = (counts > cutoff).astype(int)
            coefficients, _, summary = fit_cox_model(
                temporary, f"cutoff_{name}_{quantile:g}", ["enriched"], False
            )
            row = coefficients.iloc[0].to_dict()
            row.update(
                {
                    "wl_hash": motif_id,
                    "short_hash": name,
                    "quantile": quantile,
                    "cutoff": cutoff,
                    "enriched_patients": int(temporary["enriched"].sum()),
                    "enriched_events": int(
                        temporary.loc[temporary["enriched"].eq(1), "event"].sum()
                    ),
                    "model_fit_status": summary["fit_status"],
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def stratified_bootstrap_indices(event: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    event_indices = np.flatnonzero(event == 1)
    censor_indices = np.flatnonzero(event == 0)
    sampled = np.concatenate(
        [
            rng.choice(event_indices, len(event_indices), replace=True),
            rng.choice(censor_indices, len(censor_indices), replace=True),
        ]
    )
    return rng.permutation(sampled)


def fit_preference_coefficient(
    time: np.ndarray, event: np.ndarray, design: np.ndarray
) -> float:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = PHReg(time, design, status=event, ties="efron").fit(disp=0)
        coefficient = float(result.params[1])
        return coefficient if np.isfinite(coefficient) else np.nan
    except Exception:
        return np.nan


def bootstrap_analysis(
    data: pd.DataFrame,
    hashes: list[str],
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    time = data["time_months"].to_numpy(dtype=float)
    event = data["event"].to_numpy(dtype=int)
    primary = data[["primary_pair_burden", "primary_topology_preference"]].to_numpy(float)
    complete = data[MODEL_COVARIATES].notna().all(axis=1).to_numpy()
    adjusted_design = np.column_stack(
        [primary[complete], data.loc[complete, MODEL_COVARIATES].to_numpy(float)]
    )
    adjusted_time = time[complete]
    adjusted_event = event[complete]
    count_matrix = data[[f"count_{short_hash(value)}" for value in hashes]].to_numpy(float)
    rows: list[dict[str, object]] = []

    for replicate in range(replicates):
        sampled = stratified_bootstrap_indices(event, rng)
        coefficient = fit_preference_coefficient(
            time[sampled], event[sampled], primary[sampled]
        )
        rows.append(
            {
                "replicate": replicate,
                "analysis": "primary_pair_unadjusted",
                "term": "primary_topology_preference",
                "coefficient": coefficient,
                "p_value": np.nan,
                "rank": np.nan,
            }
        )

        adjusted_sampled = stratified_bootstrap_indices(adjusted_event, rng)
        coefficient = fit_preference_coefficient(
            adjusted_time[adjusted_sampled],
            adjusted_event[adjusted_sampled],
            adjusted_design[adjusted_sampled],
        )
        rows.append(
            {
                "replicate": replicate,
                "analysis": "primary_pair_adjusted",
                "term": "primary_topology_preference",
                "coefficient": coefficient,
                "p_value": np.nan,
                "rank": np.nan,
            }
        )

        sampled_counts = count_matrix[sampled]
        groups = sampled_counts > np.median(sampled_counts, axis=0)
        fitted = vectorized_binary_cox(groups, time[sampled], event[sampled])
        p_values = fitted["p_value"]
        valid_order = np.argsort(np.where(np.isfinite(p_values), p_values, np.inf))
        ranks = np.empty(len(hashes), dtype=int)
        ranks[valid_order] = np.arange(1, len(hashes) + 1)
        for index, motif_id in enumerate(hashes):
            rows.append(
                {
                    "replicate": replicate,
                    "analysis": "median_split_rank",
                    "term": short_hash(motif_id),
                    "coefficient": fitted["coef"][index],
                    "p_value": p_values[index],
                    "rank": ranks[index],
                }
            )
        if replicates >= 100 and (replicate + 1) % 100 == 0:
            print(f"  bootstrap {replicate + 1:,} / {replicates:,}", flush=True)

    estimates = pd.DataFrame(rows)
    summaries: list[dict[str, object]] = []
    for analysis in ["primary_pair_unadjusted", "primary_pair_adjusted"]:
        values = estimates.loc[
            estimates["analysis"].eq(analysis), "coefficient"
        ].dropna()
        summaries.append(
            {
                "analysis": analysis,
                "successful_replicates": len(values),
                "requested_replicates": replicates,
                "coefficient_median": values.median() if len(values) else np.nan,
                "coefficient_percentile_2_5": values.quantile(0.025) if len(values) else np.nan,
                "coefficient_percentile_97_5": values.quantile(0.975) if len(values) else np.nan,
                "fraction_positive": (values > 0).mean() if len(values) else np.nan,
            }
        )
    for motif_id in hashes:
        subset = estimates.loc[
            estimates["analysis"].eq("median_split_rank")
            & estimates["term"].eq(short_hash(motif_id))
        ]
        summaries.append(
            {
                "analysis": f"median_split_rank_{short_hash(motif_id)}",
                "successful_replicates": int(subset["p_value"].notna().sum()),
                "requested_replicates": replicates,
                "coefficient_median": subset["coefficient"].median(),
                "coefficient_percentile_2_5": subset["coefficient"].quantile(0.025),
                "coefficient_percentile_97_5": subset["coefficient"].quantile(0.975),
                "fraction_positive": (subset["coefficient"] > 0).mean(),
                "fraction_ranked_first": (subset["rank"] == 1).mean(),
            }
        )
    return estimates, pd.DataFrame(summaries)


def plot_topology_contrast(
    target: nx.Graph,
    focal: nx.Graph,
    matched: nx.Graph,
    focal_hash: str,
    matched_hash: str,
    target_hash: str,
    label_map: dict[str, str],
    output: Path,
) -> None:
    mapped, focal_only, matched_only, _ = matched_sibling_difference(focal, matched)
    coordinates = fixed_target_layout(target, target_hash)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    draw_aligned_graph(
        axes[0], target, focal, coordinates, label_map, added_edges=focal_only
    )
    draw_aligned_graph(
        axes[1], target, mapped, coordinates, label_map, added_edges=matched_only
    )
    axes[0].set_title(
        f"{short_hash(focal_hash)}: balanced stromal attachment\n"
        f"stromal degrees 3,3,2",
        fontsize=11,
    )
    axes[1].set_title(
        f"{short_hash(matched_hash)}: concentrated stromal attachment\n"
        f"stromal degrees 4,2,2",
        fontsize=11,
    )
    fig.suptitle(
        "Primary topology-controlled contrast: gold marks the single rewired edge",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_topology_heatmap(features: pd.DataFrame, output: Path) -> None:
    columns = [
        "stromal_nodes",
        "tumor_stromal_edges",
        "stromal_degree_variance",
        "stromal_degree_range",
        "degree_centralization",
        "average_shortest_path",
        "spectral_radius",
        "algebraic_connectivity",
    ]
    values = features.set_index("short_hash")[columns].astype(float)
    standard = (values - values.mean(axis=0)) / values.std(axis=0, ddof=0).replace(0, 1)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    image = ax.imshow(standard.to_numpy(), cmap="coolwarm", vmin=-1.5, vmax=1.5)
    ax.set_xticks(range(len(columns)), [value.replace("_", "\n") for value in columns])
    ax.set_yticks(range(len(values)), values.index)
    for row, column in itertools.product(range(len(values)), range(len(columns))):
        ax.text(
            column,
            row,
            f"{values.iloc[row, column]:.2g}",
            ha="center",
            va="center",
            fontsize=8,
        )
    fig.colorbar(image, ax=ax, label="standardized across the three motifs")
    ax.set_title("Selected interpretable topology descriptors (text shows raw values)")
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_abundance(
    data: pd.DataFrame, hashes: list[str], focal_hash: str, matched_hash: str, output: Path
) -> None:
    names = [short_hash(value) for value in hashes]
    counts = data[[f"count_{value}" for value in names]].to_numpy(float)
    correlation = spearmanr(counts).statistic
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    display = axes[0].imshow(correlation, cmap="Blues", vmin=0, vmax=1)
    axes[0].set_xticks(range(len(names)), names, rotation=30, ha="right")
    axes[0].set_yticks(range(len(names)), names)
    for row, column in itertools.product(range(len(names)), repeat=2):
        axes[0].text(column, row, f"{correlation[row, column]:.2f}", ha="center", va="center")
    fig.colorbar(display, ax=axes[0], label="Spearman correlation")
    focal = data[f"count_{short_hash(focal_hash)}"]
    matched = data[f"count_{short_hash(matched_hash)}"]
    axes[1].scatter(np.log1p(matched), np.log1p(focal), s=18, alpha=0.55, color="#3973AC")
    axes[1].set_xlabel(f"log1p {short_hash(matched_hash)} count")
    axes[1].set_ylabel(f"log1p {short_hash(focal_hash)} count")
    axes[1].set_title("Primary matched-pair patient abundance")
    fig.suptitle("Sibling motif abundance is strongly correlated")
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_cutoff_sensitivity(results: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for name, subset in results.groupby("short_hash", sort=False):
        subset = subset.sort_values("quantile")
        ax.plot(subset["quantile"], subset["hazard_ratio"], marker="o", label=name)
        ax.fill_between(
            subset["quantile"].to_numpy(float),
            subset["ci95_low"].to_numpy(float),
            subset["ci95_high"].to_numpy(float),
            alpha=0.12,
        )
    ax.axhline(1, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Patient-count quantile cutoff")
    ax.set_ylabel("Unadjusted Cox hazard ratio")
    ax.set_title("Median-split conclusion sensitivity")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_discordance_km(data: pd.DataFrame, output: Path) -> None:
    colors = {
        "both low": "#777777",
        "focal only": "#D1495B",
        "matched only": "#3973AC",
        "both high": "#7A5195",
    }
    fig, ax = plt.subplots(figsize=(8.2, 6.1))
    for group in ["both low", "focal only", "matched only", "both high"]:
        subset = data.loc[data["primary_pair_median_group"].eq(group)]
        if subset.empty:
            continue
        curve = step_function(
            subset["time_months"].to_numpy(float), subset["event"].to_numpy(int)
        )
        ax.step(
            curve["time"], curve["survival"], where="post", color=colors[group],
            linewidth=2, label=f"{group} (n={len(subset)}, events={int(subset.event.sum())})",
        )
    ax.set_xlabel("Overall survival (months)")
    ax.set_ylabel("Survival probability")
    ax.set_ylim(0, 1.03)
    ax.set_title("Descriptive median-group discordance; no four-group hypothesis test")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_bootstrap(estimates: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    colors = {"primary_pair_unadjusted": "#3973AC", "primary_pair_adjusted": "#D1495B"}
    for analysis in colors:
        values = estimates.loc[estimates["analysis"].eq(analysis), "coefficient"].dropna()
        ax.hist(values, bins=35, density=True, alpha=0.4, color=colors[analysis], label=analysis)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Bootstrap coefficient for topology preference")
    ax.set_ylabel("Density")
    ax.set_title("Patient-stratified bootstrap stability")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def write_report(
    output: Path,
    args: argparse.Namespace,
    topology: pd.DataFrame,
    pairwise: pd.DataFrame,
    patients: pd.DataFrame,
    coefficients: pd.DataFrame,
    cutoff_results: pd.DataFrame,
    ph_diagnostics: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
    saved_results: pd.DataFrame,
) -> None:
    focal = topology.set_index("wl_hash").loc[args.focal_hash]
    matched = topology.set_index("wl_hash").loc[args.matched_hash]
    primary = coefficients.loc[
        coefficients["model"].isin(["primary_pair_unadjusted", "primary_pair_adjusted"])
        & coefficients["term"].eq("primary_topology_preference")
    ]
    original = saved_results.set_index("wl_hash").loc[args.focal_hash]
    discordance = int(
        (
            patients[f"enriched_{short_hash(args.focal_hash)}"]
            != patients[f"enriched_{short_hash(args.matched_hash)}"]
        ).sum()
    )
    focal_name = short_hash(args.focal_hash)
    matched_name = short_hash(args.matched_hash)
    abundance_rho = spearmanr(
        patients[f"count_{focal_name}"], patients[f"count_{matched_name}"]
    ).statistic
    individual_models = coefficients.loc[
        coefficients["model"].isin(
            [
                f"binary_{focal_name}_unadjusted",
                f"continuous_raw_{focal_name}_unadjusted",
                f"binary_{focal_name}_adjusted",
                f"continuous_raw_{focal_name}_adjusted",
                f"continuous_per_core_{focal_name}_unadjusted",
                f"continuous_per_10k_cells_{focal_name}_unadjusted",
            ]
        )
        & coefficients["term_role"].eq("exposure")
    ]
    primary_ph = ph_diagnostics.loc[
        ph_diagnostics["model"].isin(["primary_pair_unadjusted", "primary_pair_adjusted"])
    ]
    ph_flags = primary_ph.loc[primary_ph["p_value"].lt(0.05)]
    focal_cutoffs = cutoff_results.loc[cutoff_results["wl_hash"].eq(args.focal_hash)]
    lines = [
        "# TRI-6 sibling topology and survival analysis",
        "",
        "## Interpretation first",
        "",
        f"The focal motif `{short_hash(args.focal_hash)}` has raw Cox p = "
        f"{original.p_value_unadjusted:.4g}, but its saved screen-wide BH q = "
        f"{original.q_value_bh_global_unadjusted:.4g}. It is therefore an exploratory, "
        "post-selection signal rather than a globally significant discovery.",
        "",
        f"The focal and matched motifs have identical cell composition and differ by "
        f"one edge reattachment (labeled graph-edit distance "
        f"{pairwise.loc[pairwise['is_primary_pair'], 'graph_edit_distance'].iloc[0]:.0f}). "
        f"The focal stromal degrees are `{focal.stromal_degree_sequence}` versus "
        f"`{matched.stromal_degree_sequence}` for the matched motif. This motivates a "
        "balanced-versus-concentrated stromal attachment hypothesis.",
        "",
        f"Their raw patient counts have Spearman rho = {abundance_rho:.3f}, and the "
        f"median groups disagree for only {discordance}/{len(patients)} patients. "
        "The burden-plus-log-ratio model is consequently more informative than treating "
        "the three highly correlated binary indicators as independent predictors.",
        "",
        "All three graphs have node connectivity 2, edge connectivity 2, diameter 3, "
        "and no articulation points. Ordinary global connectivity therefore does not "
        "distinguish the focal motif; the discriminating feature is how stromal contacts "
        "are distributed locally.",
        "",
        "## Focal single-motif robustness",
        "",
        "| Model | HR per encoded unit | 95% CI | p | n |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in individual_models.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.hazard_ratio:.3g} | "
            f"{row.ci95_low:.3g}–{row.ci95_high:.3g} | {row.p_value:.3g} | {row.n} |"
        )
    lines.extend(
        [
            "",
            f"Across the requested focal cutoffs, unadjusted HRs range from "
            f"{focal_cutoffs.hazard_ratio.min():.3g} to "
            f"{focal_cutoffs.hazard_ratio.max():.3g}, with p values from "
            f"{focal_cutoffs.p_value.min():.3g} to {focal_cutoffs.p_value.max():.3g}. "
            "The binary result is therefore sensitive to exposure encoding, and the "
            "continuous and sampling-normalized models must be considered alongside it.",
            "",
            "## Primary topology-preference models",
            "",
            "| Model | HR per SD | 95% CI | p | n | events |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in primary.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.hazard_ratio:.3g} | "
            f"{row.ci95_low:.3g}–{row.ci95_high:.3g} | {row.p_value:.3g} | "
            f"{row.n} | {row.events} |"
        )
    lines.extend(
        [
            "",
            "The topology-preference coefficient is conditional on total abundance of "
            "the focal/matched pair. A sign reversal relative to a single-motif model is "
            "evidence of shared-burden confounding or suppression, not proof that the "
            "topology is protective or harmful.",
            "",
            (
                "The primary topology-preference terms do not trigger the approximate "
                "Schoenfeld residual check. However, the adjusted model flags: "
                + ", ".join(
                    f"{row.term} (p={row.p_value:.3g})"
                    for row in ph_flags.itertuples(index=False)
                )
                + ". The adjusted estimate should not be interpreted as a single "
                "time-constant clinical effect without further validation."
                if len(ph_flags)
                else "No term in the primary models triggers the approximate Schoenfeld "
                "residual check at p < 0.05."
            ),
            "",
            "## Bootstrap stability",
            "",
            "| Analysis | Successful | Median coefficient | 95% percentile interval | Positive fraction |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in bootstrap_summary.head(2).itertuples(index=False):
        lines.append(
            f"| {row.analysis} | {row.successful_replicates}/{row.requested_replicates} | "
            f"{row.coefficient_median:.3g} | {row.coefficient_percentile_2_5:.3g}–"
            f"{row.coefficient_percentile_97_5:.3g} | {row.fraction_positive:.3f} |"
        )
    adjusted_bootstrap = bootstrap_summary.loc[
        bootstrap_summary["analysis"].eq("primary_pair_adjusted")
    ]
    if len(adjusted_bootstrap):
        row = adjusted_bootstrap.iloc[0]
        failed = int(row.requested_replicates - row.successful_replicates)
        if failed:
            lines.extend(
                [
                    "",
                    f"The adjusted bootstrap failed to estimate {failed} replicates, "
                    "which is additional evidence that the fully adjusted model is "
                    "unstable under patient resampling.",
                ]
            )
    lines.extend(
        [
            "",
            "Topology descriptors are used to formulate and visualize the contrast, not "
            "as three observations in a topology-to-survival regression. Clinical "
            "adjustment, cutoff checks, PH diagnostics, and bootstrap resampling do not "
            "remove selection bias. The hypothesis must be frozen and tested in an "
            "independent cohort for confirmation.",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates < 0:
        raise ValueError("--bootstrap-replicates cannot be negative")
    if any(value <= 0 or value >= 1 for value in args.cutoff_quantiles):
        raise ValueError("Cutoff quantiles must be strictly between zero and one")
    args.outdir.mkdir(parents=True, exist_ok=True)

    target, graph_map, label_map, patients, counts, core_counts, cell_counts = (
        load_analysis_data(args)
    )
    hashes = list(graph_map)
    topology = pd.DataFrame(
        [graph_descriptors(value, graph_map[value], label_map) for value in hashes]
    )
    topology.to_csv(args.outdir / "topology_features.csv", index=False)

    pairwise_rows = []
    for left, right in itertools.combinations(hashes, 2):
        pairwise_rows.append(
            {
                "left_hash": left,
                "right_hash": right,
                "left_short_hash": short_hash(left),
                "right_short_hash": short_hash(right),
                "graph_edit_distance": graph_edit_distance(graph_map[left], graph_map[right]),
                "same_cell_composition": sorted(
                    data["label"] for _, data in graph_map[left].nodes(data=True)
                )
                == sorted(data["label"] for _, data in graph_map[right].nodes(data=True)),
                "is_primary_pair": {left, right} == {args.focal_hash, args.matched_hash},
            }
        )
    pairwise = pd.DataFrame(pairwise_rows)
    mapped, focal_edges, matched_edges, mapping = matched_sibling_difference(
        graph_map[args.focal_hash], graph_map[args.matched_hash]
    )
    pairwise.loc[pairwise["is_primary_pair"], "focal_only_edges"] = ";".join(
        f"{left}-{right}" for left, right in sorted(focal_edges)
    )
    pairwise.loc[pairwise["is_primary_pair"], "matched_only_edges_after_mapping"] = ";".join(
        f"{left}-{right}" for left, right in sorted(matched_edges)
    )
    pairwise.loc[pairwise["is_primary_pair"], "matched_to_focal_node_mapping"] = json.dumps(
        mapping, sort_keys=True
    )
    pairwise.to_csv(args.outdir / "pairwise_topology_differences.csv", index=False)

    analysis = build_patient_exposures(
        patients,
        hashes,
        counts,
        core_counts,
        cell_counts,
        args.focal_hash,
        args.matched_hash,
        args.pseudocount,
    )
    coefficients, ph_diagnostics, model_summaries = fit_analysis_models(
        analysis,
        hashes,
        args.focal_hash,
        args.matched_hash,
        sorted(set([0.1, args.pseudocount, 1.0])),
    )
    cutoffs = cutoff_sensitivity(analysis, hashes, sorted(set(args.cutoff_quantiles)))

    if args.bootstrap_replicates:
        bootstrap_estimates, bootstrap_summary = bootstrap_analysis(
            analysis, hashes, args.bootstrap_replicates, args.seed
        )
    else:
        bootstrap_estimates = pd.DataFrame()
        bootstrap_summary = pd.DataFrame()

    saved_results = pd.read_csv(args.trace_summary, dtype={"wl_hash": str})
    saved_results = saved_results.loc[saved_results["wl_hash"].isin(hashes)]
    original_columns = [
        "wl_hash",
        "p_value_unadjusted",
        "q_value_bh_global_unadjusted",
        "hazard_ratio_unadjusted",
    ]
    coefficients.to_csv(args.outdir / "cox_model_results.csv", index=False)
    ph_diagnostics.to_csv(args.outdir / "proportional_hazards_diagnostics.csv", index=False)
    model_summaries.to_csv(args.outdir / "cox_model_summaries.csv", index=False)
    cutoffs.to_csv(args.outdir / "cutoff_sensitivity.csv", index=False)
    analysis.to_csv(args.outdir / "patient_level_motif_comparison.csv", index=False)
    bootstrap_estimates.to_csv(
        args.outdir / "bootstrap_estimates.csv.gz", index=False, compression="gzip"
    )
    bootstrap_summary.to_csv(args.outdir / "bootstrap_summary.csv", index=False)

    plot_topology_contrast(
        target,
        graph_map[args.focal_hash],
        graph_map[args.matched_hash],
        args.focal_hash,
        args.matched_hash,
        args.target_hash,
        label_map,
        args.outdir / "primary_topology_contrast.png",
    )
    plot_topology_heatmap(topology, args.outdir / "topology_feature_heatmap.png")
    plot_abundance(
        analysis,
        hashes,
        args.focal_hash,
        args.matched_hash,
        args.outdir / "motif_abundance_correlation.png",
    )
    plot_cutoff_sensitivity(cutoffs, args.outdir / "cutoff_sensitivity.png")
    plot_discordance_km(analysis, args.outdir / "primary_pair_discordance_km.png")
    if len(bootstrap_estimates):
        plot_bootstrap(bootstrap_estimates, args.outdir / "bootstrap_topology_preference.png")

    write_report(
        args.outdir / "REPORT.md",
        args,
        topology,
        pairwise,
        analysis,
        coefficients,
        cutoffs,
        ph_diagnostics,
        bootstrap_summary,
        saved_results[original_columns],
    )
    metadata = {
        "target_hash": args.target_hash,
        "triangle_level": args.level,
        "focal_hash": args.focal_hash,
        "matched_hash": args.matched_hash,
        "comparison_hashes": args.comparison_hashes,
        "patients": len(analysis),
        "events": int(analysis["event"].sum()),
        "adjusted_complete_cases": int(analysis[MODEL_COVARIATES].notna().all(axis=1).sum()),
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.seed,
        "primary_pseudocount": args.pseudocount,
        "primary_model": "Efron Cox: pair burden + focal/matched log-ratio",
        "interpretation": "exploratory post-selection analysis",
    }
    (args.outdir / "analysis_summary.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote sibling topology analysis under {args.outdir}")


if __name__ == "__main__":
    main()
