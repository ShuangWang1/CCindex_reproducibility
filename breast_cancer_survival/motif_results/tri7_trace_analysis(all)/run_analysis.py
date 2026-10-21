#!/usr/bin/env python3
"""Visualize all significant TRI-7 traces and compare TRI-2--TRI-7 criteria."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/breast-cancer-tri7-all-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import kruskal, levene, mannwhitneyu


PROJECT_DIR = Path(__file__).resolve().parent
MOTIF_RESULTS = PROJECT_DIR.parent
SURVIVAL_DIR = MOTIF_RESULTS.parent
REPO_ROOT = SURVIVAL_DIR.parent
SOURCE_ROOT = MOTIF_RESULTS / "significant_mtf_trace" / "tri7"
DEFAULT_OUTDIR = PROJECT_DIR
LABEL_MAP_PATH = REPO_ROOT / "breast_cancer" / "BZ_samples_intra" / "node_label_cell_type_mapping.csv"
CRITERIA_SCRIPT = MOTIF_RESULTS / "sibling_topology_analysis_new" / "run_analysis.py"
TRACE_SCRIPT = SURVIVAL_DIR / "trace_significant_motifs.py"
PLOT_SCRIPT = SURVIVAL_DIR / "trace_back_tri7_motif_survival.py"

sys.path.insert(0, str(SURVIVAL_DIR))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


criteria = load_module("tri7_all_criteria", CRITERIA_SCRIPT)
trace = load_module("tri7_all_trace", TRACE_SCRIPT)
trace_plot = load_module("tri7_all_trace_plot", PLOT_SCRIPT)

LEVELS = tuple(range(2, 8))
DISPERSION_METRICS = [
    "dispersion_composition_js_to_group_centroid",
    "dispersion_mixing_frobenius_to_group_centroid",
    "dispersion_null_corrected_mixing_frobenius_to_group_centroid",
    "dispersion_triangle_js_to_group_centroid",
]
META_COLUMNS = [
    "triangle_level", "wl_hash", "node_labels", "edge_list", "significance_group",
    "significant_raw_p", "p_value_unadjusted", "hazard_ratio_unadjusted",
    "sample_document_frequency", "median_patient_raw_count", "trace_membership_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--label-map", type=Path, default=LABEL_MAP_PATH)
    parser.add_argument("--permutations", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--panels-per-page", type=int, default=12)
    parser.add_argument("--max-targets", type=int)
    parser.add_argument("--visuals-only", action="store_true")
    parser.add_argument("--criteria-only", action="store_true")
    parser.add_argument("--rerun-visuals", action="store_true")
    return parser.parse_args()


def source_targets(source_root: Path) -> list[Path]:
    return sorted(
        path for path in source_root.iterdir()
        if path.is_dir() and (path / "trace_metadata.json").is_file()
    )


def load_union(source_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load memberships, validate duplicate records, and deduplicate level/hash motifs."""
    memberships: list[pd.DataFrame] = []
    for target_dir in source_targets(source_root):
        frame = pd.read_csv(
            target_dir / "trace_nodes.csv",
            dtype={"wl_hash": str, "node_labels": str, "edge_list": str},
        )
        frame["trace_target_hash"] = target_dir.name
        memberships.append(frame)
    all_rows = pd.concat(memberships, ignore_index=True)
    relevant = all_rows.loc[all_rows.triangle_level.isin(LEVELS)].copy()
    consistency_columns = [
        "p_value_unadjusted", "hazard_ratio_unadjusted",
        "sample_document_frequency", "median_patient_raw_count", "significant_raw_p",
    ]
    conflicts: list[str] = []
    for column in consistency_columns:
        counts = relevant.groupby(["triangle_level", "wl_hash"], dropna=False)[column].nunique(dropna=False)
        if counts.gt(1).any():
            conflicts.append(column)
    if conflicts:
        raise ValueError(f"Duplicate trace motifs disagree in columns: {conflicts}")
    encoding_counts = relevant.groupby(["triangle_level", "wl_hash"])["edge_list"].nunique()
    for (level, motif_hash), _ in encoding_counts.loc[encoding_counts.gt(1)].items():
        encodings = relevant.loc[
            relevant.triangle_level.eq(level) & relevant.wl_hash.eq(motif_hash),
            ["node_labels", "edge_list"],
        ].drop_duplicates()
        reference_row = encodings.iloc[0]
        reference = criteria.base.graph_from_encoding(
            reference_row.node_labels, reference_row.edge_list, motif_hash
        )
        for row in encodings.iloc[1:].itertuples(index=False):
            comparison = criteria.base.graph_from_encoding(row.node_labels, row.edge_list, motif_hash)
            if not nx.is_isomorphic(
                reference,
                comparison,
                node_match=lambda left, right: str(left.get("label")) == str(right.get("label")),
            ):
                raise ValueError(
                    f"Duplicate encodings are not label-isomorphic for TRI-{level} {motif_hash}"
                )
    counts = (
        relevant.groupby(["triangle_level", "wl_hash"])["trace_target_hash"]
        .nunique().rename("trace_membership_count").reset_index()
    )
    unique = (
        relevant.sort_values(["triangle_level", "wl_hash", "trace_target_hash"])
        .drop_duplicates(["triangle_level", "wl_hash"])
        .merge(counts, on=["triangle_level", "wl_hash"], validate="one_to_one")
    )
    unique["significance_group"] = np.where(
        unique["p_value_unadjusted"].lt(0.05), "significant", "nonsignificant"
    )
    membership_table = relevant[["triangle_level", "wl_hash", "trace_target_hash"]].drop_duplicates()
    return all_rows, unique.reset_index(drop=True), membership_table.reset_index(drop=True)


def reconstruct_target(target_dir: Path) -> tuple[pd.DataFrame, dict[int, dict[str, nx.Graph]], nx.Graph]:
    """Recreate target-relative graph identities and validate the saved lineage."""
    nodes = pd.read_csv(
        target_dir / "trace_nodes.csv",
        dtype={"wl_hash": str, "node_labels": str, "edge_list": str},
    )
    saved_edges = pd.read_csv(
        target_dir / "trace_edges.csv",
        dtype={"parent_hash": str, "child_hash": str, "changed_node_or_edge": str},
    )
    selected = nodes.loc[nodes.wl_hash.eq(target_dir.name) & nodes.triangle_level.eq(7)]
    if len(selected) != 1:
        raise ValueError(f"Expected one TRI-7 target row for {target_dir.name}")
    row = selected.iloc[0]
    target_graph = trace.graph_from_encoding(row.node_labels, row.edge_list, row.wl_hash)
    vocabularies = {
        level: set(nodes.loc[nodes.triangle_level.eq(level), "wl_hash"].astype(str))
        for level in range(1, 8)
    }
    labels = sorted({str(value) for value in str(row.node_labels).split("&")})
    graphs, rebuilt_edges = trace.trace_lineage(target_graph, 7, vocabularies, labels)
    saved_motifs = set(zip(nodes.triangle_level.astype(int), nodes.wl_hash.astype(str)))
    rebuilt_motifs = {
        (int(level), str(motif_hash))
        for level, level_graphs in graphs.items() for motif_hash in level_graphs
    }
    if saved_motifs != rebuilt_motifs:
        raise ValueError(f"Rebuilt motif set differs for target {target_dir.name}")
    key_columns = ["parent_level", "parent_hash", "child_level", "child_hash", "growth_operation"]
    saved_keys = set(map(tuple, saved_edges[key_columns].astype(str).to_numpy()))
    rebuilt_keys = set(map(tuple, rebuilt_edges[key_columns].astype(str).to_numpy()))
    if saved_keys != rebuilt_keys:
        raise ValueError(f"Rebuilt transition set differs for target {target_dir.name}")
    return rebuilt_edges, graphs, target_graph


def render_target_gallery(target_dir: Path, output: Path, label_map: dict[str, str]) -> dict[str, object]:
    edges, graphs, target = reconstruct_target(target_dir)
    transitions = edges.copy()
    transitions["reverse_operation"] = transitions.growth_operation.str.replace("add_", "remove_", regex=False)
    transitions["removed_node_or_edge"] = transitions.changed_node_or_edge.astype(str).str.split(":").str[0]
    transitions = transitions.sort_values(
        ["child_level", "parent_hash", "child_hash", "reverse_operation", "removed_node_or_edge"],
        kind="mergesort",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    coordinates = trace_plot.fixed_target_layout(target, target_dir.name)
    trace_plot.plot_growth_transition_gallery(
        transitions,
        graphs,
        target,
        coordinates,
        label_map,
        f"TRI-7 motif {target_dir.name[:8]}: all potential growth transitions",
        output,
    )
    return {
        "target_hash": target_dir.name,
        "motifs": int(sum(len(values) for values in graphs.values())),
        "transitions": int(len(transitions)),
        "figure_file": str(output),
        "validated": True,
    }


def generate_target_galleries(args: argparse.Namespace) -> pd.DataFrame:
    label_frame = pd.read_csv(args.label_map, dtype={"node_label": str})
    label_map = dict(label_frame[["node_label", "cell_type"]].itertuples(index=False, name=None))
    targets = source_targets(args.source_root)
    if args.max_targets is not None:
        targets = targets[: args.max_targets]
    records: list[dict[str, object]] = []
    for position, target_dir in enumerate(targets, 1):
        output = args.outdir / target_dir.name / "all_trace_back_growth_transitions.png"
        if output.is_file() and not args.rerun_visuals:
            nodes = pd.read_csv(target_dir / "trace_nodes.csv", usecols=["wl_hash"])
            edges = pd.read_csv(target_dir / "trace_edges.csv", usecols=["parent_hash"])
            records.append({
                "target_hash": target_dir.name, "motifs": len(nodes), "transitions": len(edges),
                "figure_file": str(output), "validated": True,
            })
        else:
            records.append(render_target_gallery(target_dir, output, label_map))
        if position % 10 == 0 or position == len(targets):
            print(f"completed target galleries {position}/{len(targets)}", flush=True)
    manifest = pd.DataFrame(records).sort_values("target_hash")
    (args.outdir / "tables").mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.outdir / "tables" / "target_gallery_manifest.csv", index=False)
    return manifest


def add_level_group_dispersion(features: pd.DataFrame) -> None:
    compositions = [column for column in features if column.startswith("type_") and column.endswith("_fraction")]
    contacts = [f"contact_{a}_{b}_fraction" for a, b in criteria.EDGE_TYPES]
    contact_z = [f"contact_{a}_{b}_permutation_z" for a, b in criteria.EDGE_TYPES]
    triangles = [f"typed_triangle_{a}_{b}_{c}_fraction" for a, b, c in criteria.TRIANGLE_TYPES]
    for _, indices in features.groupby(["triangle_level", "significance_group"]).groups.items():
        selected = features.loc[indices]
        composition_centroid = selected[compositions].mean().to_numpy(float)
        contact_centroid = selected[contacts].mean().to_numpy(float)
        contact_z_centroid = selected[contact_z].mean(skipna=True).fillna(0).to_numpy(float)
        triangle_centroid = selected[triangles].mean().to_numpy(float)
        features.loc[indices, DISPERSION_METRICS[0]] = [
            criteria.js_distance(row, composition_centroid)
            for row in selected[compositions].to_numpy(float)
        ]
        features.loc[indices, DISPERSION_METRICS[1]] = np.linalg.norm(
            selected[contacts].to_numpy(float) - contact_centroid, axis=1
        )
        features.loc[indices, DISPERSION_METRICS[2]] = np.linalg.norm(
            np.nan_to_num(selected[contact_z].to_numpy(float)) - contact_z_centroid, axis=1
        )
        features.loc[indices, DISPERSION_METRICS[3]] = [
            criteria.js_distance(row, triangle_centroid)
            for row in selected[triangles].to_numpy(float)
        ]


def compute_criteria(unique: pd.DataFrame, permutations: int, seed: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    total = len(unique)
    for position, row in enumerate(unique.itertuples(index=False), 1):
        graph = criteria.base.graph_from_encoding(row.node_labels, row.edge_list, row.wl_hash)
        values, _ = criteria.extra_features(graph, row.wl_hash, permutations, seed)
        rows.append({
            "triangle_level": int(row.triangle_level), "wl_hash": str(row.wl_hash),
            "node_labels": str(row.node_labels), "edge_list": str(row.edge_list),
            "significance_group": str(row.significance_group),
            "significant_raw_p": bool(row.p_value_unadjusted < 0.05),
            "p_value_unadjusted": float(row.p_value_unadjusted),
            "hazard_ratio_unadjusted": float(row.hazard_ratio_unadjusted),
            "sample_document_frequency": float(row.sample_document_frequency),
            "median_patient_raw_count": float(row.median_patient_raw_count),
            "trace_membership_count": int(row.trace_membership_count),
            **values,
        })
        if position % 100 == 0 or position == total:
            print(f"computed criteria {position}/{total}", flush=True)
    features = pd.DataFrame(rows)
    add_level_group_dispersion(features)
    return features


def metric_columns(features: pd.DataFrame) -> list[str]:
    return [column for column in features if column not in META_COLUMNS]


def distribution_and_tests(
    features: pd.DataFrame, metrics: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    distribution_rows: list[dict[str, object]] = []
    level_rows: list[dict[str, object]] = []
    within_rows: list[dict[str, object]] = []
    for metric in metrics:
        for level in LEVELS:
            subset = features.loc[features.triangle_level.eq(level)]
            for group in ["nonsignificant", "significant"]:
                values = subset.loc[subset.significance_group.eq(group), metric]
                distribution_rows.append({
                    "metric": metric, "display_name": criteria.display_name(metric),
                    "family": criteria.family_for(metric), "triangle_level": level,
                    "group": group, **criteria.describe_metric(values),
                })
            control = subset.loc[subset.significance_group.eq("nonsignificant"), metric].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
            case = subset.loc[subset.significance_group.eq("significant"), metric].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
            result: dict[str, object] = {
                "metric": metric, "display_name": criteria.display_name(metric),
                "family": criteria.family_for(metric), "triangle_level": level,
                "n_nonsignificant": len(control), "n_significant": len(case),
                "testable": False, "not_testable_reason": "",
            }
            if metric in DISPERSION_METRICS:
                result["not_testable_reason"] = "outcome_derived_descriptive_only"
            elif not len(control) or not len(case):
                result["not_testable_reason"] = "missing_significance_class"
            elif np.unique(np.concatenate([control, case])).size < 2:
                result["not_testable_reason"] = "constant_pooled_values"
            else:
                test = mannwhitneyu(case, control, alternative="two-sided", method="asymptotic")
                result.update({
                    "testable": True, "mann_whitney_u": float(test.statistic),
                    "location_p_value": float(test.pvalue),
                    "cliffs_delta": float(2 * test.statistic / (len(case) * len(control)) - 1),
                    "median_difference": float(np.median(case) - np.median(control)),
                    "mean_difference": float(np.mean(case) - np.mean(control)),
                })
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    dispersion = levene(case, control, center="median")
                if np.isfinite(dispersion.statistic):
                    result.update({
                        "brown_forsythe_statistic": float(dispersion.statistic),
                        "dispersion_p_value": float(dispersion.pvalue),
                    })
            within_rows.append(result)

        groups = [
            features.loc[features.triangle_level.eq(level), metric]
            .replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
            for level in LEVELS
        ]
        result = {
            "metric": metric, "display_name": criteria.display_name(metric),
            "family": criteria.family_for(metric), "testable": False,
            "not_testable_reason": "",
        }
        if metric in DISPERSION_METRICS:
            result["not_testable_reason"] = "outcome_derived_descriptive_only"
        elif any(not len(values) for values in groups):
            result["not_testable_reason"] = "missing_level_values"
        elif np.unique(np.concatenate(groups)).size < 2:
            result["not_testable_reason"] = "constant_across_levels"
        else:
            test = kruskal(*groups)
            sample_size = sum(map(len, groups))
            epsilon = max(0.0, float((test.statistic - len(groups) + 1) / (sample_size - len(groups))))
            result.update({
                "testable": True, "kruskal_wallis_h": float(test.statistic),
                "location_p_value": float(test.pvalue), "epsilon_squared": epsilon,
            })
        level_rows.append(result)

    within = pd.DataFrame(within_rows)
    for column in ["location_p_value", "dispersion_p_value", "cliffs_delta"]:
        if column not in within:
            within[column] = np.nan
    within["location_q_bh"] = np.nan
    within["dispersion_q_bh"] = np.nan
    for level, indices in within.groupby("triangle_level").groups.items():
        within.loc[indices, "location_q_bh"] = criteria.base.benjamini_hochberg(
            within.loc[indices, "location_p_value"]
        )
        within.loc[indices, "dispersion_q_bh"] = criteria.base.benjamini_hochberg(
            within.loc[indices, "dispersion_p_value"]
        )
    across = pd.DataFrame(level_rows)
    for column in ["location_p_value", "epsilon_squared"]:
        if column not in across:
            across[column] = np.nan
    across["location_q_bh"] = criteria.base.benjamini_hochberg(across["location_p_value"])
    return pd.DataFrame(distribution_rows), across, within


def draw_level_violin(ax: plt.Axes, features: pd.DataFrame, metric: str) -> None:
    level_colors = ["#4E79A7", "#59A14F", "#F28E2B", "#B07AA1", "#76B7B2", "#EDC948"]
    for position, (level, color) in enumerate(zip(LEVELS, level_colors), 1):
        subset = features.loc[features.triangle_level.eq(level), [metric, "significance_group"]].copy()
        clean = pd.to_numeric(subset[metric], errors="coerce").replace([np.inf, -np.inf], np.nan)
        subset = subset.loc[clean.notna()].copy()
        values = clean.dropna().to_numpy(float)
        if not len(values):
            continue
        if len(values) > 1 and np.unique(values).size > 1:
            parts = ax.violinplot([values], positions=[position], widths=.72, showextrema=False)
            for body in parts["bodies"]:
                body.set_facecolor(color); body.set_edgecolor("#444444"); body.set_alpha(.38)
        else:
            ax.hlines(values[0], position - .18, position + .18, color=color, linewidth=4)
        rng = np.random.default_rng(20260918 + position * 1009 + len(values))
        jitter = rng.uniform(-.16, .16, len(values))
        point_colors = subset.significance_group.map(
            {"nonsignificant": "#6F8499", "significant": "#D1495B"}
        ).to_numpy()
        ax.scatter(position + jitter, values, s=5, color=point_colors, alpha=.25, edgecolors="none", rasterized=True)
        q25, median, q75 = np.quantile(values, [.25, .5, .75])
        ax.vlines(position, q25, q75, color="#222222", linewidth=3)
        ax.scatter(position, median, s=23, color="#F28E2B", edgecolor="white", linewidth=.4, zorder=5)
    ax.set_xticks(range(1, 7), [f"TRI-{level}" for level in LEVELS])
    ax.tick_params(axis="x", labelrotation=25)
    ax.set_xlim(.5, 6.5)


def ordered_metrics(table: pd.DataFrame, metrics: list[str], level: int | None = None) -> list[str]:
    selected = table.loc[table.triangle_level.eq(level)].copy() if level is not None else table.copy()
    selected["effect"] = (
        selected.get("cliffs_delta", pd.Series(index=selected.index, dtype=float)).abs()
        if level is not None else selected.get("epsilon_squared", pd.Series(index=selected.index, dtype=float))
    )
    selected["metric_order"] = selected.metric.map({metric: index for index, metric in enumerate(metrics)})
    return selected.sort_values(
        ["location_q_bh", "effect", "metric_order"],
        ascending=[True, False, True], na_position="last",
    ).metric.tolist()


def plot_pages(
    features: pd.DataFrame,
    metrics: list[str],
    across: pd.DataFrame,
    within: pd.DataFrame,
    figures: Path,
    panels_per_page: int,
) -> pd.DataFrame:
    index_rows: list[dict[str, object]] = []
    rows = math.ceil(panels_per_page / 2)
    across_dir = figures / "across_levels"
    across_dir.mkdir(parents=True, exist_ok=True)
    cross_order = ordered_metrics(across, metrics)
    for page, start in enumerate(range(0, len(cross_order), panels_per_page), 1):
        selected = cross_order[start:start + panels_per_page]
        fig, axes = plt.subplots(rows, 2, figsize=(13, 3.15 * rows), squeeze=False)
        for panel, metric in enumerate(selected, 1):
            ax = axes.flat[panel - 1]; draw_level_violin(ax, features, metric)
            result = across.loc[across.metric.eq(metric)].iloc[0]
            statistic = (
                f"BH q={result.location_q_bh:.3g} | ε²={result.epsilon_squared:.2f}"
                if bool(result.testable) else f"descriptive only: {result.not_testable_reason}"
            )
            ax.set_title(f"{result.display_name}\n{statistic}", fontsize=8)
            index_rows.append({"view": "across_levels", "triangle_level": np.nan, "metric": metric, "figure_file": f"across_levels_page_{page:02d}.png", "panel": panel})
        for ax in axes.flat[len(selected):]: ax.axis("off")
        fig.suptitle(f"Deduplicated traced-motif criterion distributions across TRI levels — page {page}")
        fig.tight_layout(rect=(0, 0, 1, .98)); fig.savefig(across_dir / f"across_levels_page_{page:02d}.png", dpi=180); plt.close(fig)

    for level in LEVELS:
        level_dir = figures / "by_significance" / f"tri{level}"
        level_dir.mkdir(parents=True, exist_ok=True)
        order = ordered_metrics(within, metrics, level)
        level_features = features.loc[features.triangle_level.eq(level)]
        for page, start in enumerate(range(0, len(order), panels_per_page), 1):
            selected = order[start:start + panels_per_page]
            fig, axes = plt.subplots(rows, 2, figsize=(13, 3.15 * rows), squeeze=False)
            for panel, metric in enumerate(selected, 1):
                ax = axes.flat[panel - 1]
                criteria.draw_violin(
                    ax,
                    level_features.loc[level_features.significance_group.eq("nonsignificant"), metric],
                    level_features.loc[level_features.significance_group.eq("significant"), metric],
                )
                result = within.loc[within.triangle_level.eq(level) & within.metric.eq(metric)].iloc[0]
                statistic = (
                    f"BH q={result.location_q_bh:.3g} | Cliff δ={result.cliffs_delta:+.2f} | n={int(result.n_nonsignificant)}/{int(result.n_significant)}"
                    if bool(result.testable) else f"not testable: {result.not_testable_reason} | n={int(result.n_nonsignificant)}/{int(result.n_significant)}"
                )
                ax.set_title(f"{result.display_name}\n{statistic}", fontsize=8)
                index_rows.append({"view": "by_significance", "triangle_level": level, "metric": metric, "figure_file": f"tri{level}/by_significance_page_{page:02d}.png", "panel": panel})
            for ax in axes.flat[len(selected):]: ax.axis("off")
            fig.suptitle(f"TRI-{level} significant vs nonsignificant traced motifs — page {page}")
            fig.tight_layout(rect=(0, 0, 1, .98)); fig.savefig(level_dir / f"by_significance_page_{page:02d}.png", dpi=180); plt.close(fig)
        print(f"completed violin pages for TRI-{level}", flush=True)
    return pd.DataFrame(index_rows)


def write_readme(path: Path, summary: dict[str, object], permutations: int) -> None:
    path.write_text(f"""# Complete TRI-7 trace analysis

This project visualizes all {summary['source_targets']:,} significant TRI-7 trace folders and computes the comprehensive graph-criterion set for the deduplicated union of TRI-2 through TRI-7 motifs.

## Population

Each `(triangle_level, wl_hash)` contributes once to the criterion distributions, regardless of how many target lineages contain it. `tables/trace_memberships.csv.gz` preserves the many-to-many lineage memberships.

| Level | Unique motifs | Nonsignificant | Significant |
|---:|---:|---:|---:|
""" + "\n".join(
        f"| {row['triangle_level']} | {row['motifs']} | {row['nonsignificant']} | {row['significant']} |"
        for row in summary["level_counts"]
    ) + f"""

Raw significance means Cox `p < 0.05`. TRI-2 and TRI-3 contain no significant motifs in this traced union; TRI-7 contains no nonsignificant motifs. Their within-level significance plots are therefore descriptive and explicitly marked not testable.

## Criteria and statistics

The pipeline computes topology, composition, contacts, typed triangles, type–topology coupling, and fixed-composition label-null Z scores using {permutations} deterministic permutations per motif. Four between-motif heterogeneity distances are calculated relative to the centroid within each `(TRI level, significance group)` and are labeled outcome-derived/descriptive-only.

Cross-level intrinsic comparisons use Kruskal–Wallis tests, epsilon-squared, and BH correction. Within-level intrinsic comparisons use Mann–Whitney tests, Cliff's delta, Brown–Forsythe tests, and BH correction separately by level. Constant criteria remain in figures as horizontal segments.

## Outputs

- `<target_hash>/all_trace_back_growth_transitions.png`: fixed final-TRI-7 scaffold with each growth step highlighted in gold.
- `tables/tri2_to_tri7_motif_criteria.csv.gz`: one row per deduplicated motif.
- `tables/trace_memberships.csv.gz`: motif-to-target membership table.
- `tables/across_level_comparisons.csv` and `within_level_significance_comparisons.csv`: inferential results.
- `tables/criteria_distributions.csv.gz`: descriptive summaries.
- `figures/across_levels/`: six-level violin pages with points colored by raw significance.
- `figures/by_significance/tri2` through `tri7`: significance-comparison violin pages.

The trace relations are potential labeled-topological growth relations because original search parent IDs were not saved. Gallery coordinates are deterministic graph layouts, not physical tissue coordinates.
""", encoding="utf-8")


def run_criteria(args: argparse.Namespace) -> dict[str, object]:
    tables = args.outdir / "tables"; figures = args.outdir / "figures"
    tables.mkdir(parents=True, exist_ok=True); figures.mkdir(parents=True, exist_ok=True)
    _, unique, memberships = load_union(args.source_root)
    memberships.to_csv(tables / "trace_memberships.csv.gz", index=False, compression="gzip")
    feature_path = tables / "tri2_to_tri7_motif_criteria.csv.gz"
    features = compute_criteria(unique, args.permutations, args.seed)
    features.to_csv(feature_path, index=False, compression="gzip")
    metrics = metric_columns(features)
    registry = criteria.build_registry(metrics, args.permutations)
    registry.loc[registry.criterion.isin(DISPERSION_METRICS), "reason_or_method"] = "Distance to the centroid within the same TRI level and significance group; descriptive only."
    registry.to_csv(tables / "criteria_registry.csv", index=False)
    distributions, across, within = distribution_and_tests(features, metrics)
    distributions.to_csv(tables / "criteria_distributions.csv.gz", index=False, compression="gzip")
    across.to_csv(tables / "across_level_comparisons.csv", index=False)
    within.to_csv(tables / "within_level_significance_comparisons.csv", index=False)
    figure_index = plot_pages(features, metrics, across, within, figures, args.panels_per_page)
    figure_index.to_csv(tables / "figure_index.csv", index=False)
    counts = []
    for level in LEVELS:
        subset = features.loc[features.triangle_level.eq(level)]
        counts.append({
            "triangle_level": level, "motifs": len(subset),
            "nonsignificant": int(subset.significance_group.eq("nonsignificant").sum()),
            "significant": int(subset.significance_group.eq("significant").sum()),
        })
    return {
        "deduplicated_motifs": int(len(features)), "computed_criteria": len(metrics),
        "figure_panels": int(len(figure_index)), "level_counts": counts,
    }


def main() -> None:
    args = parse_args()
    if args.visuals_only and args.criteria_only:
        raise ValueError("Choose at most one of --visuals-only and --criteria-only")
    args.outdir.mkdir(parents=True, exist_ok=True)
    summary_path = args.outdir / "analysis_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    summary.setdefault("source_targets", len(source_targets(args.source_root)))
    if not args.criteria_only:
        manifest = generate_target_galleries(args)
        summary.update({"source_targets": len(source_targets(args.source_root)), "rendered_target_galleries": len(manifest)})
    if not args.visuals_only:
        summary.update(run_criteria(args))
    summary["status"] = "complete"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if "level_counts" in summary:
        write_readme(args.outdir / "README.md", summary, args.permutations)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
