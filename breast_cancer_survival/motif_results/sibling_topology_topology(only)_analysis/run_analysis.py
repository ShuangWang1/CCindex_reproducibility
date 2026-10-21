#!/usr/bin/env python3
"""Analyze raw Cox significance using only unlabeled motif topology."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
MOTIF_RESULTS_DIR = PROJECT_DIR.parent
SURVIVAL_DIR = MOTIF_RESULTS_DIR.parent
BASE_PATH = MOTIF_RESULTS_DIR / "sibling_topology_analysis" / "run_analysis.py"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/breast-cancer-unlabeled-topology-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy
import statsmodels

if str(SURVIVAL_DIR) not in sys.path:
    sys.path.insert(0, str(SURVIVAL_DIR))

SPEC = importlib.util.spec_from_file_location("sibling_topology_base", BASE_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Cannot load shared topology functions from {BASE_PATH}")
base = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = base
SPEC.loader.exec_module(base)

from trace_significant_motifs import graph_from_encoding


TARGET_HASH = base.TARGET_HASH
STRUCTURAL_FAMILIES = {
    "graph_organization",
    "degree_organization",
    "connectivity_distance",
    "clustering_cliques",
    "triangle_organization",
    "spectral",
}
SOURCE_METADATA_COLUMNS = [
    "triangle_level",
    "wl_hash",
    "p_value_unadjusted",
    "coef_unadjusted",
    "hazard_ratio_unadjusted",
    "fit_status_unadjusted",
    "sample_document_frequency",
    "median_patient_raw_count",
    "n_enriched_unadjusted",
    "n_non_enriched_unadjusted",
]


def parse_args() -> argparse.Namespace:
    trace = MOTIF_RESULTS_DIR / "significant_mtf_trace" / "tri1" / TARGET_HASH
    prior_cache = (
        MOTIF_RESULTS_DIR
        / "sibling_topology_analysis"
        / "results"
        / "tables"
        / "motif_topology_features.csv.gz"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, default=trace)
    parser.add_argument("--outdir", type=Path, default=PROJECT_DIR / "results")
    parser.add_argument("--validated-feature-cache", type=Path, default=prior_cache)
    parser.add_argument("--primary-level", type=int, default=5)
    parser.add_argument("--p-threshold", type=float, default=0.05)
    parser.add_argument("--min-class-size", type=int, default=20)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--layout-seed", type=int, default=20260914)
    parser.add_argument(
        "--rebuild-features",
        action="store_true",
        help="Ignore validated caches and recompute graph features from trace_nodes.csv",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def structural_metric_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if base.feature_family(column) in STRUCTURAL_FAMILIES
        and column not in SOURCE_METADATA_COLUMNS
        and column not in {"eligible_for_comparison", "significant_raw_p"}
    ]


def load_validated_cache(
    path: Path, source: pd.DataFrame, trace_hash: str
) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    audit_path = path.parents[1] / "analysis_summary.json"
    if not audit_path.is_file():
        return None
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("input_trace_nodes_sha256") != trace_hash:
        return None
    cached = pd.read_csv(path, dtype={"wl_hash": str})
    expected = set(zip(source["triangle_level"].astype(int), source["wl_hash"]))
    observed = set(zip(cached["triangle_level"].astype(int), cached["wl_hash"]))
    if expected != observed:
        return None
    metrics = structural_metric_columns(cached)
    return cached[[*SOURCE_METADATA_COLUMNS, *metrics]].copy()


def extract_features(source: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for position, (_, row) in enumerate(source.iterrows(), 1):
        graph = graph_from_encoding(str(row.node_labels), str(row.edge_list), str(row.wl_hash))
        base.validate_graph(row, graph)
        computed = base.graph_features(graph)
        structural = {
            key: value
            for key, value in computed.items()
            if base.feature_family(key) in STRUCTURAL_FAMILIES
        }
        rows.append({column: row[column] for column in SOURCE_METADATA_COLUMNS} | structural)
        if position % 10_000 == 0 or position == len(source):
            print(f"Computed unlabeled topology for {position:,}/{len(source):,} motifs", flush=True)
    return pd.DataFrame(rows)


def plot_forest(primary: pd.DataFrame, path: Path, top_n: int) -> None:
    selected = primary.loc[primary["is_representative"]].head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, max(5, 0.45 * len(selected))))
    colors = np.where(selected["topology_assoc_q_bh"].lt(0.05), "#386CB0", "#8A8A8A")
    ax.barh(range(len(selected)), selected["effect_size"], color=colors)
    ax.set_yticks(range(len(selected)), selected["display_name"], fontsize=8)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Effect size (positive = higher in significant motifs)")
    ax.set_title("TRI-5 unlabeled-topology associations")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def draw_unlabeled_graph(ax: plt.Axes, graph: nx.Graph, title: str, seed: int) -> None:
    topology = nx.Graph()
    topology.add_nodes_from(graph.nodes)
    topology.add_edges_from(graph.edges)
    positions = nx.spring_layout(topology, seed=seed)
    nx.draw_networkx(
        topology,
        pos=positions,
        ax=ax,
        node_color="#4C78A8",
        node_size=520,
        with_labels=False,
        edge_color="#555555",
        width=1.8,
    )
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
    fig, axes = plt.subplots(len(metrics), 2, figsize=(9, 3.8 * len(metrics)), squeeze=False)
    for row_index, metric in enumerate(metrics):
        display = str(primary.loc[primary["metric"].eq(metric), "display_name"].iloc[0])
        for column_index, significant in enumerate([False, True]):
            group = eligible.loc[
                eligible["significant_raw_p"].eq(significant), ["wl_hash", metric]
            ].dropna()
            median = float(group[metric].median())
            selected = group.assign(distance=(group[metric] - median).abs()).sort_values(
                ["distance", "wl_hash"]
            ).iloc[0]
            raw = source_index.loc[(primary_level, selected.wl_hash)]
            graph = graph_from_encoding(str(raw.node_labels), str(raw.edge_list), str(selected.wl_hash))
            draw_unlabeled_graph(
                axes[row_index, column_index],
                graph,
                f"{'significant' if significant else 'nonsignificant'} "
                f"{str(selected.wl_hash)[:8]}\n{display}={selected[metric]:.3g}",
                seed,
            )
    fig.suptitle("Representative TRI-5 topologies (cell-type labels discarded)")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_report(
    path: Path,
    summary: pd.DataFrame,
    primary: pd.DataFrame,
    consistency: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    overview = summary.set_index("triangle_level").loc[args.primary_level]
    representatives = primary.loc[primary["is_representative"]]
    candidates = representatives.loc[representatives["exploratory_candidate"]]
    consistent = int(consistency["cross_level_consistent"].sum()) if len(consistency) else 0
    lines = [
        "# Unlabeled sibling topology analysis",
        "",
        "## Result overview",
        "",
        f"The TRI-{args.primary_level} analysis contains {int(overview.estimable_motifs):,} "
        f"estimable motifs: {int(overview.significant_raw_p):,} with raw Cox p < "
        f"{args.p_threshold:g} and {int(overview.nonsignificant_raw_p):,} controls.",
        "",
        f"The analysis tested {len(primary)} label-free topology criteria. "
        f"{int(primary.exploratory_candidate.sum())} passed BH q < 0.05, resolving to "
        f"{len(candidates)} nonredundant candidate groups; {consistent} representatives "
        "meet the cross-level direction criterion.",
        "",
        "## Leading nonredundant criteria",
        "",
        "| Criterion | Family | Direction | Effect | Topology q | Cross-level consistent |",
        "|---|---|---|---:|---:|---|",
    ]
    consistency_index = consistency.set_index("metric") if len(consistency) else pd.DataFrame()
    for row in representatives.head(args.top_n).itertuples(index=False):
        cross = (
            bool(consistency_index.loc[row.metric, "cross_level_consistent"])
            if len(consistency_index) and row.metric in consistency_index.index
            else False
        )
        lines.append(
            f"| {row.display_name} (`{row.metric}`) | {row.family} | "
            f"{row.effect_direction} | {row.effect_size:.3g} | "
            f"{row.topology_assoc_q_bh:.3g} | {'yes' if cross else 'no'} |"
        )
    lines.extend(
        [
            "",
            "Positive effects mean larger topology values among raw-significant motifs; "
            "negative effects mean smaller values. Nonbinary effects are Cliff's delta.",
            "",
            "## What was excluded",
            "",
            "All node cell-type labels were discarded from the analysis. No cell-type node "
            "counts, label-specific centralities, labeled edge/triangle compositions, label "
            "entropy, or label assortativity enter any test. Only unlabeled adjacency structure "
            "is represented.",
            "",
            "## Limitations",
            "",
            "This remains an exploratory screen of same-level peers from a lineage selected "
            "around a significant TRI-1 motif. It excludes nonestimable Cox results and uses no "
            "abundance adjustment or resampling. All Cox results reuse the same patients, so the "
            "topology p/q-values are nominal and do not establish that graph structure causes a "
            "survival association.",
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
    missing = base.REQUIRED_COLUMNS - set(source.columns)
    if missing:
        raise ValueError(f"trace_nodes.csv is missing columns: {sorted(missing)}")
    trace_hash = sha256(nodes_path)

    tables = args.outdir / "tables"
    figures = args.outdir / "figures"
    own_cache = tables / "motif_topology_features.csv.gz"
    features = None
    cache_source = "recomputed"
    if not args.rebuild_features:
        features = load_validated_cache(own_cache, source, trace_hash)
        if features is not None:
            cache_source = "project_cache"
        else:
            features = load_validated_cache(args.validated_feature_cache, source, trace_hash)
            if features is not None:
                cache_source = "validated_labeled_project_cache_structural_columns_only"
    if features is None:
        features = extract_features(source)
    else:
        print(f"Loaded {len(features):,} motifs from {cache_source}", flush=True)

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
    metric_columns = structural_metric_columns(features)
    dictionary = base.build_metric_dictionary(metric_columns)
    if not set(dictionary["family"]).issubset(STRUCTURAL_FAMILIES):
        raise AssertionError("A label-dependent feature entered the topology-only dictionary")
    associations, exclusions = base.analyze_levels(
        features, dictionary, args.min_class_size, min_prevalence=0.01
    )
    associations = associations.merge(
        dictionary[["metric", "display_name"]], on="metric", how="left", validate="many_to_one"
    )
    summary = base.level_summary(features, args.p_threshold)
    groups, primary = base.redundancy_groups(features, associations, args.primary_level)
    consistency = base.cross_level_consistency(primary, associations, args.primary_level)

    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    features.to_csv(own_cache, index=False, compression="gzip")
    dictionary.to_csv(tables / "metric_dictionary.csv", index=False)
    summary.to_csv(tables / "level_summary.csv", index=False)
    associations.to_csv(tables / "topology_associations_all_levels.csv", index=False)
    exclusions.to_csv(tables / "excluded_metrics.csv", index=False)
    groups.to_csv(tables / "metric_redundancy_groups.csv", index=False)
    primary.to_csv(tables / "tri5_ranked_metrics.csv", index=False)
    consistency.to_csv(tables / "cross_level_consistency.csv", index=False)

    base.plot_level_overview(summary, figures / "level_overview.png")
    plot_forest(primary, figures / "tri5_metric_forest.png", args.top_n)
    base.plot_distributions(
        features, primary, args.primary_level, figures / "tri5_metric_distributions.png"
    )
    base.plot_correlation(
        features, primary, args.primary_level, figures / "tri5_metric_correlation.png"
    )
    base.plot_cross_level(consistency, figures / "cross_level_consistency.png")
    base.plot_family_summary(primary, figures / "metric_family_summary.png")
    plot_topology_examples(
        features,
        source,
        primary,
        args.primary_level,
        figures / "tri5_topology_examples.png",
        args.layout_seed,
    )
    write_report(args.outdir / "REPORT.md", summary, primary, consistency, args)

    audit = {
        "status": "complete",
        "analysis_scope": f"single lineage tri_1:{TARGET_HASH}; same-level peers",
        "primary_level": args.primary_level,
        "cox_significance_rule": f"finite estimated raw p < {args.p_threshold:g}",
        "topology_definition": "internal unlabeled graph; all cell-type labels discarded",
        "cache_source": cache_source,
        "input_trace_nodes": str(nodes_path.resolve()),
        "input_trace_nodes_sha256": trace_hash,
        "input_trace_metadata_sha256": sha256(metadata_path),
        "motifs": len(features),
        "unlabeled_metrics_computed": len(metric_columns),
        "tri5_tested_metrics": len(primary),
        "reproduction_command": (
            "python 'breast_cancer_survival/motif_results/"
            "sibling_topology_topology(only)_analysis/run_analysis.py'"
        ),
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
