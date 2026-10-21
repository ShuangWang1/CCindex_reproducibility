#!/usr/bin/env python3
"""TRI-5 topology-criterion analysis using rank-normalized Cox significance."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
MOTIF_RESULTS = PROJECT_DIR.parent
SOURCE_PROJECT = MOTIF_RESULTS / "sibling_topology_analysis_new"
TARGET_HASH = "65319d6de1808abbc65d476aa3e9436e"
TRACE_DIR = (
    MOTIF_RESULTS
    / "significant_mtf_trace_ranking_normalized"
    / "tri1"
    / f"{TARGET_HASH}_new"
)

SPEC = importlib.util.spec_from_file_location(
    "rank_sibling_topology_engine", SOURCE_PROJECT / "run_analysis.py"
)
if SPEC is None or SPEC.loader is None:
    raise ImportError("Cannot load sibling_topology_analysis_new/run_analysis.py")
engine = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(engine)

base = engine.base
LABEL_NAMES = engine.LABEL_NAMES
LABELS = engine.LABELS
RANK_METADATA = [
    "triangle_level",
    "wl_hash",
    "significance_group",
    "significant_rank_p",
    "p_value_rank_binary",
    "coef_rank_binary",
    "hazard_ratio_rank_binary",
    "ci95_low_rank_binary",
    "ci95_high_rank_binary",
    "rank_binary_fit_status",
    "median_patient_rank_score",
    "n_survival",
    "events_survival",
    "n_high_rank",
    "events_high_rank",
    "n_low_rank",
    "events_low_rank",
    "sample_document_frequency",
]
COMPATIBILITY_COLUMNS = [
    "significant_raw_p",
    "p_value_unadjusted",
    "coef_unadjusted",
    "hazard_ratio_unadjusted",
    "median_patient_raw_count",
]
METADATA = RANK_METADATA + COMPATIBILITY_COLUMNS
engine.METADATA = METADATA

COLUMN_RENAMES = {
    "significant_raw_p": "significant_rank_p",
    "p_value_unadjusted": "p_value_rank_binary",
    "coef_unadjusted": "coef_rank_binary",
    "hazard_ratio_unadjusted": "hazard_ratio_rank_binary",
    "median_patient_raw_count": "median_patient_rank_score",
}


def __getattr__(name: str) -> object:
    """Expose the established feature and plotting helpers to companion modules."""
    return getattr(engine, name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, default=TRACE_DIR)
    parser.add_argument("--outdir", type=Path, default=PROJECT_DIR / "results")
    parser.add_argument("--primary-level", type=int, default=5)
    parser.add_argument("--p-threshold", type=float, default=0.05)
    parser.add_argument("--permutations", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--panels-per-page", type=int, default=12)
    parser.add_argument("--representative-pairs", type=int, default=12)
    parser.add_argument("--plots-only", action="store_true")
    return parser.parse_args()


def select_estimable_motifs(
    source: pd.DataFrame, level: int, p_threshold: float
) -> pd.DataFrame:
    required = {
        "triangle_level",
        "wl_hash",
        "node_labels",
        "edge_list",
        "p_value_rank_binary",
        "coef_rank_binary",
        "hazard_ratio_rank_binary",
        "median_patient_rank_score",
        "rank_binary_fit_status",
    }
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"Rank trace is missing columns: {missing}")
    p_values = pd.to_numeric(source["p_value_rank_binary"], errors="coerce")
    selected = source.loc[
        source["triangle_level"].eq(level)
        & np.isfinite(p_values)
        & p_values.gt(0)
        & p_values.le(1)
        & source["rank_binary_fit_status"].eq("estimated")
    ].copy()
    if selected["wl_hash"].duplicated().any():
        raise ValueError("Estimable TRI motifs contain duplicate WL hashes")
    selected["significant_rank_p"] = selected["p_value_rank_binary"].lt(p_threshold)
    selected["significance_group"] = np.where(
        selected["significant_rank_p"], "significant", "nonsignificant"
    )
    return selected.sort_values(["p_value_rank_binary", "wl_hash"]).reset_index(drop=True)


def add_compatibility_columns(features: pd.DataFrame) -> pd.DataFrame:
    result = features.copy()
    for compatibility, native in COLUMN_RENAMES.items():
        result[compatibility] = result[native]
    return result


def clean_rank_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove or rename legacy aliases without creating duplicate columns."""
    result = frame.copy()
    for old, new in COLUMN_RENAMES.items():
        if old not in result.columns:
            continue
        if new in result.columns:
            result = result.drop(columns=old)
        else:
            result = result.rename(columns={old: new})
    return result


def rewrite_generated_metadata_tables(tables: Path) -> None:
    for path in sorted(tables.glob("*.csv*")):
        if path.name in {
            "tri5_motif_criteria.csv.gz",
            "tri5_normalized_laplacian_eigenvalues.csv.gz",
        }:
            continue
        frame = pd.read_csv(path, low_memory=False)
        cleaned = clean_rank_columns(frame)
        if list(cleaned.columns) == list(frame.columns):
            continue
        cleaned.to_csv(
            path,
            index=False,
            compression="gzip" if path.suffix == ".gz" else None,
        )


def build_features(
    selected: pd.DataFrame, permutations: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    raw_distributions: dict[tuple[str, str, float], int] = defaultdict(int)
    spectra_rows: list[dict[str, object]] = []
    audit_columns = [
        "ci95_low_rank_binary",
        "ci95_high_rank_binary",
        "rank_binary_fit_status",
        "n_survival",
        "events_survival",
        "n_high_rank",
        "events_high_rank",
        "n_low_rank",
        "events_low_rank",
    ]
    for position, row in enumerate(selected.itertuples(index=False), 1):
        motif_hash = str(row.wl_hash)
        graph = base.graph_from_encoding(
            str(row.node_labels), str(row.edge_list), motif_hash
        )
        values, raw = engine.extra_features(graph, motif_hash, permutations, seed)
        record: dict[str, object] = {
            "triangle_level": int(row.triangle_level),
            "wl_hash": motif_hash,
            "significance_group": str(row.significance_group),
            "significant_rank_p": bool(row.significant_rank_p),
            "p_value_rank_binary": float(row.p_value_rank_binary),
            "coef_rank_binary": float(row.coef_rank_binary),
            "hazard_ratio_rank_binary": float(row.hazard_ratio_rank_binary),
            "median_patient_rank_score": float(row.median_patient_rank_score),
            "sample_document_frequency": float(getattr(row, "sample_document_frequency", getattr(row, "document_frequency", np.nan))),
            **{column: getattr(row, column) for column in audit_columns},
            **values,
        }
        rows.append(record)
        for distribution, entries in raw.items():
            if distribution == "normalized_laplacian_eigenvalue":
                spectra_rows.extend(
                    {
                        "wl_hash": motif_hash,
                        "significance_group": str(row.significance_group),
                        "eigenvalue_index": index,
                        "eigenvalue": value,
                    }
                    for index, value in enumerate(entries)
                )
            else:
                for value, count in Counter(entries).items():
                    raw_distributions[(distribution, str(row.significance_group), value)] += count
        if position % 500 == 0 or position == len(selected):
            print(
                f"Computed criteria for {position:,}/{len(selected):,} rank-estimable motifs",
                flush=True,
            )
    features = pd.DataFrame(rows)
    distributions = pd.DataFrame(
        [
            {
                "distribution": key[0],
                "significance_group": key[1],
                "value": key[2],
                "count": count,
            }
            for key, count in sorted(raw_distributions.items())
        ]
    )
    return features, distributions, pd.DataFrame(spectra_rows)


def write_rank_readme(
    path: Path,
    features: pd.DataFrame,
    metrics: list[str],
    registry: pd.DataFrame,
    pages: int,
    args: argparse.Namespace,
) -> None:
    family_counts = (
        registry.loc[registry["availability"].eq("computed")]
        .groupby("family")
        .size()
        .sort_values(ascending=False)
    )
    lines = [
        "# Rank-normalized TRI-5 criterion distributions",
        "",
        "This analysis applies the established sibling-topology criteria to the "
        "rank-normalized TRI-5 lineage. Only motifs with an estimated finite binary "
        "Cox p-value are compared.",
        "",
        f"- Significant motifs (`p_value_rank_binary < {args.p_threshold:g}`): "
        f"{int(features.significant_rank_p.sum()):,}",
        f"- Nonsignificant motifs (finite `p_value_rank_binary >= {args.p_threshold:g}`): "
        f"{int((~features.significant_rank_p).sum()):,}",
        f"- Computed scalar criteria: {len(metrics):,}",
        f"- Distribution figure pages: {len(pages):,}",
        f"- Label-null permutations per motif: {args.permutations:,}",
        "",
        "The significance label is nominal and exploratory. Undefined Cox fits are "
        "excluded rather than treated as nonsignificant. Criterion-level BH values "
        "do not redefine motif significance.",
        "",
        "The tables and figures mirror `sibling_topology_analysis_new/results`, "
        "including overall and family-specific PCA plus topology-matched representative pairs.",
        "",
        "## Computed families",
        "",
    ]
    lines.extend(
        f"- {family.replace('_', ' ')}: {count}" for family, count in family_counts.items()
    )
    guide = (SOURCE_PROJECT / "criteria_family_guide.md").read_text().strip()
    guide = guide.replace("raw Cox `p < 0.05`", "rank-normalized Cox `p < 0.05`")
    guide = guide.replace("raw Cox `p >= 0.05`", "rank-normalized Cox `p >= 0.05`")
    lines += ["", guide, ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def regenerate_plots(args: argparse.Namespace) -> dict[str, object]:
    tables = args.outdir / "tables"
    figures = args.outdir / "figures"
    features = pd.read_csv(tables / "tri5_motif_criteria.csv.gz", dtype={"wl_hash": str})
    compatibility = add_compatibility_columns(features)
    comparisons = pd.read_csv(tables / "tri5_criteria_comparisons.csv")
    pages, figure_index = engine.plot_pages(
        compatibility, comparisons, figures, args.panels_per_page
    )
    figure_index.to_csv(tables / "tri5_distribution_figure_index.csv", index=False)
    engine.plot_leading_distributions(
        compatibility, comparisons, figures / "tri5_metric_distributions.png"
    )
    pca = engine.plot_all_criteria_pca(compatibility, tables, figures, args.seed)
    family = engine.plot_family_pcas(compatibility, tables, figures, args.seed)
    source = pd.read_csv(
        args.trace_dir / "trace_nodes.csv",
        dtype={"wl_hash": str, "node_labels": str, "edge_list": str},
        low_memory=False,
    )
    representatives = engine.generate_representative_gallery(
        compatibility,
        source.loc[source["triangle_level"].eq(args.primary_level)],
        tables,
        figures,
        args.seed,
        args.representative_pairs,
    )
    rewrite_generated_metadata_tables(tables)
    return {"distribution_figure_pages": len(pages), **pca, **family, **representatives}


def main() -> None:
    args = parse_args()
    if args.permutations < 2:
        raise ValueError("--permutations must be at least 2")
    if not 0 < args.p_threshold < 1:
        raise ValueError("--p-threshold must be between zero and one")
    if args.representative_pairs < 1:
        raise ValueError("--representative-pairs must be positive")
    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.plots_only:
        update = regenerate_plots(args)
        summary_path = args.outdir / "analysis_summary.json"
        summary = json.loads(summary_path.read_text())
        summary.update(update)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps({"status": "complete", "mode": "plots_only", **update}, indent=2))
        return

    source_path = args.trace_dir / "trace_nodes.csv"
    source = pd.read_csv(
        source_path,
        dtype={"wl_hash": str, "node_labels": str, "edge_list": str},
        low_memory=False,
    )
    selected = select_estimable_motifs(source, args.primary_level, args.p_threshold)
    features, raw_distributions, spectra = build_features(
        selected, args.permutations, args.seed
    )
    compatibility = add_compatibility_columns(features)
    engine.add_dispersion_metrics(compatibility)
    metrics = [column for column in compatibility.columns if column not in METADATA]
    registry = engine.build_registry(metrics, args.permutations)
    dictionary = registry.loc[
        registry["availability"].eq("computed"),
        ["criterion", "display_name", "family", "reason_or_method"],
    ].rename(columns={"criterion": "metric"})
    distribution, comparison = engine.distribution_tables(
        compatibility, metrics, registry
    )
    wide = distribution.pivot(
        index=["metric", "display_name", "family"], columns="group"
    ).reset_index()
    wide.columns = [
        "_".join(str(value) for value in column if value).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in wide.columns
    ]

    tables = args.outdir / "tables"
    figures = args.outdir / "figures"
    families = tables / "distributions_by_family"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    families.mkdir(parents=True, exist_ok=True)
    clean_rank_columns(compatibility).to_csv(
        tables / "tri5_motif_criteria.csv.gz", index=False, compression="gzip"
    )
    registry.to_csv(tables / "criteria_registry.csv", index=False)
    dictionary.to_csv(tables / "metric_dictionary.csv", index=False)
    distribution.to_csv(tables / "tri5_criteria_distribution_long.csv", index=False)
    wide.to_csv(tables / "tri5_criteria_distribution_wide.csv", index=False)
    comparison.to_csv(tables / "tri5_criteria_comparisons.csv", index=False)
    for family, frame in distribution.groupby("family"):
        frame.to_csv(families / f"{family}.csv", index=False)
    raw_distributions.to_csv(tables / "tri5_within_motif_distributions.csv", index=False)
    spectra.to_csv(
        tables / "tri5_normalized_laplacian_eigenvalues.csv.gz",
        index=False,
        compression="gzip",
    )

    pages, figure_index = engine.plot_pages(
        compatibility, comparison, figures, args.panels_per_page
    )
    figure_index.to_csv(tables / "tri5_distribution_figure_index.csv", index=False)
    engine.plot_leading_distributions(
        compatibility, comparison, figures / "tri5_metric_distributions.png"
    )
    pca_summary = engine.plot_all_criteria_pca(
        compatibility, tables, figures, args.seed
    )
    family_pca_summary = engine.plot_family_pcas(
        compatibility, tables, figures, args.seed
    )
    representative_summary = engine.generate_representative_gallery(
        compatibility,
        source.loc[source["triangle_level"].eq(args.primary_level)],
        tables,
        figures,
        args.seed,
        args.representative_pairs,
    )
    rewrite_generated_metadata_tables(tables)
    write_rank_readme(tables / "README.md", features, metrics, registry, pages, args)

    undefined = source.loc[
        source["triangle_level"].eq(args.primary_level)
        & ~np.isfinite(pd.to_numeric(source["p_value_rank_binary"], errors="coerce"))
    ]
    summary = {
        "status": "complete",
        "primary_level": args.primary_level,
        "significance_rule": f"estimated finite p_value_rank_binary < {args.p_threshold:g}",
        "significant_motifs": int(features["significant_rank_p"].sum()),
        "nonsignificant_motifs": int((~features["significant_rank_p"]).sum()),
        "excluded_undefined_p_value": int(len(undefined)),
        "computed_scalar_criteria": len(metrics),
        "unavailable_criteria": int(registry["availability"].eq("not_available").sum()),
        "label_null_permutations": args.permutations,
        "distribution_figure_pages": len(pages),
        **pca_summary,
        **family_pca_summary,
        **representative_summary,
        "input_trace": str(source_path.resolve()),
        "reproduction_command": (
            "python breast_cancer_survival/motif_results/"
            "sibling_topology_analysis_ranking_normalized/run_analysis.py"
        ),
    }
    (args.outdir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
