#!/usr/bin/env python3
"""Screen every nonzero-threshold motif using within-core rank normalization."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
os.environ.setdefault("MPLCONFIGDIR", "/tmp/bc-all-rank-motif-matplotlib")
import matplotlib.pyplot as plt

import analyze_rank_normalized_motif_survival as rankmod
from analyze_motif_survival import benjamini_hochberg
from analyze_motif_survival_unadjusted import STATUS_LABELS, vectorized_binary_cox
from normalize_significant_motif_frequencies import clinical_core_cohort
from visualize_top_motif_survival import (
    CELL_TYPE_COLORS,
    CELL_TYPE_ORDER,
    load_topologies,
    motif_graph,
)


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-results-root", type=Path, default=here / "motif_results")
    parser.add_argument("--artifacts-root", type=Path, default=repo / "breast_cancer_classification_workstation/classification_by_level/mtf_occurrence/mrmr_selection/parameter_tuning_bz/artifacts")
    parser.add_argument("--motif-root", type=Path, default=repo / "breast_cancer_meta/BZ_res_threshold_10_cpp_0811")
    parser.add_argument("--metadata", type=Path, default=repo / "breast_cancer/Basel_PatientMetadata.csv")
    parser.add_argument("--label-map", type=Path, default=repo / "breast_cancer/BZ_samples_intra/node_label_cell_type_mapping.csv")
    parser.add_argument("--outdir", type=Path, default=here / "motif_results/normalized_by_ranking_motif_analysis")
    parser.add_argument("--levels", type=int, nargs="+", default=list(range(1, 8)))
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def load_nonzero_candidates(raw_results_root: Path, level: int) -> pd.DataFrame:
    path = raw_results_root / f"tri_{level}_cox_results_sorted.csv"
    frame = pd.read_csv(path, dtype={"wl_hash": str})
    required = {"triangle_level", "wl_hash", "median_patient_raw_count", "p_value_unadjusted", "cell_type_composition"}
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError(f"TRI-{level} raw result is missing columns: {missing}")
    if not frame["triangle_level"].eq(level).all() or frame["wl_hash"].duplicated().any():
        raise ValueError(f"Invalid TRI-{level} raw result identities")
    return frame.loc[frame["median_patient_raw_count"].gt(0)].reset_index(drop=True)


def primary_screen(
    selected: pd.DataFrame,
    patient_scores: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    vocabulary_size: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    medians = np.median(patient_scores, axis=0)
    if not np.all(medians > 0):
        raise ValueError("A raw-positive-median candidate has a zero rank-score median")
    groups = patient_scores > medians
    fitted = vectorized_binary_cox(groups, time, event)
    result = selected.copy()
    result["vocabulary_size"] = vocabulary_size
    result["median_patient_rank_score"] = medians
    result["rank_group_definition"] = "patient_rank_score > cohort_median"
    result["n_survival"] = len(time)
    result["events_survival"] = int(event.sum())
    result["n_high_rank"] = fitted["n_enriched"]
    result["events_high_rank"] = fitted["events_enriched"]
    result["n_low_rank"] = fitted["n_non_enriched"]
    result["events_low_rank"] = fitted["events_non_enriched"]
    result["coef_rank_binary"] = fitted["coef"]
    result["hazard_ratio_rank_binary"] = fitted["hazard_ratio"]
    result["ci95_low_rank_binary"] = fitted["ci95_low"]
    result["ci95_high_rank_binary"] = fitted["ci95_high"]
    result["p_value_rank_binary"] = fitted["p_value"]
    result["rank_binary_fit_status"] = [STATUS_LABELS[int(code)] for code in fitted["status_code"]]
    return result, groups


def add_secondary_statistics(
    row: dict[str, object], score: np.ndarray, time: np.ndarray, event: np.ndarray
) -> dict[str, object]:
    secondary, _, _ = rankmod.analyze_rank_survival(score, time, event)
    for key in [
        "logrank_statistic", "logrank_p_value", "logrank_status", "rank_score_mean",
        "rank_score_sd", "coef_rank_continuous_per_sd",
        "hazard_ratio_rank_continuous_per_sd", "ci95_low_rank_continuous_per_sd",
        "ci95_high_rank_continuous_per_sd", "p_value_rank_continuous_per_sd",
        "rank_continuous_fit_status", "rank_continuous_fit_warning",
    ]:
        row[key] = secondary[key]
    return row


def ranked_motif_directory_name(rank_within_level: int, motif_hash: str) -> str:
    """Prefix a WL hash with its zero-padded within-TRI binary-Cox p-value rank."""
    rank = int(rank_within_level)
    if rank < 1:
        raise ValueError("Within-level motif rank must be positive")
    return f"{rank:03d}_{motif_hash}"


def patient_frame(
    level: int,
    motif_hash: str,
    aggregated: pd.DataFrame,
    column: int,
    median: float,
) -> pd.DataFrame:
    scores = np.stack(aggregated["rank_score"].to_numpy())[:, column]
    return pd.DataFrame(
        {
            "triangle_level": level,
            "wl_hash": motif_hash,
            "PID": aggregated["PID"],
            "number_of_cores": aggregated["number_of_cores"],
            "summed_raw_count": np.stack(aggregated["summed_raw_count"].to_numpy())[:, column],
            "mean_core_rank": np.stack(aggregated["mean_core_rank"].to_numpy())[:, column],
            "rank_score": scores,
            "time_months": aggregated["time_months"],
            "event": aggregated["event"],
            "median_rank_score": median,
            "rank_group": np.where(scores > median, "High rank score", "Low rank score"),
        }
    )


def load_tri1_topologies(selected: pd.DataFrame, motif_root: Path) -> pd.DataFrame:
    wanted = set(selected["wl_hash"].astype(str))
    found: list[pd.DataFrame] = []
    for path in sorted(motif_root.glob("sample*/ccindex_tri_1.csv")):
        frame = pd.read_csv(
            path,
            usecols=["wl_hash", "node_labels", "edge_list"],
            dtype={"wl_hash": str, "node_labels": str, "edge_list": str},
        )
        subset = frame.loc[frame["wl_hash"].isin(wanted)]
        if len(subset):
            found.append(subset)
            wanted -= set(subset["wl_hash"])
        if not wanted:
            break
    if wanted:
        raise ValueError(f"Missing TRI-1 topologies: {sorted(wanted)[:3]}")
    topology = pd.concat(found, ignore_index=True).drop_duplicates("wl_hash")
    return selected.merge(topology, on="wl_hash", how="left", validate="one_to_one")


def recover_qualified_topologies(
    qualified: pd.DataFrame,
    levels: list[int],
    artifacts_root: Path,
    motif_root: Path,
) -> dict[tuple[int, str], tuple[str, str]]:
    output: dict[tuple[int, str], tuple[str, str]] = {}
    for level in levels:
        selected = qualified.loc[qualified["triangle_level"].eq(level)].copy()
        if selected.empty:
            continue
        if level == 1:
            topology = load_tri1_topologies(selected, motif_root)
        else:
            topology = load_topologies(selected, level, artifacts_root, motif_root)
        for row in topology.itertuples(index=False):
            output[(level, str(row.wl_hash))] = (str(row.node_labels), str(row.edge_list))
    if len(output) != len(qualified):
        raise ValueError("Recovered topology count does not match qualified motifs")
    return output


def plot_delaunay_topology(
    level: int,
    motif_hash: str,
    composition: str,
    node_labels: str,
    edge_list: str,
    label_map: dict[str, str],
    output: Path,
    dpi: int,
) -> None:
    graph = motif_graph(node_labels, edge_list)
    planar, _ = nx.check_planarity(graph)
    if not planar:
        raise ValueError(f"TRI-{level} {motif_hash} is not planar")
    positions = nx.planar_layout(graph, scale=1.0)
    numeric_labels = str(node_labels).split("&")
    cell_types = [label_map.get(value, "unknown") for value in numeric_labels]
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    nx.draw_networkx_edges(
        graph, positions, ax=ax, edge_color="#5F6368", width=2.0, alpha=0.92
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        ax=ax,
        nodelist=list(range(len(numeric_labels))),
        node_color=[CELL_TYPE_COLORS.get(value, CELL_TYPE_COLORS["unknown"]) for value in cell_types],
        node_size=920,
        edgecolors="white",
        linewidths=2.0,
    )
    present = [value for value in CELL_TYPE_ORDER if value in set(cell_types)]
    handles = [
        plt.Line2D(
            [0], [0], marker="o", linestyle="none", markersize=10,
            markerfacecolor=CELL_TYPE_COLORS[value], markeredgecolor="white", label=value,
        )
        for value in present
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=min(4, len(handles)), frameon=False)
    ax.set_title(
        f"TRI-{level} motif topology | {motif_hash[:8]}\n{composition}",
        fontsize=12,
        pad=14,
    )
    ax.margins(0.18)
    ax.set_axis_off()
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def replace_significant_tree(
    outdir: Path,
    all_results: pd.DataFrame,
    qualified: pd.DataFrame,
    frames: dict[tuple[int, str], pd.DataFrame],
    levels: list[int],
    dpi: int,
    skip_plots: bool,
    audit: dict[str, object],
    topologies: dict[tuple[int, str], tuple[str, str]],
    label_map: dict[str, str],
) -> dict[str, object]:
    staging = Path(tempfile.mkdtemp(prefix=".all-rank-significant-", dir=outdir))
    destination = outdir / "significant"
    backup = outdir / ".significant-backup"
    try:
        for level in levels:
            (staging / f"tri{level}").mkdir(parents=True)
        for result in qualified.itertuples(index=False):
            level, motif_hash = int(result.triangle_level), str(result.wl_hash)
            directory = staging / f"tri{level}" / ranked_motif_directory_name(
                int(result.rank_within_level_rank_binary), motif_hash
            )
            directory.mkdir()
            data = frames[(level, motif_hash)]
            row = pd.Series(result._asdict())
            data.to_csv(directory / "patient_rank_scores.csv", index=False)
            with (directory / "statistics.json").open("w") as handle:
                json.dump(rankmod.json_safe(row.to_dict()), handle, indent=2)
            if not skip_plots:
                rankmod.plot_km(data, row, directory / "rank_normalized_survival_km.png", dpi)
                rankmod.plot_distribution(data, row, directory / "patient_rank_distribution.png", dpi)
                node_labels, edge_list = topologies[(level, motif_hash)]
                plot_delaunay_topology(
                    level,
                    motif_hash,
                    str(result.cell_type_composition),
                    node_labels,
                    edge_list,
                    label_map,
                    directory / "motif_delaunay_topology.png",
                    dpi,
                )

        all_results.to_csv(staging / "all_nonzero_threshold_rank_results.csv.gz", index=False, compression="gzip")
        qualified_output = qualified.copy()
        qualified_output["motif_directory"] = [
            f"tri{int(row.triangle_level)}/"
            f"{ranked_motif_directory_name(int(row.rank_within_level_rank_binary), str(row.wl_hash))}"
            for row in qualified.itertuples(index=False)
        ]
        qualified_output.to_csv(
            staging / "significant_rank_normalized_survival_results.csv", index=False
        )
        counts = {str(level): int(qualified["triangle_level"].eq(level).sum()) for level in levels}
        candidates = {str(level): int(all_results["triangle_level"].eq(level).sum()) for level in levels}
        summary = {
            **audit,
            "selection": "median_patient_rank_score > 0 and finite p_value_rank_binary < 0.05",
            "screening_family": "all motifs with nonzero patient rank-score median; no raw p-value filter",
            "candidate_motifs": int(len(all_results)),
            "candidate_motifs_by_level": candidates,
            "total_significant_motifs": int(len(qualified)),
            "significant_motifs_by_level": counts,
            "directory_layout": "triN/<zero-padded within-level p-value rank>_<wl_hash>/",
            "plots_written": not skip_plots,
        }
        with (staging / "significant_summary.json").open("w") as handle:
            json.dump(rankmod.json_safe(summary), handle, indent=2)
        (staging / "README.md").write_text(
            "# Significant motifs from the complete nonzero-threshold rank screen\n\n"
            "This collection is not restricted to the former 644 raw-significant motifs. "
            "It screens every motif whose patient median rank score is greater than zero, "
            "then retains motifs with a finite unadjusted rank-normalized binary Cox Wald "
            "p-value strictly below 0.05. Patients strictly above the motif-specific median "
            "rank score are the high-rank group; ties are low-rank.\n\n"
            "`all_nonzero_threshold_rank_results.csv.gz` contains the complete screened "
            "family. `significant_rank_normalized_survival_results.csv` and the `triN/` "
            "directories contain the qualified motifs. Folder names begin with the "
            "zero-padded within-TRI rank of `p_value_rank_binary`, followed by the full "
            "WL hash. Exact p-value ties share a rank prefix. Each motif directory also has "
            "an unlabeled, planar Delaunay-style topology using one unified cell-type "
            "color palette. The drawing preserves motif adjacency but does not represent "
            "physical tissue coordinates. These are nominal exploratory "
            "results and are not selected by a BH-adjusted threshold.\n"
        )
        if backup.exists():
            shutil.rmtree(backup)
        if destination.exists():
            destination.rename(backup)
        staging.rename(destination)
        if backup.exists():
            shutil.rmtree(backup)
        return summary
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        if backup.exists() and not destination.exists():
            backup.rename(destination)
        raise


def main() -> None:
    args = parse_args()
    levels = sorted(set(args.levels))
    if not levels or any(level < 1 or level > 7 for level in levels):
        raise ValueError("--levels must contain values from 1 through 7")
    metadata = pd.read_csv(args.metadata)
    label_frame = pd.read_csv(args.label_map, dtype={"node_label": str})
    if label_frame["node_label"].duplicated().any():
        raise ValueError("Cell-type label map contains duplicate node labels")
    label_map = dict(label_frame[["node_label", "cell_type"]].itertuples(index=False, name=None))
    cohort, patients, audit = clinical_core_cohort(args.artifacts_root, metadata)
    time = patients["time_months"].to_numpy(float)
    event = patients["event"].to_numpy(int)
    result_parts: list[pd.DataFrame] = []
    frames: dict[tuple[int, str], pd.DataFrame] = {}
    for level in levels:
        selected = load_nonzero_candidates(args.raw_results_root, level)
        print(f"TRI-{level}: screening {len(selected):,} nonzero-threshold motifs", flush=True)
        if level == 1:
            raw, ranks, scores, vocabulary_size = rankmod.tri1_core_values(selected, cohort, args.motif_root)
        else:
            raw, ranks, scores, vocabulary_size = rankmod.high_level_core_values(level, selected, cohort, args.artifacts_root)
        aggregated = rankmod.aggregate_patient_values(cohort, patients, raw, ranks, scores)
        score_matrix = np.stack(aggregated["rank_score"].to_numpy())
        result, _ = primary_screen(selected, score_matrix, time, event, vocabulary_size)
        significant_columns = np.flatnonzero(np.isfinite(result["p_value_rank_binary"]) & result["p_value_rank_binary"].lt(0.05))
        for column in significant_columns:
            record = add_secondary_statistics(result.iloc[column].to_dict(), score_matrix[:, column], time, event)
            for key, value in record.items():
                result.loc[column, key] = value
            motif_hash = str(result.iloc[column]["wl_hash"])
            frames[(level, motif_hash)] = patient_frame(level, motif_hash, aggregated, column, float(result.iloc[column]["median_patient_rank_score"]))
        result_parts.append(result)
        print(f"TRI-{level}: {len(significant_columns):,} motifs with rank p < 0.05", flush=True)

    all_results = pd.concat(result_parts, ignore_index=True)
    all_results["q_value_bh_global_rank_binary"] = benjamini_hochberg(all_results["p_value_rank_binary"])
    all_results["q_value_bh_within_level_rank_binary"] = all_results.groupby("triangle_level")["p_value_rank_binary"].transform(lambda values: benjamini_hochberg(values))
    all_results["rank_global_rank_binary"] = all_results["p_value_rank_binary"].rank(method="min", na_option="keep").astype("Int64")
    all_results["rank_within_level_rank_binary"] = all_results.groupby("triangle_level")["p_value_rank_binary"].rank(method="min", na_option="keep").astype("Int64")
    all_results = all_results.sort_values(["p_value_rank_binary", "triangle_level", "wl_hash"], na_position="last", kind="mergesort").reset_index(drop=True)
    qualified = all_results.loc[np.isfinite(all_results["p_value_rank_binary"]) & all_results["p_value_rank_binary"].lt(0.05)].copy()
    args.outdir.mkdir(parents=True, exist_ok=True)
    topologies = recover_qualified_topologies(
        qualified, levels, args.artifacts_root, args.motif_root
    )
    summary = replace_significant_tree(
        args.outdir, all_results, qualified, frames, levels, args.dpi,
        args.skip_plots, audit, topologies, label_map
    )
    root_summary_path = args.outdir / "analysis_summary.json"
    if root_summary_path.is_file():
        root_summary = json.loads(root_summary_path.read_text())
        root_summary["complete_nonzero_threshold_screen"] = summary
        root_summary_path.write_text(json.dumps(rankmod.json_safe(root_summary), indent=2))
    print(json.dumps(rankmod.json_safe(summary), indent=2), flush=True)


if __name__ == "__main__":
    main()
