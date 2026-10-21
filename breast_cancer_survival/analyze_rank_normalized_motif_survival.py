#!/usr/bin/env python3
"""Rank-normalize motif counts and rerun focal-motif survival analyses."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/bc-rank-motif-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr

from analyze_ki67_survival import COLORS, logrank_test, step_function
from analyze_motif_survival import benjamini_hochberg, fit_cox
from analyze_motif_survival_unadjusted import load_inventory
from normalize_significant_motif_frequencies import clinical_core_cohort, load_core_cell_counts


MIN_LEVEL, MAX_LEVEL = 1, 7
EXPECTED_FOCAL_COUNTS = {1: 3, 2: 3, 3: 5, 4: 9, 5: 32, 6: 142, 7: 450}


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-csv", type=Path, default=here / "motif_results/significant_mtf_trace/potential_motif_pool.csv")
    parser.add_argument("--artifacts-root", type=Path, default=repo / "breast_cancer_classification_workstation/classification_by_level/mtf_occurrence/mrmr_selection/parameter_tuning_bz/artifacts")
    parser.add_argument("--motif-root", type=Path, default=repo / "breast_cancer_meta/BZ_res_threshold_10_cpp_0811")
    parser.add_argument("--metadata", type=Path, default=repo / "breast_cancer/Basel_PatientMetadata.csv")
    parser.add_argument("--h5ad", type=Path, default=repo / "breast_cancer/BZ.h5ad")
    parser.add_argument("--outdir", type=Path, default=here / "motif_results/normalized_by_ranking_motif_analysis")
    parser.add_argument("--levels", type=int, nargs="+", default=list(range(1, 8)))
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def load_focal_pool(path: Path, levels: Iterable[int]) -> pd.DataFrame:
    levels = sorted(set(int(x) for x in levels))
    if not levels or any(x < 1 or x > 7 for x in levels):
        raise ValueError("Levels must be integers from 1 through 7")
    pool = pd.read_csv(path, dtype={"wl_hash": str})
    required = {"triangle_level", "wl_hash", "cell_type_composition", "median_patient_raw_count", "p_value_unadjusted", "hazard_ratio_unadjusted"}
    missing = sorted(required - set(pool))
    if missing:
        raise ValueError(f"Focal motif pool is missing columns: {missing}")
    if pool.duplicated(["triangle_level", "wl_hash"]).any():
        raise ValueError("Focal motif pool contains duplicate level/hash pairs")
    if not pool["p_value_unadjusted"].lt(0.05).all() or not pool["median_patient_raw_count"].gt(0).all():
        raise ValueError("Focal motifs must have raw p < 0.05 and positive raw-count medians")
    observed = pool.groupby("triangle_level").size().to_dict()
    if observed != EXPECTED_FOCAL_COUNTS:
        raise ValueError(f"Unexpected focal motif counts by level: {observed}")
    return pool.loc[pool["triangle_level"].isin(levels)].sort_values(
        ["triangle_level", "p_value_unadjusted", "wl_hash"], kind="mergesort"
    ).reset_index(drop=True)


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    left, right = np.asarray(left, float), np.asarray(right, float)
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3 or np.ptp(left[valid]) == 0 or np.ptp(right[valid]) == 0:
        return np.nan
    return float(spearmanr(left[valid], right[valid]).statistic)


def rank_scores(ranks: np.ndarray, vocabulary_size: int) -> np.ndarray:
    ranks = np.asarray(ranks, float)
    if vocabulary_size < 2 or not np.isfinite(ranks).all() or (ranks < 1).any() or (ranks > vocabulary_size).any():
        raise ValueError("Invalid vocabulary size or ranks")
    return (vocabulary_size - ranks) / (vocabulary_size - 1)


def selected_ranks_from_sparse_row(row: sparse.spmatrix, selected_columns: np.ndarray, vocabulary_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rank selected motifs against every positive motif in one sample row."""
    row = row.tocsr(copy=True)
    row.sum_duplicates()
    row.eliminate_zeros()
    if row.shape != (1, vocabulary_size):
        raise ValueError("Sparse row and vocabulary dimensions differ")
    if not np.isfinite(row.data).all() or (row.data < 0).any():
        raise ValueError("Motif counts must be finite and nonnegative")
    selected_columns = np.asarray(selected_columns, int)
    if selected_columns.ndim != 1 or (selected_columns < 0).any() or (selected_columns >= vocabulary_size).any():
        raise ValueError("Selected columns are outside the vocabulary")
    raw = row[:, selected_columns].toarray().ravel().astype(float)
    ranks = np.full(len(selected_columns), float(vocabulary_size))
    positive = raw > 0
    if positive.any():
        observed = np.sort(row.data)
        values = raw[positive]
        left = np.searchsorted(observed, values, side="left")
        right = np.searchsorted(observed, values, side="right")
        ranks[positive] = len(observed) - right + (right - left + 1) / 2
    return raw, ranks, rank_scores(ranks, vocabulary_size)


def tri1_vocabulary(motif_root: Path) -> list[str]:
    vocabulary: set[str] = set()
    files = sorted(motif_root.glob("sample*/ccindex_tri_1.csv"))
    if len(files) != 547:
        raise ValueError(f"Expected 547 TRI-1 files, found {len(files)}")
    for path in files:
        frame = pd.read_csv(path, usecols=["wl_hash", "matches"], dtype={"wl_hash": str})
        if frame["wl_hash"].duplicated().any():
            raise ValueError(f"Duplicate TRI-1 hash in {path}")
        counts = pd.to_numeric(frame["matches"], errors="raise")
        if not np.isfinite(counts).all() or counts.lt(0).any():
            raise ValueError(f"Invalid TRI-1 counts in {path}")
        vocabulary.update(frame.loc[counts.gt(0), "wl_hash"])
    if len(vocabulary) != 84:
        raise ValueError(f"Expected 84 TRI-1 motifs, found {len(vocabulary)}")
    return sorted(vocabulary)


def tri1_core_values(selected: pd.DataFrame, cohort: pd.DataFrame, motif_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    vocabulary = tri1_vocabulary(motif_root)
    lookup = {value: i for i, value in enumerate(vocabulary)}
    if not selected["wl_hash"].isin(lookup).all():
        raise ValueError("A focal TRI-1 hash is missing from the vocabulary")
    columns = np.array([lookup[x] for x in selected["wl_hash"]], int)
    raw = np.zeros((len(cohort), len(selected)))
    ranks, scores = np.empty_like(raw), np.empty_like(raw)
    for i, item in enumerate(cohort.itertuples(index=False)):
        path = motif_root / str(item.sample) / "ccindex_tri_1.csv"
        frame = pd.read_csv(path, usecols=["wl_hash", "matches"], dtype={"wl_hash": str})
        values = pd.to_numeric(frame["matches"], errors="raise").to_numpy(float)
        positions = frame["wl_hash"].map(lookup)
        if positions.isna().any() or (values < 0).any():
            raise ValueError(f"Invalid TRI-1 data in {path}")
        row = sparse.csr_matrix((values, (np.zeros(len(values), int), positions.to_numpy(int))), shape=(1, len(vocabulary)))
        raw[i], ranks[i], scores[i] = selected_ranks_from_sparse_row(row, columns, len(vocabulary))
    return raw, ranks, scores, len(vocabulary)


def high_level_core_values(level: int, selected: pd.DataFrame, cohort: pd.DataFrame, artifacts_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    level_dir = artifacts_root / f"tri_{level}"
    samples = load_inventory(artifacts_root, level)
    observed = samples.iloc[cohort["artifact_row"].to_numpy(int)][["sample", "source_sample"]].reset_index(drop=True)
    expected = cohort[["sample", "source_sample"]].reset_index(drop=True)
    if not observed.equals(expected):
        raise ValueError(f"TRI-{level} sample inventory differs")
    features = pd.read_csv(level_dir / "feature_index.csv.gz", usecols=["feature_index", "wl_hash"], dtype={"wl_hash": str})
    if features["wl_hash"].duplicated().any():
        raise ValueError(f"TRI-{level} feature hashes are duplicated")
    lookup = features.set_index("wl_hash")["feature_index"]
    if not selected["wl_hash"].isin(lookup.index).all():
        raise ValueError(f"A focal TRI-{level} hash is missing")
    columns = lookup.loc[selected["wl_hash"]].to_numpy(int)
    matrix = sparse.load_npz(level_dir / "motif_counts.npz").tocsr()
    if matrix.shape != (len(samples), len(features)):
        raise ValueError(f"TRI-{level} matrix dimensions differ from its indices")
    raw = np.zeros((len(cohort), len(selected)))
    ranks, scores = np.empty_like(raw), np.empty_like(raw)
    for i, matrix_row in enumerate(cohort["artifact_row"].to_numpy(int)):
        raw[i], ranks[i], scores[i] = selected_ranks_from_sparse_row(matrix.getrow(matrix_row), columns, len(features))
    return raw, ranks, scores, len(features)


def aggregate_patient_values(cohort: pd.DataFrame, patients: pd.DataFrame, raw: np.ndarray, ranks: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    if raw.shape != ranks.shape or raw.shape != scores.shape or raw.shape[0] != len(cohort):
        raise ValueError("Core motif matrices do not align")
    groups = cohort.groupby("PID", sort=False).indices
    rows = []
    for patient in patients.itertuples(index=False):
        if patient.PID not in groups:
            raise ValueError(f"Patient {patient.PID} lacks a core")
        indices = np.asarray(groups[patient.PID], int)
        rows.append({"PID": patient.PID, "number_of_cores": len(indices), "summed_raw_count": raw[indices].sum(axis=0), "mean_core_rank": ranks[indices].mean(axis=0), "rank_score": scores[indices].mean(axis=0), "time_months": patient.time_months, "event": patient.event})
    return pd.DataFrame(rows)


def analyze_rank_survival(patient_score: np.ndarray, time: np.ndarray, event: np.ndarray) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    score, time, event = np.asarray(patient_score, float), np.asarray(time, float), np.asarray(event, int)
    if len(score) != len(time) or len(time) != len(event) or not np.isfinite(score).all():
        raise ValueError("Patient arrays are invalid")
    median = float(np.median(score))
    high = score > median
    binary = fit_cox(time, event, high)
    if high.any() and (~high).any():
        try:
            logrank_stat, logrank_p = logrank_test(time, event, high)
            logrank_status = "estimated"
        except Exception as exc:
            logrank_stat = logrank_p = np.nan
            logrank_status = f"not_estimable_{type(exc).__name__}"
    else:
        logrank_stat = logrank_p = np.nan
        logrank_status = "not_estimable_no_group_variation"
    sd = float(np.std(score, ddof=1))
    if np.isfinite(sd) and sd > 0:
        score_z = (score - score.mean()) / sd
        continuous = fit_cox(time, event, score_z)
    else:
        score_z = np.full(len(score), np.nan)
        continuous = {"coef": np.nan, "hazard_ratio": np.nan, "ci95_low": np.nan, "ci95_high": np.nan, "p_value": np.nan, "fit_status": "not_estimable_constant_score", "fit_warning": "Rank score has zero variance"}
    result = {
        "median_patient_rank_score": median, "rank_group_definition": "patient_rank_score > cohort_median",
        "n_survival": len(score), "events_survival": int(event.sum()),
        "n_high_rank": int(high.sum()), "events_high_rank": int(event[high].sum()),
        "n_low_rank": int((~high).sum()), "events_low_rank": int(event[~high].sum()),
        "coef_rank_binary": binary["coef"], "hazard_ratio_rank_binary": binary["hazard_ratio"],
        "ci95_low_rank_binary": binary["ci95_low"], "ci95_high_rank_binary": binary["ci95_high"],
        "p_value_rank_binary": binary["p_value"], "rank_binary_fit_status": binary["fit_status"], "rank_binary_fit_warning": binary["fit_warning"],
        "logrank_statistic": logrank_stat, "logrank_p_value": logrank_p, "logrank_status": logrank_status,
        "rank_score_mean": float(score.mean()), "rank_score_sd": sd,
        "coef_rank_continuous_per_sd": continuous["coef"], "hazard_ratio_rank_continuous_per_sd": continuous["hazard_ratio"],
        "ci95_low_rank_continuous_per_sd": continuous["ci95_low"], "ci95_high_rank_continuous_per_sd": continuous["ci95_high"],
        "p_value_rank_continuous_per_sd": continuous["p_value"], "rank_continuous_fit_status": continuous["fit_status"], "rank_continuous_fit_warning": continuous["fit_warning"],
    }
    return result, high, score_z


def level_outputs(level: int, selected: pd.DataFrame, cohort: pd.DataFrame, patients: pd.DataFrame, total_cells: np.ndarray, raw: np.ndarray, ranks: np.ndarray, scores: np.ndarray, vocabulary_size: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    aggregated = aggregate_patient_values(cohort, patients, raw, ranks, scores)
    raw_patient = np.stack(aggregated["summed_raw_count"])
    rank_patient = np.stack(aggregated["mean_core_rank"])
    score_patient = np.stack(aggregated["rank_score"])
    time, event = patients["time_months"].to_numpy(float), patients["event"].to_numpy(int)
    results, patient_parts, motif_frames = [], [], {}
    for j, pool_row in enumerate(selected.to_dict("records")):
        survival, high, _ = analyze_rank_survival(score_patient[:, j], time, event)
        motif_hash = str(pool_row["wl_hash"])
        frame = pd.DataFrame({"triangle_level": level, "wl_hash": motif_hash, "PID": aggregated["PID"], "number_of_cores": aggregated["number_of_cores"], "summed_raw_count": raw_patient[:, j], "mean_core_rank": rank_patient[:, j], "rank_score": score_patient[:, j], "time_months": time, "event": event, "median_rank_score": survival["median_patient_rank_score"], "rank_group": np.where(high, "High rank score", "Low rank score")})
        patient_parts.append(frame)
        motif_frames[motif_hash] = frame
        raw_rho, rank_rho = safe_spearman(raw[:, j], total_cells), safe_spearman(scores[:, j], total_cells)
        results.append({**pool_row, "vocabulary_size": vocabulary_size, "core_rank_tie_method": "average_for_equal_positive_counts", "absent_rank": vocabulary_size, "patient_core_aggregation": "equal_weight_mean", "spearman_total_cells_raw": raw_rho, "spearman_total_cells_rank_score": rank_rho, "abs_spearman_total_cells_raw": abs(raw_rho) if np.isfinite(raw_rho) else np.nan, "abs_spearman_total_cells_rank_score": abs(rank_rho) if np.isfinite(rank_rho) else np.nan, **survival})
    core_parts = []
    for j, motif_hash in enumerate(selected["wl_hash"].astype(str)):
        core_parts.append(pd.DataFrame({"triangle_level": level, "wl_hash": motif_hash, "PID": cohort["PID"], "source_sample": cohort["source_sample"], "raw_count": raw[:, j], "within_core_rank": ranks[:, j], "rank_score": scores[:, j], "vocabulary_size": vocabulary_size, "total_cells": total_cells}))
    return pd.DataFrame(results), pd.concat(core_parts, ignore_index=True), pd.concat(patient_parts, ignore_index=True), motif_frames


def fmt(value: object) -> str:
    return "NA" if pd.isna(value) else f"{float(value):.3g}"


def motif_directory_name(motif_hash: str, p_value: float) -> str:
    """Mark nominal rank-normalized binary Cox significance with a star."""
    significant = np.isfinite(p_value) and p_value < 0.05
    return f"{motif_hash}*" if significant else motif_hash


def resolve_motif_directory(
    outdir: Path, level: int, motif_hash: str, p_value: float
) -> Path:
    """Return the significance-aware directory, migrating its old counterpart."""
    level_dir = outdir / f"tri{level}"
    destination = level_dir / motif_directory_name(motif_hash, p_value)
    alternate_name = motif_hash if destination.name.endswith("*") else f"{motif_hash}*"
    alternate = level_dir / alternate_name
    if alternate.is_dir() and not destination.exists():
        alternate.rename(destination)
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def sync_significant_results(
    outdir: Path, results: pd.DataFrame, levels: Iterable[int]
) -> dict[str, object]:
    """Create a self-contained, exactly synchronized nominal-significance tree."""
    required = {"triangle_level", "wl_hash", "p_value_rank_binary"}
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError(f"Significant-result table is missing columns: {missing}")
    selected_levels = sorted(set(int(value) for value in levels))
    significant = results.loc[
        np.isfinite(results["p_value_rank_binary"])
        & results["p_value_rank_binary"].lt(0.05)
    ].copy()
    significant = significant.sort_values(
        ["p_value_rank_binary", "triangle_level", "wl_hash"], kind="mergesort"
    ).reset_index(drop=True)
    staging = Path(tempfile.mkdtemp(prefix=".significant-staging-", dir=outdir))
    destination = outdir / "significant"
    backup = outdir / ".significant-backup"
    try:
        for level in selected_levels:
            (staging / f"tri{level}").mkdir(parents=True)
        artifact_names = [
            "rank_normalized_survival_km.png",
            "patient_rank_distribution.png",
            "patient_rank_scores.csv",
            "statistics.json",
        ]
        for row in significant.itertuples(index=False):
            level = int(row.triangle_level)
            motif_hash = str(row.wl_hash)
            source = outdir / f"tri{level}" / motif_directory_name(
                motif_hash, float(row.p_value_rank_binary)
            )
            if not source.is_dir():
                raise FileNotFoundError(f"Missing significant motif directory: {source}")
            target = staging / f"tri{level}" / motif_hash
            target.mkdir(parents=True)
            for name in artifact_names:
                source_file = source / name
                if not source_file.is_file():
                    raise FileNotFoundError(f"Missing significant motif artifact: {source_file}")
                shutil.copy2(source_file, target / name)

        significant.to_csv(
            staging / "significant_rank_normalized_survival_results.csv", index=False
        )
        counts = {
            str(level): int(significant["triangle_level"].eq(level).sum())
            for level in selected_levels
        }
        manifest = {
            "selection": "finite p_value_rank_binary < 0.05",
            "total_significant_motifs": int(len(significant)),
            "significant_motifs_by_level": counts,
            "directory_layout": "triN/<bare_wl_hash>/",
            "storage": "full copies of the four per-motif artifacts",
        }
        with (staging / "significant_summary.json").open("w") as handle:
            json.dump(manifest, handle, indent=2)
        (staging / "README.md").write_text(
            "# Nominally significant rank-normalized motifs\n\n"
            "This directory contains full copies of motif packages whose finite "
            "rank-normalized binary Cox p-value is strictly below 0.05. This is "
            "the unadjusted median-split Cox Wald p-value, not the continuous Cox, "
            "log-rank, raw-count, or BH-adjusted p-value.\n\n"
            "Motifs are stored as `triN/<wl_hash>/` without a trailing star. Each "
            "motif directory contains the Kaplan–Meier plot, patient rank-distribution "
            "plot, patient score table, and statistics JSON. These nominal results "
            "are exploratory because the focal motifs were selected in the same cohort.\n"
        )

        if backup.exists():
            shutil.rmtree(backup)
        if destination.exists():
            destination.rename(backup)
        staging.rename(destination)
        if backup.exists():
            shutil.rmtree(backup)
        return manifest
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        if backup.exists() and not destination.exists():
            backup.rename(destination)
        raise


def plot_km(data: pd.DataFrame, result: pd.Series, output: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    colors = {"Low rank score": COLORS["Non-enriched"], "High rank score": COLORS["Enriched"]}
    for group in colors:
        subset = data.loc[data["rank_group"].eq(group)]
        if subset.empty:
            continue
        km = step_function(subset["time_months"].to_numpy(float), subset["event"].to_numpy(int))
        ax.step(km["time"], km["survival"], where="post", lw=2.2, color=colors[group], label=f"{group} (n={len(subset)}, events={int(subset.event.sum())})")
        ax.fill_between(km["time"], km["lower"], km["upper"], step="post", alpha=.15, color=colors[group])
        censored = subset.loc[subset["event"].eq(0), "time_months"].to_numpy(float)
        if len(censored):
            positions = np.searchsorted(km["time"].to_numpy(), censored, side="right") - 1
            ax.scatter(censored, km["survival"].to_numpy()[positions], marker="|", s=55, color=colors[group])
    text = f"Binary Cox HR = {fmt(result['hazard_ratio_rank_binary'])}\n95% CI {fmt(result['ci95_low_rank_binary'])}–{fmt(result['ci95_high_rank_binary'])}\nCox p = {fmt(result['p_value_rank_binary'])}; log-rank p = {fmt(result['logrank_p_value'])}\nContinuous HR per SD = {fmt(result['hazard_ratio_rank_continuous_per_sd'])}"
    ax.text(.98, .98, text, transform=ax.transAxes, ha="right", va="top", fontsize=9.5, bbox={"boxstyle": "round", "facecolor": "white", "alpha": .9, "edgecolor": ".75"})
    ax.set(title=f"TRI-{int(result['triangle_level'])} {str(result['wl_hash'])[:8]}: survival by rank score\n{result['cell_type_composition']}", xlabel="Overall survival (months)", ylabel="Survival probability", ylim=(0, 1.03))
    ax.grid(axis="y", alpha=.2)
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout(); fig.savefig(output, dpi=dpi); plt.close(fig)


def plot_distribution(data: pd.DataFrame, result: pd.Series, output: Path, dpi: int) -> None:
    ordered = data.sort_values(["rank_score", "PID"], kind="mergesort").reset_index(drop=True)
    colors = np.where(ordered["rank_group"].eq("High rank score"), COLORS["Enriched"], COLORS["Non-enriched"])
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.bar(np.arange(len(ordered)), ordered["rank_score"], color=colors, width=1, linewidth=0)
    ax.axhline(float(result["median_patient_rank_score"]), color="black", ls="--", lw=1.2, label=f"Median = {float(result['median_patient_rank_score']):.4g}")
    ax.set(title=f"TRI-{int(result['triangle_level'])} {str(result['wl_hash'])[:8]}: patient rank-normalized abundance", xlabel="Patients ordered by rank score", ylabel="Mean core rank score", xlim=(-.5, len(ordered)-.5), ylim=(0, 1.01))
    ax.grid(axis="y", alpha=.2); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(output, dpi=dpi); plt.close(fig)


def plot_comparison(results: pd.DataFrame, xcol: str, ycol: str, xlabel: str, ylabel: str, title: str, output: Path, dpi: int, p_values: bool = False) -> None:
    x, y = results[xcol].to_numpy(float), results[ycol].to_numpy(float)
    if p_values:
        x, y = -np.log10(np.clip(x, 1e-300, 1)), -np.log10(np.clip(y, 1e-300, 1))
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    points = ax.scatter(x, y, c=results["triangle_level"], cmap="viridis", vmin=1, vmax=7, s=28, alpha=.72)
    if p_values:
        cutoff = -math.log10(.05); ax.axhline(cutoff, color="black", ls="--", lw=1); ax.axvline(cutoff, color="black", ls="--", lw=1)
    else:
        finite = np.concatenate([x[np.isfinite(x)], y[np.isfinite(y)]])
        limit = max(float(finite.max()) if len(finite) else 0, .05); ax.plot([0, limit], [0, limit], color="black", ls="--", lw=1); ax.set(xlim=(0, limit), ylim=(0, limit))
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title); ax.grid(alpha=.2); fig.colorbar(points, ax=ax, label="TRI level")
    fig.tight_layout(); fig.savefig(output, dpi=dpi); plt.close(fig)


def json_safe(value: object) -> object:
    if isinstance(value, dict): return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [json_safe(v) for v in value]
    if isinstance(value, np.generic): value = value.item()
    if isinstance(value, float) and not np.isfinite(value): return None
    if pd.isna(value): return None
    return value


def write_readme(path: Path, summary: dict[str, object]) -> None:
    levels = summary["triangle_levels"]
    vocabulary = summary["vocabulary_size_by_level"]
    lines = ["# Rank-normalized focal motif survival analysis", "", "## Method", "", f"This analysis reruns **{summary['focal_motifs']} focal motifs** selected by raw-count Cox `p < 0.05` and a positive patient raw-count median.", "", "Within each core and TRI level, positive counts are ranked from largest to smallest against the complete observed WL-hash vocabulary. Equal positive counts receive their average rank. Zero or absent motifs receive the fixed last rank. Vocabulary sizes are " + ", ".join(f"TRI-{x}: {vocabulary[str(x)]:,}" for x in levels) + ". These are motifs observed across the 547-sample artifacts, not all theoretically possible graphs.", "", "The score is `(vocabulary_size - rank) / (vocabulary_size - 1)`. Patient scores equally average multiple cores. Patients strictly above the motif-specific median form the high-rank group; ties are low-rank.", "", "The primary test is an unadjusted binary Cox model with Efron ties and a two-sided Wald p-value. Kaplan–Meier/log-rank comparisons use the same groups; a continuous per-SD Cox model is secondary.", "", "## Results", "", f"- Cohort: **{summary['survival_patients']} patients**, **{summary['deaths']} deaths**.", f"- Rank-normalized binary Cox `p < 0.05`: **{summary['rank_binary_p_below_0_05']} / {summary['focal_motifs']}**.", f"- Global BH `q < 0.05`: **{summary['rank_binary_global_bh_q_below_0_05']}**.", f"- Median absolute cell-count correlation, raw → rank: **{summary['median_abs_cell_correlation_raw']:.3f} → {summary['median_abs_cell_correlation_rank_score']:.3f}**.", "- Aggregate results are in `rank_normalized_survival_results.csv`.", "- Auditable scores are in the compressed core and patient tables.", "- Each `triN/<wl_hash>/` contains plots, patient scores, and statistics; a trailing `*` marks rank-normalized binary Cox `p < 0.05`.", "", "## Interpretation", "", "Ranking reduces sensitivity to absolute counts but does not guarantee removal of all cell-burden association. Because the motifs were selected using survival in this cohort, normalized p-values are exploratory rather than independent confirmation.", "", "## Reproduction", "", "```bash", "python breast_cancer_survival/analyze_rank_normalized_motif_survival.py", "```", ""]
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    if args.dpi <= 0: raise ValueError("--dpi must be positive")
    levels = sorted(set(args.levels))
    pool = load_focal_pool(args.pool_csv, levels)
    metadata = pd.read_csv(args.metadata)
    cohort, patients, audit = clinical_core_cohort(args.artifacts_root, metadata)
    cell_counts = load_core_cell_counts(args.h5ad, cohort["source_sample"].astype(str).tolist())
    total_cells = cell_counts.sum(axis=1).reindex(cohort["source_sample"]).to_numpy(float)
    if not np.isfinite(total_cells).all() or (total_cells <= 0).any(): raise ValueError("Invalid cell totals")
    args.outdir.mkdir(parents=True, exist_ok=True)
    result_parts, core_parts, patient_parts, motif_frames = [], [], [], {}
    vocabularies = {}
    for level in levels:
        selected = pool.loc[pool["triangle_level"].eq(level)].reset_index(drop=True)
        print(f"TRI-{level}: ranking {len(selected)} focal motifs", flush=True)
        if level == 1: raw, ranks, scores, size = tri1_core_values(selected, cohort, args.motif_root)
        else: raw, ranks, scores, size = high_level_core_values(level, selected, cohort, args.artifacts_root)
        result, core, patient, frames = level_outputs(level, selected, cohort, patients, total_cells, raw, ranks, scores, size)
        result_parts.append(result); core_parts.append(core); patient_parts.append(patient)
        motif_frames.update({(level, key): value for key, value in frames.items()}); vocabularies[str(level)] = size
    results = pd.concat(result_parts, ignore_index=True)
    results["q_value_bh_global_rank_binary"] = benjamini_hochberg(results["p_value_rank_binary"])
    results["q_value_bh_within_level_rank_binary"] = results.groupby("triangle_level")["p_value_rank_binary"].transform(lambda x: benjamini_hochberg(x))
    results["rank_global_rank_binary"] = results["p_value_rank_binary"].rank(method="min", na_option="keep").astype("Int64")
    results["rank_within_level_rank_binary"] = results.groupby("triangle_level")["p_value_rank_binary"].rank(method="min", na_option="keep").astype("Int64")
    results = results.sort_values(["p_value_rank_binary", "triangle_level", "wl_hash"], na_position="last", kind="mergesort").reset_index(drop=True)
    core_scores, patient_scores = pd.concat(core_parts, ignore_index=True), pd.concat(patient_parts, ignore_index=True)
    results.to_csv(args.outdir / "rank_normalized_survival_results.csv", index=False)
    core_scores.to_csv(args.outdir / "core_rank_scores.csv.gz", index=False, compression="gzip")
    patient_scores.to_csv(args.outdir / "patient_rank_scores.csv.gz", index=False, compression="gzip")
    diagnostics = results[["triangle_level", "wl_hash", "spearman_total_cells_raw", "spearman_total_cells_rank_score", "abs_spearman_total_cells_raw", "abs_spearman_total_cells_rank_score"]]
    diagnostics.to_csv(args.outdir / "rank_normalization_diagnostics.csv", index=False)
    for result in results.itertuples(index=False):
        level, motif_hash = int(result.triangle_level), str(result.wl_hash)
        directory = resolve_motif_directory(
            args.outdir, level, motif_hash, float(result.p_value_rank_binary)
        )
        data, row = motif_frames[(level, motif_hash)], pd.Series(result._asdict())
        data.to_csv(directory / "patient_rank_scores.csv", index=False)
        with (directory / "statistics.json").open("w") as handle: json.dump(json_safe(row.to_dict()), handle, indent=2)
        if not args.skip_plots:
            plot_km(data, row, directory / "rank_normalized_survival_km.png", args.dpi)
            plot_distribution(data, row, directory / "patient_rank_distribution.png", args.dpi)
    if not args.skip_plots:
        plot_comparison(results, "abs_spearman_total_cells_raw", "abs_spearman_total_cells_rank_score", "Absolute Spearman correlation: raw count vs cells", "Absolute Spearman correlation: rank score vs cells", "Cell-burden association before and after ranking", args.outdir / "cell_burden_correlation_raw_vs_rank.png", args.dpi)
        plot_comparison(results, "p_value_unadjusted", "p_value_rank_binary", "Raw binary Cox −log10(p)", "Rank-normalized binary Cox −log10(p)", "Raw versus rank-normalized survival association", args.outdir / "raw_vs_rank_normalized_survival.png", args.dpi, p_values=True)
    summary = {**audit, "pool_csv": str(args.pool_csv.resolve()), "artifacts_root": str(args.artifacts_root.resolve()), "motif_root": str(args.motif_root.resolve()), "metadata_path": str(args.metadata.resolve()), "h5ad_path": str(args.h5ad.resolve()), "triangle_levels": levels, "focal_motifs": len(results), "focal_motifs_by_level": {str(k): int(v) for k, v in results.groupby("triangle_level").size().items()}, "vocabulary_size_by_level": vocabularies, "clinical_tumor_cores": len(cohort), "clinical_patients": cohort["PID"].nunique(), "survival_patients": len(patients), "deaths": int(patients["event"].sum()), "core_rank_definition": "descending average rank for positive ties; zero/absence gets vocabulary_size", "rank_score_definition": "(vocabulary_size - rank) / (vocabulary_size - 1)", "patient_core_aggregation": "equal-weight arithmetic mean", "survival_group_definition": "patient rank score > motif-specific cohort median", "median_ties": "low rank score group", "rank_binary_estimable": int(results["p_value_rank_binary"].notna().sum()), "rank_binary_p_below_0_05": int(results["p_value_rank_binary"].lt(.05).sum()), "rank_binary_global_bh_q_below_0_05": int(results["q_value_bh_global_rank_binary"].lt(.05).sum()), "median_abs_cell_correlation_raw": float(results["abs_spearman_total_cells_raw"].median()), "median_abs_cell_correlation_rank_score": float(results["abs_spearman_total_cells_rank_score"].median()), "plot_files_written": not args.skip_plots, "selection_warning": "Focal motifs were selected using raw survival associations in the same cohort"}
    with (args.outdir / "analysis_summary.json").open("w") as handle: json.dump(json_safe(summary), handle, indent=2)
    write_readme(args.outdir / "README.md", summary)
    print(json.dumps(json_safe(summary), indent=2), flush=True)


if __name__ == "__main__":
    main()
