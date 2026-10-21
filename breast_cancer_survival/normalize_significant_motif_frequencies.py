#!/usr/bin/env python3
"""Normalize the 644 focal motif counts for cell burden and composition.

The analysis is deliberately restricted to the clinically matched BZ tumor
cores.  For each motif, a negative-binomial expected-count model uses the total
same-TRI-level motif count as an offset and adjusts for log total cells plus an
ILR representation of the eight-part cell-type composition.  Observed-versus-
expected residuals are aggregated correctly across cores before exploratory
continuous Cox and zero-split Kaplan-Meier analyses.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", "/tmp/breast-cancer-normalized-motif-matplotlib"
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import sparse
from scipy.linalg import helmert
from scipy.stats import spearmanr
from statsmodels.discrete.discrete_model import NegativeBinomial

from analyze_ki67_survival import COLORS, logrank_test, step_function
from analyze_motif_survival import benjamini_hochberg, fit_cox
from analyze_motif_survival_unadjusted import (
    load_inventory,
    prepare_patient_aggregation,
)


MIN_LEVEL = 1
MAX_LEVEL = 7
PSEUDOCOUNT = 0.5
CELL_TYPES = [
    "B cells",
    "T and B cells",
    "T cells",
    "endothelial",
    "macrophages",
    "stromal cells",
    "tumor cells",
    "unknown",
]
CORE_SCORE_COLUMNS = [
    "raw_count",
    "count_per_1000_cells",
    "within_level_fraction",
    "expected_count",
    "model_variance",
    "pearson_residual",
    "log2_observed_expected",
]


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    survival = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pool-csv",
        type=Path,
        default=survival
        / "motif_results"
        / "significant_mtf_trace"
        / "potential_motif_pool.csv",
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
        "--motif-root",
        type=Path,
        default=repo / "breast_cancer_meta" / "BZ_res_threshold_10_cpp_0811",
    )
    parser.add_argument(
        "--h5ad", type=Path, default=repo / "breast_cancer" / "BZ.h5ad"
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=repo / "breast_cancer" / "Basel_PatientMetadata.csv",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=survival / "motif_results" / "normalized_motif_analysis",
    )
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Write all tables and summaries without per-motif or summary PNGs",
    )
    return parser.parse_args()


def load_pool(path: Path) -> pd.DataFrame:
    pool = pd.read_csv(path, dtype={"wl_hash": str})
    required = {
        "triangle_level",
        "wl_hash",
        "cell_type_composition",
        "median_patient_raw_count",
        "p_value_unadjusted",
        "hazard_ratio_unadjusted",
        "n_enriched_unadjusted",
        "events_enriched_unadjusted",
        "n_non_enriched_unadjusted",
        "events_non_enriched_unadjusted",
    }
    missing = sorted(required - set(pool.columns))
    if missing:
        raise ValueError(f"Focal motif pool is missing columns: {missing}")
    if len(pool) != 644:
        raise ValueError(f"Expected 644 focal motifs, found {len(pool)}")
    if pool.duplicated(["triangle_level", "wl_hash"]).any():
        raise ValueError("Focal motif pool contains duplicate level/hash pairs")
    expected = {1: 3, 2: 3, 3: 5, 4: 9, 5: 32, 6: 142, 7: 450}
    observed = pool.groupby("triangle_level").size().to_dict()
    if observed != expected:
        raise ValueError(f"Unexpected focal motif counts by level: {observed}")
    return pool.sort_values(
        ["triangle_level", "p_value_unadjusted", "wl_hash"], kind="mergesort"
    ).reset_index(drop=True)


def clinical_core_cohort(
    artifacts_root: Path, metadata: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    samples = load_inventory(artifacts_root, 2).reset_index(names="artifact_row")
    tumor = metadata.loc[metadata["diseasestatus"].eq("tumor"), ["PID", "core"]]
    tumor = tumor.drop_duplicates()
    if tumor["core"].duplicated().any():
        raise ValueError("A tumor core maps to more than one patient")
    cohort = samples.merge(
        tumor, left_on="source_sample", right_on="core", how="inner", validate="one_to_one"
    ).sort_values("artifact_row", kind="mergesort")
    cohort = cohort.reset_index(drop=True)
    if len(cohort) != 285 or cohort["PID"].nunique() != 279:
        raise ValueError(
            f"Expected 285 tumor cores from 279 patients, found "
            f"{len(cohort)} cores from {cohort['PID'].nunique()} patients"
        )
    patients, _, audit = prepare_patient_aggregation(samples, metadata)
    if len(patients) != 278:
        raise ValueError(f"Expected 278 survival patients, found {len(patients)}")
    return cohort, patients, audit


def load_core_cell_counts(h5ad_path: Path, cores: list[str]) -> pd.DataFrame:
    import anndata as ad

    adata = ad.read_h5ad(h5ad_path, backed="r")
    try:
        required = {"core", "cluster_preprocessed"}
        missing = sorted(required - set(adata.obs.columns))
        if missing:
            raise ValueError(f"H5AD observations are missing columns: {missing}")
        obs = adata.obs[["core", "cluster_preprocessed"]].copy()
        obs["core"] = obs["core"].astype(str)
        obs = obs.loc[obs["core"].isin(cores)]
        obs["cell_type"] = (
            obs["cluster_preprocessed"]
            .astype(object)
            .where(obs["cluster_preprocessed"].notna(), "unknown")
            .astype(str)
        )
    finally:
        adata.file.close()
    unexpected = sorted(set(obs["cell_type"]) - set(CELL_TYPES))
    if unexpected:
        raise ValueError(f"Unexpected cell types: {unexpected}")
    counts = (
        obs.groupby(["core", "cell_type"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(index=cores, columns=CELL_TYPES, fill_value=0)
        .astype(int)
    )
    if (counts.sum(axis=1) <= 0).any() or len(counts) != len(cores):
        raise ValueError("One or more clinical tumor cores have no cells")
    counts.index.name = "source_sample"
    return counts


def ilr_coordinates(counts: np.ndarray, pseudocount: float = PSEUDOCOUNT) -> np.ndarray:
    values = np.asarray(counts, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(CELL_TYPES):
        raise ValueError(f"Cell counts must have {len(CELL_TYPES)} columns")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Cell counts must be finite and nonnegative")
    if pseudocount <= 0:
        raise ValueError("ILR pseudocount must be positive")
    adjusted = values + pseudocount
    proportions = adjusted / adjusted.sum(axis=1, keepdims=True)
    basis = helmert(values.shape[1], full=False)
    coordinates = np.log(proportions) @ basis.T
    if coordinates.shape != (len(values), values.shape[1] - 1):
        raise ValueError("Unexpected ILR coordinate shape")
    return coordinates


def tri1_counts(
    selected: pd.DataFrame, cohort: pd.DataFrame, motif_root: Path
) -> tuple[np.ndarray, np.ndarray]:
    wanted = selected["wl_hash"].astype(str).tolist()
    output = np.zeros((len(cohort), len(wanted)), dtype=float)
    totals = np.zeros(len(cohort), dtype=float)
    for row_index, row in enumerate(cohort.itertuples(index=False)):
        path = motif_root / str(row.sample) / "ccindex_tri_1.csv"
        frame = pd.read_csv(path, usecols=["wl_hash", "matches"], dtype={"wl_hash": str})
        if frame["wl_hash"].duplicated().any():
            raise ValueError(f"Duplicate TRI-1 WL hashes in {path}")
        matches = pd.to_numeric(frame["matches"], errors="raise")
        if matches.lt(0).any():
            raise ValueError(f"Negative TRI-1 count in {path}")
        totals[row_index] = float(matches.sum())
        lookup = frame.assign(matches=matches).set_index("wl_hash")["matches"]
        output[row_index] = lookup.reindex(wanted, fill_value=0).to_numpy(float)
    return output, totals


def high_level_counts(
    level: int,
    selected: pd.DataFrame,
    cohort: pd.DataFrame,
    artifacts_root: Path,
) -> tuple[np.ndarray, np.ndarray]:
    level_dir = artifacts_root / f"tri_{level}"
    samples = load_inventory(artifacts_root, level)
    base_identity = cohort[["artifact_row", "sample", "source_sample"]]
    observed_identity = samples.iloc[cohort["artifact_row"].to_numpy(int)][
        ["sample", "source_sample"]
    ].reset_index(drop=True)
    if not observed_identity.equals(base_identity[["sample", "source_sample"]].reset_index(drop=True)):
        raise ValueError(f"TRI-{level} sample inventory differs from TRI-2")
    features = pd.read_csv(
        level_dir / "feature_index.csv.gz",
        usecols=["feature_index", "wl_hash"],
        dtype={"wl_hash": str},
    )
    if features["wl_hash"].duplicated().any():
        raise ValueError(f"TRI-{level} feature index contains duplicate hashes")
    lookup = features.set_index("wl_hash")["feature_index"]
    missing = selected.loc[~selected["wl_hash"].isin(lookup.index), "wl_hash"]
    if len(missing):
        raise ValueError(f"TRI-{level} focal hashes are missing: {missing.tolist()[:3]}")
    columns = lookup.loc[selected["wl_hash"]].to_numpy(dtype=int)
    matrix = sparse.load_npz(level_dir / "motif_counts.npz").tocsr()
    rows = cohort["artifact_row"].to_numpy(dtype=int)
    subset = matrix[rows]
    return subset[:, columns].toarray().astype(float), subset.sum(axis=1).A1.astype(float)


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3 or np.ptp(left[valid]) == 0 or np.ptp(right[valid]) == 0:
        return np.nan
    return float(spearmanr(left[valid], right[valid]).statistic)


def motif_directory_name(motif_hash: str, normalized_p_value: float) -> str:
    """Append a literal star for nominally significant normalized Cox results."""
    significant = np.isfinite(normalized_p_value) and normalized_p_value < 0.05
    return f"{motif_hash}*" if significant else motif_hash


def resolve_plot_directory(
    outdir: Path, level: int, motif_hash: str, normalized_p_value: float
) -> Path:
    """Use the significance-aware directory, migrating an earlier counterpart."""
    level_dir = outdir / f"tri{level}"
    destination = level_dir / motif_directory_name(motif_hash, normalized_p_value)
    alternate_name = motif_hash if destination.name.endswith("*") else f"{motif_hash}*"
    alternate = level_dir / alternate_name
    if alternate.is_dir() and not destination.exists():
        alternate.rename(destination)
    elif alternate.exists() and destination.exists():
        raise FileExistsError(
            f"Both significance directory variants exist: {destination} and {alternate}"
        )
    return destination


def fit_expected_count(
    counts: np.ndarray, design: np.ndarray, exposure: np.ndarray
) -> dict[str, object]:
    y = np.asarray(counts, dtype=float)
    X = np.asarray(design, dtype=float)
    offset = np.log(np.asarray(exposure, dtype=float))
    if (y < 0).any() or not np.isfinite(y).all():
        raise ValueError("Motif counts must be finite and nonnegative")
    if (exposure <= 0).any() or not np.isfinite(exposure).all():
        raise ValueError("Same-level motif exposure must be finite and positive")
    warning_text = ""
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = NegativeBinomial(y, X, offset=offset, loglike_method="nb2")
            result = model.fit(disp=0, maxiter=250)
        converged = bool(result.mle_retvals.get("converged", False))
        alpha = float(result.params[-1])
        expected = np.asarray(result.predict(), dtype=float)
        warning_text = " | ".join(dict.fromkeys(str(item.message) for item in caught))
        if (
            not converged
            or not np.isfinite(alpha)
            or alpha <= 0
            or not np.isfinite(expected).all()
            or (expected <= 0).any()
        ):
            raise RuntimeError(
                f"invalid NB2 fit: converged={converged}, alpha={alpha!r}"
            )
        variance = expected + alpha * expected**2
        coefficients = np.asarray(result.params[:-1], dtype=float)
        return {
            "fit_method": "negative_binomial_nb2",
            "fit_status": "estimated_with_warning" if warning_text else "estimated",
            "fit_warning": warning_text,
            "dispersion_alpha": alpha,
            "scale_phi": 1.0,
            "log_likelihood": float(result.llf),
            "coefficients": coefficients,
            "expected": expected,
            "variance": variance,
        }
    except Exception as nb_error:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = sm.GLM(
                    y, X, family=sm.families.Poisson(), offset=offset
                ).fit(maxiter=250)
            expected = np.asarray(result.predict(), dtype=float)
            if not result.converged or not np.isfinite(expected).all() or (expected <= 0).any():
                raise RuntimeError("invalid Poisson fallback fit")
            df_resid = max(int(result.df_resid), 1)
            phi = max(float(np.sum((y - expected) ** 2 / expected) / df_resid), 1.0)
            warning_items = [f"NB2 failed: {type(nb_error).__name__}: {nb_error}"]
            warning_items.extend(str(item.message) for item in caught)
            return {
                "fit_method": "quasi_poisson_fallback",
                "fit_status": "estimated_fallback",
                "fit_warning": " | ".join(dict.fromkeys(warning_items)),
                "dispersion_alpha": 0.0,
                "scale_phi": phi,
                "log_likelihood": float(result.llf),
                "coefficients": np.asarray(result.params, dtype=float),
                "expected": expected,
                "variance": phi * expected,
            }
        except Exception as fallback_error:
            return {
                "fit_method": "failed",
                "fit_status": "failed",
                "fit_warning": (
                    f"NB2: {type(nb_error).__name__}: {nb_error} | "
                    f"Poisson: {type(fallback_error).__name__}: {fallback_error}"
                ),
                "dispersion_alpha": np.nan,
                "scale_phi": np.nan,
                "log_likelihood": np.nan,
                "coefficients": np.full(X.shape[1], np.nan),
                "expected": np.full(len(y), np.nan),
                "variance": np.full(len(y), np.nan),
            }


def aggregate_patient_scores(
    core_scores: pd.DataFrame, patient_order: list[str]
) -> pd.DataFrame:
    grouped = core_scores.groupby("PID", sort=False, observed=True)
    patient = grouped.agg(
        number_of_cores=("source_sample", "nunique"),
        total_cells=("total_cells", "sum"),
        raw_count=("raw_count", "sum"),
        level_total_occurrences=("level_total_occurrences", "sum"),
        expected_count=("expected_count", "sum"),
        model_variance=("model_variance", "sum"),
    ).reindex(patient_order)
    patient["count_per_1000_cells"] = 1000.0 * patient["raw_count"] / patient["total_cells"]
    patient["within_level_fraction"] = (
        patient["raw_count"] / patient["level_total_occurrences"]
    )
    patient["pearson_residual"] = (
        patient["raw_count"] - patient["expected_count"]
    ) / np.sqrt(patient["model_variance"])
    patient["log2_observed_expected"] = np.log2(
        (patient["raw_count"] + PSEUDOCOUNT)
        / (patient["expected_count"] + PSEUDOCOUNT)
    )
    return patient.reset_index(names="PID")


def validate_raw_reconstruction(
    pool_row: pd.Series,
    patient_scores: pd.DataFrame,
    survival_patients: pd.DataFrame,
) -> None:
    analysis = survival_patients[["PID", "event"]].merge(
        patient_scores[["PID", "raw_count"]], on="PID", how="left", validate="one_to_one"
    )
    if analysis["raw_count"].isna().any():
        raise ValueError("A survival patient is missing a reconstructed raw count")
    values = analysis["raw_count"].to_numpy(float)
    event = analysis["event"].to_numpy(int)
    median = float(np.median(values))
    enriched = values > median
    checks = {
        "median_patient_raw_count": median,
        "n_enriched_unadjusted": int(enriched.sum()),
        "events_enriched_unadjusted": int(event[enriched].sum()),
        "n_non_enriched_unadjusted": int((~enriched).sum()),
        "events_non_enriched_unadjusted": int(event[~enriched].sum()),
    }
    failures = [
        f"{column}: reconstructed={value!r}, saved={pool_row[column]!r}"
        for column, value in checks.items()
        if not np.isclose(value, float(pool_row[column]), rtol=0, atol=1e-9)
    ]
    if failures:
        raise ValueError(
            f"TRI-{int(pool_row['triangle_level'])} {pool_row['wl_hash']} raw "
            f"reconstruction mismatch: {'; '.join(failures)}"
        )


def continuous_survival(
    patient_scores: pd.DataFrame,
    survival_patients: pd.DataFrame,
) -> tuple[dict[str, object], pd.DataFrame]:
    data = survival_patients[["PID", "time_months", "event"]].merge(
        patient_scores[["PID", "pearson_residual"]],
        on="PID",
        how="left",
        validate="one_to_one",
    )
    score = data["pearson_residual"].to_numpy(float)
    sd = float(np.std(score, ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        cox = {
            "coef": np.nan,
            "hazard_ratio": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "p_value": np.nan,
            "fit_status": "not_estimable_constant_score",
            "fit_warning": "Normalized patient score has zero variance",
        }
        data["pearson_residual_z"] = np.nan
    else:
        data["pearson_residual_z"] = (score - float(np.mean(score))) / sd
        cox = fit_cox(
            data["time_months"].to_numpy(float),
            data["event"].to_numpy(int),
            data["pearson_residual_z"].to_numpy(float),
        )
    group = score > 0
    n_enriched = int(group.sum())
    events_enriched = int(data.loc[group, "event"].sum())
    n_non = int((~group).sum())
    events_non = int(data.loc[~group, "event"].sum())
    eligible = min(n_enriched, n_non) >= 10 and min(events_enriched, events_non) >= 3
    if eligible:
        statistic, logrank_p = logrank_test(
            data["time_months"].to_numpy(float),
            data["event"].to_numpy(int),
            group,
        )
        km_status = "estimated"
    else:
        statistic = logrank_p = np.nan
        km_status = "not_estimable_minimum_group_or_event_count"
    data["normalized_group"] = np.where(group, "Enriched", "Non-enriched")
    result = {
        "score_mean_survival": float(np.mean(score)),
        "score_sd_survival": sd,
        "n_survival": int(len(data)),
        "events_survival": int(data["event"].sum()),
        "coef_normalized_continuous": cox["coef"],
        "hazard_ratio_per_sd": cox["hazard_ratio"],
        "ci95_low_per_sd": cox["ci95_low"],
        "ci95_high_per_sd": cox["ci95_high"],
        "p_value_normalized_continuous": cox["p_value"],
        "cox_fit_status": cox["fit_status"],
        "cox_fit_warning": cox["fit_warning"],
        "n_above_expected": n_enriched,
        "events_above_expected": events_enriched,
        "n_at_or_below_expected": n_non,
        "events_at_or_below_expected": events_non,
        "km_status": km_status,
        "logrank_statistic": statistic,
        "logrank_p_value": logrank_p,
    }
    return result, data


def plot_core_diagnostic(
    frame: pd.DataFrame, row: pd.Series, diagnostic: pd.Series, output: Path, dpi: int
) -> None:
    ordered = frame.sort_values(["pearson_residual", "source_sample"], kind="mergesort")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    maximum = max(float(frame["raw_count"].max()), float(frame["expected_count"].max()), 1.0)
    axes[0].scatter(
        frame["expected_count"], frame["raw_count"], s=17, alpha=0.65, color="#4E79A7"
    )
    axes[0].plot([0, maximum], [0, maximum], color="black", linestyle="--", linewidth=1)
    axes[0].set(
        xlabel="Expected count after normalization",
        ylabel="Observed raw count",
        title="Observed versus expected",
    )
    positions = np.arange(len(ordered))
    residuals = ordered["pearson_residual"].to_numpy(float)
    axes[1].bar(
        positions,
        residuals,
        width=1.0,
        linewidth=0,
        color=np.where(residuals > 0, COLORS["Enriched"], COLORS["Non-enriched"]),
        rasterized=True,
    )
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set(
        xlabel="Tumor cores ordered by normalized residual",
        ylabel="Observed-versus-expected Pearson residual",
        title="Composition-adjusted motif enrichment",
        xlim=(-0.5, len(ordered) - 0.5),
    )
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle(
        f"TRI-{int(row['triangle_level'])} {str(row['wl_hash'])[:8]} | "
        f"{row['cell_type_composition']}\n"
        f"raw ρ(cells)={diagnostic['spearman_total_cells_raw']:.3g}; "
        f"normalized ρ(cells)={diagnostic['spearman_total_cells_residual']:.3g}",
        fontsize=11,
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def plot_km(
    data: pd.DataFrame, result: pd.Series, row: pd.Series, output: Path, dpi: int
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 6.3))
    for group in ["Non-enriched", "Enriched"]:
        subset = data.loc[data["normalized_group"].eq(group)]
        km = step_function(
            subset["time_months"].to_numpy(float), subset["event"].to_numpy(int)
        )
        label = f"{group} (n={len(subset)}, events={int(subset.event.sum())})"
        ax.step(
            km["time"], km["survival"], where="post", color=COLORS[group], lw=2.2, label=label
        )
        ax.fill_between(
            km["time"], km["lower"], km["upper"], step="post", color=COLORS[group], alpha=0.15
        )
        censored = subset.loc[subset["event"].eq(0), "time_months"].to_numpy(float)
        if len(censored):
            indices = np.searchsorted(km["time"].to_numpy(), censored, side="right") - 1
            ax.scatter(
                censored,
                km["survival"].to_numpy()[indices],
                marker="|",
                s=55,
                color=COLORS[group],
            )
    text = (
        f"Continuous Cox HR per SD = {result['hazard_ratio_per_sd']:.3g}\n"
        f"95% CI {result['ci95_low_per_sd']:.3g}–{result['ci95_high_per_sd']:.3g}\n"
        f"Cox p = {result['p_value_normalized_continuous']:.3g}; "
        f"log-rank p = {result['logrank_p_value']:.3g}"
    )
    ax.text(
        0.98,
        0.98,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.5,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.75"},
    )
    ax.set(
        title=(
            f"TRI-{int(row['triangle_level'])} {str(row['wl_hash'])[:8]}: survival by "
            "composition-adjusted enrichment\n"
            f"{row['cell_type_composition']}"
        ),
        xlabel="Overall survival (months)",
        ylabel="Survival probability",
        ylim=(0, 1.03),
    )
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def plot_nuisance_reduction(diagnostics: pd.DataFrame, output: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    pairs = [
        (
            "abs_spearman_total_cells_raw",
            "abs_spearman_total_cells_residual",
            "Absolute correlation with total cells",
        ),
        (
            "max_abs_cell_type_correlation_raw",
            "max_abs_cell_type_correlation_residual",
            "Maximum absolute cell-type correlation",
        ),
    ]
    colors = plt.cm.viridis(
        (diagnostics["triangle_level"].to_numpy(float) - MIN_LEVEL) / (MAX_LEVEL - MIN_LEVEL)
    )
    for ax, (raw_column, normalized_column, title) in zip(axes, pairs):
        ax.scatter(
            diagnostics[raw_column], diagnostics[normalized_column], c=colors, s=25, alpha=0.7
        )
        limit = max(
            float(diagnostics[raw_column].max(skipna=True)),
            float(diagnostics[normalized_column].max(skipna=True)),
            0.05,
        )
        ax.plot([0, limit], [0, limit], color="black", linestyle="--", linewidth=1)
        ax.set(xlabel="Raw count", ylabel="Normalized residual", title=title, xlim=(0, limit), ylim=(0, limit))
        ax.grid(alpha=0.2)
    fig.suptitle("Nuisance association before and after motif normalization")
    fig.tight_layout()
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def plot_survival_comparison(results: pd.DataFrame, output: Path, dpi: int) -> None:
    raw = -np.log10(np.clip(results["p_value_unadjusted"].to_numpy(float), 1e-300, 1))
    normalized = -np.log10(
        np.clip(results["p_value_normalized_continuous"].to_numpy(float), 1e-300, 1)
    )
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    scatter = ax.scatter(
        raw,
        normalized,
        c=results["triangle_level"],
        cmap="viridis",
        vmin=MIN_LEVEL,
        vmax=MAX_LEVEL,
        s=27,
        alpha=0.72,
    )
    ax.axhline(-math.log10(0.05), color="black", linestyle="--", linewidth=1)
    ax.axvline(-math.log10(0.05), color="black", linestyle="--", linewidth=1)
    ax.set(
        xlabel="−log10(raw-count Cox p)",
        ylabel="−log10(normalized continuous Cox p)",
        title="Raw versus composition-adjusted survival association",
    )
    ax.grid(alpha=0.2)
    fig.colorbar(scatter, ax=ax, label="TRI level")
    fig.tight_layout()
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def write_report(
    path: Path, summary: dict[str, object], diagnostics: pd.DataFrame, results: pd.DataFrame
) -> None:
    successful = diagnostics["fit_status"].ne("failed")
    normalized_significant = results["p_value_normalized_continuous"].lt(0.05)
    normalized_q = results["q_value_bh_global_normalized"].lt(0.05)
    text = f"""# Composition-adjusted focal motif analysis

## Cohort construction

The input pool contains **{summary['focal_motifs']} focal motifs** selected by the
upstream raw-count survival screen: 3 TRI-1, 3 TRI-2, 5 TRI-3, 9 TRI-4,
32 TRI-5, 142 TRI-6, and 450 TRI-7 motifs. A motif is identified by the
combination of its TRI level and WL hash.

The clinical metadata contains 289 unique tumor cores. Four have no matching
motif-count artifact, so the normalization cohort is their intersection with
the motif inventory: **{summary['tumor_cores']} tumor cores from
{summary['patients_before_survival_filter']} patients**. Of these patients, 273
contribute one core and 6 contribute two cores. The expected-count models are
fit at the core level so that each core retains its own cell number and cell
composition.

One of the 279 patients has missing or zero usable survival follow-up and is
excluded only from the survival models. This leaves
**{summary['survival_patients']} survival-eligible patients**: 79 observed
deaths and 199 censored observations. The motif normalization itself still
uses all 285 eligible tumor cores.

## Expected-count normalization

For focal motif `m` at TRI level `L` in core `i`, let `y(i,m)` be its observed
count. A separate expected-count model is fitted for every focal motif across
the 285 cores:

```text
log E[y(i,m)] = log T(i,L) + intercept
                + beta_cells * centered_log(total_cells(i))
                + gamma_1 * ILR_1(i) + ... + gamma_7 * ILR_7(i)
```

The terms have the following meanings:

- **Complete same-TRI-level motif opportunity, `T(i,L)`:** the sum of all motif
  occurrences in core `i` at level `L`, using the complete level-specific
  feature inventory rather than only the 644 focal motifs. `log T(i,L)` is a
  model offset with its coefficient fixed at one. It makes the focal count a
  rate relative to the amount of same-level motif structure available in that
  core.
- **Total cell count:** the number of segmented cells in the core. Its natural
  logarithm is mean-centered and fitted as a covariate with an estimated
  coefficient. This adjusts for remaining cell-burden effects that are not
  already explained by the same-level motif opportunity; centering changes
  the intercept but not fitted expectations.
- **Cell-type composition:** counts are formed for B cells, T and B cells,
  T cells, endothelial cells, macrophages, stromal cells, tumor cells, and
  unknown cells. A pseudocount of 0.5 is added before converting them to
  proportions. Because eight proportions sum to one and cannot be entered as
  eight independent predictors, an orthonormal Helmert isometric log-ratio
  (ILR) transform represents their relative composition with exactly seven
  coordinates. All seven enter the model together; no single cell type is
  treated as the reference category.

The primary model is negative binomial (NB2), whose variance is
`mu + alpha * mu^2`; a dispersion-scaled Poisson model is used if a valid NB2
fit is unavailable. The core-level normalized score is the Pearson residual
`(observed - expected) / sqrt(model variance)`. A positive residual therefore
means that the motif is more frequent than expected for a core with that
same-level motif opportunity, total cell count, and cell-type mixture; a
negative residual means that it is less frequent than expected.

## Patient aggregation and survival analysis

For a patient with multiple cores, observed counts, expected counts, and model
variances are first summed across cores, and the Pearson residual is then
calculated from those sums. This produces one normalized value per patient and
avoids treating multiple cores from one patient as independent survival
observations.

Continuous unadjusted Cox models use the normalized patient residual after
standardization to one standard deviation, so the reported hazard ratio is per
one-SD increase. Kaplan–Meier plots split patients at residual > 0 versus
residual <= 0 only as a descriptive observed-versus-expected comparison; that
split is not the continuous Cox test.

## Methodological rationale and references

No cited paper uses this exact breast-cancer motif pipeline. The model is an
application-specific combination of established statistical components:

- **NB2 regression for the motif count.** Motif occurrences are nonnegative
  integer counts, and their between-core variance can exceed their mean.
  Negative-binomial regression permits this overdispersion; NB2 specifically
  uses `Var(Y) = mu + alpha * mu^2`. See
  [Cameron and Trivedi (1986)](https://doi.org/10.1002/jae.3950010104) for
  count-regression models and
  [Ver Hoef and Boveng (2007)](https://doi.org/10.1890/07-0043.1) for the
  distinction between negative-binomial and quasi-Poisson mean–variance
  assumptions.
- **A log exposure offset for same-level opportunity.** In a log-link count
  model, adding `log(exposure)` with a coefficient fixed at one models a rate
  per unit of exposure; this is the classical construction described by
  [Frome (1983)](https://doi.org/10.2307/2531094). Here, exposure is the total
  number of motifs available at the same TRI level, so the fitted mean is for
  the focal motif relative to that core's overall same-level motif burden.
  This is analogous to the sample-size or library-size factors used with
  negative-binomial feature counts by
  [Anders and Huber (2010)](https://doi.org/10.1186/gb-2010-11-10-r106).
- **ILR coordinates for cell-type composition.** Eight raw fractions sum to
  one, so treating all eight as ordinary independent covariates produces a
  linear dependency and ignores that their information is relative.
  [Aitchison (1982)](https://doi.org/10.1111/j.2517-6161.1982.tb01195.x)
  established the log-ratio framework for compositions, and
  [Aitchison and Bacon-Shone (1984)](https://doi.org/10.1093/biomet/71.2.323)
  developed regression with compositional predictors.
  [Egozcue et al. (2003)](https://doi.org/10.1023/A:1023818214614) introduced
  the isometric log-ratio representation, which maps a `D`-part composition to
  `D - 1` orthonormal Euclidean coordinates while preserving its relative
  information. Consequently, the eight cell-type fractions become seven
  nonredundant ILR covariates that can enter the NB2 log-mean predictor.
- **Pearson residual as the observed-versus-expected score.** Dividing
  `observed - expected` by the fitted standard deviation puts discrepancies
  from differently sized cores onto a more comparable scale. Pearson and
  related generalized-linear-model residuals are discussed by
  [Pierce and Schafer (1986)](https://doi.org/10.1080/01621459.1986.10478361).
  In this analysis the residual is an enrichment score and diagnostic, not an
  individual-core p-value or an assertion that every residual is normally
  distributed.
- **Zeros in the composition.** Log-ratios require positive components, so a
  zero-handling rule is necessary; see
  [Martín-Fernández et al. (2003)](https://doi.org/10.1023/A:1023866030544).
  The added 0.5 cell is a transparent continuity correction for these integer
  counts. The literature supports treating zeros before a log-ratio transform,
  but it does not make 0.5 uniquely correct, so alternative zero-replacement
  values are a reasonable sensitivity analysis.

Together, these components define a coherent conditional mean model: NB2
handles the outcome's count scale and overdispersion, the offset adjusts for
the amount of motif opportunity, and the ILR terms adjust for relative
cell-type mixture without redundant proportions. The citations justify this
model structure; they do not by themselves prove that it is correctly
specified for these data. In particular, the offset assumes expected focal
counts scale proportionally with total same-level opportunity, and the current
model does not explicitly represent zero inflation, nonlinear covariate
effects, batch effects, or residual spatial dependence. The reported nuisance-
correlation reduction and per-motif diagnostics should therefore be considered
alongside sensitivity analyses rather than replaced by the citations.

## Results inventory

- Successful expected-count models: **{int(successful.sum())}/{len(diagnostics)}**
- Estimable continuous Cox models: **{summary['estimated_continuous_cox_models']}/{summary['focal_motifs']}**
- Raw-selected motifs retaining nominal normalized Cox p < 0.05: **{int(normalized_significant.sum())}**
- Motifs with normalized global BH q < 0.05: **{int(normalized_q.sum())}**
- Median absolute total-cell correlation, raw → residual: **{diagnostics['abs_spearman_total_cells_raw'].median():.3f} → {diagnostics['abs_spearman_total_cells_residual'].median():.3f}**
- Median maximum absolute cell-type correlation, raw → residual: **{diagnostics['max_abs_cell_type_correlation_raw'].median():.3f} → {diagnostics['max_abs_cell_type_correlation_residual'].median():.3f}**

A literal `*` suffix on a motif directory denotes a nominally significant
normalized continuous Cox result (`p < 0.05`), not BH-adjusted significance.

## Interpretation constraint

These normalized survival results are exploratory robustness analyses. The 644 motifs were selected using their original raw-count survival associations, so normalized p-values are not an independent confirmatory screen.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")
    args.outdir.mkdir(parents=True, exist_ok=True)
    pool = load_pool(args.pool_csv)
    metadata = pd.read_csv(args.metadata, dtype={"PID": str, "core": str})
    cohort, survival_patients, cohort_audit = clinical_core_cohort(
        args.artifacts_root, metadata
    )
    cell_counts = load_core_cell_counts(args.h5ad, cohort["source_sample"].tolist())
    total_cells = cell_counts.sum(axis=1).to_numpy(float)
    cell_proportions = cell_counts.to_numpy(float) / total_cells[:, None]
    ilr = ilr_coordinates(cell_counts.to_numpy(float))
    centered_log_cells = np.log(total_cells) - float(np.mean(np.log(total_cells)))
    design = np.column_stack([np.ones(len(cohort)), centered_log_cells, ilr])
    design_names = ["intercept", "centered_log_total_cells", *[f"ilr_{i}" for i in range(1, 8)]]

    core_metadata = cohort[["sample", "source_sample", "PID"]].copy()
    core_metadata["total_cells"] = total_cells.astype(int)
    for index, cell_type in enumerate(CELL_TYPES):
        core_metadata[f"cells_{cell_type}"] = cell_counts.iloc[:, index].to_numpy(int)
        core_metadata[f"fraction_{cell_type}"] = cell_proportions[:, index]
    core_metadata.to_csv(args.outdir / "core_cell_composition.csv", index=False)

    patient_order = sorted(cohort["PID"].unique())
    core_parts: list[pd.DataFrame] = []
    patient_parts: list[pd.DataFrame] = []
    diagnostics_rows: list[dict[str, object]] = []
    survival_rows: list[dict[str, object]] = []
    plot_payloads: list[tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]] = []

    for level in range(MIN_LEVEL, MAX_LEVEL + 1):
        selected = pool.loc[pool["triangle_level"].eq(level)].reset_index(drop=True)
        if level == 1:
            raw, exposure = tri1_counts(selected, cohort, args.motif_root)
        else:
            raw, exposure = high_level_counts(
                level, selected, cohort, args.artifacts_root
            )
        if (exposure <= 0).any():
            raise ValueError(f"TRI-{level} has a nonpositive same-level exposure")

        for column, (_, motif) in enumerate(selected.iterrows()):
            values = raw[:, column]
            fit = fit_expected_count(values, design, exposure)
            expected = np.asarray(fit["expected"], dtype=float)
            variance = np.asarray(fit["variance"], dtype=float)
            residual = (values - expected) / np.sqrt(variance)
            log2_oe = np.log2((values + PSEUDOCOUNT) / (expected + PSEUDOCOUNT))
            core_frame = core_metadata[["sample", "source_sample", "PID", "total_cells"]].copy()
            core_frame.insert(0, "wl_hash", str(motif["wl_hash"]))
            core_frame.insert(0, "triangle_level", level)
            core_frame["level_total_occurrences"] = exposure
            core_frame["raw_count"] = values
            core_frame["count_per_1000_cells"] = 1000.0 * values / total_cells
            core_frame["within_level_fraction"] = values / exposure
            core_frame["expected_count"] = expected
            core_frame["model_variance"] = variance
            core_frame["pearson_residual"] = residual
            core_frame["log2_observed_expected"] = log2_oe
            core_parts.append(core_frame)

            patient_frame = aggregate_patient_scores(core_frame, patient_order)
            validate_raw_reconstruction(motif, patient_frame, survival_patients)
            survival, survival_data = continuous_survival(
                patient_frame, survival_patients
            )
            z_lookup = survival_data.set_index("PID")["pearson_residual_z"]
            group_lookup = survival_data.set_index("PID")["normalized_group"]
            patient_frame.insert(0, "wl_hash", str(motif["wl_hash"]))
            patient_frame.insert(0, "triangle_level", level)
            patient_frame["survival_eligible"] = patient_frame["PID"].isin(z_lookup.index)
            patient_frame["pearson_residual_z_survival"] = patient_frame["PID"].map(z_lookup)
            patient_frame["normalized_group_survival"] = patient_frame["PID"].map(group_lookup)
            patient_parts.append(patient_frame)

            raw_cell_corr = safe_spearman(values, total_cells)
            residual_cell_corr = safe_spearman(residual, total_cells)
            raw_type_corrs = [
                safe_spearman(values, cell_proportions[:, index])
                for index in range(len(CELL_TYPES))
            ]
            residual_type_corrs = [
                safe_spearman(residual, cell_proportions[:, index])
                for index in range(len(CELL_TYPES))
            ]
            coefficients = np.asarray(fit["coefficients"], dtype=float)
            diagnostic = {
                "triangle_level": level,
                "wl_hash": str(motif["wl_hash"]),
                "fit_method": fit["fit_method"],
                "fit_status": fit["fit_status"],
                "fit_warning": fit["fit_warning"],
                "dispersion_alpha": fit["dispersion_alpha"],
                "scale_phi": fit["scale_phi"],
                "log_likelihood": fit["log_likelihood"],
                "spearman_total_cells_raw": raw_cell_corr,
                "spearman_total_cells_residual": residual_cell_corr,
                "abs_spearman_total_cells_raw": abs(raw_cell_corr),
                "abs_spearman_total_cells_residual": abs(residual_cell_corr),
                "max_abs_cell_type_correlation_raw": float(np.nanmax(np.abs(raw_type_corrs))),
                "max_abs_cell_type_correlation_residual": float(np.nanmax(np.abs(residual_type_corrs))),
                **{
                    f"spearman_{cell_type}_raw": raw_type_corrs[index]
                    for index, cell_type in enumerate(CELL_TYPES)
                },
                **{
                    f"spearman_{cell_type}_residual": residual_type_corrs[index]
                    for index, cell_type in enumerate(CELL_TYPES)
                },
                **{
                    f"coef_{name}": coefficients[index] if index < len(coefficients) else np.nan
                    for index, name in enumerate(design_names)
                },
            }
            diagnostics_rows.append(diagnostic)
            survival_row = {
                "triangle_level": level,
                "wl_hash": str(motif["wl_hash"]),
                "cell_type_composition": motif["cell_type_composition"],
                "raw_hazard_ratio": motif["hazard_ratio_unadjusted"],
                "p_value_unadjusted": motif["p_value_unadjusted"],
                **survival,
            }
            survival_rows.append(survival_row)
            if not args.skip_plots:
                plot_payloads.append(
                    (
                        core_frame,
                        survival_data,
                        motif,
                        pd.Series(diagnostic),
                        pd.Series(survival_row),
                    )
                )
        print(f"TRI-{level}: normalized {len(selected)} focal motifs", flush=True)

    diagnostics = pd.DataFrame(diagnostics_rows)
    if diagnostics["fit_status"].eq("failed").any():
        failed = diagnostics.loc[diagnostics["fit_status"].eq("failed"), "wl_hash"].tolist()
        raise RuntimeError(f"Expected-count modeling failed for {len(failed)} motifs: {failed[:3]}")
    core_scores = pd.concat(core_parts, ignore_index=True)
    patient_scores = pd.concat(patient_parts, ignore_index=True)
    results = pd.DataFrame(survival_rows)
    results["q_value_bh_global_normalized"] = benjamini_hochberg(
        results["p_value_normalized_continuous"]
    )
    results["q_value_bh_within_level_normalized"] = np.nan
    for level, indices in results.groupby("triangle_level").groups.items():
        results.loc[indices, "q_value_bh_within_level_normalized"] = benjamini_hochberg(
            results.loc[indices, "p_value_normalized_continuous"]
        )
    results["rank_global_normalized"] = results["p_value_normalized_continuous"].rank(
        method="min", na_option="keep"
    )
    results["rank_within_level_normalized"] = results.groupby("triangle_level")[
        "p_value_normalized_continuous"
    ].rank(method="min", na_option="keep")
    results["motif_directory"] = results.apply(
        lambda row: str(
            Path(f"tri{int(row['triangle_level'])}")
            / motif_directory_name(
                str(row["wl_hash"]), float(row["p_value_normalized_continuous"])
            )
        ),
        axis=1,
    )
    results = results.sort_values(
        ["p_value_normalized_continuous", "triangle_level", "wl_hash"],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)

    # Keep existing plot directories synchronized even when plots are skipped.
    # On a fresh run these paths do not exist yet, so this is a no-op.
    for row in results.itertuples(index=False):
        resolve_plot_directory(
            args.outdir,
            int(row.triangle_level),
            str(row.wl_hash),
            float(row.p_value_normalized_continuous),
        )

    core_scores.to_csv(
        args.outdir / "core_normalized_motif_scores.csv.gz",
        index=False,
        compression="gzip",
    )
    patient_scores.to_csv(
        args.outdir / "patient_normalized_motif_scores.csv.gz",
        index=False,
        compression="gzip",
    )
    diagnostics.to_csv(args.outdir / "motif_normalization_diagnostics.csv", index=False)
    results.to_csv(args.outdir / "normalized_motif_survival_results.csv", index=False)

    if not args.skip_plots:
        for position, (core_frame, survival_data, motif, diagnostic, result) in enumerate(
            plot_payloads, start=1
        ):
            level = int(motif["triangle_level"])
            motif_hash = str(motif["wl_hash"])
            destination = resolve_plot_directory(
                args.outdir,
                level,
                motif_hash,
                float(result["p_value_normalized_continuous"]),
            )
            plot_core_diagnostic(
                core_frame,
                motif,
                diagnostic,
                destination / "normalized_core_residuals.png",
                args.dpi,
            )
            if result["km_status"] == "estimated":
                plot_km(
                    survival_data,
                    result,
                    motif,
                    destination / "normalized_survival_km.png",
                    args.dpi,
                )
            if position % 100 == 0 or position == len(plot_payloads):
                print(f"Rendered plots for {position}/{len(plot_payloads)} motifs", flush=True)
        plot_nuisance_reduction(
            diagnostics, args.outdir / "normalization_nuisance_reduction.png", args.dpi
        )
        plot_survival_comparison(
            results, args.outdir / "raw_vs_normalized_survival.png", args.dpi
        )

    summary = {
        "analysis_status": "complete",
        "focal_motifs": int(len(pool)),
        "motifs_by_level": {
            str(int(level)): int(count)
            for level, count in pool.groupby("triangle_level").size().items()
        },
        "tumor_cores": int(len(cohort)),
        "patients_before_survival_filter": int(cohort["PID"].nunique()),
        "survival_patients": int(len(survival_patients)),
        "survival_events": int(survival_patients["event"].sum()),
        "patients_with_two_cores": int((cohort.groupby("PID").size() == 2).sum()),
        "cell_types": CELL_TYPES,
        "normalization_model": (
            "motif-specific NB2 count model with log complete same-level motif total offset, "
            "centered log total cells, and seven ILR cell-composition coordinates"
        ),
        "ilr_zero_replacement": PSEUDOCOUNT,
        "primary_core_score": "observed-versus-expected Pearson residual",
        "patient_aggregation": "sum observed, expected, and variance across cores before residual",
        "survival_model": "unadjusted Efron Cox model using standardized continuous patient residual",
        "km_split": "patient residual > 0 versus <= 0; descriptive only",
        "successful_nb2_models": int(
            diagnostics["fit_method"].eq("negative_binomial_nb2").sum()
        ),
        "quasi_poisson_fallback_models": int(
            diagnostics["fit_method"].eq("quasi_poisson_fallback").sum()
        ),
        "nominal_normalized_cox_p_below_0_05": int(
            results["p_value_normalized_continuous"].lt(0.05).sum()
        ),
        "global_bh_q_below_0_05": int(
            results["q_value_bh_global_normalized"].lt(0.05).sum()
        ),
        "estimated_continuous_cox_models": int(
            results["p_value_normalized_continuous"].notna().sum()
        ),
        "nonestimable_continuous_cox_models": int(
            results["p_value_normalized_continuous"].isna().sum()
        ),
        "selection_caveat": (
            "exploratory robustness analysis because motifs were selected using raw-count survival p values"
        ),
        "cohort_audit": cohort_audit,
    }
    (args.outdir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    write_report(args.outdir / "README.md", summary, diagnostics, results)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
