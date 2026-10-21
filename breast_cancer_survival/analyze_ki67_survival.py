#!/usr/bin/env python3
"""Patient-level Ki-67 overall-survival analysis for the Basel IMC cohort.

Primary exposure:
  1. Restrict to cells classified as tumor cells in metadata-designated tumor cores.
  2. Define a Ki-67-high cell as above the pooled tumor-cell 75th percentile.
  3. Define each patient's score as their fraction of Ki-67-high tumor cells.
  4. Split patients at the patient-level median score (outcome-blind).

The script creates Kaplan-Meier curves, Cox models, diagnostics, and auditable
patient-level data. Cells are never treated as independent survival observations.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from pathlib import Path

import anndata as ad
os.environ.setdefault("MPLCONFIGDIR", "/tmp/breast-cancer-survival-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2, spearmanr
from statsmodels.duration.hazard_regression import PHReg


KI67_MARKER = "1441101Er168Di Ki67"
DEATH_STATUSES = {"death", "death by primary disease"}
LIVING_STATUSES = {"alive", "alive w metastases"}
COLORS = {"Non-enriched": "#3568A8", "Enriched": "#D1495B"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", type=Path, default=Path("breast_cancer/BZ.h5ad"))
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("breast_cancer/Basel_PatientMetadata.csv"),
    )
    parser.add_argument("--outdir", type=Path, default=Path("breast_cancer_survival/results"))
    parser.add_argument("--cell-quantile", type=float, default=0.75)
    return parser.parse_args()


def step_function(time: np.ndarray, event: np.ndarray) -> pd.DataFrame:
    """Kaplan-Meier estimate with log-log pointwise 95% confidence intervals."""
    event_times = np.sort(np.unique(time[event == 1]))
    survival = 1.0
    greenwood_sum = 0.0
    rows = [{"time": 0.0, "survival": 1.0, "lower": 1.0, "upper": 1.0}]
    for t in event_times:
        at_risk = int(np.sum(time >= t))
        deaths = int(np.sum((time == t) & (event == 1)))
        survival *= 1.0 - deaths / at_risk
        if at_risk > deaths:
            greenwood_sum += deaths / (at_risk * (at_risk - deaths))
        if 0.0 < survival < 1.0 and greenwood_sum > 0:
            se_loglog = math.sqrt(greenwood_sum) / abs(math.log(survival))
            loglog = math.log(-math.log(survival))
            lower = math.exp(-math.exp(loglog + 1.96 * se_loglog))
            upper = math.exp(-math.exp(loglog - 1.96 * se_loglog))
        else:
            lower = upper = survival
        rows.append({"time": float(t), "survival": survival, "lower": lower, "upper": upper})
    return pd.DataFrame(rows)


def logrank_test(time: np.ndarray, event: np.ndarray, group: np.ndarray) -> tuple[float, float]:
    """Two-sided, two-group Mantel-Cox log-rank test."""
    observed = expected = variance = 0.0
    for t in np.sort(np.unique(time[event == 1])):
        at_risk = time >= t
        n = int(at_risk.sum())
        n1 = int((at_risk & group).sum())
        d = int(((time == t) & (event == 1)).sum())
        d1 = int(((time == t) & (event == 1) & group).sum())
        observed += d1
        expected += d * n1 / n
        if n > 1:
            variance += n1 * (n - n1) * d * (n - d) / (n * n * (n - 1))
    statistic = (observed - expected) ** 2 / variance if variance > 0 else np.nan
    return float(statistic), float(chi2.sf(statistic, 1))


def collapse_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    """Validate invariant survival/clinical fields and keep one row per patient."""
    fields = [
        "Patientstatus", "OSmonth", "age", "grade", "tumor_size", "PTNM_M",
        "PTNM_N", "ERStatus", "HER2Status",
    ]
    inconsistent = {
        c: metadata.groupby("PID")[c].nunique(dropna=False).loc[lambda x: x > 1].index.tolist()
        for c in fields
    }
    inconsistent = {k: v for k, v in inconsistent.items() if v}
    if inconsistent:
        raise ValueError(f"Patient metadata are inconsistent across cores: {inconsistent}")
    return metadata.drop_duplicates("PID")[["PID", *fields]].copy()


def make_patient_scores(
    h5ad_path: Path, metadata: pd.DataFrame, cell_quantile: float
) -> tuple[pd.DataFrame, dict]:
    adata = ad.read_h5ad(h5ad_path, backed="r")
    if KI67_MARKER not in adata.var_names:
        raise KeyError(f"Marker not found: {KI67_MARKER}")

    tumor_cores = set(metadata.loc[metadata["diseasestatus"].eq("tumor"), "core"])
    core = adata.obs["core"].astype(str).to_numpy()
    cell_type = adata.obs["cluster_preprocessed"].astype(str).to_numpy()
    keep = np.isin(core, list(tumor_cores)) & (cell_type == "tumor cells")
    if not keep.any():
        raise ValueError("No tumor cells matched between H5AD and metadata")

    marker_idx = int(adata.var_names.get_loc(KI67_MARKER))
    # Reading one column is much smaller than loading the complete 1.1M x 34 matrix.
    ki67 = np.asarray(adata.X[:, marker_idx]).ravel()[keep]
    kept_obs = adata.obs.loc[keep, ["PID", "core"]].copy()
    kept_obs["PID"] = kept_obs["PID"].astype(str).str.removeprefix("b").astype(int)
    kept_obs["ki67"] = ki67

    cell_cutoff = float(np.quantile(ki67, cell_quantile))
    kept_obs["ki67_high"] = kept_obs["ki67"] > cell_cutoff
    patient = (
        kept_obs.groupby("PID", sort=True)
        .agg(
            ki67_score=("ki67_high", "mean"),
            ki67_median=("ki67", "median"),
            ki67_mean=("ki67", "mean"),
            tumor_cell_count=("ki67", "size"),
            tumor_core_count=("core", "nunique"),
        )
        .reset_index()
    )
    matched_tumor_cores = tumor_cores.intersection(set(adata.obs["core"].astype(str)))
    audit = {
        "h5ad_cells": int(adata.n_obs),
        "h5ad_markers": int(adata.n_vars),
        "metadata_rows": int(len(metadata)),
        "metadata_patients": int(metadata["PID"].nunique()),
        "metadata_tumor_cores": int(len(tumor_cores)),
        "matched_tumor_cores": int(len(matched_tumor_cores)),
        "analyzed_tumor_cells": int(len(kept_obs)),
        "patients_with_ki67_score": int(len(patient)),
        "ki67_high_cell_quantile": float(cell_quantile),
        "ki67_high_cell_cutoff": cell_cutoff,
    }
    adata.file.close()
    return patient, audit


def prepare_analysis(patient_scores: pd.DataFrame, metadata: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    patient = collapse_metadata(metadata).merge(patient_scores, on="PID", how="inner", validate="one_to_one")
    statuses = set(patient["Patientstatus"].dropna().unique())
    unexpected = statuses - DEATH_STATUSES - LIVING_STATUSES
    if unexpected:
        raise ValueError(f"Unexpected Patientstatus values: {sorted(unexpected)}")
    patient["event"] = patient["Patientstatus"].isin(DEATH_STATUSES).astype(int)
    patient["time_months"] = pd.to_numeric(patient["OSmonth"], errors="coerce")
    # Zero follow-up contributes no survival information and log(time) is used in
    # the PH diagnostic. Exclude rather than manufacture an arbitrary duration.
    patient = patient.loc[patient["time_months"].notna() & patient["time_months"].gt(0)].copy()
    if patient.empty:
        raise ValueError("No patients have positive, non-missing OSmonth")

    group_cutoff = float(patient["ki67_score"].median())
    patient["ki67_enriched"] = (patient["ki67_score"] > group_cutoff).astype(int)
    patient["ki67_group"] = np.where(patient["ki67_enriched"].eq(1), "Enriched", "Non-enriched")
    sd = float(patient["ki67_score"].std(ddof=1))
    patient["ki67_score_z"] = (patient["ki67_score"] - patient["ki67_score"].mean()) / sd

    patient["age_z"] = (patient["age"] - patient["age"].mean()) / patient["age"].std(ddof=1)
    patient["tumor_size_z"] = (
        (patient["tumor_size"] - patient["tumor_size"].mean()) / patient["tumor_size"].std(ddof=1)
    )
    patient["grade_z"] = (patient["grade"] - patient["grade"].mean()) / patient["grade"].std(ddof=1)
    node = patient["PTNM_N"].astype(str).str.strip().str.lower()
    patient["node_positive"] = np.where(node.str.startswith("0"), 0.0, np.where(node.eq("x"), np.nan, 1.0))
    patient["metastatic"] = pd.to_numeric(patient["PTNM_M"], errors="coerce")
    patient["er_positive"] = patient["ERStatus"].map({"positive": 1.0, "negative": 0.0})
    patient["her2_positive"] = patient["HER2Status"].map({"positive": 1.0, "negative": 0.0})
    return patient, group_cutoff


def fit_cox(data: pd.DataFrame, name: str, columns: list[str]) -> tuple[pd.DataFrame, dict, object, pd.DataFrame]:
    model_data = data[["time_months", "event", *columns]].dropna().copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = PHReg(
            model_data["time_months"].to_numpy(),
            model_data[columns].to_numpy(),
            status=model_data["event"].to_numpy(),
            ties="efron",
        )
        result = model.fit(disp=0)
    ci = np.asarray(result.conf_int())
    coefficients = pd.DataFrame(
        {
            "model": name,
            "term": columns,
            "coef": result.params,
            "hazard_ratio": np.exp(result.params),
            "ci95_low": np.exp(ci[:, 0]),
            "ci95_high": np.exp(ci[:, 1]),
            "p_value": result.pvalues,
            "n": len(model_data),
            "events": int(model_data["event"].sum()),
        }
    )

    # Approximate per-term PH check: correlate Schoenfeld residuals with log event time.
    schoenfeld = np.asarray(result.schoenfeld_residuals)
    event_mask = model_data["event"].to_numpy().astype(bool)
    ph_rows = []
    for j, term in enumerate(columns):
        resid = schoenfeld[:, j]
        valid = event_mask & np.isfinite(resid)
        rho, pvalue = spearmanr(np.log(model_data.loc[valid, "time_months"]), resid[valid])
        ph_rows.append({"model": name, "term": term, "spearman_rho": rho, "p_value": pvalue})
    diagnostics = pd.DataFrame(ph_rows)
    summary = {
        "model": name,
        "n": int(len(model_data)),
        "events": int(model_data["event"].sum()),
        "log_likelihood": float(result.llf),
    }
    return coefficients, summary, result, diagnostics


def plot_km(patient: pd.DataFrame, cox_row: pd.Series, logrank_p: float, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 6.3))
    for group in ["Non-enriched", "Enriched"]:
        d = patient.loc[patient["ki67_group"].eq(group)]
        km = step_function(d["time_months"].to_numpy(), d["event"].to_numpy())
        label = f"{group} (n={len(d)}, events={int(d.event.sum())})"
        ax.step(km.time, km.survival, where="post", color=COLORS[group], lw=2.2, label=label)
        ax.fill_between(km.time, km.lower, km.upper, step="post", color=COLORS[group], alpha=0.15)
        censored_times = d.loc[d.event.eq(0), "time_months"].to_numpy()
        if len(censored_times):
            idx = np.searchsorted(km.time.to_numpy(), censored_times, side="right") - 1
            ax.scatter(censored_times, km.survival.to_numpy()[idx], marker="|", s=55, color=COLORS[group])

    text = (
        f"Log-rank p = {logrank_p:.3g}\n"
        f"Unadjusted Cox HR = {cox_row.hazard_ratio:.2f} "
        f"(95% CI {cox_row.ci95_low:.2f}–{cox_row.ci95_high:.2f}), p = {cox_row.p_value:.3g}"
    )
    ax.text(0.98, 0.98, text, transform=ax.transAxes, ha="right", va="top", fontsize=10,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.75"})
    ax.set(title="Overall survival by tumor-cell Ki-67 enrichment", xlabel="Overall survival (months)", ylabel="Survival probability")
    ax.set_ylim(0, 1.03)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=220)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def plot_score(patient: pd.DataFrame, cutoff: float, output: Path) -> None:
    order = patient.sort_values("ki67_score").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    colors = order["ki67_group"].map(COLORS)
    ax.scatter(np.arange(len(order)), order["ki67_score"], c=colors, s=13, alpha=0.8, linewidth=0)
    ax.axhline(cutoff, color="black", ls="--", lw=1, label=f"Patient median = {cutoff:.3f}")
    ax.set(xlabel="Patients, ordered by Ki-67 score", ylabel="Fraction of Ki-67-high tumor cells",
           title="Patient-level Ki-67 enrichment score")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=220)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def write_report(
    output: Path,
    audit: dict,
    coefficients: pd.DataFrame,
    ph_diagnostics: pd.DataFrame,
) -> None:
    def row(model: str, term: str) -> pd.Series:
        return coefficients.loc[coefficients.model.eq(model) & coefficients.term.eq(term)].iloc[0]

    binary_u = row("binary_unadjusted", "ki67_enriched")
    binary_a = row("binary_adjusted", "ki67_enriched")
    continuous_u = row("continuous_unadjusted", "ki67_score_z")
    continuous_a = row("continuous_adjusted", "ki67_score_z")
    ph_u = ph_diagnostics.loc[
        ph_diagnostics.model.eq("binary_unadjusted") & ph_diagnostics.term.eq("ki67_enriched"), "p_value"
    ].iloc[0]

    significant = binary_u.p_value < 0.05
    conclusion = (
        "The prespecified binary Ki-67 comparison is statistically significant at α = 0.05."
        if significant
        else "The prespecified binary Ki-67 comparison is not statistically significant at α = 0.05."
    )
    report = f"""# Ki-67 overall-survival results

## Result

{conclusion} The point estimate is in the direction of higher mortality for the enriched group, but its confidence interval includes both no association and clinically meaningful effects.

| Analysis | N / events | Hazard ratio (95% CI) | P value |
|---|---:|---:|---:|
| Enriched vs non-enriched, unadjusted | {int(binary_u['n'])} / {int(binary_u.events)} | {binary_u.hazard_ratio:.2f} ({binary_u.ci95_low:.2f}–{binary_u.ci95_high:.2f}) | {binary_u.p_value:.3g} |
| Enriched vs non-enriched, adjusted | {int(binary_a['n'])} / {int(binary_a.events)} | {binary_a.hazard_ratio:.2f} ({binary_a.ci95_low:.2f}–{binary_a.ci95_high:.2f}) | {binary_a.p_value:.3g} |
| Ki-67 score per SD, unadjusted | {int(continuous_u['n'])} / {int(continuous_u.events)} | {continuous_u.hazard_ratio:.2f} ({continuous_u.ci95_low:.2f}–{continuous_u.ci95_high:.2f}) | {continuous_u.p_value:.3g} |
| Ki-67 score per SD, adjusted | {int(continuous_a['n'])} / {int(continuous_a.events)} | {continuous_a.hazard_ratio:.2f} ({continuous_a.ci95_low:.2f}–{continuous_a.ci95_high:.2f}) | {continuous_a.p_value:.3g} |

The two-sided log-rank test gives p = {audit['logrank_p_value']:.3g}. There are {audit['analyzed_patients']} analyzed patients ({audit['deaths']} deaths): {audit['enriched_patients']} enriched and {audit['non_enriched_patients']} non-enriched. The adjusted complete-case analysis has {int(binary_a['n'])} patients and {int(binary_a.events)} deaths.

## Thresholds and cohort flow

- Ki-67-high tumor cell cutoff (pooled tumor-cell 75th percentile): `{audit['ki67_high_cell_cutoff']:.6g}`.
- Enriched patient cutoff (median fraction of Ki-67-high tumor cells): `{audit['ki67_patient_median_cutoff']:.6g}`; enriched is strictly above this value.
- {audit['matched_tumor_cores']} of {audit['metadata_tumor_cores']} metadata-designated tumor cores matched the H5AD.
- {audit['analyzed_tumor_cells']:,} tumor cells produced scores for {audit['patients_with_ki67_score']} patients.
- {audit['patients_excluded_missing_or_zero_followup']} patient was excluded for missing or zero follow-up.

## Caution

The approximate Schoenfeld residual/time check for the unadjusted binary Ki-67 term gives p = {ph_u:.3g}, which flags possible non-proportional hazards. The Kaplan–Meier curves and continuous-score models should therefore be considered alongside the single Cox hazard ratio. This exploratory result is not evidence that Ki-67 has no effect; its confidence interval is too broad for that conclusion. An independent cohort with cutoffs frozen in advance is needed for confirmation.
"""
    output.write_text(report)


def main() -> None:
    args = parse_args()
    if not 0 < args.cell_quantile < 1:
        raise ValueError("--cell-quantile must be between 0 and 1")
    args.outdir.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(args.metadata)
    scores, audit = make_patient_scores(args.h5ad, metadata, args.cell_quantile)
    patient, group_cutoff = prepare_analysis(scores, metadata)

    stat, logrank_p = logrank_test(
        patient["time_months"].to_numpy(), patient["event"].to_numpy(),
        patient["ki67_enriched"].to_numpy().astype(bool),
    )
    adjusted = ["age_z", "grade_z", "tumor_size_z", "node_positive", "metastatic", "er_positive", "her2_positive"]
    specifications = {
        "binary_unadjusted": ["ki67_enriched"],
        "binary_adjusted": ["ki67_enriched", *adjusted],
        "continuous_unadjusted": ["ki67_score_z"],
        "continuous_adjusted": ["ki67_score_z", *adjusted],
    }
    coefs, summaries, diagnostics = [], [], []
    for name, columns in specifications.items():
        coefficient, summary, _, diagnostic = fit_cox(patient, name, columns)
        coefs.append(coefficient)
        summaries.append(summary)
        diagnostics.append(diagnostic)
    coefficients = pd.concat(coefs, ignore_index=True)
    ph_diagnostics = pd.concat(diagnostics, ignore_index=True)

    audit.update(
        {
            "analyzed_patients": int(len(patient)),
            "patients_excluded_missing_or_zero_followup": int(len(scores) - len(patient)),
            "deaths": int(patient["event"].sum()),
            "censored": int((1 - patient["event"]).sum()),
            "ki67_patient_median_cutoff": group_cutoff,
            "enriched_patients": int(patient["ki67_enriched"].sum()),
            "non_enriched_patients": int((1 - patient["ki67_enriched"]).sum()),
            "logrank_chi_square": stat,
            "logrank_p_value": logrank_p,
            "cox_models": summaries,
        }
    )
    patient.sort_values("PID").to_csv(args.outdir / "patient_level_analysis.csv", index=False)
    coefficients.to_csv(args.outdir / "cox_coefficients.csv", index=False)
    ph_diagnostics.to_csv(args.outdir / "proportional_hazards_diagnostics.csv", index=False)
    pd.DataFrame(
        [
            {
                "group": group,
                "n": len(d),
                "events": int(d.event.sum()),
                "median_ki67_score": d.ki67_score.median(),
                "median_os_months_observed": d.time_months.median(),
            }
            for group in ["Non-enriched", "Enriched"]
            for d in [patient.loc[patient.ki67_group.eq(group)]]
        ]
    ).to_csv(args.outdir / "group_summary.csv", index=False)
    with (args.outdir / "analysis_summary.json").open("w") as handle:
        json.dump(audit, handle, indent=2)

    unadjusted = coefficients.loc[
        coefficients.model.eq("binary_unadjusted") & coefficients.term.eq("ki67_enriched")
    ].iloc[0]
    plot_km(patient, unadjusted, logrank_p, args.outdir / "ki67_kaplan_meier")
    plot_score(patient, group_cutoff, args.outdir / "ki67_score_distribution")
    write_report(args.outdir / "REPORT.md", audit, coefficients, ph_diagnostics)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
