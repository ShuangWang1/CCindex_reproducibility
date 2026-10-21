#!/usr/bin/env python3
"""Patient-level, unadjusted overall-survival Cox screen for METABRIC motifs.

All motif-bearing images belonging to a patient are summed. For each motif,
patients strictly above its cohort median are enriched and patients at or below
the median are non-enriched. The model is Surv(T, Death) ~ motif_enriched,
using Efron ties and a two-sided Wald p-value.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import norm


STATUS_LABELS = {
    0: "not_estimable_no_group_variation",
    1: "not_estimable_event_separation",
    2: "estimated",
    3: "not_estimable_no_convergence",
}


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=repo / "breast_cancer_classification_workstation" / "artifacts" / "metabric",
    )
    parser.add_argument(
        "--clinical",
        type=Path,
        default=repo / "breast_cancer" / "41586_2019_1007_MOESM7_ESM.txt",
    )
    parser.add_argument(
        "--data-dictionary",
        type=Path,
        default=repo / "breast_cancer" / "41586_2019_1007_MOESM10_ESM.xls",
    )
    parser.add_argument(
        "--label-map",
        type=Path,
        default=repo / "breast_cancer" / "metabric_samples_intra" / "node_label_cell_type_mapping.csv",
    )
    parser.add_argument(
        "--outdir", type=Path, default=Path(__file__).resolve().parent / "results"
    )
    parser.add_argument("--levels", type=int, nargs="+", default=[2, 3, 4, 5, 6, 7])
    parser.add_argument("--chunk-size", type=int, default=20_000)
    parser.add_argument("--top-n", type=int, default=30)
    return parser.parse_args()


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    output = np.full(values.shape, np.nan, dtype=float)
    finite_positions = np.flatnonzero(np.isfinite(values))
    if not len(finite_positions):
        return output
    finite_values = values[finite_positions]
    order = np.argsort(finite_values, kind="mergesort")
    ranked = finite_values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)
    output[finite_positions[order]] = adjusted
    return output


def cell_type_composition(value: object, label_map: dict[str, str]) -> str:
    labels = str(value).split("&")
    cell_types = [label_map.get(label, f"Unknown label {label}") for label in labels]
    counts = pd.Series(cell_types).value_counts(sort=False)
    return "; ".join(f"{name} x{count}" for name, count in sorted(counts.items()))


def prepare_cohort(
    samples: pd.DataFrame, clinical: pd.DataFrame
) -> tuple[pd.DataFrame, sparse.csr_matrix, pd.DataFrame, dict[str, object]]:
    required_samples = {"row_index", "patient_id"}
    required_clinical = {"METABRIC.ID", "T", "Death"}
    if not required_samples.issubset(samples.columns):
        raise ValueError(f"Sample table lacks {sorted(required_samples - set(samples.columns))}")
    if not required_clinical.issubset(clinical.columns):
        raise ValueError(f"Clinical table lacks {sorted(required_clinical - set(clinical.columns))}")
    if samples["row_index"].duplicated().any() or samples["row_index"].isna().any():
        raise ValueError("Sample row_index values must be complete and unique")
    expected_rows = np.arange(len(samples))
    if not np.array_equal(samples["row_index"].to_numpy(dtype=int), expected_rows):
        raise ValueError("Sample row_index does not match motif matrix row order")
    if clinical["METABRIC.ID"].duplicated().any():
        raise ValueError("Clinical METABRIC.ID values must be unique")

    motif_ids = pd.Index(samples["patient_id"].dropna().astype(str).unique()).sort_values()
    clinical_work = clinical[["METABRIC.ID", "T", "Death"]].copy()
    clinical_work["METABRIC.ID"] = clinical_work["METABRIC.ID"].astype(str)
    clinical_work["time_days"] = pd.to_numeric(clinical_work["T"], errors="coerce")
    clinical_work["event"] = pd.to_numeric(clinical_work["Death"], errors="coerce")
    clinical_by_id = clinical_work.set_index("METABRIC.ID")

    linked = pd.DataFrame({"patient_id": motif_ids})
    linked = linked.join(clinical_by_id[["time_days", "event"]], on="patient_id")
    linked["in_clinical_table"] = linked["patient_id"].isin(clinical_by_id.index)
    linked["valid_time"] = linked["time_days"].gt(0)
    linked["valid_event"] = linked["event"].isin([0, 1])
    linked["included"] = linked[["in_clinical_table", "valid_time", "valid_event"]].all(axis=1)
    linked["exclusion_reason"] = ""
    linked.loc[~linked["in_clinical_table"], "exclusion_reason"] = "patient_id_not_in_clinical_table"
    linked.loc[linked["in_clinical_table"] & ~linked["valid_time"], "exclusion_reason"] = "missing_or_nonpositive_T"
    linked.loc[
        linked["in_clinical_table"] & linked["valid_time"] & ~linked["valid_event"],
        "exclusion_reason",
    ] = "missing_or_nonbinary_Death"

    patients = linked.loc[linked["included"], ["patient_id", "time_days", "event"]].copy()
    patients["event"] = patients["event"].astype(int)
    patients = patients.sort_values("patient_id").reset_index(drop=True)
    if patients.empty or patients["event"].sum() == 0:
        raise ValueError("No usable METABRIC survival cohort or no deaths")

    patient_position = {pid: row for row, pid in enumerate(patients["patient_id"])}
    patient_rows: list[int] = []
    sample_columns: list[int] = []
    for sample_row, patient_id in enumerate(samples["patient_id"].astype(str)):
        if patient_id in patient_position:
            patient_rows.append(patient_position[patient_id])
            sample_columns.append(sample_row)
    aggregation = sparse.csr_matrix(
        (np.ones(len(patient_rows)), (patient_rows, sample_columns)),
        shape=(len(patients), len(samples)),
    )
    patients["n_images"] = np.asarray(aggregation.sum(axis=1)).ravel().astype(int)
    linked = linked.merge(
        samples.groupby(samples["patient_id"].astype(str)).size().rename("n_images"),
        left_on="patient_id",
        right_index=True,
        how="left",
    )
    audit = {
        "motif_images": int(len(samples)),
        "motif_mapped_patients": int(len(motif_ids)),
        "clinical_table_patients": int(len(clinical)),
        "analyzed_patients": int(len(patients)),
        "deaths": int(patients["event"].sum()),
        "censored": int((patients["event"] == 0).sum()),
        "excluded_motif_patients": int((~linked["included"]).sum()),
        "excluded_not_in_clinical_table": int((~linked["in_clinical_table"]).sum()),
        "patients_with_multiple_images": int((patients["n_images"] > 1).sum()),
        "analyzed_images": int(aggregation.nnz),
        "time_days_min": float(patients["time_days"].min()),
        "time_days_max": float(patients["time_days"].max()),
    }
    return patients, aggregation, linked, audit


def survival_risk_data(
    time: np.ndarray, event: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    event_times = np.sort(np.unique(time[event == 1]))
    order = np.argsort(time, kind="mergesort")
    sorted_time = time[order]
    risk_start = np.searchsorted(sorted_time, event_times, side="left")
    total_at_risk = np.array([(time >= value).sum() for value in event_times], dtype=float)
    total_deaths = np.array(
        [((time == value) & (event == 1)).sum() for value in event_times], dtype=int
    )
    return event_times, order, risk_start, np.column_stack([total_at_risk, total_deaths])


def vectorized_binary_cox(
    groups: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    tolerance: float = 1e-10,
    max_iterations: int = 100,
) -> dict[str, np.ndarray]:
    """Fit binary, unadjusted Cox models with Efron handling of tied deaths."""
    groups = np.asarray(groups, dtype=bool)
    if groups.ndim == 1:
        groups = groups[:, None]
    if groups.shape[0] != len(time) or len(time) != len(event):
        raise ValueError("Groups, time, and event lengths do not agree")

    event_times, order, risk_start, totals = survival_risk_data(time, event)
    ordered_groups = groups[order]
    cumulative = np.cumsum(ordered_groups[::-1], axis=0, dtype=np.int32)[::-1]
    enriched_at_risk = cumulative[risk_start].astype(float)
    enriched_deaths = np.vstack(
        [groups[(time == value) & (event == 1)].sum(axis=0) for value in event_times]
    ).astype(float)

    n_features = groups.shape[1]
    enriched_n = groups.sum(axis=0).astype(np.int32)
    enriched_event_n = enriched_deaths.sum(axis=0).astype(np.int32)
    non_enriched_event_n = int(event.sum()) - enriched_event_n
    varying = (enriched_n > 0) & (enriched_n < len(time))
    estimable = varying & (enriched_event_n > 0) & (non_enriched_event_n > 0)
    status = np.zeros(n_features, dtype=np.int8)
    status[varying & ~estimable] = 1
    beta = np.zeros(n_features, dtype=float)
    converged = np.zeros(n_features, dtype=bool)
    active = estimable.copy()

    for _ in range(max_iterations):
        exp_beta = np.exp(np.clip(beta, -40.0, 40.0))
        score = np.zeros(n_features, dtype=float)
        information = np.zeros(n_features, dtype=float)
        for row, (_, death_count) in enumerate(totals):
            deaths = int(death_count)
            risk_one = enriched_at_risk[row] * exp_beta
            death_one = enriched_deaths[row] * exp_beta
            risk_zero = totals[row, 0] - enriched_at_risk[row]
            death_zero = deaths - enriched_deaths[row]
            score += enriched_deaths[row]
            for tied_index in range(deaths):
                fraction = tied_index / deaths
                denominator = risk_zero + risk_one - fraction * (death_zero + death_one)
                numerator = risk_one - fraction * death_one
                probability = np.divide(
                    numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0
                )
                score -= probability
                information += probability * (1.0 - probability)
        valid = active & np.isfinite(score) & np.isfinite(information) & (information > 1e-12)
        delta = np.zeros(n_features, dtype=float)
        delta[valid] = np.clip(score[valid] / information[valid], -5.0, 5.0)
        beta[valid] += delta[valid]
        newly_converged = valid & (np.abs(delta) < tolerance)
        converged |= newly_converged
        active &= ~newly_converged
        active &= valid
        active &= np.abs(beta) < 30.0
        if not active.any():
            break

    successful = estimable & converged & (np.abs(beta) < 30.0)
    status[successful] = 2
    status[estimable & ~successful] = 3

    exp_beta = np.exp(np.clip(beta, -40.0, 40.0))
    final_information = np.zeros(n_features, dtype=float)
    for row, (_, death_count) in enumerate(totals):
        deaths = int(death_count)
        risk_one = enriched_at_risk[row] * exp_beta
        death_one = enriched_deaths[row] * exp_beta
        risk_zero = totals[row, 0] - enriched_at_risk[row]
        death_zero = deaths - enriched_deaths[row]
        for tied_index in range(deaths):
            fraction = tied_index / deaths
            denominator = risk_zero + risk_one - fraction * (death_zero + death_one)
            numerator = risk_one - fraction * death_one
            probability = np.divide(
                numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0
            )
            final_information += probability * (1.0 - probability)

    standard_error = np.full(n_features, np.nan)
    standard_error[successful] = np.sqrt(1.0 / final_information[successful])
    p_value = np.full(n_features, np.nan)
    p_value[successful] = 2.0 * norm.sf(np.abs(beta[successful] / standard_error[successful]))
    z95 = norm.ppf(0.975)
    return {
        "coef": np.where(successful, beta, np.nan),
        "hazard_ratio": np.where(successful, np.exp(beta), np.nan),
        "ci95_low": np.where(successful, np.exp(beta - z95 * standard_error), np.nan),
        "ci95_high": np.where(successful, np.exp(beta + z95 * standard_error), np.nan),
        "p_value": p_value,
        "status_code": status,
        "n_enriched": enriched_n,
        "events_enriched": enriched_event_n,
        "n_non_enriched": len(time) - enriched_n,
        "events_non_enriched": int(event.sum()) - enriched_event_n,
    }


def analyze_sparse(
    patient_counts: sparse.spmatrix,
    time: np.ndarray,
    event: np.ndarray,
    chunk_size: int,
    checkpoint_dir: Path,
) -> dict[str, np.ndarray]:
    matrix = patient_counts.tocsc()
    n_features = matrix.shape[1]
    keys_float = ["median_patient_count", "coef", "hazard_ratio", "ci95_low", "ci95_high", "p_value"]
    output: dict[str, np.ndarray] = {key: np.empty(n_features, dtype=float) for key in keys_float}
    output.update(
        {
            "status_code": np.empty(n_features, dtype=np.int8),
            "n_enriched": np.empty(n_features, dtype=np.int32),
            "events_enriched": np.empty(n_features, dtype=np.int32),
            "n_non_enriched": np.empty(n_features, dtype=np.int32),
            "events_non_enriched": np.empty(n_features, dtype=np.int32),
        }
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, n_features, chunk_size):
        stop = min(start + chunk_size, n_features)
        checkpoint = checkpoint_dir / f"features_{start}_{stop}.npz"
        if checkpoint.is_file():
            with np.load(checkpoint) as saved:
                fitted = {key: saved[key] for key in saved.files}
        else:
            dense = matrix[:, start:stop].toarray()
            medians = np.median(dense, axis=0)
            fitted = vectorized_binary_cox(dense > medians, time, event)
            fitted["median_patient_count"] = medians
            temporary = checkpoint.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, **fitted)
            temporary.replace(checkpoint)
        for key in output:
            output[key][start:stop] = fitted[key]
        print(f"fitted {stop:,} / {n_features:,} motifs", flush=True)
    return output


def write_report(path: Path, results: pd.DataFrame, summary: dict[str, object], top_n: int) -> None:
    lines = [
        "# METABRIC motif overall-survival screen",
        "",
        f"The analysis includes **{summary['analyzed_patients']} patients**, "
        f"**{summary['deaths']} deaths**, and **{summary['total_motifs']:,} motifs**. "
        "The unadjusted model is `Surv(T, Death) ~ motif_enriched`, using T in days, "
        "Efron ties, and two-sided Wald p-values.",
        "",
        "Counts from repeated images were summed within patient. Enriched means strictly "
        "above the motif-specific patient median; median ties are non-enriched.",
        "",
        "## Counts by level",
        "",
        "| Level | Motifs | Estimated | Raw p < 0.05 | Within-level BH q < 0.05 | Global BH q < 0.05 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for level, item in summary["level_summary"].items():
        lines.append(
            f"| TRI-{level} | {item['motifs']:,} | {item['estimated_models']:,} | "
            f"{item['p_below_0_05']:,} | {item['within_level_bh_q_below_0_05']:,} | "
            f"{item['global_bh_q_below_0_05']:,} |"
        )
    for level in summary["triangle_levels"]:
        subset = results.loc[results["triangle_level"].eq(level)].head(top_n)
        lines.extend(
            [
                "",
                f"## Top TRI-{level} motifs",
                "",
                "| Rank | WL hash | Cell types | Enriched (deaths) / non-enriched (deaths) | HR (95% CI) | Raw p | Global BH q |",
                "|---:|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in subset.itertuples(index=False):
            lines.append(
                f"| {row.rank_within_level} | `{row.wl_hash}` | {row.cell_type_composition} | "
                f"{row.n_enriched} ({row.events_enriched}) / {row.n_non_enriched} "
                f"({row.events_non_enriched}) | "
                f"{row.hazard_ratio:.3g} ({row.ci95_low:.3g}-{row.ci95_high:.3g}) | "
                f"{row.p_value:.3g} | {row.q_value_bh_global:.3g} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "HR > 1 indicates higher hazard in enriched patients; HR < 1 indicates lower "
            "hazard. Raw p-values are exploratory. Use BH q-values when interpreting this "
            "large motif screen. No clinical covariates or prevalence/event filters were used.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    levels = sorted(set(args.levels))
    if not levels or any(level not in range(2, 8) for level in levels):
        raise ValueError("Levels must be selected from 2 through 7")
    if args.chunk_size <= 0 or args.top_n <= 0:
        raise ValueError("chunk-size and top-n must be positive")
    args.outdir.mkdir(parents=True, exist_ok=True)

    samples = pd.read_csv(args.artifacts_root / "samples.csv")
    clinical = pd.read_csv(args.clinical, sep="\t", low_memory=False)
    features = pd.read_csv(args.artifacts_root / "feature_index.csv.gz")
    triangle_levels = np.load(args.artifacts_root / "feature_triangle_levels.npy")
    counts = sparse.load_npz(args.artifacts_root / "motif_counts.npz").tocsr()
    if counts.shape != (len(samples), len(features)):
        raise ValueError("Motif matrix shape does not agree with sample/feature tables")
    if len(triangle_levels) != len(features) or not np.array_equal(
        triangle_levels.astype(int), features["triangle_level"].to_numpy(dtype=int)
    ):
        raise ValueError("Triangle-level vector does not agree with feature metadata")

    patients, aggregation, linkage, cohort_audit = prepare_cohort(samples, clinical)
    selected = features["triangle_level"].isin(levels).to_numpy()
    selected_features = features.loc[selected].reset_index(drop=True)
    patient_counts = (aggregation @ counts[:, selected]).tocsr()
    patients.to_csv(args.outdir / "patient_level_survival.csv", index=False)
    linkage.to_csv(args.outdir / "patient_linkage_audit.csv", index=False)
    print(
        f"Analyzing {len(patients)} patients, {int(patients['event'].sum())} deaths, "
        f"{patient_counts.shape[1]:,} motifs",
        flush=True,
    )
    fitted = analyze_sparse(
        patient_counts,
        patients["time_days"].to_numpy(dtype=float),
        patients["event"].to_numpy(dtype=int),
        args.chunk_size,
        args.outdir / "checkpoints" / ("levels_" + "-".join(map(str, levels))),
    )

    result = selected_features.rename(columns={"document_frequency": "image_document_frequency"}).copy()
    for key, values in fitted.items():
        if key != "status_code":
            result[key] = values
    result["fit_status"] = [STATUS_LABELS[int(code)] for code in fitted["status_code"]]
    result["n_patients"] = len(patients)
    result["n_deaths"] = int(patients["event"].sum())
    result["q_value_bh_global"] = benjamini_hochberg(result["p_value"].to_numpy())
    result["q_value_bh_within_level"] = np.nan
    result["rank_within_level"] = pd.array([pd.NA] * len(result), dtype="Int64")
    for level in levels:
        mask = result["triangle_level"].eq(level)
        result.loc[mask, "q_value_bh_within_level"] = benjamini_hochberg(
            result.loc[mask, "p_value"].to_numpy()
        )
        result.loc[mask, "rank_within_level"] = (
            result.loc[mask, "p_value"].rank(method="min", na_option="keep").astype("Int64")
        )
    result["rank_global"] = result["p_value"].rank(method="min", na_option="keep").astype("Int64")
    label_frame = pd.read_csv(args.label_map, dtype={"node_label": str})
    label_map = dict(label_frame[["node_label", "cell_type"]].itertuples(index=False, name=None))
    result["cell_type_composition"] = result["representative_node_labels"].map(
        lambda value: cell_type_composition(value, label_map)
    )
    result = result.sort_values(
        ["p_value", "triangle_level", "wl_hash"], na_position="last", kind="mergesort"
    ).reset_index(drop=True)
    result.to_csv(args.outdir / "motif_cox_results_sorted.csv.gz", index=False)
    result.head(1000).to_csv(args.outdir / "top_1000_motif_cox_results.csv", index=False)
    for level in levels:
        level_result = result.loc[result["triangle_level"].eq(level)].copy()
        level_result.to_csv(args.outdir / f"tri_{level}_cox_results_sorted.csv.gz", index=False)

    level_summary: dict[str, dict[str, int]] = {}
    for level in levels:
        subset = result.loc[result["triangle_level"].eq(level)]
        level_summary[str(level)] = {
            "motifs": int(len(subset)),
            "estimated_models": int(subset["p_value"].notna().sum()),
            "p_below_0_05": int(subset["p_value"].lt(0.05).sum()),
            "within_level_bh_q_below_0_05": int(subset["q_value_bh_within_level"].lt(0.05).sum()),
            "global_bh_q_below_0_05": int(subset["q_value_bh_global"].lt(0.05).sum()),
        }
    summary = {
        **cohort_audit,
        "artifacts_root": str(args.artifacts_root.resolve()),
        "clinical_path": str(args.clinical.resolve()),
        "data_dictionary_path": str(args.data_dictionary.resolve()),
        "label_map_path": str(args.label_map.resolve()),
        "triangle_levels": levels,
        "total_motifs": int(len(result)),
        "estimated_models": int(result["p_value"].notna().sum()),
        "p_below_0_05": int(result["p_value"].lt(0.05).sum()),
        "global_bh_q_below_0_05": int(result["q_value_bh_global"].lt(0.05).sum()),
        "level_summary": level_summary,
        "endpoint": {"time": "T", "time_unit": "days", "event": "Death", "event_definition": "all-cause death"},
        "cox_model": "Surv(T, Death) ~ motif_enriched",
        "cox_ties": "efron",
        "cox_p_value": "two-sided Wald test",
        "clinical_covariates": None,
        "enrichment_definition": "summed patient motif count > motif-specific cohort median",
        "median_ties": "non-enriched",
        "minimum_prevalence_group_or_event_filter": None,
    }
    with (args.outdir / "analysis_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    write_report(args.outdir / "REPORT.md", result, summary, args.top_n)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
