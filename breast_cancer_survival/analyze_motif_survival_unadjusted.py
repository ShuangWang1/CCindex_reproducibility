#!/usr/bin/env python3
"""Fast, checkpointed unadjusted Cox screen for high-level BZ motifs.

The sparse motif artifacts are exact derivatives of the raw per-sample CSVs.
For each motif, sample counts are summed to the patient level, split strictly
above the patient-count median, and fit with a one-predictor Cox model using
Efron ties. The vectorized solver produces the same Wald test as PHReg while
avoiding per-model Python optimizer overhead.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import norm

from analyze_motif_survival import (
    benjamini_hochberg,
    cell_type_composition,
    prepare_patients,
)


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
        default=repo / "breast_cancer" / "BZ_samples_intra" / "node_label_cell_type_mapping.csv",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).resolve().parent / "motif_results_tri4_tri7_unadjusted",
    )
    parser.add_argument("--levels", type=int, nargs="+", default=[4, 5, 6, 7])
    parser.add_argument("--chunk-size", type=int, default=20_000)
    parser.add_argument("--top-n", type=int, default=20)
    return parser.parse_args()


def load_inventory(artifacts_root: Path, level: int) -> pd.DataFrame:
    samples = pd.read_csv(artifacts_root / f"tri_{level}" / "samples.csv")
    required = {"sample", "source_sample"}
    if not required.issubset(samples):
        raise ValueError(f"TRI-{level} samples are missing {sorted(required-set(samples))}")
    return samples


def prepare_patient_aggregation(
    samples: pd.DataFrame, metadata: pd.DataFrame
) -> tuple[pd.DataFrame, sparse.csr_matrix, dict[str, object]]:
    inventory = samples[["sample", "source_sample"]].copy()
    patients, core_to_pid, audit = prepare_patients(metadata, inventory)
    patient_position = {pid: position for position, pid in enumerate(patients["PID"])}
    patient_rows: list[int] = []
    sample_columns: list[int] = []
    for sample_column, source in enumerate(samples["source_sample"]):
        pid = core_to_pid.get(source)
        if pid in patient_position:
            patient_rows.append(patient_position[pid])
            sample_columns.append(sample_column)
    aggregation = sparse.csr_matrix(
        (
            np.ones(len(patient_rows), dtype=np.float64),
            (patient_rows, sample_columns),
        ),
        shape=(len(patients), len(samples)),
    )
    expected = sum(source in core_to_pid for source in samples["source_sample"])
    if aggregation.nnz != expected:
        raise ValueError("Patient/sample aggregation did not reconcile")
    return patients, aggregation, audit


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
    """Fit many binary, unadjusted Cox models with the Efron method."""
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
    information = np.full(n_features, np.nan, dtype=float)
    for _ in range(max_iterations):
        exp_beta = np.exp(np.clip(beta, -40.0, 40.0))
        score = np.zeros(n_features, dtype=float)
        info = np.zeros(n_features, dtype=float)
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
                    numerator,
                    denominator,
                    out=np.zeros_like(numerator),
                    where=denominator > 0,
                )
                score -= probability
                info += probability * (1.0 - probability)
        information = info
        valid_step = active & np.isfinite(score) & np.isfinite(info) & (info > 1e-12)
        delta = np.zeros(n_features, dtype=float)
        delta[valid_step] = score[valid_step] / info[valid_step]
        delta = np.clip(delta, -5.0, 5.0)
        beta[valid_step] += delta[valid_step]
        newly_converged = valid_step & (np.abs(delta) < tolerance)
        converged |= newly_converged
        active &= ~newly_converged
        active &= valid_step
        active &= np.abs(beta) < 30.0
        if not active.any():
            break

    successful = estimable & converged & (np.abs(beta) < 30.0)
    status[successful] = 2
    status[estimable & ~successful] = 3

    # Refresh observed information at the final coefficient values.
    exp_beta = np.exp(np.clip(beta, -40.0, 40.0))
    final_info = np.zeros(n_features, dtype=float)
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
                numerator,
                denominator,
                out=np.zeros_like(numerator),
                where=denominator > 0,
            )
            final_info += probability * (1.0 - probability)

    standard_error = np.full(n_features, np.nan, dtype=float)
    standard_error[successful] = np.sqrt(1.0 / final_info[successful])
    p_value = np.full(n_features, np.nan, dtype=float)
    p_value[successful] = 2.0 * norm.sf(
        np.abs(beta[successful] / standard_error[successful])
    )
    coefficient = np.where(successful, beta, np.nan)
    hazard_ratio = np.where(successful, np.exp(beta), np.nan)
    ci_low = np.where(successful, np.exp(beta - norm.ppf(0.975) * standard_error), np.nan)
    ci_high = np.where(successful, np.exp(beta + norm.ppf(0.975) * standard_error), np.nan)
    return {
        "coef": coefficient,
        "hazard_ratio": hazard_ratio,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "p_value": p_value,
        "status_code": status,
        "n_enriched": enriched_n,
        "events_enriched": enriched_event_n,
        "n_non_enriched": len(time) - enriched_n,
        "events_non_enriched": int(event.sum()) - enriched_event_n,
    }


def analyze_level_sparse(
    patient_counts: sparse.spmatrix,
    time: np.ndarray,
    event: np.ndarray,
    chunk_size: int,
) -> dict[str, np.ndarray]:
    matrix = patient_counts.tocsc()
    n_features = matrix.shape[1]
    output = {
        "median_patient_raw_count": np.empty(n_features, dtype=float),
        "coef": np.empty(n_features, dtype=float),
        "hazard_ratio": np.empty(n_features, dtype=float),
        "ci95_low": np.empty(n_features, dtype=float),
        "ci95_high": np.empty(n_features, dtype=float),
        "p_value": np.empty(n_features, dtype=float),
        "status_code": np.empty(n_features, dtype=np.int8),
        "n_enriched": np.empty(n_features, dtype=np.int32),
        "events_enriched": np.empty(n_features, dtype=np.int32),
        "n_non_enriched": np.empty(n_features, dtype=np.int32),
        "events_non_enriched": np.empty(n_features, dtype=np.int32),
    }
    for start in range(0, n_features, chunk_size):
        stop = min(start + chunk_size, n_features)
        dense = matrix[:, start:stop].toarray()
        medians = np.median(dense, axis=0)
        fitted = vectorized_binary_cox(dense > medians, time, event)
        output["median_patient_raw_count"][start:stop] = medians
        for key, values in fitted.items():
            output[key][start:stop] = values
        print(f"  fitted {stop:,} / {n_features:,}", flush=True)
    return output


def save_checkpoint(path: Path, results: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **results)
    temporary.replace(path)


def load_checkpoint(path: Path, expected_features: int) -> dict[str, np.ndarray] | None:
    if not path.is_file():
        return None
    with np.load(path) as stored:
        results = {key: stored[key] for key in stored.files}
    if len(results.get("p_value", [])) != expected_features:
        return None
    return results


def feature_table(
    path: Path, label_map: dict[str, str], expected_level: int
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if len(frame) and not frame["triangle_level"].eq(expected_level).all():
        raise ValueError(f"Feature levels do not match TRI-{expected_level}")
    frame = frame.rename(columns={"document_frequency": "sample_document_frequency"})
    frame["cell_type_composition"] = frame["representative_node_labels"].map(
        lambda value: cell_type_composition(value, label_map)
    )
    return frame


def assemble_level_results(
    features: pd.DataFrame,
    fitted: dict[str, np.ndarray],
    global_q: np.ndarray,
    global_rank: np.ndarray,
    n_patients: int,
    n_events: int,
) -> pd.DataFrame:
    result = features.copy()
    for key, values in fitted.items():
        if key != "status_code":
            result[key if key.startswith("median_") else f"{key}_unadjusted"] = values
    result["fit_status_unadjusted"] = [
        STATUS_LABELS[int(value)] for value in fitted["status_code"]
    ]
    result["n_unadjusted"] = n_patients
    result["events_unadjusted"] = n_events
    result["q_value_bh_global_unadjusted"] = global_q
    result["q_value_bh_within_level_unadjusted"] = benjamini_hochberg(
        fitted["p_value"]
    )
    result["rank_global_unadjusted"] = pd.array(global_rank, dtype="Int64")
    result["rank_within_level_unadjusted"] = result["p_value_unadjusted"].rank(
        method="min", na_option="keep"
    ).astype("Int64")
    return result.sort_values(
        ["p_value_unadjusted", "wl_hash"], na_position="last", kind="mergesort"
    ).reset_index(drop=True)


def write_report(path: Path, combined: pd.DataFrame, audit: dict[str, object], top_n: int) -> None:
    lines = [
        "# TRI-4–TRI-7 unadjusted motif Cox results",
        "",
        f"The screen includes **{audit['analyzed_patients']} patients**, "
        f"**{audit['deaths']} deaths**, and **{audit['total_motifs']:,} motifs**. "
        "Each model is `Surv(OSmonth, event) ~ motif_enriched`, with Efron ties and "
        "a two-sided Wald p-value.",
        "",
    ]
    for level in audit["triangle_levels"]:
        subset = combined.loc[combined["triangle_level"].eq(level)].head(top_n)
        lines.extend(
            [
                f"## Top TRI-{level} motifs",
                "",
                "| Rank | WL hash | Composition | Enriched (events) / non-enriched (events) | HR (95% CI) | P value | Global BH q |",
                "|---:|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in subset.itertuples(index=False):
            lines.append(
                f"| {row.rank_within_level_unadjusted} | `{row.wl_hash}` | "
                f"{row.cell_type_composition} | {row.n_enriched_unadjusted} "
                f"({row.events_enriched_unadjusted}) / {row.n_non_enriched_unadjusted} "
                f"({row.events_non_enriched_unadjusted}) | {row.hazard_ratio_unadjusted:.3g} "
                f"({row.ci95_low_unadjusted:.3g}–{row.ci95_high_unadjusted:.3g}) | "
                f"{row.p_value_unadjusted:.3g} | {row.q_value_bh_global_unadjusted:.3g} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Multiple testing and interpretation",
            "",
            f"{audit['p_below_0_05']:,} motifs have raw p < 0.05; "
            f"{audit['global_bh_q_below_0_05']:,} have global BH q < 0.05. "
            "No prevalence, group-size, or minimum-event filter was applied. Motifs "
            "without group variation, with event separation, or without a converged "
            "finite estimate remain in the output with NA statistics and an explicit "
            "fit status.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    levels = sorted(set(args.levels))
    if not levels or args.chunk_size <= 0 or args.top_n <= 0:
        raise ValueError("Levels, chunk size, and top-n must be positive")
    args.outdir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.outdir / "checkpoints"
    metadata = pd.read_csv(args.metadata)
    label_frame = pd.read_csv(args.label_map, dtype={"node_label": str})
    label_map = dict(label_frame[["node_label", "cell_type"]].itertuples(index=False, name=None))

    level_results: dict[int, dict[str, np.ndarray]] = {}
    level_features: dict[int, Path] = {}
    base_audit: dict[str, object] | None = None
    patients_reference: pd.DataFrame | None = None
    sample_order_reference: list[str] | None = None
    level_summary: dict[str, dict[str, int]] = {}
    for level in levels:
        level_dir = args.artifacts_root / f"tri_{level}"
        samples = load_inventory(args.artifacts_root, level)
        sample_order = samples["source_sample"].astype(str).tolist()
        if sample_order_reference is not None and sample_order != sample_order_reference:
            raise ValueError(f"TRI-{level} sample ordering differs from earlier levels")
        sample_order_reference = sample_order
        patients, aggregation, audit = prepare_patient_aggregation(samples, metadata)
        if patients_reference is not None and not patients["PID"].equals(patients_reference["PID"]):
            raise ValueError(f"TRI-{level} patient cohort differs from earlier levels")
        patients_reference = patients
        base_audit = audit
        matrix = sparse.load_npz(level_dir / "motif_counts.npz").tocsr()
        if matrix.shape[0] != len(samples):
            raise ValueError(f"TRI-{level} count matrix/sample mismatch")
        patient_counts = (aggregation @ matrix).tocsr()
        checkpoint = checkpoint_dir / f"tri_{level}_unadjusted.npz"
        fitted = load_checkpoint(checkpoint, matrix.shape[1])
        if fitted is None:
            print(f"TRI-{level}: fitting {matrix.shape[1]:,} motifs", flush=True)
            fitted = analyze_level_sparse(
                patient_counts,
                patients["time_months"].to_numpy(dtype=float),
                patients["event"].to_numpy(dtype=int),
                args.chunk_size,
            )
            save_checkpoint(checkpoint, fitted)
            print(f"TRI-{level}: checkpoint written", flush=True)
        else:
            print(f"TRI-{level}: reused checkpoint", flush=True)
        level_results[level] = fitted
        level_features[level] = level_dir / "feature_index.csv.gz"
        level_summary[str(level)] = {
            "motifs": int(matrix.shape[1]),
            "motifs_present_in_survival_cohort": int(
                np.count_nonzero(np.asarray(patient_counts.getnnz(axis=0)))
            ),
            "estimated_models": int(np.count_nonzero(fitted["status_code"] == 2)),
        }

    assert patients_reference is not None and base_audit is not None
    all_p = np.concatenate([level_results[level]["p_value"] for level in levels])
    all_q = benjamini_hochberg(all_p)
    ranks = pd.Series(all_p).rank(method="min", na_option="keep").to_numpy()
    offsets = np.cumsum([0, *[len(level_results[level]["p_value"]) for level in levels]])
    combined_parts: list[pd.DataFrame] = []
    for position, level in enumerate(levels):
        start, stop = int(offsets[position]), int(offsets[position + 1])
        features = feature_table(level_features[level], label_map, level)
        if len(features) != stop - start:
            raise ValueError(f"TRI-{level} feature metadata/count mismatch")
        table = assemble_level_results(
            features,
            level_results[level],
            all_q[start:stop],
            ranks[start:stop],
            len(patients_reference),
            int(patients_reference["event"].sum()),
        )
        table.to_csv(args.outdir / f"tri_{level}_cox_results_sorted.csv", index=False)
        combined_parts.append(
            table[
                [
                    "rank_global_unadjusted",
                    "triangle_level",
                    "rank_within_level_unadjusted",
                    "wl_hash",
                    "cell_type_composition",
                    "median_patient_raw_count",
                    "n_enriched_unadjusted",
                    "events_enriched_unadjusted",
                    "n_non_enriched_unadjusted",
                    "events_non_enriched_unadjusted",
                    "hazard_ratio_unadjusted",
                    "ci95_low_unadjusted",
                    "ci95_high_unadjusted",
                    "p_value_unadjusted",
                    "q_value_bh_global_unadjusted",
                    "q_value_bh_within_level_unadjusted",
                    "fit_status_unadjusted",
                ]
            ]
        )
        print(f"TRI-{level}: sorted CSV written", flush=True)

    combined = pd.concat(combined_parts, ignore_index=True).sort_values(
        ["p_value_unadjusted", "triangle_level", "wl_hash"],
        na_position="last",
        kind="mergesort",
    )
    combined.to_csv(args.outdir / "motif_cox_results_sorted.csv", index=False)
    audit = {
        **base_audit,
        "artifacts_root": str(args.artifacts_root.resolve()),
        "metadata_path": str(args.metadata.resolve()),
        "triangle_levels": levels,
        "level_summary": level_summary,
        "total_motifs": int(len(all_p)),
        "estimated_models": int(np.isfinite(all_p).sum()),
        "non_estimable_models": int(np.isnan(all_p).sum()),
        "p_below_0_05": int(np.count_nonzero(all_p < 0.05)),
        "global_bh_q_below_0_05": int(np.count_nonzero(all_q < 0.05)),
        "enrichment_definition": "summed patient raw count > motif-specific cohort median",
        "median_ties": "non-enriched",
        "cox_model": "Surv(OSmonth, event) ~ motif_enriched",
        "cox_ties": "efron",
        "cox_p_value": "two-sided Wald test",
        "clinical_covariates": None,
        "minimum_prevalence_group_or_event_filter": None,
        "chunk_size": args.chunk_size,
    }
    with (args.outdir / "analysis_summary.json").open("w") as handle:
        json.dump(audit, handle, indent=2)
    write_report(args.outdir / "REPORT.md", combined, audit, args.top_n)
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
