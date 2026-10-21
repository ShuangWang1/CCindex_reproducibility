#!/usr/bin/env python3
"""Run the TRI-1 patient-level unadjusted Cox survival screen.

TRI-1 frequencies are read from the per-sample ``ccindex_tri_1.csv`` files.
Counts from all available tumor cores are summed for each patient.  For each
motif, patients strictly above its cohort median are compared with patients at
or below the median using an Efron one-predictor Cox model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_motif_survival import (
    benjamini_hochberg,
    discover_samples,
    load_level_counts,
    prepare_patients,
)
from analyze_motif_survival_unadjusted import (
    assemble_level_results,
    vectorized_binary_cox,
)


LEVEL = 1
OUTPUT_COLUMNS = [
    "feature_index",
    "triangle_level",
    "wl_hash",
    "sample_document_frequency",
    "nodes",
    "edges",
    "representative_node_labels",
    "cell_type_composition",
    "median_patient_raw_count",
    "coef_unadjusted",
    "hazard_ratio_unadjusted",
    "ci95_low_unadjusted",
    "ci95_high_unadjusted",
    "p_value_unadjusted",
    "n_enriched_unadjusted",
    "events_enriched_unadjusted",
    "n_non_enriched_unadjusted",
    "events_non_enriched_unadjusted",
    "fit_status_unadjusted",
    "n_unadjusted",
    "events_unadjusted",
    "q_value_bh_global_unadjusted",
    "q_value_bh_within_level_unadjusted",
    "rank_global_unadjusted",
    "rank_within_level_unadjusted",
]


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--motif-root",
        type=Path,
        default=repo / "breast_cancer_meta" / "BZ_res_threshold_10_cpp_0811",
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
        "--outdir",
        type=Path,
        default=Path(__file__).resolve().parent / "motif_results",
    )
    return parser.parse_args()


def analyze_tri1(
    features: pd.DataFrame, counts: np.ndarray, patients: pd.DataFrame
) -> pd.DataFrame:
    """Fit and rank all TRI-1 motifs using the high-level screen's solver."""
    medians = np.median(counts, axis=0)
    fitted = vectorized_binary_cox(
        counts > medians,
        patients["time_months"].to_numpy(dtype=float),
        patients["event"].to_numpy(dtype=int),
    )
    fitted["median_patient_raw_count"] = medians
    p_values = fitted["p_value"]
    q_values = benjamini_hochberg(p_values)
    ranks = pd.Series(p_values).rank(method="min", na_option="keep").to_numpy()

    indexed = features.copy()
    indexed.insert(0, "feature_index", np.arange(len(indexed), dtype=int))
    result = assemble_level_results(
        indexed,
        fitted,
        q_values,
        ranks,
        len(patients),
        int(patients["event"].sum()),
    )
    return result[OUTPUT_COLUMNS]


def main() -> None:
    args = parse_args()
    inventory = discover_samples(args.motif_root)
    metadata = pd.read_csv(args.metadata)
    patients, core_to_pid, audit = prepare_patients(metadata, inventory)
    label_frame = pd.read_csv(args.label_map, dtype={"node_label": str})
    label_map = dict(
        label_frame[["node_label", "cell_type"]].itertuples(index=False, name=None)
    )
    features, counts = load_level_counts(
        inventory, patients, core_to_pid, LEVEL, label_map
    )
    results = analyze_tri1(features, counts, patients)

    args.outdir.mkdir(parents=True, exist_ok=True)
    output = args.outdir / "tri_1_cox_results_sorted.csv"
    temporary = output.with_suffix(".tmp")
    results.to_csv(temporary, index=False)
    temporary.replace(output)

    p_values = results["p_value_unadjusted"]
    level_audit = {
        **audit,
        "motif_root": str(args.motif_root.resolve()),
        "metadata_path": str(args.metadata.resolve()),
        "triangle_level": LEVEL,
        "motifs": int(len(results)),
        "motifs_present_in_survival_cohort": int((counts > 0).any(axis=0).sum()),
        "estimated_models": int(p_values.notna().sum()),
        "non_estimable_models": int(p_values.isna().sum()),
        "raw_p_below_0_05": int(p_values.lt(0.05).sum()),
        "within_tri1_bh_q_below_0_05": int(
            results["q_value_bh_within_level_unadjusted"].lt(0.05).sum()
        ),
        "enrichment_definition": (
            "summed patient raw count > motif-specific cohort median"
        ),
        "median_ties": "non-enriched",
        "cox_model": "Surv(OSmonth, event) ~ motif_enriched",
        "cox_ties": "efron",
        "cox_p_value": "two-sided Wald test",
        "multiple_testing_family": "TRI-1 motifs only",
        "output_file": str(output.resolve()),
    }
    audit_path = args.outdir / "tri_1_cox_analysis_summary.json"
    temporary_audit = audit_path.with_suffix(".tmp")
    temporary_audit.write_text(json.dumps(level_audit, indent=2) + "\n")
    temporary_audit.replace(audit_path)
    print(json.dumps(level_audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
