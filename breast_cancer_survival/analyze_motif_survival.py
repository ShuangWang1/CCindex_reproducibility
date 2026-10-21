#!/usr/bin/env python3
"""Rank TRI-2/TRI-3 motifs by patient-level overall-survival association.

For each motif, raw match counts are summed over a patient's available Basel
tumor cores. Patients strictly above the cohort median count are called
enriched; patients at or below the median are non-enriched. Each motif is fit
with an unadjusted and a prespecified clinically adjusted Cox model.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import re
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from statsmodels.duration.hazard_regression import PHReg


DEATH_STATUSES = {"death", "death by primary disease"}
LIVING_STATUSES = {"alive", "alive w metastases"}
MODEL_COVARIATES = [
    "age_z",
    "grade_z",
    "tumor_size_z",
    "node_positive",
    "metastatic",
    "er_positive",
    "her2_positive",
]
MOTIF_COLUMNS = [
    "triangles",
    "nodes",
    "edges",
    "node_labels",
    "edge_list",
    "wl_hash",
    "matches",
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
        default=repo / "breast_cancer" / "BZ_samples_intra" / "node_label_cell_type_mapping.csv",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).resolve().parent / "motif_results",
    )
    parser.add_argument("--levels", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    return parser.parse_args()


def sample_number(path: Path) -> int:
    match = re.fullmatch(r"sample(\d+)", path.name)
    if not match:
        raise ValueError(f"Unexpected sample directory name: {path.name}")
    return int(match.group(1))


def discover_samples(motif_root: Path) -> pd.DataFrame:
    """Use per-sample summaries, not the potentially partial root manifest."""
    rows: list[dict[str, object]] = []
    sample_dirs = sorted(motif_root.glob("sample*"), key=sample_number)
    for sample_dir in sample_dirs:
        summary_path = sample_dir / "run_summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"Missing run summary: {summary_path}")
        summary = json.loads(summary_path.read_text())
        sample = str(summary.get("sample", "")).strip()
        source_sample = str(summary.get("source_sample", "")).strip()
        if sample != sample_dir.name or not source_sample:
            raise ValueError(f"Invalid sample identity in {summary_path}")
        rows.append(
            {
                "sample": sample,
                "source_sample": source_sample,
                "sample_dir": sample_dir,
                "completed_triangle_level": int(summary["completed_triangle_level"]),
            }
        )
    inventory = pd.DataFrame(rows)
    if inventory.empty:
        raise ValueError(f"No sample directories found under {motif_root}")
    for column in ["sample", "source_sample"]:
        duplicate = inventory.loc[inventory[column].duplicated(), column].tolist()
        if duplicate:
            raise ValueError(f"Duplicate {column} values: {duplicate[:5]}")
    return inventory


def collapse_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    fields = [
        "Patientstatus",
        "OSmonth",
        "age",
        "grade",
        "tumor_size",
        "PTNM_M",
        "PTNM_N",
        "ERStatus",
        "HER2Status",
    ]
    required = {"PID", "core", "diseasestatus", *fields}
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"Metadata columns are missing: {missing}")
    inconsistent = {
        field: metadata.groupby("PID")[field]
        .nunique(dropna=False)
        .loc[lambda counts: counts > 1]
        .index.tolist()
        for field in fields
    }
    inconsistent = {key: value for key, value in inconsistent.items() if value}
    if inconsistent:
        raise ValueError(f"Patient metadata are inconsistent across cores: {inconsistent}")
    return metadata.drop_duplicates("PID")[["PID", *fields]].copy()


def zscore(series: pd.Series, name: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    sd = float(numeric.std(ddof=1))
    if not np.isfinite(sd) or sd == 0:
        raise ValueError(f"Cannot standardize {name}: non-positive standard deviation")
    return (numeric - numeric.mean()) / sd


def prepare_patients(
    metadata: pd.DataFrame, inventory: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, object], dict[str, object]]:
    tumor = metadata.loc[metadata["diseasestatus"].eq("tumor")].copy()
    available_sources = set(inventory["source_sample"])
    matched = tumor.loc[tumor["core"].isin(available_sources)].copy()
    patient_metadata = collapse_metadata(metadata)
    patients = patient_metadata.loc[
        patient_metadata["PID"].isin(matched["PID"])
    ].copy()

    statuses = set(patients["Patientstatus"].dropna().astype(str))
    unexpected = statuses - DEATH_STATUSES - LIVING_STATUSES
    if unexpected:
        raise ValueError(f"Unexpected Patientstatus values: {sorted(unexpected)}")
    patients["event"] = patients["Patientstatus"].isin(DEATH_STATUSES).astype(int)
    patients["time_months"] = pd.to_numeric(patients["OSmonth"], errors="coerce")
    patients = patients.loc[
        patients["time_months"].notna() & patients["time_months"].gt(0)
    ].copy()
    patients = patients.sort_values("PID").reset_index(drop=True)
    if patients.empty or patients["event"].sum() == 0:
        raise ValueError("No usable patients or events remain")

    patients["age_z"] = zscore(patients["age"], "age")
    patients["grade_z"] = zscore(patients["grade"], "grade")
    patients["tumor_size_z"] = zscore(patients["tumor_size"], "tumor_size")
    node = patients["PTNM_N"].astype(str).str.strip().str.lower()
    patients["node_positive"] = np.where(
        node.str.startswith("0"), 0.0, np.where(node.eq("x"), np.nan, 1.0)
    )
    patients["metastatic"] = pd.to_numeric(patients["PTNM_M"], errors="coerce")
    patients["er_positive"] = patients["ERStatus"].map(
        {"positive": 1.0, "negative": 0.0}
    )
    patients["her2_positive"] = patients["HER2Status"].map(
        {"positive": 1.0, "negative": 0.0}
    )

    usable_pids = set(patients["PID"])
    core_to_pid = dict(
        matched.loc[matched["PID"].isin(usable_pids), ["core", "PID"]].itertuples(
            index=False, name=None
        )
    )
    audit = {
        "metadata_rows": int(len(metadata)),
        "metadata_patients": int(metadata["PID"].nunique()),
        "metadata_tumor_cores": int(tumor["core"].nunique()),
        "matched_tumor_cores": int(matched["core"].nunique()),
        "missing_tumor_cores": sorted(set(tumor["core"]) - available_sources),
        "patients_with_matched_motif_cores": int(matched["PID"].nunique()),
        "analyzed_patients": int(len(patients)),
        "deaths": int(patients["event"].sum()),
        "censored": int((1 - patients["event"]).sum()),
        "patients_excluded_missing_or_zero_followup": int(
            matched["PID"].nunique() - len(patients)
        ),
        "patients_with_multiple_motif_cores": int(
            (matched.loc[matched["PID"].isin(usable_pids)].groupby("PID").size() > 1).sum()
        ),
    }
    return patients, core_to_pid, audit


def read_motif_csv(path: Path, expected_level: int) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=MOTIF_COLUMNS, dtype={"wl_hash": str})
    if frame["wl_hash"].duplicated().any():
        duplicate = frame.loc[frame["wl_hash"].duplicated(), "wl_hash"].iloc[0]
        raise ValueError(f"Duplicate wl_hash {duplicate} in {path}")
    if not frame.empty and not frame["triangles"].eq(expected_level).all():
        raise ValueError(f"Unexpected triangle level in {path}")
    frame["matches"] = pd.to_numeric(frame["matches"], errors="raise")
    if frame["matches"].lt(0).any() or not np.isfinite(frame["matches"]).all():
        raise ValueError(f"Invalid motif match counts in {path}")
    # Search output can contain explicit zero-match candidates. They represent
    # absence and must neither enter the vocabulary nor its document frequency.
    return frame.loc[frame["matches"].gt(0)].copy()


def cell_type_composition(node_labels: object, label_map: dict[str, str]) -> str:
    labels = str(node_labels).split("&")
    names = [label_map.get(label, f"label {label}") for label in labels]
    counts: dict[str, int] = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1
    return "; ".join(
        f"{name} x{count}" if count > 1 else name for name, count in counts.items()
    )


def load_level_counts(
    inventory: pd.DataFrame,
    patients: pd.DataFrame,
    core_to_pid: dict[str, object],
    level: int,
    label_map: dict[str, str],
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return the complete motif vocabulary and summed patient count matrix."""
    feature_metadata: dict[str, dict[str, object]] = {}
    for row in inventory.itertuples(index=False):
        path = Path(row.sample_dir) / f"ccindex_tri_{level}.csv"
        if not path.is_file():
            if row.completed_triangle_level < level:
                continue
            raise FileNotFoundError(f"Missing completed motif file: {path}")
        frame = read_motif_csv(path, level)
        for motif in frame.itertuples(index=False):
            motif_hash = str(motif.wl_hash)
            current = feature_metadata.get(motif_hash)
            if current is None:
                feature_metadata[motif_hash] = {
                    "triangle_level": level,
                    "wl_hash": motif_hash,
                    "triangles": int(motif.triangles),
                    "nodes": int(motif.nodes),
                    "edges": int(motif.edges),
                    "representative_node_labels": str(motif.node_labels),
                    "representative_edge_list": str(motif.edge_list),
                    "cell_type_composition": cell_type_composition(
                        motif.node_labels, label_map
                    ),
                    "sample_document_frequency": 1,
                }
            else:
                if current["nodes"] != int(motif.nodes) or current["edges"] != int(
                    motif.edges
                ):
                    raise ValueError(f"Inconsistent topology for TRI-{level} {motif_hash}")
                current["sample_document_frequency"] += 1

    features = pd.DataFrame(feature_metadata.values()).sort_values("wl_hash").reset_index(
        drop=True
    )
    if features.empty:
        raise ValueError(f"No TRI-{level} motifs were found")
    feature_position = {value: i for i, value in enumerate(features["wl_hash"])}
    patient_position = {value: i for i, value in enumerate(patients["PID"])}
    counts = np.zeros((len(patients), len(features)), dtype=np.float64)

    matched_inventory = inventory.loc[inventory["source_sample"].isin(core_to_pid)]
    for row in matched_inventory.itertuples(index=False):
        pid = core_to_pid[row.source_sample]
        if pid not in patient_position:
            continue
        path = Path(row.sample_dir) / f"ccindex_tri_{level}.csv"
        if not path.is_file():
            if row.completed_triangle_level < level:
                continue
            raise FileNotFoundError(f"Missing completed motif file: {path}")
        frame = read_motif_csv(path, level)
        columns = frame["wl_hash"].map(feature_position).to_numpy(dtype=int)
        counts[patient_position[pid], columns] += frame["matches"].to_numpy(dtype=float)
    return features, counts


def fit_cox(
    time: np.ndarray,
    event: np.ndarray,
    enriched: np.ndarray,
    covariates: np.ndarray | None = None,
) -> dict[str, object]:
    if len(np.unique(enriched)) < 2:
        return {
            "coef": np.nan,
            "hazard_ratio": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "p_value": np.nan,
            "fit_status": "not_estimable_no_group_variation",
            "fit_warning": "",
        }
    design = enriched.astype(float)[:, None]
    if covariates is not None:
        design = np.column_stack([design, covariates])
    caught: list[warnings.WarningMessage]
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = PHReg(time, design, status=event, ties="efron")
            result = model.fit(disp=0)
        coefficient = float(result.params[0])
        interval = np.asarray(result.conf_int(), dtype=float)[0]
        p_value = float(result.pvalues[0])
        values = np.array([coefficient, interval[0], interval[1], p_value])
        warning_text = " | ".join(dict.fromkeys(str(item.message) for item in caught))
        if not np.isfinite(values).all():
            return {
                "coef": np.nan,
                "hazard_ratio": np.nan,
                "ci95_low": np.nan,
                "ci95_high": np.nan,
                "p_value": np.nan,
                "fit_status": "not_estimable_nonfinite_result",
                "fit_warning": warning_text or "Cox fit returned non-finite statistics",
            }
        if warning_text:
            status = "estimated_with_warning"
        else:
            status = "estimated"
        return {
            "coef": coefficient,
            "hazard_ratio": math.exp(coefficient) if coefficient < 710 else np.inf,
            "ci95_low": math.exp(float(interval[0])) if interval[0] < 710 else np.inf,
            "ci95_high": math.exp(float(interval[1])) if interval[1] < 710 else np.inf,
            "p_value": p_value,
            "fit_status": status,
            "fit_warning": warning_text,
        }
    except Exception as exc:  # Numerical failures vary by motif and LAPACK build.
        return {
            "coef": np.nan,
            "hazard_ratio": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "p_value": np.nan,
            "fit_status": "not_estimable_fit_error",
            "fit_warning": f"{type(exc).__name__}: {exc}",
        }


def benjamini_hochberg(p_values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(p_values), dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    valid_indices = np.flatnonzero(np.isfinite(values))
    if not len(valid_indices):
        return adjusted
    order = valid_indices[np.argsort(values[valid_indices], kind="mergesort")]
    count = len(order)
    ranked = values[order] * count / np.arange(1, count + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted


def analyze_feature(
    feature: dict[str, object],
    patient_counts: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    complete: np.ndarray,
    adjusted_covariates: np.ndarray,
) -> dict[str, object]:
    cutoff = float(np.median(patient_counts))
    enriched = patient_counts > cutoff
    enriched_events = int(event[enriched].sum())
    non_enriched = ~enriched
    unadjusted = fit_cox(time, event, enriched)
    adjusted = fit_cox(
        time[complete], event[complete], enriched[complete], adjusted_covariates
    )
    row = dict(feature)
    row.update(
        {
            "median_patient_raw_count": cutoff,
            "enriched_definition": "patient_raw_count > cohort_median",
            "n_unadjusted": int(len(time)),
            "events_unadjusted": int(event.sum()),
            "n_enriched": int(enriched.sum()),
            "events_enriched": enriched_events,
            "n_non_enriched": int(non_enriched.sum()),
            "events_non_enriched": int(event[non_enriched].sum()),
            "coef_unadjusted": unadjusted["coef"],
            "hazard_ratio_unadjusted": unadjusted["hazard_ratio"],
            "ci95_low_unadjusted": unadjusted["ci95_low"],
            "ci95_high_unadjusted": unadjusted["ci95_high"],
            "p_value_unadjusted": unadjusted["p_value"],
            "fit_status_unadjusted": unadjusted["fit_status"],
            "fit_warning_unadjusted": unadjusted["fit_warning"],
            "n_adjusted": int(complete.sum()),
            "events_adjusted": int(event[complete].sum()),
            "n_enriched_adjusted": int(enriched[complete].sum()),
            "events_enriched_adjusted": int(event[complete & enriched].sum()),
            "n_non_enriched_adjusted": int((complete & non_enriched).sum()),
            "events_non_enriched_adjusted": int(event[complete & non_enriched].sum()),
            "coef_adjusted": adjusted["coef"],
            "hazard_ratio_adjusted": adjusted["hazard_ratio"],
            "ci95_low_adjusted": adjusted["ci95_low"],
            "ci95_high_adjusted": adjusted["ci95_high"],
            "p_value_adjusted": adjusted["p_value"],
            "fit_status_adjusted": adjusted["fit_status"],
            "fit_warning_adjusted": adjusted["fit_warning"],
        }
    )
    return row


_WORKER_CONTEXT: tuple[np.ndarray, ...] | None = None


def initialize_worker(
    counts: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    complete: np.ndarray,
    adjusted_covariates: np.ndarray,
) -> None:
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = (counts, time, event, complete, adjusted_covariates)


def analyze_feature_worker(task: tuple[int, dict[str, object]]) -> dict[str, object]:
    if _WORKER_CONTEXT is None:
        raise RuntimeError("Motif worker was not initialized")
    counts, time, event, complete, adjusted_covariates = _WORKER_CONTEXT
    column, feature = task
    return analyze_feature(
        feature,
        counts[:, column],
        time,
        event,
        complete,
        adjusted_covariates,
    )


def analyze_level(
    features: pd.DataFrame,
    counts: np.ndarray,
    patients: pd.DataFrame,
    workers: int = 1,
) -> pd.DataFrame:
    time = patients["time_months"].to_numpy(dtype=float)
    event = patients["event"].to_numpy(dtype=int)
    complete = patients[MODEL_COVARIATES].notna().all(axis=1).to_numpy()
    adjusted_covariates = patients.loc[complete, MODEL_COVARIATES].to_numpy(dtype=float)
    feature_records = features.to_dict(orient="records")
    tasks = list(enumerate(feature_records))
    if workers == 1:
        rows = [
            analyze_feature(
                feature,
                counts[:, column],
                time,
                event,
                complete,
                adjusted_covariates,
            )
            for column, feature in tasks
        ]
    else:
        context = mp.get_context("fork")
        with context.Pool(
            processes=workers,
            initializer=initialize_worker,
            initargs=(counts, time, event, complete, adjusted_covariates),
        ) as pool:
            rows = list(pool.imap(analyze_feature_worker, tasks, chunksize=20))
    return pd.DataFrame(rows)


def add_multiplicity_and_ranks(results: pd.DataFrame) -> pd.DataFrame:
    results = results.copy()
    results["q_value_bh_global_unadjusted"] = benjamini_hochberg(
        results["p_value_unadjusted"]
    )
    results["q_value_bh_global_adjusted"] = benjamini_hochberg(
        results["p_value_adjusted"]
    )
    results["rank_global_unadjusted"] = results["p_value_unadjusted"].rank(
        method="min", na_option="keep"
    ).astype("Int64")
    results["rank_within_level_unadjusted"] = results.groupby("triangle_level")[
        "p_value_unadjusted"
    ].rank(method="min", na_option="keep").astype("Int64")
    return results.sort_values(
        ["p_value_unadjusted", "triangle_level", "wl_hash"],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)


def format_number(value: object, digits: int = 3) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}g}"


def markdown_top_table(frame: pd.DataFrame, top_n: int) -> str:
    lines = [
        "| Rank | WL hash | Cell-type composition | Enriched / non-enriched | HR (95% CI) | Unadjusted p | Adjusted p | Global BH q |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in frame.head(top_n).itertuples(index=False):
        lines.append(
            f"| {row.rank_within_level_unadjusted} | `{row.wl_hash}` | "
            f"{row.cell_type_composition} | {row.n_enriched} / {row.n_non_enriched} | "
            f"{format_number(row.hazard_ratio_unadjusted)} "
            f"({format_number(row.ci95_low_unadjusted)}–{format_number(row.ci95_high_unadjusted)}) | "
            f"{format_number(row.p_value_unadjusted)} | "
            f"{format_number(row.p_value_adjusted)} | "
            f"{format_number(row.q_value_bh_global_unadjusted)} |"
        )
    return "\n".join(lines)


def write_report(
    path: Path, results: pd.DataFrame, audit: dict[str, object], top_n: int
) -> None:
    sections = [
        "# TRI-2/TRI-3 motif overall-survival results",
        "",
        "## Analysis",
        "",
        f"The analysis includes **{audit['analyzed_patients']} patients** and "
        f"**{audit['deaths']} deaths**. Raw motif counts were summed across each "
        "patient's available tumor cores. For each motif, enriched means strictly "
        "above its patient-level cohort median; ties are non-enriched. The primary "
        "ranking is the unadjusted Cox Wald p-value.",
        "",
    ]
    for level in sorted(results["triangle_level"].unique()):
        subset = results.loc[results["triangle_level"].eq(level)].sort_values(
            "p_value_unadjusted", na_position="last"
        )
        sections.extend(
            [
                f"## Top TRI-{level} motifs",
                "",
                markdown_top_table(subset, top_n),
                "",
            ]
        )
    raw_significant = int(results["p_value_unadjusted"].lt(0.05).sum())
    fdr_significant = int(results["q_value_bh_global_unadjusted"].lt(0.05).sum())
    unadjusted_failed = int(results["p_value_unadjusted"].isna().sum())
    adjusted_failed = int(results["p_value_adjusted"].isna().sum())
    sections.extend(
        [
            "## Multiplicity and cautions",
            "",
            f"Across {len(results):,} motifs, {raw_significant:,} have an unadjusted "
            f"p-value below 0.05 and {fdr_significant:,} have a global Benjamini–Hochberg "
            "q-value below 0.05. The global correction treats TRI-2 and TRI-3 as one "
            "search family.",
            "",
            f"Unadjusted models were non-estimable for {unadjusted_failed:,} motifs; "
            f"adjusted models were non-estimable for {adjusted_failed:,}. No minimum "
            "group-size or event filter was imposed, so tiny groups and separation can "
            "produce unstable estimates. Review group sizes, events, confidence intervals, "
            "fit warnings, adjusted results, and FDR—not only the raw p-value. This is an "
            "exploratory association screen and does not establish causality.",
            "",
        ]
    )
    path.write_text("\n".join(sections))


def root_manifest_rows(motif_root: Path) -> int | None:
    path = motif_root / "run_manifest.csv"
    if not path.is_file():
        return None
    return int(len(pd.read_csv(path)))


def main() -> None:
    args = parse_args()
    levels = sorted(set(args.levels))
    if not levels or any(level < 2 for level in levels):
        raise ValueError("--levels must contain triangle levels >= 2")
    if args.top_n <= 0:
        raise ValueError("--top-n must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")

    inventory = discover_samples(args.motif_root)
    metadata = pd.read_csv(args.metadata)
    patients, core_to_pid, audit = prepare_patients(metadata, inventory)
    label_frame = pd.read_csv(args.label_map, dtype={"node_label": str})
    label_map = dict(
        label_frame[["node_label", "cell_type"]].itertuples(index=False, name=None)
    )

    all_results: list[pd.DataFrame] = []
    level_audit: dict[str, dict[str, int]] = {}
    for level in levels:
        print(f"Loading TRI-{level} motif counts", flush=True)
        features, counts = load_level_counts(
            inventory, patients, core_to_pid, level, label_map
        )
        print(f"Fitting {len(features):,} TRI-{level} motifs", flush=True)
        level_results = analyze_level(features, counts, patients, workers=args.workers)
        all_results.append(level_results)
        level_audit[str(level)] = {
            "motifs": int(len(features)),
            "motifs_present_in_survival_cohort": int((counts > 0).any(axis=0).sum()),
        }

    results = add_multiplicity_and_ranks(pd.concat(all_results, ignore_index=True))
    complete = patients[MODEL_COVARIATES].notna().all(axis=1)
    audit.update(
        {
            "motif_root": str(args.motif_root.resolve()),
            "metadata_path": str(args.metadata.resolve()),
            "sample_summaries_discovered": int(len(inventory)),
            "root_run_manifest_rows": root_manifest_rows(args.motif_root),
            "triangle_levels": levels,
            "level_summary": level_audit,
            "total_motifs": int(len(results)),
            "adjusted_complete_case_patients": int(complete.sum()),
            "adjusted_complete_case_events": int(patients.loc[complete, "event"].sum()),
            "enrichment_definition": "summed patient raw count > motif-specific cohort median",
            "median_ties": "non-enriched",
            "cox_ties": "efron",
            "cox_p_value": "two-sided Wald test",
            "primary_ranking": "unadjusted Cox p-value ascending",
            "workers": args.workers,
            "multiple_testing": "Benjamini-Hochberg across all estimable TRI-2/TRI-3 motifs",
            "minimum_group_or_event_filter": None,
            "unadjusted_non_estimable": int(results["p_value_unadjusted"].isna().sum()),
            "adjusted_non_estimable": int(results["p_value_adjusted"].isna().sum()),
            "unadjusted_p_below_0_05": int(results["p_value_unadjusted"].lt(0.05).sum()),
            "unadjusted_global_bh_q_below_0_05": int(
                results["q_value_bh_global_unadjusted"].lt(0.05).sum()
            ),
        }
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.outdir / "motif_cox_results_sorted.csv", index=False)
    for level in levels:
        results.loc[results["triangle_level"].eq(level)].to_csv(
            args.outdir / f"tri_{level}_cox_results_sorted.csv", index=False
        )
    with (args.outdir / "analysis_summary.json").open("w") as handle:
        json.dump(audit, handle, indent=2)
    write_report(args.outdir / "REPORT.md", results, audit, args.top_n)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
