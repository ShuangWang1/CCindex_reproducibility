#!/usr/bin/env python3
"""Plot Kaplan-Meier curves for the existing TRI-2--TRI-7 top-30 motifs.

The selected motifs and displayed Cox statistics come from the consolidated
top-30 table. Patient motif counts are reconstructed from the sparse artifacts
and checked against the saved medians, group sizes, and event counts before any
plot is written.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/breast-cancer-top30-km-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse

from analyze_ki67_survival import COLORS, step_function
from analyze_motif_survival_unadjusted import load_inventory, prepare_patient_aggregation


TOP30_REQUIRED_COLUMNS = [
    "triangle_level",
    "top_rank",
    "wl_hash",
    "representative_node_labels",
    "median_patient_raw_count",
    "hazard_ratio_unadjusted",
    "ci95_low_unadjusted",
    "ci95_high_unadjusted",
    "p_value_unadjusted",
    "n_enriched_unadjusted",
    "events_enriched_unadjusted",
    "n_non_enriched_unadjusted",
    "events_non_enriched_unadjusted",
]


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    survival_root = Path(__file__).resolve().parent
    default_visualizations = (
        survival_root
        / "motif_results_tri4_tri7_unadjusted"
        / "top30_visualizations"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--top30-csv",
        type=Path,
        default=default_visualizations / "top30_motifs_by_level.csv",
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
    parser.add_argument("--outdir", type=Path, default=default_visualizations)
    parser.add_argument("--levels", type=int, nargs="+", default=list(range(2, 8)))
    return parser.parse_args()


def load_top30(path: Path, levels: list[int]) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        dtype={"wl_hash": str, "representative_node_labels": str},
    )
    missing = sorted(set(TOP30_REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Top-30 table is missing columns: {missing}")
    frame = frame.loc[frame["triangle_level"].isin(levels), TOP30_REQUIRED_COLUMNS].copy()
    if frame.empty:
        raise ValueError(f"No requested triangle levels were found in {path}")
    for level in levels:
        selected = frame.loc[frame["triangle_level"].eq(level)]
        if len(selected) != 30:
            raise ValueError(f"Expected 30 selected TRI-{level} motifs, found {len(selected)}")
        if selected["wl_hash"].duplicated().any():
            raise ValueError(f"TRI-{level} contains duplicate WL hashes")
        ranks = selected["top_rank"].astype(int).sort_values().tolist()
        if ranks != list(range(1, 31)):
            raise ValueError(f"TRI-{level} top ranks are not exactly 1 through 30")
    return frame.sort_values(["triangle_level", "top_rank"], kind="mergesort").reset_index(
        drop=True
    )


def motif_name(node_labels: str, label_map: dict[str, str]) -> str:
    labels = str(node_labels).split("&")
    missing = sorted(set(labels) - set(label_map))
    if missing:
        raise ValueError(f"Node labels are missing from the cell-type map: {missing}")
    return "&".join(label_map[label] for label in labels)


def safe_filename(value: str) -> str:
    """Preserve readable cell-type names while preventing path traversal."""
    result = value.replace("/", "-").replace("\\", "-")
    result = "".join(character for character in result if character.isprintable())
    if not result or result in {".", ".."}:
        raise ValueError(f"Invalid motif filename: {value!r}")
    return result


def unique_plot_stems(names: list[str]) -> list[str]:
    """Append _2, _3, ... to repeated names in their current rank order."""
    occurrences: Counter[str] = Counter()
    used: set[str] = set()
    stems: list[str] = []
    for raw_name in names:
        name = safe_filename(raw_name)
        occurrences[name] += 1
        number = occurrences[name]
        stem = name if number == 1 else f"{name}_{number}"
        while stem in used:
            number += 1
            occurrences[name] = number
            stem = f"{name}_{number}"
        if len(stem.encode("utf-8")) > 240:
            raise ValueError(f"Motif filename is too long: {stem}")
        used.add(stem)
        stems.append(stem)
    return stems


def wrap_motif_name(name: str, max_characters: int = 64) -> str:
    """Wrap an A&B&C motif name only between complete cell-type labels."""
    if max_characters <= 0:
        raise ValueError("max_characters must be positive")
    lines: list[str] = []
    current = ""
    for cell_type in name.split("&"):
        candidate = cell_type if not current else f"{current}&{cell_type}"
        if current and len(candidate) > max_characters:
            lines.append(current)
            current = cell_type
        else:
            current = candidate
    if current:
        lines.append(current)
    return "\n".join(lines)


def validate_reconstruction(
    selected: pd.DataFrame,
    medians: np.ndarray,
    enriched: np.ndarray,
    event: np.ndarray,
) -> None:
    enriched_n = enriched.sum(axis=0)
    enriched_events = (enriched * event[:, None]).sum(axis=0)
    observed = {
        "median_patient_raw_count": medians,
        "n_enriched_unadjusted": enriched_n,
        "events_enriched_unadjusted": enriched_events,
        "n_non_enriched_unadjusted": len(event) - enriched_n,
        "events_non_enriched_unadjusted": int(event.sum()) - enriched_events,
    }
    failures: list[str] = []
    for column, values in observed.items():
        expected = selected[column].to_numpy(dtype=float)
        if not np.allclose(values, expected, rtol=1e-10, atol=1e-12, equal_nan=True):
            positions = np.flatnonzero(~np.isclose(values, expected, rtol=1e-10, atol=1e-12))
            for position in positions[:3]:
                row = selected.iloc[int(position)]
                failures.append(
                    f"rank {int(row.top_rank)} {column}: reconstructed={values[position]!r}, "
                    f"saved={expected[position]!r}"
                )
    if failures:
        raise ValueError("Top-30 reconstruction did not match saved results: " + "; ".join(failures))


def plot_survival(
    patients: pd.DataFrame,
    enriched: np.ndarray,
    row: pd.Series,
    display_name: str,
    output: Path,
) -> None:
    plot_data = patients[["time_months", "event"]].copy()
    plot_data["motif_group"] = np.where(enriched, "Enriched", "Non-enriched")
    fig, ax = plt.subplots(figsize=(8.2, 6.3))
    for group in ["Non-enriched", "Enriched"]:
        data = plot_data.loc[plot_data["motif_group"].eq(group)]
        km = step_function(
            data["time_months"].to_numpy(dtype=float),
            data["event"].to_numpy(dtype=int),
        )
        label = f"{group} (n={len(data)}, events={int(data.event.sum())})"
        ax.step(
            km["time"],
            km["survival"],
            where="post",
            color=COLORS[group],
            linewidth=2.2,
            label=label,
        )
        ax.fill_between(
            km["time"],
            km["lower"],
            km["upper"],
            step="post",
            color=COLORS[group],
            alpha=0.15,
        )
        censored_times = data.loc[data["event"].eq(0), "time_months"].to_numpy(dtype=float)
        if len(censored_times):
            indices = np.searchsorted(km["time"].to_numpy(), censored_times, side="right") - 1
            ax.scatter(
                censored_times,
                km["survival"].to_numpy()[indices],
                marker="|",
                s=55,
                color=COLORS[group],
            )

    annotation = (
        f"Unadjusted Cox HR = {row['hazard_ratio_unadjusted']:.3g}\n"
        f"95% CI {row['ci95_low_unadjusted']:.3g}–{row['ci95_high_unadjusted']:.3g}\n"
        f"p = {row['p_value_unadjusted']:.3g}"
    )
    ax.text(
        0.98,
        0.98,
        annotation,
        transform=ax.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        fontsize=10,
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.9,
            "edgecolor": "0.75",
        },
    )
    level = int(row["triangle_level"])
    rank = int(row["top_rank"])
    motif_title = wrap_motif_name(display_name)
    ax.set_title(
        f"TRI-{level} rank {rank}: overall survival by motif enrichment\n"
        f"{motif_title}\nWL hash {str(row['wl_hash'])[:8]}",
        fontsize=11,
        pad=12,
    )
    ax.set_xlabel("Overall survival (months)")
    ax.set_ylabel("Survival probability")
    ax.set_ylim(0, 1.03)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def threshold_summary(
    row: pd.Series,
    display_name: str,
    median: float,
    enriched: np.ndarray,
    event: np.ndarray,
) -> str:
    enriched_events = int(event[enriched].sum())
    non_enriched = ~enriched
    return "\n".join(
        [
            f"triangle_level: {int(row['triangle_level'])}",
            f"top_rank: {int(row['top_rank'])}",
            f"wl_hash: {row['wl_hash']}",
            f"motif_cell_types: {display_name}",
            "patient_count_definition: motif matches summed across all available patient cores",
            f"threshold_cohort_median_patient_raw_count: {median:.15g}",
            f"enriched_definition: patient_raw_count > {median:.15g}",
            f"non_enriched_definition: patient_raw_count <= {median:.15g}",
            "median_ties_assigned_to: non-enriched",
            f"enriched_patients: {int(enriched.sum())}",
            f"enriched_events: {enriched_events}",
            f"non_enriched_patients: {int(non_enriched.sum())}",
            f"non_enriched_events: {int(event[non_enriched].sum())}",
            "",
        ]
    )


def process_level(
    level: int,
    selected: pd.DataFrame,
    artifacts_root: Path,
    metadata: pd.DataFrame,
    label_map: dict[str, str],
    outdir: Path,
) -> pd.DataFrame:
    level_dir = artifacts_root / f"tri_{level}"
    samples = load_inventory(artifacts_root, level)
    patients, aggregation, _ = prepare_patient_aggregation(samples, metadata)
    features = pd.read_csv(
        level_dir / "feature_index.csv.gz",
        usecols=["feature_index", "wl_hash"],
        dtype={"wl_hash": str},
    )
    if features["wl_hash"].duplicated().any():
        raise ValueError(f"TRI-{level} feature index contains duplicate WL hashes")
    feature_lookup = features.set_index("wl_hash")["feature_index"]
    missing_hashes = selected.loc[~selected["wl_hash"].isin(feature_lookup.index), "wl_hash"]
    if len(missing_hashes):
        raise ValueError(f"TRI-{level} selected WL hashes are missing: {missing_hashes.tolist()[:3]}")
    feature_indices = feature_lookup.loc[selected["wl_hash"]].to_numpy(dtype=int)
    matrix = sparse.load_npz(level_dir / "motif_counts.npz").tocsr()
    if matrix.shape[0] != len(samples):
        raise ValueError(f"TRI-{level} motif matrix/sample count mismatch")
    patient_counts = (aggregation @ matrix[:, feature_indices]).toarray()
    medians = np.median(patient_counts, axis=0)
    enriched = patient_counts > medians
    event = patients["event"].to_numpy(dtype=int)
    validate_reconstruction(selected, medians, enriched, event)

    names = [
        motif_name(value, label_map)
        for value in selected["representative_node_labels"].astype(str)
    ]
    stems = unique_plot_stems(names)
    level_outdir = outdir / f"tri{level}"
    level_outdir.mkdir(parents=True, exist_ok=True)
    for position, (_, row) in enumerate(selected.iterrows()):
        output_stem = level_outdir / stems[position]
        plot_survival(
            patients,
            enriched[:, position],
            row,
            names[position],
            output_stem.with_suffix(".png"),
        )
        output_stem.with_suffix(".txt").write_text(
            threshold_summary(
                row,
                names[position],
                float(medians[position]),
                enriched[:, position],
                event,
            )
        )
    print(
        f"TRI-{level}: wrote {len(selected)} plots and threshold files to {level_outdir}",
        flush=True,
    )
    return patients[["PID", "time_months", "event"]].reset_index(drop=True)


def main() -> None:
    args = parse_args()
    levels = sorted(set(args.levels))
    if not levels or any(level < 1 for level in levels):
        raise ValueError("--levels must contain positive integers")
    top30 = load_top30(args.top30_csv, levels)
    metadata = pd.read_csv(args.metadata)
    label_frame = pd.read_csv(args.label_map, dtype={"node_label": str})
    if label_frame["node_label"].duplicated().any():
        raise ValueError("Cell-type map contains duplicate node labels")
    label_map = dict(
        label_frame[["node_label", "cell_type"]].itertuples(index=False, name=None)
    )

    reference_patients: pd.DataFrame | None = None
    for level in levels:
        selected = top30.loc[top30["triangle_level"].eq(level)].reset_index(drop=True)
        patients = process_level(
            level,
            selected,
            args.artifacts_root,
            metadata,
            label_map,
            args.outdir,
        )
        if reference_patients is not None and not patients.equals(reference_patients):
            raise ValueError(f"TRI-{level} patient cohort differs from earlier levels")
        reference_patients = patients
    print(
        f"Wrote {len(top30)} Kaplan-Meier PNGs and {len(top30)} threshold TXT files "
        f"under {args.outdir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
