#!/usr/bin/env python3
"""Plot the patient-level frequency split for every significant focal motif.

Counts from all available tumor cores are summed per patient, exactly as in the
unadjusted Cox screen.  Each plot orders patients by raw motif count, colors
patients at or below the motif-specific median as non-enriched, and colors
patients strictly above the median as enriched.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", "/tmp/breast-cancer-significant-frequency-matplotlib"
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy import sparse

from analyze_motif_survival import discover_samples, load_level_counts, prepare_patients
from analyze_motif_survival_unadjusted import load_inventory, prepare_patient_aggregation


NON_ENRICHED_COLOR = "#4C78A8"
ENRICHED_COLOR = "#F28E2B"
REQUIRED_POOL_COLUMNS = {
    "triangle_level",
    "wl_hash",
    "cell_type_composition",
    "median_patient_raw_count",
    "p_value_unadjusted",
    "n_enriched_unadjusted",
    "n_non_enriched_unadjusted",
}


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    survival = Path(__file__).resolve().parent
    trace_root = survival / "motif_results" / "significant_mtf_trace"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pool-csv", type=Path, default=trace_root / "potential_motif_pool.csv"
    )
    parser.add_argument(
        "--trace-root", type=Path, default=trace_root,
        help="Plots are written inside each triN/<wl_hash> trace directory",
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
    parser.add_argument("--levels", type=int, nargs="+", default=list(range(1, 8)))
    parser.add_argument("--dpi", type=int, default=190)
    return parser.parse_args()


def load_pool(path: Path, levels: list[int]) -> pd.DataFrame:
    pool = pd.read_csv(path, dtype={"wl_hash": str})
    missing = sorted(REQUIRED_POOL_COLUMNS - set(pool.columns))
    if missing:
        raise ValueError(f"Significant motif pool is missing columns: {missing}")
    pool = pool.loc[pool["triangle_level"].isin(levels)].copy()
    if pool.empty:
        raise ValueError("No significant motifs matched the requested levels")
    if pool.duplicated(["triangle_level", "wl_hash"]).any():
        raise ValueError("Significant motif pool contains duplicate level/hash pairs")
    return pool.sort_values(
        ["triangle_level", "p_value_unadjusted", "wl_hash"], kind="mergesort"
    ).reset_index(drop=True)


def ordered_frequency_frame(
    patients: pd.DataFrame, counts: np.ndarray, median: float
) -> pd.DataFrame:
    if len(patients) != len(counts):
        raise ValueError("Patient and motif-count lengths differ")
    frame = pd.DataFrame(
        {
            "PID": patients["PID"].to_numpy(),
            "motif_count": np.asarray(counts, dtype=float),
        }
    )
    if not np.isfinite(frame["motif_count"]).all() or frame["motif_count"].lt(0).any():
        raise ValueError("Motif counts must be finite and nonnegative")
    frame["enrichment_group"] = np.where(
        frame["motif_count"].gt(median), "Enriched", "Non-enriched"
    )
    return frame.sort_values(
        ["motif_count", "PID"], kind="mergesort"
    ).reset_index(drop=True)


def validate_split(row: pd.Series, ordered: pd.DataFrame, median: float) -> None:
    saved_median = float(row["median_patient_raw_count"])
    if not np.isclose(median, saved_median, rtol=0, atol=1e-9):
        raise ValueError(
            f"TRI-{int(row['triangle_level'])} {row['wl_hash']} median mismatch: "
            f"reconstructed {median:g}, saved {saved_median:g}"
        )
    counts = ordered["enrichment_group"].value_counts()
    reconstructed_enriched = int(counts.get("Enriched", 0))
    reconstructed_non_enriched = int(counts.get("Non-enriched", 0))
    saved_enriched = int(row["n_enriched_unadjusted"])
    saved_non_enriched = int(row["n_non_enriched_unadjusted"])
    if (reconstructed_enriched, reconstructed_non_enriched) != (
        saved_enriched,
        saved_non_enriched,
    ):
        raise ValueError(
            f"TRI-{int(row['triangle_level'])} {row['wl_hash']} group-size "
            f"mismatch: reconstructed {(reconstructed_enriched, reconstructed_non_enriched)}, "
            f"saved {(saved_enriched, saved_non_enriched)}"
        )


def plot_frequency_histogram(
    ordered: pd.DataFrame, row: pd.Series, output: Path, dpi: int
) -> None:
    level = int(row["triangle_level"])
    motif_hash = str(row["wl_hash"])
    median = float(row["median_patient_raw_count"])
    enriched = ordered["enrichment_group"].eq("Enriched").to_numpy()
    values = ordered["motif_count"].to_numpy(dtype=float)
    positions = np.arange(len(ordered))
    colors = np.where(enriched, ENRICHED_COLOR, NON_ENRICHED_COLOR)
    non_enriched_n = int((~enriched).sum())

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.bar(positions, values, width=1.0, color=colors, linewidth=0, rasterized=True)
    ax.axhline(
        median,
        color="black",
        linestyle="--",
        linewidth=1.35,
        label=f"Patient median = {median:g}",
        zorder=4,
    )
    if 0 < non_enriched_n < len(ordered):
        ax.axvline(
            non_enriched_n - 0.5,
            color="black",
            linestyle=":",
            linewidth=1.2,
            zorder=4,
        )
    composition = str(row["cell_type_composition"])
    ax.set_title(
        f"TRI-{level} significant motif {motif_hash[:8]}: ordered patient frequencies\n"
        f"{composition} | raw Cox p = {float(row['p_value_unadjusted']):.3g}",
        fontsize=11,
    )
    ax.set_xlabel("Patients ordered by summed raw motif count")
    ax.set_ylabel("Motif occurrences across available patient cores")
    ax.set_xlim(-0.5, len(ordered) - 0.5)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.2)
    handles = [
        Patch(
            facecolor=NON_ENRICHED_COLOR,
            label=f"Non-enriched (count ≤ median; n={non_enriched_n})",
        ),
        Patch(
            facecolor=ENRICHED_COLOR,
            label=f"Enriched (count > median; n={int(enriched.sum())})",
        ),
        plt.Line2D([], [], color="black", linestyle="--", label=f"Median = {median:g}"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper left")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def tri1_counts(
    selected: pd.DataFrame,
    motif_root: Path,
    metadata: pd.DataFrame,
    label_map: dict[str, str],
) -> tuple[pd.DataFrame, np.ndarray]:
    inventory = discover_samples(motif_root)
    patients, core_to_pid, _ = prepare_patients(metadata, inventory)
    features, all_counts = load_level_counts(
        inventory, patients, core_to_pid, 1, label_map
    )
    lookup = features.reset_index().set_index("wl_hash")["index"]
    missing = selected.loc[~selected["wl_hash"].isin(lookup.index), "wl_hash"]
    if len(missing):
        raise ValueError(f"TRI-1 selected hashes are missing: {missing.tolist()}")
    columns = lookup.loc[selected["wl_hash"]].to_numpy(dtype=int)
    return patients, all_counts[:, columns]


def high_level_counts(
    level: int,
    selected: pd.DataFrame,
    artifacts_root: Path,
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    level_dir = artifacts_root / f"tri_{level}"
    samples = load_inventory(artifacts_root, level)
    patients, aggregation, _ = prepare_patient_aggregation(samples, metadata)
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
        raise ValueError(f"TRI-{level} selected hashes are missing: {missing.tolist()[:3]}")
    columns = lookup.loc[selected["wl_hash"]].to_numpy(dtype=int)
    matrix = sparse.load_npz(level_dir / "motif_counts.npz").tocsr()
    if matrix.shape[0] != len(samples):
        raise ValueError(f"TRI-{level} motif matrix/sample count mismatch")
    return patients, (aggregation @ matrix[:, columns]).toarray()


def main() -> None:
    args = parse_args()
    levels = sorted(set(args.levels))
    if not levels or any(level < 1 or level > 7 for level in levels):
        raise ValueError("--levels must be between 1 and 7")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")
    pool = load_pool(args.pool_csv, levels)
    metadata = pd.read_csv(args.metadata)
    label_frame = pd.read_csv(args.label_map, dtype={"node_label": str})
    label_map = dict(
        label_frame[["node_label", "cell_type"]].itertuples(index=False, name=None)
    )

    manifest_rows: list[dict[str, object]] = []
    reference_pids: list[object] | None = None
    for level in levels:
        selected = pool.loc[pool["triangle_level"].eq(level)].reset_index(drop=True)
        if selected.empty:
            continue
        if level == 1:
            patients, counts = tri1_counts(
                selected, args.motif_root, metadata, label_map
            )
        else:
            patients, counts = high_level_counts(
                level, selected, args.artifacts_root, metadata
            )
        pids = patients["PID"].tolist()
        if reference_pids is not None and pids != reference_pids:
            raise ValueError(f"TRI-{level} patient cohort/order differs from prior levels")
        reference_pids = pids

        for position, (_, row) in enumerate(selected.iterrows()):
            motif_counts = counts[:, position]
            median = float(np.median(motif_counts))
            ordered = ordered_frequency_frame(patients, motif_counts, median)
            validate_split(row, ordered, median)
            motif_hash = str(row["wl_hash"])
            relative_output = (
                Path(f"tri{level}") / motif_hash / "patient_frequency_histogram.png"
            )
            output = args.trace_root / relative_output
            plot_frequency_histogram(ordered, row, output, args.dpi)
            manifest_rows.append(
                {
                    "triangle_level": level,
                    "wl_hash": motif_hash,
                    "median_patient_raw_count": median,
                    "n_non_enriched": int(
                        ordered["enrichment_group"].eq("Non-enriched").sum()
                    ),
                    "n_enriched": int(
                        ordered["enrichment_group"].eq("Enriched").sum()
                    ),
                    "minimum_patient_count": float(ordered["motif_count"].min()),
                    "maximum_patient_count": float(ordered["motif_count"].max()),
                    "plot": str(relative_output),
                }
            )
        print(f"TRI-{level}: wrote {len(selected)} frequency histograms", flush=True)

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(args.trace_root / "frequency_histogram_manifest.csv", index=False)
    summary = {
        "plots": int(len(manifest)),
        "plots_by_level": {
            str(int(level)): int(count)
            for level, count in manifest.groupby("triangle_level").size().items()
        },
        "patients_per_plot": 0 if reference_pids is None else len(reference_pids),
        "count_definition": "sum of raw motif occurrences across all available patient cores",
        "ordering": "ascending patient motif count, then PID",
        "non_enriched_definition": "patient motif count <= motif-specific cohort median",
        "enriched_definition": "patient motif count > motif-specific cohort median",
        "plot_filename": "patient_frequency_histogram.png",
    }
    (args.trace_root / "frequency_histogram_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
