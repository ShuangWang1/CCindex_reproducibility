#!/usr/bin/env python3
"""Summarize and plot cell-type composition and cell counts for all BZ patients."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/breast-cancer-patient-analysis-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CELL_TYPE_COLORS = {
    "B cells": "#4E79A7",
    "T and B cells": "#76B7B2",
    "T cells": "#59A14F",
    "endothelial": "#EDC948",
    "macrophages": "#B07AA1",
    "stromal cells": "#FF9DA7",
    "tumor cells": "#E15759",
    "unknown": "#9C9C9C",
}


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--h5ad", type=Path, default=repo / "breast_cancer" / "BZ.h5ad"
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
        default=Path(__file__).resolve().parent
        / "motif_results"
        / "patient_analysis",
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def load_cell_observations(path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    import anndata as ad

    adata = ad.read_h5ad(path, backed="r")
    try:
        metadata = adata.uns.get("metadata", {})
        patient_column = metadata.get("patient_col", "PID")
        core_column = metadata.get("image_col", "core")
        cell_type_column = metadata.get(
            "cluster_col_preprocessed", "cluster_preprocessed"
        )
        required = {patient_column, core_column, cell_type_column}
        missing = sorted(required - set(adata.obs.columns))
        if missing:
            raise ValueError(f"H5AD observations are missing columns: {missing}")
        cells = adata.obs[[patient_column, core_column, cell_type_column]].copy()
        cells.columns = ["PID", "core", "cell_type"]
        if cells["PID"].isna().any() or cells["core"].isna().any():
            raise ValueError("Patient or core identifiers are missing")
        cells["PID"] = cells["PID"].astype(str)
        cells["core"] = cells["core"].astype(str)
        cells["cell_type"] = (
            cells["cell_type"].astype(object).where(cells["cell_type"].notna(), "unknown")
        ).astype(str)
        source = {
            "h5ad_rows": int(adata.n_obs),
            "h5ad_variables": int(adata.n_vars),
            "patient_column": patient_column,
            "core_column": core_column,
            "cell_type_column": cell_type_column,
        }
        return cells.reset_index(drop=True), source
    finally:
        adata.file.close()


def cell_type_order(label_map_path: Path, observed: set[str]) -> list[str]:
    mapping = pd.read_csv(label_map_path, dtype={"node_label": str})
    if "cell_type" not in mapping:
        raise ValueError("Cell-type label map is missing the cell_type column")
    if mapping["cell_type"].duplicated().any():
        raise ValueError("Cell-type label map contains duplicate names")
    ordered = mapping["cell_type"].astype(str).tolist()
    return [*ordered, *sorted(observed - set(ordered))]


def summarize_patients(
    cells: pd.DataFrame, ordered_cell_types: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = {"PID", "core", "cell_type"}
    missing = sorted(required - set(cells.columns))
    if missing:
        raise ValueError(f"Cell table is missing columns: {missing}")
    unexpected = sorted(set(cells["cell_type"]) - set(ordered_cell_types))
    if unexpected:
        raise ValueError(f"Cell types are missing from the requested order: {unexpected}")

    counts = (
        cells.groupby(["PID", "cell_type"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=ordered_cell_types, fill_value=0)
        .astype(int)
    )
    cores = cells.groupby("PID", observed=True)["core"].nunique().astype(int)
    wide = counts.copy()
    wide.insert(0, "number_of_cores", cores.reindex(wide.index))
    wide.insert(1, "total_cells", counts.sum(axis=1))
    wide = wide.reset_index()
    if int(wide["total_cells"].sum()) != len(cells):
        raise ValueError("Patient cell counts do not reconcile with the input")

    long = counts.rename_axis(index="PID", columns="cell_type").stack().rename("cell_count")
    long = long.reset_index()
    totals = long.groupby("PID", observed=True)["cell_count"].transform("sum")
    long["cell_fraction"] = long["cell_count"] / totals
    long["cell_percent"] = 100.0 * long["cell_fraction"]

    core_summary = (
        cells.groupby(["PID", "core"], observed=True)
        .size()
        .rename("cell_count")
        .reset_index()
    )
    return wide, long, core_summary


def patient_order(summary: pd.DataFrame) -> list[str]:
    return (
        summary.sort_values(["total_cells", "PID"], kind="mergesort")["PID"]
        .astype(str)
        .tolist()
    )


def sparse_patient_ticks(ax: plt.Axes, patients: list[str]) -> None:
    step = max(1, int(np.ceil(len(patients) / 20)))
    positions = np.arange(0, len(patients), step)
    ax.set_xticks(positions)
    ax.set_xticklabels([patients[index] for index in positions], rotation=60, ha="right")


def plot_composition(
    long: pd.DataFrame,
    patients: list[str],
    cell_types: list[str],
    output: Path,
    dpi: int,
) -> None:
    pivot = long.pivot(index="PID", columns="cell_type", values="cell_percent")
    pivot = pivot.reindex(index=patients, columns=cell_types, fill_value=0.0)
    x = np.arange(len(patients))
    bottom = np.zeros(len(patients), dtype=float)
    fig, ax = plt.subplots(figsize=(18, 7.5))
    for cell_type in cell_types:
        values = pivot[cell_type].to_numpy(dtype=float)
        ax.bar(
            x,
            values,
            bottom=bottom,
            width=1.0,
            color=CELL_TYPE_COLORS.get(cell_type, "#BAB0AC"),
            label=cell_type,
            linewidth=0,
            rasterized=True,
        )
        bottom += values
    ax.set(
        title="Cell-type composition for all BZ patients",
        xlabel="Patients ordered by total cell count",
        ylabel="Cell composition (%)",
        xlim=(-0.5, len(patients) - 0.5),
        ylim=(0, 100),
    )
    sparse_patient_ticks(ax, patients)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(
        title="Cell type",
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def plot_cell_counts(
    summary: pd.DataFrame, patients: list[str], output: Path, dpi: int
) -> None:
    indexed = summary.set_index("PID").reindex(patients)
    values = indexed["total_cells"].to_numpy(dtype=int)
    median = float(np.median(values))
    x = np.arange(len(patients))
    fig, ax = plt.subplots(figsize=(18, 7.0))
    ax.bar(x, values, width=1.0, color="#4E79A7", linewidth=0, rasterized=True)
    ax.axhline(
        median,
        color="black",
        linestyle="--",
        linewidth=1.3,
        label=f"Patient median = {median:,.0f} cells",
    )
    ax.set(
        title="Number of cells for all BZ patients",
        xlabel="Patients ordered by total cell count",
        ylabel="Number of cells across all available cores",
        xlim=(-0.5, len(patients) - 0.5),
        ylim=(0, None),
    )
    sparse_patient_ticks(ax, patients)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")
    args.outdir.mkdir(parents=True, exist_ok=True)
    cells, source = load_cell_observations(args.h5ad)
    cell_types = cell_type_order(args.label_map, set(cells["cell_type"]))
    summary, long, core_summary = summarize_patients(cells, cell_types)
    patients = patient_order(summary)

    summary.to_csv(args.outdir / "patient_cell_type_counts.csv", index=False)
    long.to_csv(args.outdir / "patient_cell_type_composition_long.csv", index=False)
    core_summary.to_csv(args.outdir / "patient_core_cell_counts.csv", index=False)
    plot_composition(
        long,
        patients,
        cell_types,
        args.outdir / "patient_cell_type_composition.png",
        args.dpi,
    )
    plot_cell_counts(
        summary,
        patients,
        args.outdir / "patient_cell_counts.png",
        args.dpi,
    )

    audit = {
        **source,
        "h5ad": str(args.h5ad.resolve()),
        "patients": int(summary["PID"].nunique()),
        "cores": int(core_summary["core"].nunique()),
        "total_cells": int(summary["total_cells"].sum()),
        "cell_types": cell_types,
        "patient_order": "ascending total_cells, then PID",
        "composition_definition": "cell-type percentage among all cells across each patient's available BZ cores",
        "unknown_definition": "missing cluster_preprocessed values",
        "outputs": {
            "composition_plot": "patient_cell_type_composition.png",
            "cell_count_plot": "patient_cell_counts.png",
            "wide_counts": "patient_cell_type_counts.csv",
            "long_composition": "patient_cell_type_composition_long.csv",
            "core_counts": "patient_core_cell_counts.csv",
        },
    }
    (args.outdir / "analysis_summary.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
