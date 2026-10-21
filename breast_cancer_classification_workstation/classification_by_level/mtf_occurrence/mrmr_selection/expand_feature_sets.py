#!/usr/bin/env python3
"""Create large nested feature sets from the Feature-engine MRMR core and count MI."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_selection import mutual_info_classif


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACTS = PROJECT_DIR.parents[1] / "artifacts"
DEFAULT_OUTPUT = PROJECT_DIR / "expanded_feature_sets"
TASKS = ("multiclass", "binary_01_vs_2")


def target_for_task(grades: pd.Series, task: str) -> np.ndarray:
    if task == "multiclass":
        return grades.astype(int).to_numpy()
    if task == "binary_01_vs_2":
        return grades.eq(2).astype(np.int8).to_numpy()
    raise ValueError(f"Unknown task: {task}")


def expand_level(
    artifacts: Path,
    core_dir: Path,
    output_dir: Path,
    level: int,
    *,
    sizes: tuple[int, ...],
    task: str,
    seed: int,
    n_jobs: int,
) -> list[dict[str, object]]:
    """Create nested expanded feature sets for one TRI level."""
    source = artifacts / f"tri_{level}"
    samples = pd.read_csv(source / "samples.csv")
    features = pd.read_csv(source / "feature_index.csv.gz")
    matrix = sparse.load_npz(source / "motif_counts.npz").tocsr()
    labeled = (
        samples["is_labeled"].astype(bool).to_numpy()
        & samples["grade"].notna().to_numpy()
    )
    target = target_for_task(samples.loc[labeled, "grade"], task)
    labeled_counts = matrix[labeled].tocsc()
    relevance = mutual_info_classif(
        labeled_counts,
        target,
        discrete_features=True,
        random_state=seed,
        n_jobs=n_jobs,
    )

    core_path = core_dir / f"tri_{level}_selected_features.csv"
    core = pd.read_csv(core_path)
    if "selection_task" in core and set(core["selection_task"]) != {task}:
        raise ValueError(f"{core_path} was not selected for task {task}")
    core_indices = core["feature_index"].astype(int).to_numpy()
    is_core = np.zeros(matrix.shape[1], dtype=bool)
    is_core[core_indices] = True
    remainder = np.flatnonzero(~is_core)
    remainder_order = np.lexsort((remainder, -relevance[remainder]))
    nested_order = np.concatenate([core_indices, remainder[remainder_order]])

    summary_rows: list[dict[str, object]] = []
    for requested_size in sorted(set(sizes)):
        actual_size = min(requested_size, matrix.shape[1])
        selected_indices = nested_order[:actual_size]
        selected = features.iloc[selected_indices].copy().reset_index(drop=True)
        selected.insert(0, "selection_order", np.arange(1, actual_size + 1))
        selected["target_mutual_information"] = relevance[selected_indices]
        selected["selection_source"] = np.where(
            np.arange(actual_size) < len(core_indices),
            "feature_engine_mrmr_core",
            "raw_count_mutual_information_expansion",
        )
        selected["selection_task"] = task
        selected["requested_feature_count"] = requested_size
        selected["actual_feature_count"] = actual_size
        selected.to_csv(
            output_dir / f"tri_{level}_features_{requested_size}.csv", index=False
        )
        summary_rows.append(
            {
                "triangle_level": level,
                "selection_task": task,
                "requested_feature_count": requested_size,
                "actual_feature_count": actual_size,
                "available_motifs": matrix.shape[1],
                "feature_engine_mrmr_core": min(len(core_indices), actual_size),
                "mi_expansion_features": max(0, actual_size - len(core_indices)),
            }
        )
        print(f"  requested {requested_size:,}; retained {actual_size:,}", flush=True)
    return summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--core", type=Path, default=PROJECT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--levels", type=int, nargs="+", default=range(2, 8))
    parser.add_argument("--sizes", type=int, nargs="+", default=(1000, 3000, 5000))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--task", choices=TASKS, default="multiclass")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for level in sorted(set(args.levels)):
        print(f"Ranking raw-count motifs for TRI {level}...", flush=True)
        summary_rows.extend(
            expand_level(
                args.artifacts,
                args.core,
                args.output,
                level,
                sizes=tuple(args.sizes),
                task=args.task,
                seed=args.seed,
                n_jobs=args.n_jobs,
            )
        )
    pd.DataFrame(summary_rows).to_csv(args.output / "feature_set_summary.csv", index=False)


if __name__ == "__main__":
    main()
