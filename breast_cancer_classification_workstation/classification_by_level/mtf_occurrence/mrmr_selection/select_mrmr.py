#!/usr/bin/env python3
"""Select predictive count features with Feature-engine MRMR at each TRI level."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import feature_engine
import numpy as np
import pandas as pd
from feature_engine.selection import MRMR
from scipy import sparse
from sklearn.feature_selection import mutual_info_classif


OUTPUT_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACTS = OUTPUT_DIR.parents[1] / "artifacts"
DEFAULT_LEVELS = tuple(range(2, 8))
TASKS = ("multiclass", "binary_01_vs_2")


def target_for_task(grades: pd.Series, task: str) -> pd.Series:
    """Return the supervised target used for selection."""
    if task == "multiclass":
        return grades.astype(int).reset_index(drop=True)
    if task == "binary_01_vs_2":
        return grades.eq(2).astype(np.int8).reset_index(drop=True)
    raise ValueError(f"Unknown task: {task}")


def select_level(
    artifacts_root: Path,
    output_dir: Path,
    level: int,
    *,
    feature_count: int,
    candidate_count: int,
    min_prevalence: float,
    random_state: int,
    n_jobs: int,
    task: str = "multiclass",
) -> dict[str, object]:
    source = artifacts_root / f"tri_{level}"
    samples = pd.read_csv(source / "samples.csv")
    features = pd.read_csv(source / "feature_index.csv.gz")
    matrix = sparse.load_npz(source / "motif_counts.npz").tocsr()

    labeled = (
        samples["is_labeled"].astype(bool).to_numpy()
        & samples["grade"].notna().to_numpy()
    )
    target = target_for_task(samples.loc[labeled, "grade"], task)
    labeled_counts = matrix[labeled].tocsr()

    # Prevalence is only a filter. All relevance and MRMR calculations below use
    # the original integer occurrence counts, without converting them to 0/1.
    minimum_count = max(1, math.ceil(min_prevalence * len(target)))
    labeled_document_frequency = np.asarray(
        (labeled_counts > 0).sum(axis=0)
    ).ravel()
    eligible = np.flatnonzero(labeled_document_frequency >= minimum_count)
    if eligible.size < feature_count:
        raise ValueError(
            f"TRI {level} has only {eligible.size} eligible motifs, fewer than "
            f"the requested {feature_count}"
        )

    # Feature-engine MRMR requires a dense pandas DataFrame. Screen very wide
    # levels by count-based target MI before constructing that DataFrame.
    eligible_counts = labeled_counts[:, eligible].tocsc()
    screening_relevance = mutual_info_classif(
        eligible_counts,
        target,
        discrete_features=True,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    screened_count = min(candidate_count, eligible.size)
    screening_order = np.lexsort((eligible, -screening_relevance))[:screened_count]
    candidates = eligible[screening_order]

    candidate_names = [f"feature_{index}" for index in candidates]
    candidate_frame = pd.DataFrame(
        labeled_counts[:, candidates].toarray().astype(np.int32, copy=False),
        columns=candidate_names,
    )
    selector = MRMR(
        method="MID",
        max_features=feature_count,
        discrete_features=True,
        regression=False,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    selector.fit(candidate_frame, target)
    selected_candidate_positions = selector.get_support(indices=True)
    selected = candidates[selected_candidate_positions]
    selected_relevance = selector.relevance_[selected_candidate_positions]

    selected_features = features.iloc[selected].copy().reset_index(drop=True)
    selected_features.insert(0, "feature_engine_output_order", np.arange(1, len(selected) + 1))
    selected_features["target_mutual_information"] = selected_relevance
    selected_features["selection_method"] = "feature_engine.MRMR_MID"
    selected_features["selection_task"] = task
    selected_features.to_csv(output_dir / f"tri_{level}_selected_features.csv", index=False)

    selected_counts = pd.DataFrame(
        matrix[:, selected].toarray().astype(np.int64),
        columns=[f"motif_{value}" for value in selected_features["wl_hash"]],
    )
    occurrence_table = pd.concat(
        [samples.reset_index(drop=True), selected_counts], axis=1
    )
    occurrence_table.to_csv(
        output_dir / f"tri_{level}_selected_mtf_occurrence.csv", index=False
    )

    return {
        "triangle_level": level,
        "labeled_samples": int(labeled.sum()),
        "minimum_labeled_prevalence": minimum_count,
        "eligible_features": int(eligible.size),
        "screened_candidates": screened_count,
        "selected_features": int(len(selected)),
        "mrmr_method": "MID",
        "mrmr_input": "original_occurrence_counts",
        "selection_task": task,
        "feature_engine_version": feature_engine.__version__,
        "maximum_selected_target_mutual_information": float(selected_relevance.max()),
        "mean_selected_target_mutual_information": float(selected_relevance.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--levels", type=int, nargs="+", default=DEFAULT_LEVELS)
    parser.add_argument("--features", type=int, default=100)
    parser.add_argument("--candidates", type=int, default=1000)
    parser.add_argument("--min-prevalence", type=float, default=0.05)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--task", choices=TASKS, default="multiclass")
    args = parser.parse_args()

    if args.features < 2:
        parser.error("--features must be at least 2")
    if args.candidates < args.features:
        parser.error("--candidates must be at least --features")
    if not 0 < args.min_prevalence <= 1:
        parser.error("--min-prevalence must be in (0, 1]")

    args.output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for level in sorted(set(args.levels)):
        print(f"Selecting TRI {level} from original frequencies...", flush=True)
        summary = select_level(
            args.artifacts,
            args.output,
            level,
            feature_count=args.features,
            candidate_count=args.candidates,
            min_prevalence=args.min_prevalence,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
            task=args.task,
        )
        summaries.append(summary)
        print(
            f"  Feature-engine MRMR selected {summary['selected_features']} of "
            f"{summary['screened_candidates']:,} screened count features",
            flush=True,
        )
    pd.DataFrame(summaries).to_csv(args.output / "selection_summary.csv", index=False)


if __name__ == "__main__":
    main()
