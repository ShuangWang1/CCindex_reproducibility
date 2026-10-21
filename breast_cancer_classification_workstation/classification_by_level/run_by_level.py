#!/usr/bin/env python3
"""Evaluate METABRIC motif-frequency classifiers one TRI level at a time."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "metabric-level-matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse


PROJECT_DIR = Path(__file__).resolve().parent
WORKSTATION_DIR = PROJECT_DIR.parent
sys.path.insert(0, str(WORKSTATION_DIR))

from metabric_ml import create_report, evaluate_baselines  # noqa: E402


DEFAULT_SOURCE_ARTIFACTS = WORKSTATION_DIR / "artifacts" / "metabric"
DEFAULT_ARTIFACTS = PROJECT_DIR / "artifacts"
DEFAULT_RESULTS = PROJECT_DIR / "results"
DEFAULT_LEVELS = tuple(range(2, 8))
TASK_NAMES = {
    "multiclass": "Grade 0/1/2",
    "binary_01_vs_2": "Grade 0/1 vs grade 2",
}


def _validate_levels(levels: tuple[int, ...]) -> None:
    if not levels:
        raise ValueError("At least one TRI level is required")
    if tuple(sorted(set(levels))) != levels:
        raise ValueError("TRI levels must be unique and in ascending order")


def prepare_level_artifacts(
    source_dir: Path,
    output_root: Path,
    levels: tuple[int, ...],
) -> None:
    """Slice the existing prepared matrix without changing samples or counts."""
    _validate_levels(levels)
    matrix = sparse.load_npz(source_dir / "motif_counts.npz").tocsr()
    feature_levels = np.load(source_dir / "feature_triangle_levels.npy")
    feature_index = pd.read_csv(source_dir / "feature_index.csv.gz")
    with (source_dir / "audit.json").open(encoding="utf-8") as source:
        source_audit = json.load(source)

    if matrix.shape[1] != len(feature_levels) or len(feature_index) != len(feature_levels):
        raise ValueError("Source feature artifacts have inconsistent dimensions")

    available = set(int(value) for value in np.unique(feature_levels))
    unavailable = set(levels) - available
    if unavailable:
        raise ValueError(f"TRI levels are absent from source artifacts: {sorted(unavailable)}")

    output_root.mkdir(parents=True, exist_ok=True)
    for level in levels:
        destination = output_root / f"tri_{level}"
        destination.mkdir(parents=True, exist_ok=True)
        mask = feature_levels == level
        level_matrix = matrix[:, mask].tocsr()
        level_features = feature_index.loc[mask].copy().reset_index(drop=True)
        level_features["feature_index"] = np.arange(len(level_features), dtype=np.int64)

        sparse.save_npz(destination / "motif_counts.npz", level_matrix, compressed=True)
        np.save(
            destination / "feature_triangle_levels.npy",
            np.full(level_matrix.shape[1], level, dtype=feature_levels.dtype),
        )
        level_features.to_csv(
            destination / "feature_index.csv.gz", index=False, compression="gzip"
        )
        shutil.copy2(source_dir / "samples.csv", destination / "samples.csv")

        missing_suffix = f"ccindex_tri_{level}.csv"
        audit = {
            **source_audit,
            "source_artifacts": str(source_dir.resolve()),
            "triangle_levels": [level],
            "feature_count": int(level_matrix.shape[1]),
            "nonzero_count": int(level_matrix.nnz),
            "matrix_shape": list(level_matrix.shape),
            "search_exhausted_missing_level_files": [
                path
                for path in source_audit.get("search_exhausted_missing_level_files", [])
                if str(path).endswith(missing_suffix)
            ],
        }
        with (destination / "audit.json").open("w", encoding="utf-8") as output:
            json.dump(audit, output, indent=2)
            output.write("\n")
        print(
            f"Prepared TRI {level}: {level_matrix.shape[1]:,} features, "
            f"{level_matrix.nnz:,} nonzeros"
        )


def evaluate_levels(
    artifacts_root: Path,
    results_root: Path,
    levels: tuple[int, ...],
    *,
    folds: int,
    repeats: int,
    seed: int,
    min_prevalence: float,
    max_features: int,
    trees: int,
    n_jobs: int,
) -> None:
    for level in levels:
        print(f"\n=== Evaluating TRI {level} ===")
        evaluate_baselines(
            artifacts_root / f"tri_{level}",
            results_root / f"tri_{level}",
            folds=folds,
            repeats=repeats,
            seed=seed,
            min_prevalence=min_prevalence,
            max_features=max_features,
            trees=trees,
            n_jobs=n_jobs,
        )


def _combined_metrics(results_root: Path, levels: tuple[int, ...]) -> pd.DataFrame:
    frames = []
    for level in levels:
        frame = pd.read_csv(results_root / f"tri_{level}" / "summary_metrics.csv")
        frame.insert(0, "triangle_level", level)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _best_level_rows(all_metrics: pd.DataFrame) -> pd.DataFrame:
    motif = all_metrics[all_metrics["representation"].isin(["raw_log", "level_relative"])]
    ordered = motif.sort_values(
        ["task", "triangle_level", "roc_auc_mean"],
        ascending=[True, True, False],
    )
    return ordered.groupby(["task", "triangle_level"], as_index=False).first()


def _plot_level_comparison(best: pd.DataFrame, path: Path) -> None:
    tasks = list(TASK_NAMES)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    for ax, task in zip(axes, tasks):
        subset = best[best["task"] == task].sort_values("triangle_level")
        ax.errorbar(
            subset["triangle_level"],
            subset["roc_auc_mean"],
            yerr=subset["roc_auc_std"].fillna(0),
            marker="o",
            linewidth=2,
            capsize=4,
            color="#2878B5",
        )
        for row in subset.itertuples(index=False):
            ax.annotate(
                f"{row.roc_auc_mean:.3f}",
                (row.triangle_level, row.roc_auc_mean),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
        ax.axhline(0.5, color="#888888", linestyle=":", linewidth=1.5)
        ax.set_xticks(subset["triangle_level"])
        ax.set_ylim(0.4, 0.8)
        ax.set_xlabel("TRI level used as the only motif input")
        ax.set_title(TASK_NAMES[task])
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Mean fold ROC AUC ± SD")
    fig.suptitle("METABRIC motif classification by individual TRI level")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = [
        "triangle_level",
        "task",
        "representation",
        "model",
        "roc_auc_mean",
        "roc_auc_std",
        "average_precision_mean",
        "macro_f1_mean",
        "balanced_accuracy_mean",
        "accuracy_mean",
    ]
    display = frame[columns].copy()
    for column in columns[4:]:
        display[column] = display[column].map(lambda value: f"{value:.3f}")
    rows = ["| " + " | ".join(columns) + " |"]
    rows.append("| " + " | ".join("---" for _ in columns) + " |")
    rows.extend(
        "| " + " | ".join(str(value) for value in values) + " |"
        for values in display.itertuples(index=False, name=None)
    )
    return "\n".join(rows)


def _feature_markdown_table(rows: list[dict[str, object]]) -> str:
    columns = ["TRI level", "features", "nonzeros", "missing exhausted files"]
    lines = ["| " + " | ".join(columns) + " |"]
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    lines.extend(
        "| " + " | ".join(str(row[column]) for column in columns) + " |"
        for row in rows
    )
    return "\n".join(lines)


def _finding_lines(best: pd.DataFrame) -> list[str]:
    lines = []
    for task, display_name in TASK_NAMES.items():
        winner = (
            best[best["task"] == task]
            .sort_values("roc_auc_mean", ascending=False)
            .iloc[0]
        )
        lines.append(
            f"- **{display_name}:** TRI {int(winner['triangle_level'])} was highest at "
            f"{winner['roc_auc_mean']:.3f} ± {winner['roc_auc_std']:.3f} ROC AUC "
            f"using `{winner['representation']} / {winner['model']}`."
        )
    return lines


def _original_comparison(best: pd.DataFrame) -> str:
    original_path = WORKSTATION_DIR / "results" / "metabric" / "summary_metrics.csv"
    if not original_path.is_file():
        return ""
    original = pd.read_csv(original_path)
    original = original[
        original["representation"].isin(["raw_log", "level_relative"])
    ]
    original = (
        original.sort_values(["task", "roc_auc_mean"], ascending=[True, False])
        .groupby("task", as_index=False)
        .first()
    )
    rows = []
    for task, display_name in TASK_NAMES.items():
        level_row = (
            best[best["task"] == task]
            .sort_values("roc_auc_mean", ascending=False)
            .iloc[0]
        )
        original_row = original[original["task"] == task].iloc[0]
        rows.append(
            {
                "task": display_name,
                "best isolated level": f"TRI {int(level_row['triangle_level'])}",
                "isolated AUC": f"{level_row['roc_auc_mean']:.3f} ± {level_row['roc_auc_std']:.3f}",
                "all-level AUC": f"{original_row['roc_auc_mean']:.3f} ± {original_row['roc_auc_std']:.3f}",
                "AUC difference": f"{level_row['roc_auc_mean'] - original_row['roc_auc_mean']:+.3f}",
            }
        )
    columns = list(rows[0])
    lines = ["| " + " | ".join(columns) + " |"]
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    lines.extend(
        "| " + " | ".join(str(row[column]) for column in columns) + " |"
        for row in rows
    )
    return """## Comparison with the original all-level run

The original run combined TRI 2 through TRI 7 before its training-fold feature
selection. Using only the best isolated level gave the following descriptive change:

{table}

These are exploratory comparisons on the same repeated folds. Fold standard deviations
overlap and folds are correlated across repeats, so the differences should not be read as
a formal significance test.
""".format(table="\n".join(lines))


def create_combined_report(
    artifacts_root: Path,
    results_root: Path,
    levels: tuple[int, ...],
) -> Path:
    """Create per-level diagnostics and a concise cross-level comparison."""
    results_root.mkdir(parents=True, exist_ok=True)
    for level in levels:
        create_report(artifacts_root / f"tri_{level}", results_root / f"tri_{level}")

    all_metrics = _combined_metrics(results_root, levels)
    all_metrics.to_csv(results_root / "all_model_metrics.csv", index=False)
    best = _best_level_rows(all_metrics)
    best.to_csv(results_root / "level_summary.csv", index=False)
    comparison_path = results_root / "roc_auc_by_level.png"
    _plot_level_comparison(best, comparison_path)

    with (results_root / f"tri_{levels[0]}" / "evaluation.json").open(
        encoding="utf-8"
    ) as source:
        evaluation = json.load(source)

    feature_rows = []
    for level in levels:
        with (artifacts_root / f"tri_{level}" / "audit.json").open(
            encoding="utf-8"
        ) as source:
            audit = json.load(source)
        feature_rows.append(
            {
                "TRI level": level,
                "features": f"{audit['feature_count']:,}",
                "nonzeros": f"{audit['nonzero_count']:,}",
                "missing exhausted files": len(
                    audit.get("search_exhausted_missing_level_files", [])
                ),
            }
        )
    feature_table = _feature_markdown_table(feature_rows)

    diagnostics = "\n".join(
        f"- TRI {level}: [multiclass ROC](tri_{level}/figures/roc_multiclass.png), "
        f"[binary ROC](tri_{level}/figures/roc_binary_01_vs_2.png), "
        f"[full report](tri_{level}/metabric_report.md)"
        for level in levels
    )
    findings = "\n".join(_finding_lines(best))
    original_comparison = _original_comparison(best)
    report = f"""# METABRIC classification by individual TRI level

Each experiment uses motif occurrence frequencies from exactly one TRI level. Samples,
labels, model families, representations, fold assignments, and leakage-safe training-fold
feature selection match the original workstation. Grade 2 is the positive class in the
binary task.

## Validation

- Samples evaluated: 487 labeled images (grade counts: 37 / 181 / 269)
- Cross-validation: {evaluation['folds']}-fold stratified CV repeated {evaluation['repeats']} times
- Random seed: {evaluation['seed']}
- Primary selection metric: mean fold ROC AUC
- Feature cap: {evaluation['max_features']:,}; minimum training prevalence: {evaluation['min_prevalence']:.1%}

![Best ROC AUC by TRI level](roc_auc_by_level.png)

## Main findings

{findings}

- TRI 2–5 were tightly grouped for both tasks; performance declined at TRI 6–7,
  despite their much larger vocabularies.
- The best isolated-level AUCs remain modest. Level isolation improves the original
  binary AUC descriptively, but it does not turn this into a high-accuracy classifier.

{original_comparison}

## Best motif model at each level

{_markdown_table(best)}

## Input size by level

{feature_table}

## Per-level diagnostics

{diagnostics}

## Interpretation notes

- Results compare isolated levels; they do not combine TRI 2 through TRI 7 features.
- `raw_log` is log(1 + motif count). For a single-level experiment, `level_relative`
  divides every motif count by that sample's total motif count at that TRI level.
- Model selection and performance reporting use the same repeated cross-validation run,
  so comparisons are exploratory rather than an untouched external-test estimate.
- Validation remains image/sample-level, as in the original run. Images from the same
  patient can occur in training and validation folds, which can make results optimistic.
"""
    report_path = results_root / "classification_by_level_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"Saved combined report to {report_path}")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("all", "prepare", "evaluate", "report"), nargs="?", default="all"
    )
    parser.add_argument("--source-artifacts", type=Path, default=DEFAULT_SOURCE_ARTIFACTS)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--levels", type=int, nargs="+", default=list(DEFAULT_LEVELS))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20_260_818)
    parser.add_argument("--min-prevalence", type=float, default=0.05)
    parser.add_argument("--max-features", type=int, default=5_000)
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    levels = tuple(args.levels)
    _validate_levels(levels)
    if args.command in ("all", "prepare"):
        prepare_level_artifacts(args.source_artifacts, args.artifacts, levels)
    if args.command in ("all", "evaluate"):
        evaluate_levels(
            args.artifacts,
            args.results,
            levels,
            folds=args.folds,
            repeats=args.repeats,
            seed=args.seed,
            min_prevalence=args.min_prevalence,
            max_features=args.max_features,
            trees=args.trees,
            n_jobs=args.n_jobs,
        )
    if args.command in ("all", "report"):
        create_combined_report(args.artifacts, args.results, levels)


if __name__ == "__main__":
    main()
