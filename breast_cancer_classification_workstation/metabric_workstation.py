#!/usr/bin/env python3
"""Prepare, evaluate, and report METABRIC motif classification baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

from metabric_ml import create_report, evaluate_baselines, prepare_metabric_features


WORKSTATION_DIR = Path(__file__).resolve().parent
REPO_DIR = WORKSTATION_DIR.parent
DEFAULT_MOTIFS = REPO_DIR / "breast_cancer_meta" / "metabric_res_threshold_10_cpp_0811"
DEFAULT_GRADES = WORKSTATION_DIR / "meta_grade.csv"
DEFAULT_H5AD = REPO_DIR / "breast_cancer" / "metabric.h5ad"
DEFAULT_ARTIFACTS = WORKSTATION_DIR / "artifacts" / "metabric"
DEFAULT_RESULTS = WORKSTATION_DIR / "results" / "metabric"


def _add_prepare_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--motif-root", type=Path, default=DEFAULT_MOTIFS)
    parser.add_argument("--grades", type=Path, default=DEFAULT_GRADES)
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_H5AD)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument(
        "--triangle-levels",
        type=int,
        nargs="+",
        default=[2, 3, 4, 5, 6, 7],
        help="Triangle levels to include (default: 2 3 4 5 6 7)",
    )


def _add_evaluation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20_260_818)
    parser.add_argument("--min-prevalence", type=float, default=0.05)
    parser.add_argument("--max-features", type=int, default=5_000)
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--n-jobs", type=int, default=-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="Build sparse motif features")
    _add_prepare_arguments(prepare)

    evaluate = commands.add_parser("evaluate", help="Run fixed repeated-CV baselines")
    _add_evaluation_arguments(evaluate)

    report = commands.add_parser("report", help="Create plots and Markdown report")
    report.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    report.add_argument("--results", type=Path, default=DEFAULT_RESULTS)

    all_steps = commands.add_parser("all", help="Run prepare, evaluate, and report")
    _add_prepare_arguments(all_steps)
    all_steps.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    all_steps.add_argument("--folds", type=int, default=5)
    all_steps.add_argument("--repeats", type=int, default=5)
    all_steps.add_argument("--seed", type=int, default=20_260_818)
    all_steps.add_argument("--min-prevalence", type=float, default=0.05)
    all_steps.add_argument("--max-features", type=int, default=5_000)
    all_steps.add_argument("--trees", type=int, default=300)
    all_steps.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def _prepare(args: argparse.Namespace) -> None:
    prepare_metabric_features(
        motif_root=args.motif_root,
        grades_path=args.grades,
        h5ad_path=args.h5ad,
        output_dir=args.artifacts,
        triangle_levels=tuple(args.triangle_levels),
    )


def _evaluate(args: argparse.Namespace) -> None:
    evaluate_baselines(
        artifacts_dir=args.artifacts,
        results_dir=args.results,
        folds=args.folds,
        repeats=args.repeats,
        seed=args.seed,
        min_prevalence=args.min_prevalence,
        max_features=args.max_features,
        trees=args.trees,
        n_jobs=args.n_jobs,
    )


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        _prepare(args)
    elif args.command == "evaluate":
        _evaluate(args)
    elif args.command == "report":
        create_report(args.artifacts, args.results)
    elif args.command == "all":
        _prepare(args)
        _evaluate(args)
        create_report(args.artifacts, args.results)
    else:  # pragma: no cover
        raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
