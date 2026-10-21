#!/usr/bin/env python3
"""Prepare BZ motifs and run binary expanded-style mRMR parameter tuning."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
MRMR_DIR = PROJECT_DIR.parent
WORKSTATION_DIR = MRMR_DIR.parents[2]
REPO_DIR = WORKSTATION_DIR.parent
TUNING_DIR = MRMR_DIR / "parameter_tuning"
for path in (WORKSTATION_DIR, MRMR_DIR, TUNING_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from metabric_ml.data import prepare_metabric_features, read_grades  # noqa: E402
from select_mrmr import select_level  # noqa: E402
from expand_feature_sets import expand_level  # noqa: E402
import tune_mrmr_grid as tuning  # noqa: E402


DEFAULT_MOTIF_ROOT = REPO_DIR / "breast_cancer_meta" / "BZ_res_threshold_10_cpp_0811"
DEFAULT_GRADES = WORKSTATION_DIR / "bz_grade.csv"
DEFAULT_SAMPLE_MANIFEST = REPO_DIR / "breast_cancer" / "BZ_samples_intra" / "sample_manifest.csv"
DEFAULT_ARTIFACTS = PROJECT_DIR / "artifacts"
DEFAULT_SELECTIONS = PROJECT_DIR / "feature_selection"
DEFAULT_EXPANDED = PROJECT_DIR / "expanded_feature_sets"
DEFAULT_RESULTS = PROJECT_DIR / "results" / "expanded_style"
DEFAULT_PARTS_ROOT = PROJECT_DIR / "results" / "parts"
DEFAULT_LEVELS = tuple(range(2, 8))
RUN_SUMMARY_RE = re.compile(r"^sample([1-9][0-9]*)$")


def build_source_inventory(
    motif_root: Path,
    grades_path: Path,
    sample_manifest_path: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Reconstruct the authoritative inventory from completed sample directories."""
    grades = read_grades(grades_path)
    rows: list[dict[str, object]] = []
    for sample_dir in motif_root.iterdir():
        match = RUN_SUMMARY_RE.fullmatch(sample_dir.name)
        if not sample_dir.is_dir() or match is None:
            continue
        summary_path = sample_dir / "run_summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"Missing run summary: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("sample") != sample_dir.name:
            raise ValueError(f"Sample mismatch in {summary_path}")
        source_sample = str(summary.get("source_sample", "")).strip()
        if source_sample not in grades or grades[source_sample] is None:
            raise ValueError(f"Missing usable grade for {source_sample!r}")
        completed_level = int(summary["completed_triangle_level"])
        for level in range(2, completed_level + 1):
            path = sample_dir / f"ccindex_tri_{level}.csv"
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing TRI {level} at or below completed level: {path}"
                )
        rows.append(
            {
                "dataset": summary["dataset"],
                "sample": sample_dir.name,
                "source_sample": source_sample,
                "graph_file": summary["graph_file"],
                "nodes": int(summary["nodes"]),
                "edges": int(summary["edges"]),
                "source_triangle_types": int(summary["source_triangle_types"]),
                "retained_triangle_types": int(summary["retained_triangle_types"]),
                "completed_triangle_level": completed_level,
                "seconds": float(summary["seconds"]),
                "_sample_number": int(match.group(1)),
                "_grade": int(grades[source_sample]),
            }
        )
    if not rows:
        raise ValueError(f"No completed sample directories found below {motif_root}")
    rows.sort(key=lambda row: int(row["_sample_number"]))
    inventory = pd.DataFrame(rows)
    if inventory["sample"].duplicated().any() or inventory["source_sample"].duplicated().any():
        raise ValueError("BZ inventory contains duplicate sample identifiers")

    expected = pd.read_csv(sample_manifest_path, dtype={"source_sample": str})
    expected_sources = set(expected["source_sample"])
    inventory_sources = set(inventory["source_sample"])
    unknown = sorted(inventory_sources - expected_sources)
    if unknown:
        raise ValueError(f"Completed BZ samples are absent from sample manifest: {unknown}")
    excluded = expected.loc[
        ~expected["source_sample"].isin(inventory_sources),
        ["sample_index", "gml_file", "source_sample"],
    ].sort_values("sample_index")

    output_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = output_dir / "source_inventory.csv"
    excluded_path = output_dir / "excluded_samples.csv"
    inventory.drop(columns=["_sample_number", "_grade"]).to_csv(
        inventory_path, index=False
    )
    excluded.to_csv(excluded_path, index=False)

    grade_counts = inventory["_grade"].value_counts().sort_index()
    binary_counts = {
        "0": int(grade_counts.get(0, 0) + grade_counts.get(1, 0)),
        "1": int(grade_counts.get(2, 0)),
    }
    metadata = {
        "source_motif_root": str(motif_root.resolve()),
        "source_inventory": str(inventory_path.resolve()),
        "source_top_level_manifest_rows": int(
            len(pd.read_csv(motif_root / "run_manifest.csv"))
        ),
        "labeled_sample_count": int(len(inventory)),
        "original_grade_class_counts": {
            str(label): int(grade_counts.get(label, 0)) for label in range(3)
        },
        "binary_class_counts": binary_counts,
        "excluded_missing_motif_samples": int(len(excluded)),
        "exhausted_before_tri_7_samples": int(
            (inventory["completed_triangle_level"] < 7).sum()
        ),
    }
    metadata_path = output_dir / "source_inventory_audit.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return inventory_path, metadata_path


def prepare(args: argparse.Namespace) -> None:
    inventory, _ = build_source_inventory(
        args.motif_root, args.grades, args.sample_manifest, args.artifacts
    )
    for level in args.levels:
        destination = args.artifacts / f"tri_{level}"
        required = (
            destination / "motif_counts.npz",
            destination / "samples.csv",
            destination / "feature_index.csv.gz",
            destination / "audit.json",
        )
        if not args.force and all(path.is_file() for path in required):
            print(f"Skipping prepared TRI {level}; artifacts already exist", flush=True)
            continue
        prepare_metabric_features(
            motif_root=args.motif_root,
            grades_path=args.grades,
            output_dir=destination,
            triangle_levels=(level,),
            manifest_path=inventory,
        )


def select(args: argparse.Namespace) -> None:
    args.selections.mkdir(parents=True, exist_ok=True)
    summaries = []
    for level in args.levels:
        output = args.selections / f"tri_{level}_selected_features.csv"
        if not args.force and output.is_file():
            existing = pd.read_csv(output)
            if set(existing.get("selection_task", [])) == {"binary_01_vs_2"}:
                print(f"Skipping selected TRI {level}; binary selection exists", flush=True)
                continue
        print(f"Selecting binary mRMR core for TRI {level}...", flush=True)
        summaries.append(
            select_level(
                args.artifacts,
                args.selections,
                level,
                feature_count=100,
                candidate_count=1000,
                min_prevalence=0.05,
                random_state=args.seed,
                n_jobs=args.n_jobs,
                task="binary_01_vs_2",
            )
        )
    if summaries:
        summary_path = args.selections / "selection_summary.csv"
        old = pd.read_csv(summary_path) if summary_path.is_file() else pd.DataFrame()
        combined = pd.concat([old, pd.DataFrame(summaries)], ignore_index=True)
        combined = combined.drop_duplicates("triangle_level", keep="last").sort_values(
            "triangle_level"
        )
        combined.to_csv(summary_path, index=False)


def expand(args: argparse.Namespace) -> None:
    args.expanded.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    for level in args.levels:
        outputs = [args.expanded / f"tri_{level}_features_{size}.csv" for size in (1000, 3000, 5000)]
        if not args.force and all(path.is_file() for path in outputs):
            print(f"Skipping expanded TRI {level}; feature sets exist", flush=True)
            continue
        print(f"Expanding binary-ranked TRI {level} feature sets...", flush=True)
        summaries.extend(
            expand_level(
                args.artifacts,
                args.selections,
                args.expanded,
                level,
                sizes=(1000, 3000, 5000),
                task="binary_01_vs_2",
                seed=args.seed,
                n_jobs=args.n_jobs,
            )
        )
    if summaries:
        summary_path = args.expanded / "feature_set_summary.csv"
        old = pd.read_csv(summary_path) if summary_path.is_file() else pd.DataFrame()
        combined = pd.concat([old, pd.DataFrame(summaries)], ignore_index=True)
        combined = combined.drop_duplicates(
            ["triangle_level", "requested_feature_count"], keep="last"
        ).sort_values(["triangle_level", "requested_feature_count"])
        combined.to_csv(summary_path, index=False)


def tune(args: argparse.Namespace) -> None:
    argv = [
        "--artifacts", str(args.artifacts),
        "--feature-sets", str(args.expanded),
        "--results", str(args.results),
        "--levels", *map(str, args.levels),
        "--tasks", "binary_01_vs_2",
        "--sizes", *map(str, args.sizes),
        "--preprocessing", *args.preprocessing,
        "--models", *args.models,
        "--folds", str(args.folds),
        "--repeats", str(args.repeats),
        "--seed", str(args.seed),
        "--trees", str(args.trees),
        "--xgb-estimators", str(args.xgb_estimators),
        "--n-jobs", str(args.n_jobs),
        "--dataset", "BZ",
        "--selection-task", "binary_01_vs_2",
        "--samples-metadata", str(args.artifacts / "source_inventory_audit.json"),
    ]
    if not args.resume:
        argv.append("--no-resume")
    tuning.evaluate(tuning.parse_args(argv))


def report(args: argparse.Namespace) -> None:
    config = json.loads((args.results / "evaluation_config.json").read_text(encoding="utf-8"))
    tuning.write_report(args.results, config)


def merge(args: argparse.Namespace) -> None:
    """Merge independently checkpointed TRI-level runs into the final result set."""
    frames: list[pd.DataFrame] = []
    canonical = args.results / "fold_metrics.csv"
    if canonical.is_file():
        frames.append(pd.read_csv(canonical))
    for level in args.levels:
        part_dir = args.parts_root / f"tri_{level}"
        part = part_dir / "fold_metrics.csv"
        if part.is_file():
            frames.append(pd.read_csv(part))
            selected_dir = args.results / "selected_features"
            selected_dir.mkdir(parents=True, exist_ok=True)
            for selected in (part_dir / "selected_features").glob("*.csv"):
                shutil.copy2(selected, selected_dir / selected.name)
    if not frames:
        raise FileNotFoundError("No completed canonical or per-level fold metrics found")
    folds = pd.concat(frames, ignore_index=True)
    key = [
        "triangle_level", "task", "requested_feature_count", "preprocessing",
        "model", "repeat", "fold",
    ]
    folds = folds.drop_duplicates(key, keep="last").sort_values(key).reset_index(drop=True)
    folds = folds[folds["triangle_level"].isin(args.levels)]
    expected = (
        len(args.levels) * len(args.sizes) * len(args.preprocessing)
        * len(args.models) * args.folds * args.repeats
    )
    if len(folds) != expected:
        raise ValueError(f"Merged grid has {len(folds):,} rows; expected {expected:,}")
    args.results.mkdir(parents=True, exist_ok=True)
    folds.to_csv(args.results / "fold_metrics.partial.csv", index=False)
    folds.to_csv(args.results / "fold_metrics.csv", index=False)
    tuning.summarize_results(folds, args.results)
    config_path = args.results / "evaluation_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["levels"] = list(args.levels)
    config["sizes"] = list(args.sizes)
    config["preprocessing"] = list(args.preprocessing)
    config["models"] = list(args.models)
    config["folds"] = args.folds
    config["repeats"] = args.repeats
    config["total_fold_metric_rows"] = len(folds)
    config["results_per_configuration"] = args.folds * args.repeats
    config["execution"] = "independent TRI-level checkpoints merged by composite key"
    metadata = json.loads(
        (args.artifacts / "source_inventory_audit.json").read_text(encoding="utf-8")
    )
    config.update(metadata)
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    tuning.write_report(args.results, config)
    print(f"Merged and summarized {len(folds):,} records in {args.results}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "select", "expand", "tune", "merge", "report", "all"))
    parser.add_argument("--motif-root", type=Path, default=DEFAULT_MOTIF_ROOT)
    parser.add_argument("--grades", type=Path, default=DEFAULT_GRADES)
    parser.add_argument("--sample-manifest", type=Path, default=DEFAULT_SAMPLE_MANIFEST)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--selections", type=Path, default=DEFAULT_SELECTIONS)
    parser.add_argument("--expanded", type=Path, default=DEFAULT_EXPANDED)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--parts-root", type=Path, default=DEFAULT_PARTS_ROOT)
    parser.add_argument("--levels", type=int, nargs="+", default=DEFAULT_LEVELS)
    parser.add_argument("--sizes", type=int, nargs="+", default=(10, *range(50, 451, 50)))
    parser.add_argument("--preprocessing", nargs="+", choices=tuning.PREPROCESSORS, default=tuning.PREPROCESSORS)
    parser.add_argument("--models", nargs="+", choices=tuning.MODELS, default=tuning.MODELS)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trees", type=int, default=50)
    parser.add_argument("--xgb-estimators", type=int, default=50)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    args.levels = tuple(sorted(set(args.levels)))
    args.sizes = tuple(sorted(set(args.sizes)))
    return args


def main() -> None:
    args = parse_args()
    if args.command in ("prepare", "all"):
        prepare(args)
    if args.command in ("select", "all"):
        select(args)
    if args.command in ("expand", "all"):
        expand(args)
    if args.command in ("tune", "all"):
        tune(args)
    elif args.command == "merge":
        merge(args)
    elif args.command == "report":
        report(args)


if __name__ == "__main__":
    main()
