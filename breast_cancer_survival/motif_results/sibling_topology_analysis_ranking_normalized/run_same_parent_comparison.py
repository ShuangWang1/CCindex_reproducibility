#!/usr/bin/env python3
"""Compare rank-significant and nonsignificant TRI-5 motifs sharing TRI-4 parents."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import tempfile

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
REFERENCE_DIR = PROJECT_DIR.parent / "sibling_topology_analysis_new"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


analysis = load_module("rank_sibling_topology_analysis", PROJECT_DIR / "run_analysis.py")
engine = load_module(
    "rank_same_parent_engine", REFERENCE_DIR / "run_same_parent_comparison.py"
)
# The established engine resolves criterion metadata and graph helpers through
# its analysis module. Point it at the rank-native adapter for this run.
engine.analysis = analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, default=analysis.TRACE_DIR)
    parser.add_argument(
        "--features",
        type=Path,
        default=PROJECT_DIR / "results" / "tables" / "tri5_motif_criteria.csv.gz",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=PROJECT_DIR / "results" / "same_parent_comparsion",
    )
    parser.add_argument("--min-per-class", type=int, default=3)
    parser.add_argument("--top-criteria", type=int, default=24)
    parser.add_argument("--panels-per-page", type=int, default=12)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--motif-galleries-only", action="store_true")
    mode.add_argument("--clustered-heatmap-only", action="store_true")
    return parser.parse_args()


def rewrite_rank_native_csvs(outdir: Path) -> None:
    """Remove internal raw-column aliases from every exported CSV table."""
    for path in sorted(outdir.rglob("*.csv")) + sorted(outdir.rglob("*.csv.gz")):
        frame = pd.read_csv(path, low_memory=False)
        cleaned = analysis.clean_rank_columns(frame)
        if list(frame.columns) != list(cleaned.columns):
            cleaned.to_csv(
                path,
                index=False,
                compression="gzip" if path.suffix == ".gz" else None,
            )


def rewrite_documentation(outdir: Path, args: argparse.Namespace) -> dict[str, object]:
    summary_path = outdir / "analysis_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "significance_rule": "estimated finite p_value_rank_binary < 0.05",
            "undefined_rank_p_values_excluded": True,
            "input_trace": str(args.trace_dir.resolve()),
            "input_features": str(args.features.resolve()),
            "reproduction_command": (
                "python breast_cancer_survival/motif_results/"
                "sibling_topology_analysis_ranking_normalized/"
                "run_same_parent_comparison.py"
            ),
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    readme_path = outdir / "README.md"
    if readme_path.exists():
        text = readme_path.read_text(encoding="utf-8")
        text = text.replace(
            "Significance is the established raw Cox threshold (`p < 0.05`) in the TRI-5\n"
            "criterion table.",
            "Significance is the rank-normalized binary Cox threshold\n"
            "(`p_value_rank_binary < 0.05`) in the TRI-5 criterion table. Motifs with\n"
            "undefined or non-estimable rank Cox p-values are excluded.",
        )
        text = text.replace("p_value_unadjusted", "p_value_rank_binary")
        text = text.replace("hazard_ratio_unadjusted", "hazard_ratio_rank_binary")
        text = text.replace("median_patient_raw_count", "median_patient_rank_score")
        readme_path.write_text(text, encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    if args.motif_galleries_only:
        result = engine.generate_existing_motif_galleries(args)
        print(json.dumps(result, indent=2))
        return
    if args.clustered_heatmap_only:
        result = engine.generate_clustered_heatmap_from_existing(args)
        print(json.dumps(result, indent=2))
        return

    features = pd.read_csv(args.features, dtype={"wl_hash": str}, low_memory=False)
    compatibility = analysis.add_compatibility_columns(features)
    with tempfile.TemporaryDirectory(prefix="rank-sibling-topology-") as temporary:
        compatibility_path = Path(temporary) / "tri5_motif_criteria.csv.gz"
        compatibility.to_csv(compatibility_path, index=False, compression="gzip")
        engine_args = argparse.Namespace(
            trace_dir=args.trace_dir,
            features=compatibility_path,
            outdir=args.outdir,
            min_per_class=args.min_per_class,
            top_criteria=args.top_criteria,
            panels_per_page=args.panels_per_page,
            motif_galleries_only=False,
            clustered_heatmap_only=False,
        )
        engine.run(engine_args)

    rewrite_rank_native_csvs(args.outdir)
    summary = rewrite_documentation(args.outdir, args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
