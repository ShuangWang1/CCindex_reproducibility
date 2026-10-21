"""Run cumulative optimization ablations during motif tree search.

The five stages intentionally add one optimization at a time:

1. baseline (Python, correctness checks only)
2. triangle filtering
3. triangle filtering + candidate prefiltering
4. triangle filtering + candidate prefiltering + search reordering
5. the same fully optimized search implemented in C++

Every stage sees the same candidate patterns.  Match counts are checked for
parity, and only the final C++ stage controls pruning of the next tree level.
"""

import argparse
import copy
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import igraph as ig

from ablation_matcher import (
    PythonGraphCache,
    find_pattern_matches_python_with_options,
    igraph_to_cpp,
)
from tree_search_ablation import (
    construct_queue,
    extend_graph,
    prepare_triangle_index,
    read_queue,
    summarize_level,
    target_edge_frequencies,
    write_csv,
)


VARIANT_OPTIONS = {
    "baseline": {
        "label_index": False,
        "degree_filter": False,
        "triangle_filter": False,
        "reorder": False,
        "cpp": False,
    },
    "triangle_filter": {
        "label_index": False,
        "degree_filter": False,
        "triangle_filter": True,
        "reorder": False,
        "cpp": False,
    },
    "triangle_candidate_filter": {
        "label_index": True,
        "degree_filter": True,
        "triangle_filter": True,
        "reorder": False,
        "cpp": False,
    },
    "triangle_candidate_reorder": {
        "label_index": True,
        "degree_filter": True,
        "triangle_filter": True,
        "reorder": True,
        "cpp": False,
    },
    "triangle_candidate_reorder_cpp": {
        "label_index": True,
        "degree_filter": True,
        "triangle_filter": True,
        "reorder": True,
        "cpp": True,
    },
}
VALID_VARIANTS = tuple(VARIANT_OPTIONS)
REFERENCE_VARIANT = "triangle_candidate_reorder_cpp"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cumulative CC-index tree-search ablation"
    )
    parser.add_argument("--MaxDepth", type=int, default=10)
    parser.add_argument("--threshold", type=int, default=10)
    parser.add_argument("--SM_stop_early", action="store_true")
    parser.add_argument("--LevelStartFrom", type=int, default=0)
    parser.add_argument("--GraphQueuePath", default="")
    parser.add_argument("--data_csv", default="ad_data.csv")
    parser.add_argument(
        "--target_graph", default="spatial_13months-disease-replicate_1.gml"
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VALID_VARIANTS,
        default=list(VALID_VARIANTS),
        help="Cumulative stages to run; the final C++ stage is required",
    )
    parser.add_argument("--output_dir", default="ad_ablation_res_v2")
    parser.add_argument("--run_name", default="")
    return parser.parse_args()


def run_variant(graph, patterns, variant, result_path, threshold=0, edge_dict=None):
    """Run one cumulative stage and write its per-pattern results."""
    if variant not in VARIANT_OPTIONS:
        raise ValueError(f"Unknown cumulative ablation variant: {variant}")

    options = VARIANT_OPTIONS[variant]
    cache_started = time.perf_counter()
    if options["cpp"]:
        import graph_pattern_ablation_cpp as cpp

        cache = cpp.GraphCache(igraph_to_cpp(graph, cpp))
    else:
        cpp = None
        cache = PythonGraphCache(graph)
    cache_seconds = time.perf_counter() - cache_started

    matcher_options = {
        key: options[key]
        for key in ("label_index", "degree_filter", "triangle_filter", "reorder")
    }
    rows = []
    updated_patterns = copy.deepcopy(patterns)
    for pattern in updated_patterns:
        started = time.perf_counter()
        if options["cpp"]:
            cpp_pattern = igraph_to_cpp(pattern, cpp)
            count, matches, stats = cpp.find_pattern_matches_ablation(
                cache, cpp_pattern, True, threshold, edge_dict, "full"
            )
        else:
            count, matches, stats = find_pattern_matches_python_with_options(
                cache,
                pattern,
                threshold,
                edge_dict,
                options=matcher_options,
            )
        elapsed = time.perf_counter() - started
        pattern["count"] = count
        pattern["mtf_ids"] = matches
        pattern["time"] = elapsed
        rows.append(
            {
                "variant": variant,
                "level": pattern["ablation_level"],
                "pattern_id": pattern["ablation_id"],
                "matches": count,
                "threshold_reached": bool(threshold and count >= threshold),
                "search_seconds": elapsed,
                **dict(stats),
            }
        )

    result_path = Path(result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["variant", "level", "pattern_id", "matches"]
    with result_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return updated_patterns, rows, cache_seconds


def main():
    args = parse_args()
    if REFERENCE_VARIANT not in args.variants:
        raise ValueError(
            f"--variants must include {REFERENCE_VARIANT} because it controls tree pruning"
        )

    run_name = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    config = vars(args) | {
        "run_name": run_name,
        "counting": "threshold" if args.SM_stop_early else "exact",
        "variant_options": VARIANT_OPTIONS,
        "reference_variant": REFERENCE_VARIANT,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    triangles, edge_pool = prepare_triangle_index(args.data_csv, args.threshold)
    queue = (
        construct_queue(triangles)
        if args.LevelStartFrom == 0
        else read_queue(args.GraphQueuePath, args.LevelStartFrom)
    )
    target = ig.read(args.target_graph)
    target.vs["label"] = [str(int(label)) for label in target.vs["label"]]
    edge_dict = target_edge_frequencies(target)
    effective_threshold = args.threshold if args.SM_stop_early else 0
    pattern_rows = []
    level_rows = []

    for loop_level in range(args.LevelStartFrom, args.MaxDepth - 1):
        level = loop_level + 2
        generated = {}
        for graph in queue:
            extend_graph(graph, edge_pool, generated)
        candidates = list(generated.values())
        for position, graph in enumerate(candidates):
            graph["ablation_level"] = level
            graph["ablation_id"] = (
                f"L{level}-{position:07d}-{graph['hashvalue']}"
            )

        variant_results = {}
        for variant in args.variants:
            result_path = run_dir / variant / f"ccindex_tri_{level}.csv"
            matched, rows, cache_seconds = run_variant(
                target,
                candidates,
                variant,
                result_path,
                effective_threshold,
                edge_dict,
            )
            variant_results[variant] = matched
            pattern_rows.extend(rows)
            level_rows.append(summarize_level(variant, level, rows, cache_seconds))

        reference = {
            graph["ablation_id"]: graph["count"]
            for graph in variant_results[REFERENCE_VARIANT]
        }
        for variant, matched in variant_results.items():
            observed = {graph["ablation_id"]: graph["count"] for graph in matched}
            if observed != reference:
                raise RuntimeError(
                    f"Count mismatch at level {level}: "
                    f"{REFERENCE_VARIANT} versus {variant}"
                )

        queue = [
            graph
            for graph in variant_results[REFERENCE_VARIANT]
            if graph["count"] >= args.threshold
        ]
        write_csv(run_dir / "pattern_results.csv", pattern_rows)
        write_csv(run_dir / "level_summary.csv", level_rows)
        print(f"level={level} candidates={len(candidates)} retained={len(queue)}")
        if not queue:
            break


if __name__ == "__main__":
    main()
