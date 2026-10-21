import csv
import subprocess
import sys

import igraph as ig

from ablation_matcher import VALID_VARIANTS, run_cc_index_ablation


def make_graph(vertex_count, edges, labels):
    graph = ig.Graph(n=vertex_count, edges=edges)
    graph.vs["label"] = labels
    return graph


def patterns_with_ids():
    patterns = [
        make_graph(3, [(0, 1), (1, 2), (2, 0)], ["0", "1", "2"]),
        make_graph(3, [(0, 1), (1, 2)], ["0", "1", "2"]),
    ]
    for index, pattern in enumerate(patterns):
        pattern["ablation_level"] = 2
        pattern["ablation_id"] = f"pattern-{index}"
    return patterns


def test_variants_preserve_exact_and_threshold_counts(tmp_path):
    target = make_graph(
        7,
        [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 2), (4, 5), (5, 6), (6, 4)],
        ["0", "1", "2", "3", "2", "0", "1"],
    )
    for threshold in (0, 1):
        counts = {}
        for variant in VALID_VARIANTS:
            _, rows, _ = run_cc_index_ablation(
                target,
                patterns_with_ids(),
                variant,
                tmp_path / f"{variant}-{threshold}.csv",
                threshold,
                {},
            )
            counts[variant] = [row["matches"] for row in rows]
        assert len({tuple(value) for value in counts.values()}) == 1


def test_each_filter_ablation_changes_only_its_counters(tmp_path):
    target = make_graph(
        5,
        [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4)],
        ["0", "1", "2", "0", "9"],
    )
    pattern = patterns_with_ids()[0]
    results = {}
    for variant in ("full", "no_triangle_filter", "no_candidate_prefilter"):
        _, rows, _ = run_cc_index_ablation(
            target, [pattern], variant, tmp_path / f"{variant}.csv", 0, {}
        )
        results[variant] = rows[0]
    assert results["no_triangle_filter"]["rejected_triangle_count"] == 0
    assert results["no_triangle_filter"]["rejected_signature"] == 0
    assert results["no_candidate_prefilter"]["rejected_degree"] == 0
    assert results["no_candidate_prefilter"]["rejected_label"] > 0


def test_driver_writes_requested_layout(tmp_path):
    data_csv = tmp_path / "ad_data.csv"
    with data_csv.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["a&b&c"])
        writer.writerow([2])
    target = make_graph(4, [(0, 1), (1, 2), (2, 0), (2, 3)], ["0", "1", "2", "0"])
    target_path = tmp_path / "target.gml"
    target.write_gml(str(target_path))
    output_dir = tmp_path / "ad_ablation_res"

    subprocess.run(
        [
            sys.executable,
            "tree_search_ablation.py",
            "--MaxDepth", "2",
            "--threshold", "1",
            "--data_csv", str(data_csv),
            "--target_graph", str(target_path),
            "--output_dir", str(output_dir),
            "--run_name", "smoke",
        ],
        check=True,
    )
    run_dir = output_dir / "smoke"
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "pattern_results.csv").is_file()
    assert (run_dir / "level_summary.csv").is_file()
    for variant in VALID_VARIANTS:
        assert (run_dir / variant / "ccindex_tri_2.csv").is_file()
