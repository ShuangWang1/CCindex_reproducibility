"""Exercise CLI workflows using synthetic inputs outside the source tree."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import igraph as ig
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def run(script, cwd, *args):
    completed = subprocess.run([sys.executable, str(ROOT / script), *map(str, args)],
                               cwd=cwd, capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed


def target_graph():
    g = ig.Graph(n=4, edges=[(0, 1), (1, 2), (2, 0), (0, 3), (1, 3)])
    g.vs["label"] = ["0"] * 4
    return g


def test_ad_benchmark_and_bfs_from_external_working_directory(tmp_path):
    target_graph().write_gml(str(tmp_path / "spatial_13months-disease-replicate_1.gml"))
    (tmp_path / "ad_data.csv").write_text("a&a&a\n2\n")
    run("benchmark_matchers.py", tmp_path, "--MaxDepth", 2, "--threshold", 1,
        "--ResultDir", tmp_path / "benchmark")
    result = pd.read_csv(tmp_path / "benchmark/ccindex_tri_2.csv")
    assert result.Matches.tolist() == [1]
    run("tree_search_ablation_bfs.py", tmp_path, "--MaxDepth", 2, "--threshold", 1)
    result = pd.read_csv(tmp_path / "ad_output_graph_ccindex_1_bfs_ablation/bfs_mtf_records_ablation.csv")
    assert result.trinum.tolist() == [2]
    assert result.iso_mtf.iloc[0] > 0


def test_breast_cancer_exact_counts_for_both_cohorts(tmp_path):
    inputs = tmp_path / "breast_cancer"
    for dirname in ["BZ_samples_intra", "metabric_samples_intra"]:
        folder = inputs / dirname
        folder.mkdir(parents=True)
        target_graph().write_gml(str(folder / "sample1.gml"))
        pd.DataFrame([{"sample_index": 1, "gml_file": "sample1.gml", "source_sample": "synthetic-1"}]).to_csv(folder / "sample_manifest.csv", index=False)
        pd.DataFrame([{"node_label": 0, "cell_type": "synthetic"}]).to_csv(folder / "node_label_cell_type_mapping.csv", index=False)
    outputs = tmp_path / "motifs"
    run("tree_search_all_level_threshold_cpp_breast_cancer_intra.py", tmp_path,
        "--dataset", "both", "--MaxDepth", 2, "--threshold", 1,
        "--breast-cancer-dir", inputs, "--output-root", outputs)
    for dirname in ["BZ_res_threshold_1_cpp_0811", "metabric_res_threshold_1_cpp_0811"]:
        folder = outputs / dirname / "sample1"
        rows = pd.read_csv(folder / "ccindex_tri_2.csv")
        assert rows.matches.tolist() == [1]
        summary = json.loads((folder / "run_summary.json").read_text())
        assert summary["completed_triangle_level"] == 2


def test_benchmark_plot_validates_counts_and_uses_neutral_labels(tmp_path):
    for version in ["reference", "current"]:
        folder = tmp_path / version
        folder.mkdir()
        with (folder / "ccindex_tri_2.csv").open("w", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(["Pattern", "Matches", "MatchingTimeSeconds"])
            writer.writerows([["2_0", 1, 0.1], ["2_1", 2, 0.2]])
    run("plot_matching_time.py", tmp_path, "--old-dir", tmp_path / "reference",
        "--current-dir", tmp_path / "current", "--max-level", 2,
        "--output", tmp_path / "comparison.png", "--summary", tmp_path / "summary.csv")
    assert (tmp_path / "comparison.png").stat().st_size > 0
    assert "MedianSpeedupOldOverCurrent" in pd.read_csv(tmp_path / "summary.csv").columns
