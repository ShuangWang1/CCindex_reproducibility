from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
import os
import pytest

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("unlabeled_topology_pipeline", PROJECT / "run_analysis.py")
pipeline = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = pipeline
SPEC.loader.exec_module(pipeline)


class UnlabeledTopologyPipelineTests(unittest.TestCase):
    def test_structural_families_exclude_all_label_features(self) -> None:
        columns = [
            "graph_nodes",
            "degree_gini",
            "triangle_pairs_share_edge",
            "label_4_node_count",
            "edge_4_4_count",
            "labeled_triangle_2_4_4_count",
        ]
        frame = pd.DataFrame(columns=columns)
        observed = pipeline.structural_metric_columns(frame)
        self.assertEqual(
            observed,
            ["graph_nodes", "degree_gini", "triangle_pairs_share_edge"],
        )

    @pytest.mark.integration
    @unittest.skipUnless(os.environ.get("CCINDEX_RUN_INTEGRATION") == "1", "Requires saved study artifacts and CCINDEX_RUN_INTEGRATION=1")
    def test_generated_ranked_table_contains_no_label_metrics(self) -> None:
        path = PROJECT / "results" / "tables" / "tri5_ranked_metrics.csv"
        if not path.is_file():
            self.skipTest("optional generated results are unavailable")
        frame = pd.read_csv(path)
        self.assertTrue(set(frame["family"]).issubset(pipeline.STRUCTURAL_FAMILIES))
        self.assertFalse(frame["metric"].str.startswith("label_").any())
        self.assertFalse(frame["metric"].str.startswith("edge_").any())
        self.assertFalse(frame["metric"].str.startswith("labeled_triangle_").any())

    @pytest.mark.integration
    @unittest.skipUnless(os.environ.get("CCINDEX_RUN_INTEGRATION") == "1", "Requires saved study artifacts and CCINDEX_RUN_INTEGRATION=1")
    def test_generated_results_match_tri5_population(self) -> None:
        path = PROJECT / "results" / "tables" / "level_summary.csv"
        if not path.is_file():
            self.skipTest("optional generated results are unavailable")
        row = pd.read_csv(path).set_index("triangle_level").loc[5]
        self.assertEqual(row["estimable_motifs"], 6099)
        self.assertEqual(row["significant_raw_p"], 472)
        self.assertEqual(row["nonsignificant_raw_p"], 5627)


if __name__ == "__main__":
    unittest.main()
