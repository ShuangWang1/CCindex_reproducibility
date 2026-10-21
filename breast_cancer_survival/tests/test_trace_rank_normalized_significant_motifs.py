from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


SURVIVAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVIVAL_DIR))

import trace_rank_normalized_significant_motifs as ranktrace  # noqa: E402
import trace_significant_motifs as rawtrace  # noqa: E402


class RankNormalizedSignificantMotifTraceTests(unittest.TestCase):
    def test_pool_uses_rank_p_and_positive_rank_median_without_raw_p_filter(self) -> None:
        frame = pd.DataFrame(
            {
                "triangle_level": [7, 7, 7, 7],
                "wl_hash": ["keep", "zero", "large", "missing"],
                "median_patient_rank_score": [0.5, 0.0, 0.5, 0.5],
                "p_value_rank_binary": [0.01, 0.01, 0.10, np.nan],
                "p_value_unadjusted": [0.90, 0.01, 0.01, 0.01],
            }
        )

        observed = ranktrace.significant_rank_pool(frame, 0.05)

        self.assertEqual(observed["wl_hash"].tolist(), ["keep"])
        self.assertEqual(observed["motif_id"].tolist(), ["tri_7:keep"])

    def test_rank_lineage_nodes_retains_unanalyzed_relatives(self) -> None:
        parent = nx.Graph([(0, 1), (0, 2), (1, 2)])
        nx.set_node_attributes(parent, {0: "5", 1: "6", 2: "6"}, "label")
        child = parent.copy()
        child.add_node(3, label="5")
        child.add_edges_from([(0, 3), (1, 3)])
        parent_hash = rawtrace.motif_hash(parent)
        child_hash = rawtrace.motif_hash(child)
        graphs = {1: {parent_hash: parent}, 2: {child_hash: child}}

        raw_rows = []
        for level, motif_hash in [(1, parent_hash), (2, child_hash)]:
            row = {column: np.nan for column in rawtrace.SQL_COLUMNS}
            row.update(
                {
                    "triangle_level": level,
                    "wl_hash": motif_hash,
                    "median_patient_raw_count": 1.0,
                    "p_value_unadjusted": 0.5,
                }
            )
            raw_rows.append(row)
        raw_cox = pd.DataFrame(raw_rows)
        ranked_source = pd.DataFrame(
            {
                "triangle_level": [2],
                "wl_hash": [child_hash],
                "median_patient_rank_score": [0.4],
                "p_value_rank_binary": [0.01],
                "hazard_ratio_rank_binary": [2.0],
            }
        )

        observed = ranktrace.rank_lineage_nodes(
            graphs,
            raw_cox,
            ranktrace.rank_lookup(ranked_source),
            2,
            child_hash,
            0.05,
        ).set_index("wl_hash")

        self.assertFalse(bool(observed.loc[parent_hash, "rank_result_available"]))
        self.assertTrue(pd.isna(observed.loc[parent_hash, "p_value_rank_binary"]))
        self.assertTrue(bool(observed.loc[child_hash, "selected_focal_rule"]))

    def test_rank_p_lineage_plot_smoke(self) -> None:
        nodes = pd.DataFrame(
            {
                "triangle_level": [1, 2],
                "wl_hash": ["parent", "target"],
                "lineage_role": ["ancestor", "target"],
                "p_value_rank_binary": [np.nan, 0.01],
            }
        )
        edges = pd.DataFrame(
            {
                "parent_level": [1],
                "parent_hash": ["parent"],
                "child_level": [2],
                "child_hash": ["target"],
                "growth_operation": ["add_vertex"],
                "changed_node_or_edge": ["3:6@0-1"],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "rank_cox.png"
            ranktrace.plot_rank_p_lineage(
                nodes, edges, 2, "target", output, 0.05
            )
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)

    def test_cumulative_manifest_preserves_prior_levels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prior = {
                "status": "complete",
                "motif_id": "tri_7:old",
                "target_level": 7,
                "target_hash": "old",
            }
            path = root / "tri7" / "old" / "trace_metadata.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(prior), encoding="utf-8")
            current = {
                "status": "complete",
                "motif_id": "tri_5:new",
                "target_level": 5,
                "target_hash": "new",
                "run_action": "generated",
            }

            observed = ranktrace.cumulative_manifest(root, [current])

            self.assertEqual(set(observed["motif_id"]), {"tri_7:old", "tri_5:new"})


if __name__ == "__main__":
    unittest.main()
