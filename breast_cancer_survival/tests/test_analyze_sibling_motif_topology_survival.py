from __future__ import annotations

import sys
import unittest
from pathlib import Path
import os
import pytest

import networkx as nx
import numpy as np
import pandas as pd


SURVIVAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVIVAL_DIR))


def trace_back_dir() -> Path:
    """Locate optional generated artifacts in the current or legacy result root."""
    for result_name in ("motif_results", "motif_results_tri4_tri7_unadjusted"):
        candidate = SURVIVAL_DIR / result_name / "tri-7_trace_back"
        if candidate.is_dir():
            return candidate
    return SURVIVAL_DIR / "motif_results" / "tri-7_trace_back"

import analyze_sibling_motif_topology_survival as sibling  # noqa: E402


class SiblingTopologySurvivalTests(unittest.TestCase):
    @staticmethod
    def triangle_pair() -> tuple[nx.Graph, nx.Graph]:
        focal = nx.Graph([(0, 1), (0, 2), (1, 2), (0, 3), (1, 3)])
        matched = nx.Graph([(0, 1), (0, 2), (1, 2), (0, 4), (2, 4)])
        nx.set_node_attributes(focal, {0: "6", 1: "6", 2: "5", 3: "6"}, "label")
        nx.set_node_attributes(matched, {0: "6", 1: "6", 2: "5", 4: "6"}, "label")
        return focal, matched

    def test_graph_descriptors_are_cell_type_stratified(self) -> None:
        focal, _ = self.triangle_pair()
        observed = sibling.graph_descriptors(
            "abc", focal, {"6": "tumor cells", "5": "stromal cells"}
        )
        self.assertEqual(observed["triangles"], 2)
        self.assertEqual(observed["tumor_nodes"], 3)
        self.assertEqual(observed["stromal_nodes"], 1)
        self.assertEqual(observed["tumor_tumor_tumor_triangles"], 1)
        self.assertEqual(observed["stromal_tumor_tumor_triangles"], 1)

    def test_matched_sibling_difference_finds_one_edge_rewire(self) -> None:
        focal, matched = self.triangle_pair()
        mapped, focal_only, matched_only, mapping = sibling.matched_sibling_difference(
            focal, matched
        )
        self.assertEqual(mapping[4], 3)
        self.assertEqual(focal_only, {(1, 3)})
        self.assertEqual(matched_only, {(2, 3)})
        self.assertEqual(set(mapped), set(focal))

    def test_patient_exposures_use_strict_median_and_log_ratio(self) -> None:
        patients = pd.DataFrame(
            {
                "PID": [1, 2, 3, 4],
                "time_months": [2.0, 4.0, 6.0, 8.0],
                "event": [1, 0, 1, 0],
            }
        )
        counts = np.array([[0, 0], [1, 2], [3, 3], [7, 4]], dtype=float)
        observed = sibling.build_patient_exposures(
            patients,
            ["focal000", "match000"],
            counts,
            np.ones(4),
            np.full(4, 1000.0),
            "focal000",
            "match000",
            0.5,
        )
        self.assertEqual(observed["median_cutoff_focal000"].iloc[0], 2.0)
        self.assertEqual(observed["enriched_focal000"].tolist(), [0, 0, 1, 1])
        expected = np.log((counts[:, 0] + 0.5) / (counts[:, 1] + 0.5))
        np.testing.assert_allclose(
            observed["primary_topology_preference"], sibling.safe_zscore(expected)
        )

    def test_bootstrap_is_deterministic(self) -> None:
        size = 24
        data = pd.DataFrame(
            {
                "time_months": np.arange(1, size + 1, dtype=float),
                "event": np.tile([0, 1], size // 2),
                "primary_pair_burden": np.linspace(-1, 1, size),
                "primary_topology_preference": np.sin(np.arange(size)),
                "count_focal000": np.arange(size) % 5,
                "count_match000": np.arange(size) % 7,
            }
        )
        for column in sibling.MODEL_COVARIATES:
            data[column] = np.linspace(-1, 1, size) + np.arange(size) % 3
        first, _ = sibling.bootstrap_analysis(
            data, ["focal000", "match000"], 3, seed=17
        )
        second, _ = sibling.bootstrap_analysis(
            data, ["focal000", "match000"], 3, seed=17
        )
        pd.testing.assert_frame_equal(first, second)

    def test_saved_primary_motifs_have_expected_structural_contrast(self) -> None:
        summary_path = trace_back_dir() / "trace_back_summary.csv"
        if not summary_path.is_file():
            self.skipTest("optional generated trace-back summary is unavailable")
        summary = pd.read_csv(
            summary_path, dtype={"wl_hash": str, "node_labels": str, "edge_list": str}
        ).set_index("wl_hash")

        def graph_from_summary(motif_id: str) -> nx.Graph:
            row = summary.loc[motif_id]
            graph = nx.Graph()
            for node, label in enumerate(row.node_labels.split("&")):
                graph.add_node(node, label=label)
            for encoded in row.edge_list.split(";"):
                left, right = map(int, encoded.split("-"))
                graph.add_edge(left, right)
            return graph

        focal = graph_from_summary(sibling.FOCAL_HASH)
        matched = graph_from_summary(sibling.MATCHED_HASH)
        self.assertEqual(sibling.graph_edit_distance(focal, matched), 2.0)

    @pytest.mark.integration
    @unittest.skipUnless(os.environ.get("CCINDEX_RUN_INTEGRATION") == "1", "Requires saved study artifacts and CCINDEX_RUN_INTEGRATION=1")
    def test_generated_analysis_reproduces_saved_focal_cox_result(self) -> None:
        output = trace_back_dir() / "tri6_sibling_topology_analysis"
        if not (output / "cox_model_results.csv").is_file():
            self.skipTest("optional generated sibling-analysis output is unavailable")
        coefficients = pd.read_csv(output / "cox_model_results.csv")
        observed = coefficients.loc[
            coefficients["model"].eq("binary_7a7e7ea4_unadjusted")
            & coefficients["term"].eq("enriched_7a7e7ea4")
        ].iloc[0]
        summary = pd.read_csv(
            output.parent / "trace_back_summary.csv", dtype={"wl_hash": str}
        ).set_index("wl_hash").loc[sibling.FOCAL_HASH]
        self.assertEqual(observed["n"], 278)
        self.assertAlmostEqual(observed["hazard_ratio"], summary.hazard_ratio_unadjusted)
        self.assertAlmostEqual(observed["p_value"], summary.p_value_unadjusted)


if __name__ == "__main__":
    unittest.main()
