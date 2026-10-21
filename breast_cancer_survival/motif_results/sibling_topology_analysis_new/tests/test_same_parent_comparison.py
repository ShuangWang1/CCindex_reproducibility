import importlib.util
import unittest
from pathlib import Path
import os
import pytest

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "same_parent_comparison", PROJECT / "run_same_parent_comparison.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SameParentComparisonTests(unittest.TestCase):
    def test_relationships_are_immediate_and_unique(self):
        edges = pd.DataFrame({
            "parent_level": [4, 4, 3, 4],
            "parent_hash": ["p1", "p1", "p0", "p2"],
            "child_level": [5, 5, 4, 5],
            "child_hash": ["c1", "c1", "p1", "c2"],
            "growth_operation": ["add_vertex", "add_vertex", "add_vertex", "add_edge"],
            "changed_node_or_edge": ["x", "x", "y", "z"],
        })
        result = MODULE.prepare_relationships(edges)
        self.assertEqual(len(result), 2)
        self.assertEqual(set(zip(result.parent_hash, result.child_hash)), {("p1", "c1"), ("p2", "c2")})

    def test_parent_eligibility_and_exclusion_reasons(self):
        relationships = pd.DataFrame({
            "parent_hash": ["eligible"] * 6 + ["small"] * 3 + ["one_class"] * 3,
            "child_hash": [f"c{i}" for i in range(12)],
            "growth_operation": ["add_vertex"] * 12,
            "changed_node_or_edge": ["x"] * 12,
        })
        features = pd.DataFrame({
            "wl_hash": [f"c{i}" for i in range(12)],
            "significance_group": (
                ["nonsignificant"] * 3 + ["significant"] * 3
                + ["nonsignificant"] * 2 + ["significant"]
                + ["nonsignificant"] * 3
            ),
        })
        index, _ = MODULE.build_parent_index(relationships, features, 3)
        index = index.set_index("parent_hash")
        self.assertTrue(index.loc["eligible", "eligible"])
        self.assertEqual(
            index.loc["small", "exclusion_reason"],
            "fewer_than_3_children_in_one_or_both_classes",
        )
        self.assertEqual(index.loc["one_class", "exclusion_reason"], "missing_significance_class")

    def test_comparison_uses_q_first_then_effect_ranking(self):
        group = pd.DataFrame({
            "significance_group": ["nonsignificant"] * 5 + ["significant"] * 5,
            "strong": [0, 0, 0, 1, 1, 4, 4, 5, 5, 5],
            "weak": [0, 1, 0, 1, 0, 0, 1, 1, 1, 0],
            "constant": [2] * 10,
        })
        distributions, comparisons = MODULE.compare_parent(
            group, ["strong", "weak", "constant"], 3
        )
        self.assertEqual(len(distributions), 6)
        self.assertNotIn("constant", comparisons.metric.tolist())
        self.assertEqual(comparisons.iloc[0].metric, "strong")
        self.assertGreater(comparisons.iloc[0].cliffs_delta, 0)
        self.assertEqual(comparisons.rank_within_parent.tolist(), list(range(1, len(comparisons) + 1)))

    def test_intrinsic_metrics_exclude_metadata_and_dispersion(self):
        features = pd.DataFrame({
            "wl_hash": ["a", "b"],
            "significance_group": ["nonsignificant", "significant"],
            "p_value_unadjusted": [0.2, 0.01],
            "size_nodes": [7.0, 7.0],
            "dispersion_triangle_js_to_group_centroid": [0.1, 0.2],
        })
        self.assertEqual(MODULE.intrinsic_metrics(features), ["size_nodes"])

    def test_location_q_matrix_and_cluster_order_are_deterministic(self):
        comparisons = pd.DataFrame({
            "parent_hash": ["p2", "p1", "p2", "p1", "p3"],
            "metric": ["m1", "m1", "m2", "m2", "m2"],
            "display_name": ["metric one", "metric one", "metric two", "metric two", "metric two"],
            "family": ["f1", "f1", "f2", "f2", "f2"],
            "location_q_bh": [0.01, 0.5, 0.2, 0.05, 1.0],
            "n_nonsignificant": [4, 3, 4, 3, 5],
            "n_significant": [3, 3, 3, 3, 4],
        })
        matrix, metric_metadata, parent_metadata = MODULE.build_location_q_matrix(comparisons)
        self.assertEqual(matrix.shape, (2, 3))
        self.assertAlmostEqual(matrix.loc["m1", "p2"], 0.01)
        self.assertTrue(np.isnan(matrix.loc["m1", "p3"]))
        self.assertEqual(metric_metadata.loc["m2", "family"], "f2")
        self.assertEqual(parent_metadata.loc["p3", "n_significant"], 4)
        first = MODULE.clustered_matrix_order(matrix)[:2]
        second = MODULE.clustered_matrix_order(matrix)[:2]
        self.assertEqual(first, second)
        self.assertEqual(set(first[0]), {"m1", "m2"})
        self.assertEqual(set(first[1]), {"p1", "p2", "p3"})

    @pytest.mark.integration
    @unittest.skipUnless(os.environ.get("CCINDEX_RUN_INTEGRATION") == "1", "Requires saved study lineage")
    def test_current_lineage_parent_counts(self):
        edges = pd.read_csv(
            MODULE.TRACE_DIR / "trace_edges.csv",
            dtype={"parent_hash": str, "child_hash": str},
        )
        features = pd.read_csv(MODULE.DEFAULT_FEATURES, dtype={"wl_hash": str})
        relationships = MODULE.prepare_relationships(edges)
        index, _ = MODULE.build_parent_index(relationships, features, 3)
        self.assertEqual(len(index), 2225)
        self.assertEqual(int(index.mixed_significance_group.sum()), 348)
        self.assertEqual(int(index.eligible.sum()), 93)

    @pytest.mark.integration
    @unittest.skipUnless(os.environ.get("CCINDEX_RUN_INTEGRATION") == "1", "Requires saved study artifacts and CCINDEX_RUN_INTEGRATION=1")
    def test_generated_study_is_complete_when_present(self):
        outdir = MODULE.DEFAULT_OUTDIR
        summary_path = outdir / "analysis_summary.json"
        if not summary_path.is_file():
            self.skipTest("Generated same-parent study is not present")
        index = pd.read_csv(outdir / "parent_group_index.csv", dtype={"parent_hash": str})
        eligible = index.loc[index.eligible, "parent_hash"].tolist()
        gallery_index = pd.read_csv(
            outdir / "motif_gallery_index.csv",
            dtype={"parent_hash": str, "child_hash": str},
        )
        self.assertEqual(len(eligible), 93)
        self.assertEqual(len(gallery_index), 1908)
        self.assertFalse(gallery_index.duplicated(["parent_hash", "child_hash"]).any())
        self.assertEqual(
            {path.name for path in outdir.iterdir() if path.is_dir()},
            set(eligible),
        )
        for parent_hash in eligible:
            parent_dir = outdir / parent_hash
            self.assertTrue((parent_dir / "child_motifs.csv").is_file())
            self.assertTrue((parent_dir / "criteria_comparisons.csv").is_file())
            self.assertTrue((parent_dir / "tri5_child_motif_gallery.png").is_file())
            top = pd.read_csv(parent_dir / "top_criteria.csv")
            self.assertLessEqual(len(top), 24)
            self.assertEqual(top.rank_within_parent.tolist(), list(range(1, len(top) + 1)))
            expected_pages = int(np.ceil(len(top) / 12))
            self.assertEqual(len(list(parent_dir.glob("top_criteria_distributions_page_*.png"))), expected_pages)
            children = pd.read_csv(parent_dir / "child_motifs.csv", dtype={"child_hash": str})
            indexed = gallery_index.loc[gallery_index.parent_hash.eq(parent_hash)]
            self.assertEqual(set(children.child_hash), set(indexed.child_hash))

        comparisons = pd.read_csv(
            outdir / "all_parent_criteria_comparisons.csv.gz", dtype={"parent_hash": str}
        )
        matrix = pd.read_csv(
            outdir / "clustered_location_bh_q_matrix.csv.gz", index_col="metric"
        )
        order = pd.read_csv(outdir / "clustered_location_bh_q_order.csv", dtype={"id": str})
        self.assertEqual(matrix.shape, (263, 93))
        self.assertEqual(int(matrix.notna().sum().sum()), 11199)
        self.assertEqual(set(matrix.index), set(comparisons.metric))
        self.assertEqual(set(matrix.columns), set(comparisons.parent_hash))
        self.assertEqual(len(order), 263 + 93)
        self.assertEqual(order.loc[order.axis.eq("criterion"), "position"].tolist(), list(range(1, 264)))
        self.assertEqual(order.loc[order.axis.eq("parent"), "position"].tolist(), list(range(1, 94)))
        parent_gallery = pd.read_csv(
            outdir / "clustered_tri4_parent_motif_gallery_index.csv",
            dtype={"parent_hash": str},
        )
        self.assertEqual(len(parent_gallery), 93)
        self.assertEqual(parent_gallery.cluster_position.tolist(), list(range(1, 94)))
        self.assertEqual(
            parent_gallery.parent_hash.tolist(),
            order.loc[order.axis.eq("parent")].sort_values("position").id.tolist(),
        )
        for filename in [
            "clustered_location_bh_q_heatmap.png",
            "clustered_location_bh_q_heatmap.pdf",
            "clustered_tri4_parent_motif_gallery.png",
            "clustered_tri4_parent_motif_gallery.pdf",
        ]:
            self.assertGreater((outdir / filename).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
