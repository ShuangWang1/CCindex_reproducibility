import os
import pytest
import importlib.util
import unittest
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[2]
PROJECT = REPO / "breast_cancer_survival" / "motif_results" / "tri7_trace_analysis(all)"
SPEC = importlib.util.spec_from_file_location("tri7_trace_analysis_all", PROJECT / "run_analysis.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.integration
@unittest.skipUnless(os.environ.get("CCINDEX_RUN_INTEGRATION") == "1", "Requires restored study traces and CCINDEX_RUN_INTEGRATION=1")
class Tri7TraceAnalysisAllTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.unique, cls.memberships = MODULE.load_union(MODULE.SOURCE_ROOT)

    def test_source_inventory_and_deduplicated_counts(self):
        self.assertEqual(len(MODULE.source_targets(MODULE.SOURCE_ROOT)), 450)
        counts = self.unique.groupby("triangle_level").size().to_dict()
        self.assertEqual(counts, {2: 26, 3: 44, 4: 157, 5: 420, 6: 727, 7: 450})
        significant = self.unique.groupby("triangle_level").significance_group.apply(
            lambda values: int(values.eq("significant").sum())
        ).to_dict()
        self.assertEqual(significant, {2: 0, 3: 0, 4: 2, 5: 15, 6: 76, 7: 450})
        self.assertEqual(len(self.unique), 1824)

    def test_memberships_are_many_to_many_and_unique(self):
        self.assertFalse(
            self.memberships.duplicated(
                ["triangle_level", "wl_hash", "trace_target_hash"]
            ).any()
        )
        self.assertGreater(
            self.memberships.groupby(["triangle_level", "wl_hash"]).size().max(), 1
        )

    def test_first_target_reconstructs_saved_lineage(self):
        target = MODULE.source_targets(MODULE.SOURCE_ROOT)[0]
        edges, graphs, graph = MODULE.reconstruct_target(target)
        saved_edges = pd.read_csv(target / "trace_edges.csv")
        saved_nodes = pd.read_csv(target / "trace_nodes.csv")
        self.assertEqual(len(edges), len(saved_edges))
        self.assertEqual(sum(len(values) for values in graphs.values()), len(saved_nodes))
        self.assertEqual(MODULE.trace.motif_hash(graph), target.name)

    def test_small_feature_computation_and_level_group_dispersion(self):
        sample = pd.concat(
            [group.head(2) for _, group in self.unique.groupby("triangle_level")],
            ignore_index=True,
        )
        features = MODULE.compute_criteria(sample, permutations=4, seed=17)
        self.assertEqual(len(features), 12)
        self.assertTrue(set(MODULE.DISPERSION_METRICS).issubset(features.columns))
        self.assertFalse(features[MODULE.DISPERSION_METRICS].isna().all().any())

    def test_one_sided_significance_comparison_is_not_testable(self):
        features = pd.DataFrame({
            "triangle_level": [2, 2, 3, 3],
            "significance_group": ["nonsignificant"] * 4,
            "metric_a": [0.0, 1.0, 1.0, 2.0],
        })
        _, _, within = MODULE.distribution_and_tests(features, ["metric_a"])
        self.assertFalse(within.testable.any())
        self.assertTrue(
            within.loc[within.triangle_level.isin([2, 3]), "not_testable_reason"]
            .eq("missing_significance_class").all()
        )


if __name__ == "__main__":
    unittest.main()
