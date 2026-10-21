from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.duration.hazard_regression import PHReg


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze_motif_survival as motif  # noqa: E402


class MotifSurvivalTests(unittest.TestCase):
    def test_benjamini_hochberg_preserves_nan_and_order(self) -> None:
        observed = motif.benjamini_hochberg([0.01, 0.04, np.nan, 0.03])
        np.testing.assert_allclose(observed[[0, 1, 3]], [0.03, 0.04, 0.04])
        self.assertTrue(np.isnan(observed[2]))

    def test_fit_cox_matches_direct_phreg(self) -> None:
        time = np.array([2, 3, 5, 6, 8, 9, 12, 15], dtype=float)
        event = np.array([1, 1, 0, 1, 0, 1, 0, 1], dtype=int)
        enriched = np.array([1, 0, 1, 0, 1, 0, 0, 1], dtype=bool)
        expected = PHReg(time, enriched.astype(float)[:, None], status=event, ties="efron").fit(
            disp=0
        )
        observed = motif.fit_cox(time, event, enriched)
        self.assertAlmostEqual(observed["coef"], expected.params[0])
        self.assertAlmostEqual(observed["p_value"], expected.pvalues[0])

    def test_analyze_level_uses_strict_median_and_sums_are_precomputed(self) -> None:
        patients = pd.DataFrame(
            {
                "PID": range(1, 9),
                "time_months": [2, 3, 5, 6, 8, 9, 12, 15],
                "event": [1, 1, 0, 1, 0, 1, 0, 1],
                **{column: np.linspace(-1, 1, 8) for column in motif.MODEL_COVARIATES},
            }
        )
        features = pd.DataFrame(
            [
                {
                    "triangle_level": 2,
                    "wl_hash": "abc",
                    "triangles": 2,
                    "nodes": 4,
                    "edges": 5,
                    "representative_node_labels": "0&1&1&2",
                    "representative_edge_list": "0-1;0-2",
                    "cell_type_composition": "B cells; T and B cells x2; T cells",
                    "sample_document_frequency": 4,
                }
            ]
        )
        # These values represent already-summed core counts per patient.
        counts = np.array([[0], [0], [2], [2], [3], [4], [5], [9]], dtype=float)
        result = motif.analyze_level(features, counts, patients).iloc[0]
        self.assertEqual(result.median_patient_raw_count, 2.5)
        self.assertEqual(result.n_enriched, 4)
        self.assertEqual(result.n_non_enriched, 4)

    def test_no_variation_is_retained_as_non_estimable(self) -> None:
        result = motif.fit_cox(
            np.array([1, 2, 3], dtype=float),
            np.array([1, 0, 1]),
            np.array([False, False, False]),
        )
        self.assertTrue(np.isnan(result["p_value"]))
        self.assertEqual(result["fit_status"], "not_estimable_no_group_variation")

    def test_explicit_zero_match_rows_are_absence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ccindex_tri_2.csv"
            pd.DataFrame(
                {
                    "triangles": [2, 2],
                    "nodes": [4, 4],
                    "edges": [5, 5],
                    "node_labels": ["0&0&0&0", "0&0&0&1"],
                    "edge_list": ["0-1;0-2", "0-1;0-2"],
                    "wl_hash": ["zero", "positive"],
                    "matches": [0, 3],
                }
            ).to_csv(path, index=False)
            observed = motif.read_motif_csv(path, 2)
            self.assertEqual(observed["wl_hash"].tolist(), ["positive"])


if __name__ == "__main__":
    unittest.main()
