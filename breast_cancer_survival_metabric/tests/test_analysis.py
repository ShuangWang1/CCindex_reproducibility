from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from statsmodels.duration.hazard_regression import PHReg


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from breast_cancer_survival_metabric.analyze_motif_survival import (  # noqa: E402
    benjamini_hochberg,
    prepare_cohort,
    vectorized_binary_cox,
)


class AnalysisTests(unittest.TestCase):
    def test_vectorized_cox_matches_phreg_with_ties(self) -> None:
        time = np.array([2, 2, 5, 6, 8, 8, 12, 15, 18, 18], dtype=float)
        event = np.array([1, 1, 0, 1, 0, 1, 0, 1, 1, 0], dtype=int)
        groups = np.array(
            [[1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [0, 1], [1, 1], [0, 1], [1, 0]],
            dtype=bool,
        )
        observed = vectorized_binary_cox(groups, time, event)
        for column in range(groups.shape[1]):
            expected = PHReg(
                time, groups[:, column].astype(float)[:, None], status=event, ties="efron"
            ).fit(disp=0)
            self.assertAlmostEqual(observed["coef"][column], expected.params[0], places=9)
            self.assertAlmostEqual(observed["p_value"][column], expected.pvalues[0], places=9)

    def test_prepare_cohort_sums_repeated_patient_images(self) -> None:
        samples = pd.DataFrame(
            {"row_index": [0, 1, 2, 3], "patient_id": ["MB-1", "MB-1", "MB-2", "MB-X"]}
        )
        clinical = pd.DataFrame(
            {"METABRIC.ID": ["MB-1", "MB-2"], "T": [10, 20], "Death": [1, 0]}
        )
        patients, aggregation, linkage, audit = prepare_cohort(samples, clinical)
        counts = sparse.csr_matrix([[1], [3], [5], [7]])
        observed = np.asarray((aggregation @ counts).todense()).ravel()
        self.assertEqual(patients["patient_id"].tolist(), ["MB-1", "MB-2"])
        self.assertEqual(observed.tolist(), [4.0, 5.0])
        self.assertEqual(audit["excluded_not_in_clinical_table"], 1)
        self.assertEqual(linkage.loc[~linkage["included"], "patient_id"].tolist(), ["MB-X"])

    def test_bh_preserves_missing_values(self) -> None:
        observed = benjamini_hochberg(np.array([0.01, 0.04, np.nan, 0.03]))
        np.testing.assert_allclose(observed[[0, 1, 3]], [0.03, 0.04, 0.04])
        self.assertTrue(np.isnan(observed[2]))


if __name__ == "__main__":
    unittest.main()
