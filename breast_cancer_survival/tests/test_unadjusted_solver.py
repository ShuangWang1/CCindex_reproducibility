from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from statsmodels.duration.hazard_regression import PHReg


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_motif_survival_unadjusted import vectorized_binary_cox  # noqa: E402


class VectorizedCoxTests(unittest.TestCase):
    def test_matches_phreg_with_tied_times(self) -> None:
        time = np.array([2, 2, 5, 6, 8, 8, 12, 15, 18, 18], dtype=float)
        event = np.array([1, 1, 0, 1, 0, 1, 0, 1, 1, 0], dtype=int)
        groups = np.array(
            [
                [1, 0, 1],
                [0, 1, 1],
                [1, 0, 0],
                [0, 1, 0],
                [1, 0, 1],
                [0, 1, 0],
                [0, 0, 1],
                [1, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
            ],
            dtype=bool,
        )
        observed = vectorized_binary_cox(groups, time, event)
        for column in range(groups.shape[1]):
            expected = PHReg(
                time, groups[:, column].astype(float)[:, None], status=event, ties="efron"
            ).fit(disp=0)
            self.assertAlmostEqual(observed["coef"][column], expected.params[0], places=9)
            self.assertAlmostEqual(observed["p_value"][column], expected.pvalues[0], places=9)

    def test_marks_no_variation_and_event_separation(self) -> None:
        time = np.array([1, 2, 3, 4], dtype=float)
        event = np.array([1, 1, 0, 0], dtype=int)
        groups = np.array(
            [
                [0, 1],
                [0, 1],
                [0, 0],
                [0, 0],
            ],
            dtype=bool,
        )
        observed = vectorized_binary_cox(groups, time, event)
        self.assertEqual(observed["status_code"].tolist(), [0, 1])
        self.assertTrue(np.isnan(observed["p_value"]).all())


if __name__ == "__main__":
    unittest.main()
