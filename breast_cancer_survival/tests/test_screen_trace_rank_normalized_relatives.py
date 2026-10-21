from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse


SURVIVAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVIVAL_DIR))

import analyze_rank_normalized_motif_survival as rankmod  # noqa: E402
import screen_trace_rank_normalized_relatives as extended  # noqa: E402


class TraceRelativeRankScreenTests(unittest.TestCase):
    def test_sparse_rank_scores_match_existing_selected_rank_definition(self) -> None:
        counts = sparse.csr_matrix(
            np.array(
                [
                    [0, 7, 7, 2, 0, 1],
                    [4, 0, 2, 2, 0, 0],
                ],
                dtype=float,
            )
        )
        observed = extended.rank_score_sparse(counts, vocabulary_size=6).toarray()

        for row in range(counts.shape[0]):
            _, _, expected = rankmod.selected_ranks_from_sparse_row(
                counts.getrow(row), np.arange(6), vocabulary_size=6
            )
            np.testing.assert_allclose(observed[row], expected)

    def test_patient_aggregation_equally_averages_multiple_cores(self) -> None:
        import pandas as pd

        cohort = pd.DataFrame({"PID": ["A", "A", "B"]})
        patients = pd.DataFrame({"PID": ["A", "B"]})
        aggregation = extended.patient_aggregation_matrix(cohort, patients)
        core_scores = sparse.csr_matrix([[0.2, 0.0], [0.6, 0.8], [1.0, 0.4]])

        observed = (aggregation @ core_scores).toarray()

        np.testing.assert_allclose(observed, [[0.4, 0.4], [1.0, 0.4]])


if __name__ == "__main__":
    unittest.main()
