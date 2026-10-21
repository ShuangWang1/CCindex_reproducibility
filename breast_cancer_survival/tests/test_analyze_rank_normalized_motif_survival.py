from __future__ import annotations

import sys
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze_rank_normalized_motif_survival as ranked  # noqa: E402


class RankNormalizedMotifSurvivalTests(unittest.TestCase):
    def test_positive_ties_average_and_absence_is_last(self) -> None:
        row = sparse.csr_matrix([[10.0, 5.0, 5.0, 0.0, 1.0, 0.0]])
        raw, ranks, scores = ranked.selected_ranks_from_sparse_row(row, np.arange(6), 6)
        np.testing.assert_allclose(raw, [10, 5, 5, 0, 1, 0])
        np.testing.assert_allclose(ranks, [1, 2.5, 2.5, 6, 4, 6])
        np.testing.assert_allclose(scores, [1, 0.7, 0.7, 0, 0.4, 0])

    def test_unselected_motifs_affect_selected_rank(self) -> None:
        row = sparse.csr_matrix([[20.0, 10.0, 5.0, 0.0]])
        raw, ranks, _ = ranked.selected_ranks_from_sparse_row(row, np.array([2, 3]), 4)
        np.testing.assert_allclose(raw, [5, 0])
        np.testing.assert_allclose(ranks, [3, 4])

    def test_multiple_cores_are_equally_averaged(self) -> None:
        cohort = pd.DataFrame({"PID": ["p1", "p1", "p2"]})
        patients = pd.DataFrame({"PID": ["p1", "p2"], "time_months": [4.0, 8.0], "event": [1, 0]})
        observed = ranked.aggregate_patient_values(
            cohort,
            patients,
            np.array([[2.0], [8.0], [3.0]]),
            np.array([[2.0], [6.0], [4.0]]),
            np.array([[0.8], [0.2], [0.6]]),
        )
        self.assertEqual(observed.loc[0, "number_of_cores"], 2)
        self.assertEqual(observed.loc[0, "summed_raw_count"][0], 10.0)
        self.assertEqual(observed.loc[0, "mean_core_rank"][0], 4.0)
        self.assertEqual(observed.loc[0, "rank_score"][0], 0.5)

    def test_survival_split_is_strictly_above_median(self) -> None:
        score = np.array([0.0, 0.2, 0.2, 0.8, 0.9, 1.0])
        result, high, score_z = ranked.analyze_rank_survival(
            score, np.arange(1, 7, dtype=float), np.array([1, 0, 1, 0, 1, 1])
        )
        self.assertEqual(result["median_patient_rank_score"], 0.5)
        np.testing.assert_array_equal(high, [False, False, False, True, True, True])
        self.assertAlmostEqual(float(np.std(score_z, ddof=1)), 1.0)

    def test_constant_score_is_explicitly_non_estimable(self) -> None:
        result, high, score_z = ranked.analyze_rank_survival(
            np.zeros(6), np.arange(1, 7, dtype=float), np.array([1, 0, 1, 0, 1, 1])
        )
        self.assertFalse(high.any())
        self.assertEqual(result["rank_binary_fit_status"], "not_estimable_no_group_variation")
        self.assertEqual(result["rank_continuous_fit_status"], "not_estimable_constant_score")
        self.assertTrue(np.isnan(score_z).all())

    def test_significant_directory_gets_star_and_migrates(self) -> None:
        motif_hash = "3a6e04025f7ced7d44f8ee2d3d03846a"
        with tempfile.TemporaryDirectory() as temporary:
            outdir = Path(temporary)
            original = outdir / "tri7" / motif_hash
            original.mkdir(parents=True)
            (original / "statistics.json").touch()
            resolved = ranked.resolve_motif_directory(outdir, 7, motif_hash, 0.049)
            self.assertEqual(resolved.name, motif_hash + "*")
            self.assertFalse(original.exists())
            self.assertTrue((resolved / "statistics.json").is_file())

    def test_nonsignificant_directory_has_no_star(self) -> None:
        self.assertEqual(ranked.motif_directory_name("abc", 0.05), "abc")
        self.assertEqual(ranked.motif_directory_name("abc", np.nan), "abc")

    def test_significant_collection_copies_packages_and_removes_stale_entries(self) -> None:
        artifacts = [
            "rank_normalized_survival_km.png",
            "patient_rank_distribution.png",
            "patient_rank_scores.csv",
            "statistics.json",
        ]
        with tempfile.TemporaryDirectory() as temporary:
            outdir = Path(temporary)
            significant_source = outdir / "tri2" / "sig*"
            nonsignificant_source = outdir / "tri2" / "not_sig"
            significant_source.mkdir(parents=True)
            nonsignificant_source.mkdir(parents=True)
            for name in artifacts:
                (significant_source / name).write_text(name)
                (nonsignificant_source / name).write_text(name)
            stale = outdir / "significant" / "tri2" / "stale"
            stale.mkdir(parents=True)
            (stale / "old.txt").write_text("old")
            results = pd.DataFrame(
                {
                    "triangle_level": [2, 2],
                    "wl_hash": ["sig", "not_sig"],
                    "p_value_rank_binary": [0.049, 0.05],
                }
            )

            manifest = ranked.sync_significant_results(outdir, results, [1, 2])

            copied = outdir / "significant" / "tri2" / "sig"
            self.assertTrue((outdir / "significant" / "tri1").is_dir())
            self.assertTrue(copied.is_dir())
            self.assertEqual(sorted(path.name for path in copied.iterdir()), sorted(artifacts))
            self.assertFalse((outdir / "significant" / "tri2" / "stale").exists())
            self.assertFalse((outdir / "significant" / "tri2" / "not_sig").exists())
            self.assertTrue(significant_source.is_dir())
            self.assertTrue(nonsignificant_source.is_dir())
            self.assertEqual(manifest["total_significant_motifs"], 1)
            saved = json.loads(
                (outdir / "significant" / "significant_summary.json").read_text()
            )
            self.assertEqual(saved["significant_motifs_by_level"], {"1": 0, "2": 1})


if __name__ == "__main__":
    unittest.main()
