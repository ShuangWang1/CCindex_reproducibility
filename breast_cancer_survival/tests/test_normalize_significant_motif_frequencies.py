from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SURVIVAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVIVAL_DIR))

import normalize_significant_motif_frequencies as normalize  # noqa: E402


class NormalizeSignificantMotifFrequencyTests(unittest.TestCase):
    def test_ilr_is_finite_with_zeros_and_scale_invariant_without_pseudocount_effect(self) -> None:
        counts = np.array([[0, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8]])
        observed = normalize.ilr_coordinates(counts)
        self.assertEqual(observed.shape, (2, 7))
        self.assertTrue(np.isfinite(observed).all())

        positive = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=float)
        first = normalize.ilr_coordinates(positive, pseudocount=1e-12)
        second = normalize.ilr_coordinates(positive * 10, pseudocount=1e-12)
        self.assertTrue(np.allclose(first, second, atol=1e-10))

    def test_expected_count_model_removes_known_cell_composition_effect(self) -> None:
        rng = np.random.default_rng(17)
        n = 100
        exposure = rng.integers(500, 2000, size=n).astype(float)
        log_cells = rng.normal(size=n)
        ilr = rng.normal(size=(n, 7))
        design = np.column_stack([np.ones(n), log_cells, ilr])
        linear = -5.0 + 0.35 * log_cells + 0.8 * ilr[:, 0] - 0.5 * ilr[:, 1]
        mean = exposure * np.exp(linear)
        counts = rng.poisson(mean)
        fitted = normalize.fit_expected_count(counts, design, exposure)
        self.assertNotEqual(fitted["fit_status"], "failed")
        expected = np.asarray(fitted["expected"])
        variance = np.asarray(fitted["variance"])
        residual = (counts - expected) / np.sqrt(variance)
        self.assertTrue(np.isfinite(residual).all())
        self.assertLess(abs(normalize.safe_spearman(residual, ilr[:, 0])), 0.2)

    def test_patient_aggregation_sums_counts_expectation_and_variance_first(self) -> None:
        core = pd.DataFrame(
            {
                "PID": ["p1", "p1", "p2"],
                "source_sample": ["a", "b", "c"],
                "total_cells": [100, 200, 150],
                "raw_count": [4.0, 8.0, 3.0],
                "level_total_occurrences": [20.0, 30.0, 25.0],
                "expected_count": [5.0, 5.0, 2.0],
                "model_variance": [4.0, 5.0, 2.0],
            }
        )
        observed = normalize.aggregate_patient_scores(core, ["p1", "p2"]).set_index("PID")
        self.assertEqual(observed.loc["p1", "raw_count"], 12.0)
        self.assertEqual(observed.loc["p1", "expected_count"], 10.0)
        self.assertEqual(observed.loc["p1", "model_variance"], 9.0)
        self.assertAlmostEqual(observed.loc["p1", "pearson_residual"], 2.0 / 3.0)
        self.assertEqual(observed.loc["p1", "number_of_cores"], 2)

    def test_continuous_survival_uses_zero_for_descriptive_split(self) -> None:
        n = 40
        scores = pd.DataFrame(
            {"PID": [f"p{i}" for i in range(n)], "pearson_residual": np.linspace(-2, 2, n)}
        )
        patients = pd.DataFrame(
            {
                "PID": [f"p{i}" for i in range(n)],
                "time_months": np.arange(1, n + 1, dtype=float),
                "event": np.tile([0, 1], n // 2),
            }
        )
        result, data = normalize.continuous_survival(scores, patients)
        self.assertEqual(result["n_above_expected"], 20)
        self.assertEqual(result["n_at_or_below_expected"], 20)
        self.assertEqual(result["km_status"], "estimated")
        self.assertTrue(np.isclose(data["pearson_residual_z"].std(ddof=1), 1.0))

    def test_motif_directory_name_marks_only_finite_nominal_significance(self) -> None:
        motif_hash = "004d6427908d448b25be00dac0671ef2"
        self.assertEqual(
            normalize.motif_directory_name(motif_hash, 0.049), motif_hash + "*"
        )
        self.assertEqual(normalize.motif_directory_name(motif_hash, 0.05), motif_hash)
        self.assertEqual(normalize.motif_directory_name(motif_hash, np.nan), motif_hash)

    def test_resolve_plot_directory_migrates_existing_directory(self) -> None:
        motif_hash = "004d6427908d448b25be00dac0671ef2"
        with tempfile.TemporaryDirectory() as temporary_directory:
            outdir = Path(temporary_directory)
            original = outdir / "tri5" / motif_hash
            original.mkdir(parents=True)
            (original / "plot.png").touch()

            resolved = normalize.resolve_plot_directory(outdir, 5, motif_hash, 0.01)

            self.assertEqual(resolved.name, motif_hash + "*")
            self.assertFalse(original.exists())
            self.assertTrue((resolved / "plot.png").is_file())


if __name__ == "__main__":
    unittest.main()
