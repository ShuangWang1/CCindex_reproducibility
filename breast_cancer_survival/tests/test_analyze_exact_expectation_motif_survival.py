from __future__ import annotations

import sys
import tempfile
import unittest
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd


SURVIVAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVIVAL_DIR))

import analyze_exact_expectation_motif_survival as exact  # noqa: E402


class ExactExpectationMotifSurvivalTests(unittest.TestCase):
    def test_falling_factorial_handles_boundaries(self) -> None:
        self.assertEqual(exact.falling(5, 0), 1)
        self.assertEqual(exact.falling(5, 3), 60)
        self.assertEqual(exact.falling(2, 3), 0)
        with self.assertRaises(ValueError):
            exact.falling(-1, 1)

    def test_expected_count_matches_hand_calculation(self) -> None:
        # A triangle with two A labels and one B label has three distinct
        # placements: |Aut(S)|/|Aut(m)| = 6/2.
        observed = exact.expected_count(
            shape_count=100,
            aut_shape=6,
            aut_labeled=2,
            motif_type_counts={0: 2, 1: 1},
            sample_type_counts={0: 4, 1: 6},
            total_cells=10,
        )
        self.assertAlmostEqual(observed, 30.0)

    def test_impossible_composition_has_zero_expectation(self) -> None:
        observed = exact.expected_count(20, 6, 2, {0: 2, 1: 1}, {0: 1, 1: 9}, 10)
        self.assertEqual(observed, 0.0)

    def test_canonical_shape_ignores_vertex_numbering(self) -> None:
        first = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]
        permutation = {0: 3, 1: 0, 2: 2, 3: 1}
        second = [(permutation[a], permutation[b]) for a, b in first]
        left = exact.canonical_shape(4, first)
        right = exact.canonical_shape(4, second)
        self.assertEqual(left, right)
        self.assertEqual(left[2], 4)

    def test_colored_automorphism_ratio(self) -> None:
        edges = [(0, 1), (0, 2), (1, 2)]
        _, _, aut_shape = exact.canonical_shape(3, edges)
        self.assertEqual(aut_shape, 6)
        self.assertEqual(exact.labeled_automorphisms("0&0&1", edges), 2)
        self.assertEqual(exact.labeled_automorphisms("0&1&2", edges), 1)

    def test_exact_isomorphism_mapping_handles_vertex_permutation(self) -> None:
        edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 4)]
        permutation = [3, 0, 4, 2, 1]
        permuted = [tuple(sorted((permutation[a], permutation[b]))) for a, b in edges]
        mapping = exact._exact_isomorphism_mapping(5, edges, permuted)
        permuted_set = set(permuted)
        self.assertEqual(len(set(mapping)), 5)
        self.assertTrue(
            all(tuple(sorted((mapping[a], mapping[b]))) in permuted_set for a, b in edges)
        )

    def test_label_orbit_canonicalization_respects_shape_symmetry(self) -> None:
        # Reversing a four-node path is a shape automorphism.
        automorphisms = [(0, 1, 2, 3), (3, 2, 1, 0)]
        left = exact._canonical_label_orbit(("A", "B", "C", "D"), automorphisms)
        right = exact._canonical_label_orbit(("D", "C", "B", "A"), automorphisms)
        different = exact._canonical_label_orbit(("A", "C", "B", "D"), automorphisms)
        self.assertEqual(left, right)
        self.assertNotEqual(left, different)

    def test_formula_equals_exhaustive_label_assignments(self) -> None:
        # Four fixed cells with two labels of each type; enumerate all six
        # distinct permutations and inspect one fixed triangle occurrence.
        assignments = set(permutations([0, 0, 1, 1]))
        empirical = np.mean(
            [sorted(values[:3]) == [0, 0, 1] for values in assignments]
        )
        analytic = exact.expected_count(1, 6, 2, {0: 2, 1: 1}, {0: 2, 1: 2}, 4)
        self.assertAlmostEqual(empirical, analytic)

    def test_enrichment_score_and_strict_grouping(self) -> None:
        observed = np.array([0.0, 2.0, 4.0, 8.0, 3.0, 7.0])
        expected = np.array([0.0, 3.0, 4.0, 2.0, 3.1, 1.0])
        score = exact.enrichment_scores(observed, expected)
        result, enriched, score_z = exact.survival_statistics(
            score,
            observed,
            expected,
            np.arange(1, 7, dtype=float),
            np.array([1, 0, 1, 0, 1, 1]),
        )
        np.testing.assert_array_equal(enriched, [False, False, False, True, False, True])
        self.assertEqual(result["n_enriched"], 2)
        self.assertAlmostEqual(float(np.std(score_z, ddof=1)), 1.0)

    def test_significance_directory_name(self) -> None:
        self.assertEqual(exact.motif_directory_name("abc", 0.049), "abc*")
        self.assertEqual(exact.motif_directory_name("abc", 0.05), "abc")
        self.assertEqual(exact.motif_directory_name("abc", np.nan), "abc")
        with tempfile.TemporaryDirectory() as temporary:
            directory = exact.resolve_motif_directory(Path(temporary), 4, "abc", 0.01)
            self.assertEqual(directory.name, "abc*")
            self.assertTrue(directory.is_dir())


if __name__ == "__main__":
    unittest.main()
