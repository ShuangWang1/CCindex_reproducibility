import unittest

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import tune_mrmr_grid as grid


class ParameterTuningTests(unittest.TestCase):
    def test_default_grid_and_reproducibility_settings(self):
        args = grid.parse_args([])
        self.assertEqual(args.sizes, (10, 50, 100, 150, 200, 250, 300, 350, 400, 450))
        self.assertEqual((args.folds, args.repeats, args.seed), (5, 5, 42))
        self.assertEqual(grid.MODELS[-1], "linear_svm")

    def test_preprocessing_is_finite_and_shape_preserving(self):
        train = np.array([[0, 1, 4], [2, 1, 0], [4, 3, 1]], dtype=np.float32)
        test = np.array([[1, 0, 8], [0, 2, 2]], dtype=np.float32)
        combined = np.vstack([train, test])
        train_indices = np.arange(3)
        test_indices = np.arange(3, 5)
        for method in grid.PREPROCESSORS:
            precomputed = {
                "raw": combined,
                "rank": grid.nonzero_rank(combined),
                "log1p": np.log1p(combined),
            }
            transformed_train, transformed_test = grid.transform_fold(
                combined,
                precomputed,
                method,
                train_indices,
                test_indices,
                seed=7,
            )
            self.assertEqual(transformed_train.shape, train.shape)
            self.assertEqual(transformed_test.shape, test.shape)
            self.assertTrue(np.isfinite(transformed_train).all())
            self.assertTrue(np.isfinite(transformed_test).all())
        logged, _ = grid.transform_fold(
            combined,
            {"log1p": np.log1p(combined)},
            "log1p",
            train_indices,
            test_indices,
            seed=7,
        )
        np.testing.assert_allclose(logged, np.log1p(train))

    def test_expanded_prefix_is_verified_and_exported(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            features = root / "features"
            results = root / "results"
            features.mkdir()
            frame = pd.DataFrame(
                {"feature_index": range(12), "selection_source": ["core"] * 12}
            )
            for size in (1000, 3000, 5000):
                frame.to_csv(features / f"tri_2_features_{size}.csv", index=False)
            indices, output = grid.load_expanded_prefix(features, results, 2, 10)
            np.testing.assert_array_equal(indices, np.arange(10))
            self.assertTrue(output.exists())
            self.assertEqual(len(pd.read_csv(output)), 10)


if __name__ == "__main__":
    unittest.main()
