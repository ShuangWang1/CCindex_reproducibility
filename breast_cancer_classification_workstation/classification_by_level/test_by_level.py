from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from run_by_level import prepare_level_artifacts


class LevelArtifactTests(unittest.TestCase):
    def test_prepare_slices_columns_and_reindexes_features(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            matrix = sparse.csr_matrix(
                np.asarray([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)
            )
            sparse.save_npz(source / "motif_counts.npz", matrix)
            np.save(source / "feature_triangle_levels.npy", np.asarray([2, 3, 2, 3]))
            with gzip.open(source / "feature_index.csv.gz", "wt", encoding="utf-8") as out:
                pd.DataFrame(
                    {
                        "feature_index": range(4),
                        "triangle_level": [2, 3, 2, 3],
                        "wl_hash": ["a", "b", "c", "d"],
                    }
                ).to_csv(out, index=False)
            pd.DataFrame({"sample": ["sample1", "sample2"]}).to_csv(
                source / "samples.csv", index=False
            )
            (source / "audit.json").write_text(
                json.dumps(
                    {
                        "feature_count": 4,
                        "nonzero_count": 8,
                        "search_exhausted_missing_level_files": [
                            "sample2/ccindex_tri_3.csv"
                        ],
                    }
                ),
                encoding="utf-8",
            )

            output = root / "levels"
            prepare_level_artifacts(source, output, (2, 3))

            level_two = sparse.load_npz(output / "tri_2" / "motif_counts.npz")
            self.assertTrue(np.array_equal(level_two.toarray(), [[1, 3], [5, 7]]))
            features = pd.read_csv(output / "tri_2" / "feature_index.csv.gz")
            self.assertEqual(features["feature_index"].tolist(), [0, 1])
            self.assertEqual(features["wl_hash"].tolist(), ["a", "c"])
            audit = json.loads((output / "tri_3" / "audit.json").read_text())
            self.assertEqual(audit["matrix_shape"], [2, 2])
            self.assertEqual(audit["search_exhausted_missing_level_files"], [
                "sample2/ccindex_tri_3.csv"
            ])


if __name__ == "__main__":
    unittest.main()
