from __future__ import annotations

import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from metabric_ml.data import parse_grade, prepare_metabric_features
from metabric_ml.modeling import (
    _select_features,
    _target_labels,
    evaluate_baselines,
    normalize_within_triangle_levels,
)
from metabric_ml.reporting import create_report


HEADER = [
    "pattern_id",
    "triangles",
    "nodes",
    "edges",
    "node_labels",
    "edge_list",
    "wl_hash",
    "matches",
    "seconds",
]


def _write_motif(path: Path, level: int, rows: list[tuple[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(HEADER)
        for index, (motif_hash, matches) in enumerate(rows, start=1):
            writer.writerow(
                [f"tri_{level}_{index}", level, 4, 5, "0&1&1&2", "0-1;0-2", motif_hash, matches, 0.1]
            )


class MetabricWorkstationTests(unittest.TestCase):
    def test_parse_grade(self) -> None:
        self.assertEqual(parse_grade("[1. 0. 0.]"), 0)
        self.assertEqual(parse_grade("[0. 1. 0.]"), 1)
        self.assertEqual(parse_grade("[0. 0. 1.]"), 2)
        self.assertIsNone(parse_grade("[nan nan nan]"))
        with self.assertRaisesRegex(ValueError, "one-hot"):
            parse_grade("[1. 1. 0.]")

    def test_prepare_uses_source_sample_and_sparse_zero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            root = tmp_path / "motifs"
            root.mkdir()
            with (root / "run_manifest.csv").open("w", newline="", encoding="utf-8") as output:
                writer = csv.writer(output)
                writer.writerow(
                    [
                        "dataset",
                        "sample",
                        "source_sample",
                        "graph_file",
                        "nodes",
                        "edges",
                        "source_triangle_types",
                        "retained_triangle_types",
                        "completed_triangle_level",
                        "seconds",
                    ]
                )
                writer.writerow(["metabric", "sample1", "10", "a.gml", 100, 200, 5, 3, 2, 1])
                writer.writerow(["metabric", "sample2", "20", "b.gml", 120, 240, 6, 4, 2, 1])
            grades = tmp_path / "grades.csv"
            grades.write_text("id,grade\n10,[1. 0. 0.]\n20,[0. 0. 1.]\n", encoding="utf-8")
            _write_motif(root / "sample1" / "ccindex_tri_2.csv", 2, [("aaa", 3), ("zero", 0)])
            _write_motif(root / "sample2" / "ccindex_tri_2.csv", 2, [("bbb", 4)])

            prepared = prepare_metabric_features(
                root, grades, tmp_path / "artifacts", h5ad_path=None, triangle_levels=(2,)
            )
            matrix = sparse.load_npz(prepared.matrix_path)
            samples = pd.read_csv(prepared.samples_path)
            self.assertEqual(matrix.shape, (2, 2))
            self.assertEqual(matrix.nnz, 2)
            self.assertEqual(samples["grade"].tolist(), [0, 2])
            self.assertEqual(samples["source_sample"].tolist(), [10, 20])

    def test_relative_normalization_and_training_only_selection(self) -> None:
        matrix = sparse.csr_matrix(
            np.asarray([[2, 2, 6], [0, 5, 5], [1, 0, 0]], dtype=np.float32)
        )
        levels = np.asarray([2, 2, 3], dtype=np.int8)
        normalized = normalize_within_triangle_levels(matrix, levels).toarray()
        self.assertTrue(np.allclose(normalized[0], [0.5, 0.5, 1.0]))
        self.assertTrue(np.allclose(normalized[1], [0.0, 1.0, 1.0]))
        selected = _select_features(matrix[:2], np.asarray([0, 1]), 0.5, 2)
        self.assertLessEqual(len(selected), 2)

    def test_target_mapping(self) -> None:
        grades = np.asarray([0, 1, 2, 2])
        self.assertEqual(_target_labels(grades, "multiclass").tolist(), [0, 1, 2, 2])
        self.assertEqual(_target_labels(grades, "binary_01_vs_2").tolist(), [0, 0, 1, 1])

    def test_small_evaluation_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            artifacts = tmp_path / "artifacts"
            results = tmp_path / "results"
            artifacts.mkdir()
            rng = np.random.default_rng(3)
            grades = np.repeat([0, 1, 2], 4)
            dense = rng.poisson(2, size=(12, 18)).astype(np.float32)
            dense[np.arange(12), grades * 3] += 8
            sparse.save_npz(artifacts / "motif_counts.npz", sparse.csr_matrix(dense))
            np.save(artifacts / "feature_triangle_levels.npy", np.repeat([2, 3, 4], 6))
            pd.DataFrame(
                {
                    "row_index": np.arange(12),
                    "sample": [f"sample{i + 1}" for i in range(12)],
                    "source_sample": np.arange(1, 13),
                    "patient_id": [f"p{i + 1}" for i in range(12)],
                    "grade": grades,
                    "is_labeled": True,
                    "nodes": 100 + np.arange(12),
                    "edges": 200 + np.arange(12),
                    "source_triangle_types": 10 + grades,
                    "retained_triangle_types": 5 + grades,
                }
            ).to_csv(artifacts / "samples.csv", index=False)
            with gzip.open(artifacts / "feature_index.csv.gz", "wt", newline="", encoding="utf-8") as output:
                writer = csv.writer(output)
                writer.writerow(
                    [
                        "feature_index",
                        "triangle_level",
                        "wl_hash",
                        "document_frequency",
                        "nodes",
                        "edges",
                        "representative_node_labels",
                    ]
                )
                for index in range(18):
                    writer.writerow([index, 2 + index // 6, f"hash{index}", 12, 4, 5, "0&1&2"])
            (artifacts / "audit.json").write_text(
                json.dumps(
                    {
                        "sample_count": 12,
                        "labeled_sample_count": 12,
                        "unlabeled_sample_count": 0,
                        "class_counts": {"0": 4, "1": 4, "2": 4},
                        "feature_count": 18,
                        "nonzero_count": int(np.count_nonzero(dense)),
                        "triangle_levels": [2, 3, 4],
                    }
                ),
                encoding="utf-8",
            )

            config = evaluate_baselines(
                artifacts,
                results,
                folds=2,
                repeats=1,
                max_features=8,
                min_prevalence=0.1,
                trees=5,
                n_jobs=1,
            )
            self.assertEqual(set(config["best"]), {"multiclass", "binary_01_vs_2"})
            self.assertEqual(len(pd.read_csv(results / "fold_metrics.csv")), 32)
            report = create_report(artifacts, results)
            self.assertTrue(report.is_file())
            self.assertIn("sample-level", report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
