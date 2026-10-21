import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import run_bz_parameter_tuning as bz


class BZParameterTuningTests(unittest.TestCase):
    def test_default_grid_is_binary_expanded_style(self):
        args = bz.parse_args(["all"])
        self.assertEqual(args.levels, (2, 3, 4, 5, 6, 7))
        self.assertEqual(args.sizes, (10, 50, 100, 150, 200, 250, 300, 350, 400, 450))
        self.assertEqual((args.folds, args.repeats, args.seed), (5, 5, 42))

    def test_inventory_uses_run_summaries_and_records_missing_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            motifs = root / "motifs"
            output = root / "artifacts"
            motifs.mkdir()
            pd.DataFrame(
                [{"sample": "sample2", "source_sample": "b", "completed_triangle_level": 5}]
            ).to_csv(motifs / "run_manifest.csv", index=False)
            grades = root / "grades.csv"
            grades.write_text(
                "id,grade\na,[1. 0. 0.]\nb,[0. 0. 1.]\nc,[0. 1. 0.]\n",
                encoding="utf-8",
            )
            sample_manifest = root / "sample_manifest.csv"
            pd.DataFrame(
                {
                    "sample_index": [1, 2, 3],
                    "gml_file": ["sample1.gml", "sample2.gml", "sample3.gml"],
                    "source_sample": ["a", "b", "c"],
                }
            ).to_csv(sample_manifest, index=False)
            for number, source, completed in ((1, "a", 7), (2, "b", 5)):
                sample_dir = motifs / f"sample{number}"
                sample_dir.mkdir()
                summary = {
                    "dataset": "BZ",
                    "sample": f"sample{number}",
                    "source_sample": source,
                    "graph_file": f"sample{number}.gml",
                    "nodes": 10,
                    "edges": 20,
                    "source_triangle_types": 3,
                    "retained_triangle_types": 2,
                    "completed_triangle_level": completed,
                    "seconds": 1.0,
                }
                (sample_dir / "run_summary.json").write_text(json.dumps(summary), encoding="utf-8")
                for level in range(2, completed + 1):
                    (sample_dir / f"ccindex_tri_{level}.csv").touch()

            inventory_path, audit_path = bz.build_source_inventory(
                motifs, grades, sample_manifest, output
            )
            inventory = pd.read_csv(inventory_path)
            self.assertEqual(inventory["sample"].tolist(), ["sample1", "sample2"])
            excluded = pd.read_csv(output / "excluded_samples.csv")
            self.assertEqual(excluded["source_sample"].tolist(), ["c"])
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            self.assertEqual(audit["source_top_level_manifest_rows"], 1)
            self.assertEqual(audit["labeled_sample_count"], 2)
            self.assertEqual(audit["binary_class_counts"], {"0": 1, "1": 1})
            self.assertEqual(audit["exhausted_before_tri_7_samples"], 1)

    def test_binary_target_combines_grade_zero_and_one(self):
        grades = pd.Series([0, 1, 2, 2])
        self.assertEqual(
            bz.tuning.target_for_task(grades.to_numpy(), "binary_01_vs_2").tolist(),
            [0, 0, 1, 1],
        )


if __name__ == "__main__":
    unittest.main()
