from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

from enumerate_topology.enumerate_topologies import expand_code, seed_code
from enumerate_topology.selective_res.selective_enumerate import (
    ParentLinks,
    extract_features,
    select_diverse_beam,
    write_level_archive,
)


class SelectiveEnumerationTests(unittest.TestCase):
    def test_feature_extraction_seed(self) -> None:
        feature = extract_features(seed_code())
        self.assertEqual(3, feature.nodes)
        self.assertEqual(3, feature.boundary_vertices)
        self.assertEqual(0, feature.interior_vertices)
        self.assertEqual(1, len(feature.dual_degree_sequence))

    def test_all_candidates_retained_below_cap(self) -> None:
        parent = seed_code()
        children, _ = expand_code(parent)
        pool = {
            child: ParentLinks({0}, set())
            for child in children
        }
        features = {code: extract_features(code) for code in pool}
        result = select_diverse_beam(pool, features, 1, 10, 42)
        self.assertEqual(set(pool), result.selected)
        self.assertEqual(100.0, result.parent_coverage)

    def test_branch_coverage_and_determinism(self) -> None:
        states = {seed_code()}
        for _level in range(2, 9):
            children = set()
            for parent in states:
                local, _ = expand_code(parent)
                children.update(local)
            states = children
        parents = sorted(states)
        pool: dict[bytes, ParentLinks] = {}
        for parent_index, parent in enumerate(parents):
            children, _ = expand_code(parent)
            for child in children:
                links = pool.setdefault(child, ParentLinks.empty())
                (links.attach if child[0] == parent[0] + 1 else links.close).add(
                    parent_index
                )
        features = {code: extract_features(code) for code in pool}
        result_a = select_diverse_beam(pool, features, len(parents), 200, 42)
        reversed_pool = dict(reversed(list(pool.items())))
        result_b = select_diverse_beam(reversed_pool, features, len(parents), 200, 42)
        self.assertEqual(result_a.selected, result_b.selected)
        self.assertEqual(100.0, result_a.parent_coverage)

    def test_tri12_integration_count_and_coverage(self) -> None:
        expected = (1, 1, 1, 4, 6, 18, 41, 134, 390, 1339, 4429)
        states = {seed_code()}
        observed = [1]
        for _level in range(2, 12):
            children = set()
            for parent in states:
                local, _ = expand_code(parent)
                children.update(local)
            states = children
            observed.append(len(states))
        self.assertEqual(expected, tuple(observed))

        parents = sorted(states)
        pool: dict[bytes, ParentLinks] = {}
        for parent_index, parent in enumerate(parents):
            children, _ = expand_code(parent)
            for child in children:
                links = pool.setdefault(child, ParentLinks.empty())
                (links.attach if child[0] == parent[0] + 1 else links.close).add(
                    parent_index
                )
        self.assertEqual(15_645, len(pool))
        features = {code: extract_features(code) for code in pool}
        result = select_diverse_beam(pool, features, len(parents), 10_000, 54)
        self.assertEqual(10_000, len(result.selected))
        self.assertEqual(100.0, result.parent_coverage)
        self.assertEqual(100.0, result.operation_coverage)

    def test_archive_round_trip(self) -> None:
        code = seed_code()
        features = {code: extract_features(code)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "level.jsonl.gz"
            write_level_archive(
                path,
                1,
                {code},
                [],
                {code: ParentLinks.empty()},
                features,
                {code: "all"},
            )
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                row = json.loads(handle.readline())
            self.assertEqual(code, bytes.fromhex(row["canonical_code_hex"]))
            self.assertEqual(1, row["triangle_level"])


if __name__ == "__main__":
    unittest.main()
