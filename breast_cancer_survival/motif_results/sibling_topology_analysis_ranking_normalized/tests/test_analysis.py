"""Regression checks for the rank-normalized sibling-topology adapter."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import os
import pytest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("rank_analysis_tested", ROOT / "run_analysis.py")
assert SPEC is not None and SPEC.loader is not None
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def test_selection_excludes_undefined_and_nonestimated() -> None:
    source = pd.DataFrame(
        {
            "triangle_level": [5, 5, 5, 5, 4],
            "wl_hash": list("abcde"),
            "node_labels": ["x"] * 5,
            "edge_list": ["x"] * 5,
            "p_value_rank_binary": [0.01, 0.05, np.nan, 0.01, 0.01],
            "coef_rank_binary": [1.0] * 5,
            "hazard_ratio_rank_binary": [2.0] * 5,
            "median_patient_rank_score": [0.4] * 5,
            "rank_binary_fit_status": ["estimated", "estimated", "undefined", "failed", "estimated"],
        }
    )
    selected = analysis.select_estimable_motifs(source, 5, 0.05)
    assert selected.wl_hash.tolist() == ["a", "b"]
    assert selected.significance_group.tolist() == ["significant", "nonsignificant"]


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("CCINDEX_RUN_INTEGRATION") != "1", reason="Requires CCINDEX_RUN_INTEGRATION=1 and saved study outputs")
def test_generated_cohort_and_rank_native_columns() -> None:
    summary_path = ROOT / "results" / "analysis_summary.json"
    if not summary_path.exists():
        pytest.skip("Restore generated study outputs for this integration check")
    summary = json.loads(summary_path.read_text())
    assert summary["significant_motifs"] == 176
    assert summary["nonsignificant_motifs"] == 1859
    assert summary["excluded_undefined_p_value"] == 3264
    assert summary["representative_pairs"] == 12
    columns = pd.read_csv(
        ROOT / "results" / "tables" / "tri5_motif_criteria.csv.gz", nrows=0
    ).columns
    assert "p_value_rank_binary" in columns
    assert not set(analysis.COMPATIBILITY_COLUMNS).intersection(columns)
