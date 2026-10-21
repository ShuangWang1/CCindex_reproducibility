"""Output regression checks for the rank-normalized same-parent analysis."""

from __future__ import annotations

import json
from pathlib import Path
import os
import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("CCINDEX_RUN_INTEGRATION") != "1", reason="Requires CCINDEX_RUN_INTEGRATION=1 and saved study outputs")
def test_generated_same_parent_counts() -> None:
    summary_path = ROOT / "results" / "same_parent_comparsion" / "analysis_summary.json"
    if not summary_path.exists():
        pytest.skip("Restore generated study outputs for this integration check")
    summary = json.loads(summary_path.read_text())
    assert summary["trace_parents"] == 693
    assert summary["analyzable_parents"] == 364
    assert summary["mixed_parents"] == 128
    assert summary["eligible_parents"] == 40
    assert summary["undefined_rank_p_values_excluded"] is True
