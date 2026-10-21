"""Make source-tree scripts importable and gate external-data integration tests."""

import os
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent
for directory in [ROOT, ROOT / "breast_cancer_survival", ROOT / "breast_cancer_survival_common"]:
    sys.path.insert(0, str(directory))


def pytest_collection_modifyitems(items):
    if os.environ.get("CCINDEX_RUN_INTEGRATION") == "1":
        return
    skip = pytest.mark.skip(reason="Set CCINDEX_RUN_INTEGRATION=1 and restore study artifacts")
    for item in items:
        if item.get_closest_marker("integration"):
            item.add_marker(skip)
