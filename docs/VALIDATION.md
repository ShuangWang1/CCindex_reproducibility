# Validation

From a clean checkout, install dependencies and build both extensions:

```bash
python -m pip install -r requirements-dev.txt
python setup.py build_ext --inplace
python setup_ablation.py build_ext --inplace
python -m pytest -q -m 'not integration'
```

Tests cover exact matcher parity with independent VF2 enumeration, automorphism deduplication, threshold counting, empty/no-match workloads, instrumented ablation parity, synthetic workflow outputs, patient aggregation, rank normalization, Cox calculations, structural growth/lineage, and topology comparisons. All portable graph/clinical fixtures are synthetic and created in temporary directories.

Checks against saved study results are marked `integration` and require `CCINDEX_RUN_INTEGRATION=1` as well as the corresponding external artifacts. After restoring those artifacts, run:

```bash
CCINDEX_RUN_INTEGRATION=1 python -m pytest -q -m integration
```

The original direct unittest checks remain available:

```bash
python -m unittest discover -s breast_cancer_survival/tests
python -m unittest discover -s breast_cancer_survival_metabric/tests
python breast_cancer_survival_common/test_compare_common_same_parent.py
```

## Publication validation

The publication copy was validated on Python 3.13 with locally built production,
reference, and instrumented ablation extensions. The portable suite completed
with **160 passed, 1 skipped, and 14 external-artifact integration tests
deselected**. Separate-process synthetic current/reference benchmarks agreed on
candidate identities and exact counts. The principal experiment/preparation
commands also passed `--help` checks without study inputs present.

This checks source packaging and synthetic behavior. It does not constitute a
rerun of the full AD benchmark or the patient-level study analyses.

For a real-data run, retain full configuration, package versions, target identity, threshold/counting mode, and complete output directories. Benchmark comparisons require identical candidate identities and exact counts; interpret timing only within equivalent workloads on the same machine. Generated outputs remain excluded from Git.
