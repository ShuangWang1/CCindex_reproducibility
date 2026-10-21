# CC-index reproducibility

CC-index mines labeled spatial graph motifs by growing triangles, pruning candidate patterns, and counting their occurrences with a triangle-indexed C++ matcher. This repository collects the code for AD benchmarking and two breast cancer case studies, with survival and topology analyses organized by triangle level (TRI).

## Experiments

| Workflow | Purpose | Main entry points |
| --- | --- | --- |
| AD benchmarking | Search `spatial_13months-disease-replicate_1.gml`; compare optimized/reference C++ matchers and VF2 | `benchmark_matchers.py`, `plot_matching_time.py` |
| Subgraph ablation | Measure triangle filtering, candidate prefiltering, search ordering, and native execution | `tree_search_ablation.py`, `tree_search_ablation_v2.py` |
| BFS ablation | Record candidate counts before and after extendability, pool, and isomorphism filtering | `tree_search_ablation_bfs.py` |
| Basel/Zurich (BZ) | Rank-normalized patient-level motif survival screening, structural lineage, and sibling topology | `breast_cancer_survival/` |
| METABRIC | Corresponding patient-level survival and topology workflows | `breast_cancer_survival_metabric/` |
| Cross-cohort comparison | Exact labeled matching of common motifs and same-parent topology comparisons by level | `breast_cancer_survival_common/` |

The **current matcher is `graph_pattern_cpp.cpp`**, built as `graph_pattern_cpp`. Standard AD, BFS, breast cancer, topology-restricted, and exact-expectation callers use this implementation. The earlier implementation is retained as `graph_pattern_cpp_reference.cpp` for explicit benchmarks. Optimization ablations use their separate instrumented engine; their timings are not timings of the current production matcher.

## Installation

Use Linux, Python 3.11 or newer, and a C++17 compiler. Validation of this export uses Python 3.13; dependencies are resolved for your Python version.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python setup.py build_ext --inplace
python setup_ablation.py build_ext --inplace
```

The optional historical BZ sparse-artifact preparation command also needs `python -m pip install -r requirements-artifacts.txt`. Existing classification helpers are retained because that command imports them; classification model tuning is outside the reproduction sequence described here.

## Reproduce

1. Supply external inputs using the [input manifest](docs/INPUTS.md).
2. Follow the [AD benchmarking and ablation guide](docs/AD_EXPERIMENTS.md).
3. Follow the [breast cancer survival and topology guide](docs/BREAST_CANCER.md), from preprocessing through per-level comparisons.
4. Run the [validation commands](docs/VALIDATION.md).

This is a **code-and-documentation release**. Study graphs, AnnData files, clinical records, sparse matrices, measured result tables, figures, and compiled extensions are excluded. Synthetic tests generate temporary fixtures locally. Full study runs require the external inputs and can be computationally substantial. No performance or survival results are regenerated merely by installing or testing the code.

## Analysis conventions

TRI-N denotes triangle-growth level, rather than a fixed number of vertices. Counts deduplicate matches by their target vertex set; matching is non-induced unless an analysis explicitly applies an induced-topology check. Positive stopping thresholds censor counts after that many distinct sets, so survival frequency extraction uses exact counting.

Rank normalization is performed within each image and TRI vocabulary. Positive count ties receive average ranks; absent motifs receive the last rank. Patient scores equally average available images. Raw-count analyses instead sum image counts within patients. High/enriched groups are strictly above the cohort median, with ties assigned low/non-enriched. BZ survival time is in months; METABRIC survival time is in days.

Primary positive-median screens and expanded analyses of structural relatives have different eligibility rules; zero-median relatives remain in the expanded analysis. Nominal Cox significance and Benjamini–Hochberg correction are reported separately. Structural parents are potential graph-growth relationships reconstructed from topology, not a unique recorded search history or physical tissue ancestry. Cohort-specific sibling studies focus on TRI-5; shared-parent cross-cohort comparisons cover TRI-2–TRI-7.
