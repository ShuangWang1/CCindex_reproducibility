# AD benchmark and ablations

The AD workload uses `spatial_13months-disease-replicate_1.gml` and its accompanying base-triangle table `ad_data.csv`. Supply both inputs as described in [INPUTS.md](INPUTS.md).

## Matcher benchmark

Build the current and earlier reference implementations:

```bash
python setup_benchmark.py build_ext --inplace
python benchmark_matchers.py --MaxDepth 6 --threshold 10 \
  --CppModule graph_pattern_cpp --ResultDir ad_matcher_results_current
python benchmark_matchers.py --MaxDepth 6 --threshold 10 \
  --CppModule graph_pattern_cpp_reference --ResultDir ad_matcher_results_reference
python plot_matching_time.py \
  --old-dir ad_matcher_results_reference --current-dir ad_matcher_results_current \
  --output figures/matcher_comparison.png --summary figures/matcher_summary.csv
```

Run each implementation in a separate process. Use fresh result directories. The plotting command checks that pattern identifiers and exact match counts agree before comparing per-level timing distributions. Keep the target, threshold, depth, and initial vocabulary fixed across runs. Leave early stopping disabled for exact count comparisons. The shared driver is deterministic for a fixed input; its independently grown queues must pass this identity check before timings are compared.

The same tree-search driver exposes the VF2 baseline:

```bash
python benchmark_matchers.py --MaxDepth 6 --threshold 10 \
  --SM_method vf2 --ResultDir ad_matcher_results_vf2
```

The violin plotting tool accepts the C++ benchmark CSV schema. VF2 has a separate timing output layout; inspect its output rather than passing it to that plotting tool.

## Subgraph optimization ablation

```bash
python setup_ablation.py build_ext --inplace
python tree_search_ablation.py --MaxDepth 6 --threshold 10 --run_name exact
python tree_search_ablation_v2.py --MaxDepth 6 --threshold 10 --run_name cumulative
```

The first run removes one optimization at a time: C++ execution, triangle filtering, candidate prefiltering, or search reordering. The cumulative run adds those optimizations in stages. Both provide identical candidate queues to all variants, verify count parity, and let only the full reference variant determine pruning. Default counting is exact; `--SM_stop_early` explicitly enables threshold-censored counting.

Outputs include run configuration, per-pattern timing/search counters, per-level summaries, and variant-specific count tables. See [ablation.md](../ablation.md) for the detailed instrumented experimental design. The full ablation variant is the instrumented reference engine, not the current production extension.

## BFS candidate-generation ablation

```bash
python tree_search_ablation_bfs.py --MaxDepth 6 --threshold 10
```

This records `base_mtf`, `extendable_mtf`, `pool_mtf`, and `iso_mtf` at each level in the generated `bfs_mtf_records_ablation.csv`. They measure candidate-generation/filtering stages; they are not independent matcher implementations or four separate timing runs. Candidate occurrence counting uses the current matcher. Keep exact counting enabled for comparable frequency/pruning results.

The optional `tree_search_topology_restrict.py` uses root-centered face-hop shells to constrain motif growth. Its topology restriction is a distinct experiment from the BFS stage-count ablation. `enumerate_topology/` separately enumerates unlabeled topologies without searching an AD target.
