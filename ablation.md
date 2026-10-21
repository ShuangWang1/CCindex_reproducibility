# CC-Index Tree-Search Ablation

## Purpose

This experiment measures which optimizations contribute to the performance of
CC-index subgraph matching during motif tree search. It evaluates the same
candidate patterns with several one-at-a-time ablations, checks that every
variant returns the same result, and uses only the complete implementation to
decide which patterns continue to the next tree level.

The experiment is implemented in `tree_search_ablation.py`. The original
`tree_search_all_level_threshold_cpp_0806.py` is not imported or modified.

## Experimental design

At each tree level, the driver performs the following sequence:

1. Generate the next-level candidate patterns once.
2. Assign every candidate a stable pattern ID.
3. Give a deep copy of the same candidate queue to every selected variant.
4. Build a fresh target-graph cache and run each variant sequentially on one
   CPU.
5. Compare every variant's match count with `full`.
6. Stop with an error if any count differs.
7. Use only the `full` results to apply the occurrence threshold and construct
   the next tree level.

This design prevents a faster or slower variant from changing the candidate
patterns seen by later variants. Multiprocessing is intentionally excluded so
that scheduling and worker count do not confound the comparison.

## Ablation variants

The default run evaluates all five variants.

| Variant | Engine | Disabled optimization | Behavior retained |
| --- | --- | --- | --- |
| `full` | C++ | None | All CC-index filters and ordering heuristics |
| `no_cpp` | Python | Native C++ execution | Same filters and search policy as `full` |
| `no_triangle_filter` | C++ | Local triangle-count rejection and labeled triangle-signature rejection | Triangle statistics remain available for candidate ordering |
| `no_candidate_prefilter` | C++ | Label-to-vertex candidate index and degree lower-bound filter | All target vertices are scanned and labels are checked explicitly |
| `no_reorder` | C++ | Pattern-vertex and target-candidate ordering heuristics | Vertices and candidates are visited in deterministic ID order |

The three index-related subtypes isolate different sources of speed:

- `no_triangle_filter` measures pruning from the CC triangle index.
- `no_candidate_prefilter` measures initial candidate-pool reduction from the
  label index and degree bound.
- `no_reorder` measures the benefit of visiting selective pattern vertices and
  promising target candidates first.

### Checks that are never disabled

Every variant retains the following correctness conditions:

- Pattern and target vertex labels must match.
- A target vertex can map to at most one pattern vertex.
- Edges to already mapped pattern neighbors must exist in the target graph.

Removing these checks would change the definition of a valid subgraph match,
so they are not performance ablations.

## Counting modes

Exact counting is the default. Every unique matched target-vertex set is
enumerated and all variants must return the same exact count.

Use `--SM_stop_early` to stop after `--threshold` unique matches. In this mode,
the reported count is threshold-censored and is sufficient for the tree's
retain-or-prune decision. The same cutoff is applied to every variant.

Matches are deduplicated by their target-vertex set. This avoids counting
different automorphisms of the same target subgraph as separate occurrences.

## Build

The ablation extension has a separate module name, so building it does not
replace the existing `graph_pattern_cpp` binary:

```bash
python setup_ablation.py build_ext --inplace
```

This creates `graph_pattern_ablation_cpp` with C++17 and `-O3` optimization.
The extension source is
`CCindex/Claude_sonnet4.5/graph_pattern_cpp.cpp`.

## Running the experiment

Run every variant with exact counting:

```bash
python tree_search_ablation.py \
  --MaxDepth 10 \
  --threshold 10 \
  --run_name threshold_10_exact
```

Run every variant with threshold-based early stopping:

```bash
python tree_search_ablation.py \
  --MaxDepth 10 \
  --threshold 10 \
  --SM_stop_early \
  --run_name threshold_10_stop_early
```

Run only a subset of variants. `full` is mandatory because it supplies the
reference result and controls pruning:

```bash
python tree_search_ablation.py \
  --MaxDepth 8 \
  --threshold 10 \
  --variants full no_triangle_filter no_reorder \
  --run_name selected_variants
```

Use a saved queue as the starting point:

```bash
python tree_search_ablation.py \
  --MaxDepth 10 \
  --LevelStartFrom 6 \
  --GraphQueuePath ad_output_graph_ccindex_10_0811 \
  --threshold 10 \
  --run_name resume_level_6
```

### Command-line options

| Option | Default | Description |
| --- | --- | --- |
| `--MaxDepth` | `10` | Maximum tree-search depth |
| `--threshold` | `10` | Minimum occurrence count used for tree pruning |
| `--SM_stop_early` | off | Stop matching when the occurrence threshold is reached |
| `--LevelStartFrom` | `0` | Start from an existing tree level instead of the initial triangles |
| `--GraphQueuePath` | empty | Parent directory containing saved level directories |
| `--data_csv` | `ad_data.csv` | Triangle-frequency input used to construct the motif tree |
| `--target_graph` | `spatial_13months-disease-replicate_1.gml` | Target graph searched for each pattern |
| `--variants` | all variants | Space-separated variants to execute; must include `full` |
| `--output_dir` | `ad_ablation_res` | Root directory for ablation results |
| `--run_name` | timestamp | Unique experiment directory name |

The directory `ad_ablation_res/<run_name>` must not already exist. This avoids
mixing measurements from different executions.

## Result layout

```text
ad_ablation_res/
└── <run_name>/
    ├── config.json
    ├── pattern_results.csv
    ├── level_summary.csv
    ├── full/
    │   └── ccindex_tri_<level>.csv
    ├── no_cpp/
    │   └── ccindex_tri_<level>.csv
    ├── no_triangle_filter/
    │   └── ccindex_tri_<level>.csv
    ├── no_candidate_prefilter/
    │   └── ccindex_tri_<level>.csv
    └── no_reorder/
        └── ccindex_tri_<level>.csv
```

### `config.json`

Records the complete command configuration, selected variants, run name, and
whether counting was exact or threshold-censored.

### `pattern_results.csv`

Contains one row per variant and candidate pattern:

| Column | Meaning |
| --- | --- |
| `variant` | Ablation variant |
| `level` | Motif tree level |
| `pattern_id` | Stable candidate identifier shared across variants |
| `matches` | Exact or threshold-censored unique match count |
| `threshold_reached` | Whether early stopping reached the threshold |
| `search_seconds` | Per-pattern matching time, including C++ pattern conversion |
| `states_visited` | Number of recursive search states entered |
| `vertices_examined` | Target vertices examined as possible candidates |
| `candidates_accepted` | Candidates surviving all active filters |
| `rejected_label` | Candidates rejected by explicit label comparison |
| `rejected_injective` | Candidates already used in the current mapping |
| `rejected_degree` | Candidates rejected by the degree lower bound |
| `rejected_triangle_count` | Candidates rejected by local triangle count |
| `rejected_signature` | Candidates rejected by labeled triangle signature |
| `rejected_adjacency` | Candidates missing an edge to a mapped neighbor |

### `level_summary.csv`

Aggregates each variant at each level. It reports candidate count, cache-build
time, total search time, mean/median/p95 pattern time, and sums of all search
counters.

### Per-variant CSV files

Each variant directory contains its per-level pattern records. These files make
it possible to inspect a single variant without filtering the combined output.

## Interpreting results

For an ablated variant `A`, a simple speed contribution relative to `full` is:

```text
slowdown(A) = search_seconds(A) / search_seconds(full)
```

- A slowdown greater than `1` indicates that the removed optimization helped.
- A value near `1` indicates little effect for that level or workload.
- A value below `1` indicates that the optimization's overhead exceeded its
  pruning or ordering benefit for that workload.

Use the search counters to explain timing differences. For example:

- More `vertices_examined` in `no_candidate_prefilter` shows the value of the
  label candidate index.
- More `candidates_accepted` or `states_visited` in `no_triangle_filter` shows
  how much search the triangle checks eliminate.
- More `states_visited` in `no_reorder`, especially with early stopping, shows
  the benefit of exploring selective branches first.
- `no_cpp` should have similar search counters to `full`; its timing difference
  primarily measures implementation-language acceleration.

Compare variants within the same run and tree level. Avoid comparing timing
across machines, concurrent system loads, different counting modes, or
different candidate queues.

## Validation

Regression coverage is provided in `test_ablation.py` for:

- Exact-count parity across all variants.
- Stop-early parity across all variants.
- Disabled-filter counters.
- Required result-directory structure.

If `pytest` is installed, run:

```bash
pytest -q test_ablation.py
```

The driver also performs runtime parity checking at every tree level. A count
mismatch raises an exception before the full implementation's results can be
used to continue the tree search.
