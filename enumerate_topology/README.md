> Workflow reference retained from the research workspace. Generated outputs and historical measured results are excluded from this release. Start with the root reproduction guides for the current execution sequence.

# Triangle topology enumeration

This directory enumerates the pure, unlabeled graph topologies produced by the
triangle-growth rules in `tree_search_all_level_threshold_cpp_0806.py`.
Every exported vertex has label `"1"`; there is no target graph, occurrence
threshold, CC-index, or VF2 subgraph matching step.

## Definition

The search starts from one triangle. Each BFS level adds exactly one triangle
using one of the original operations:

1. attach a new vertex to both endpoints of a boundary edge; or
2. close two consecutive boundary edges when the new opposite edge creates
   exactly one triangle.

An edge is on the boundary exactly when it belongs to one 3-clique. The
resulting boundary is required to be a single cycle. Motifs are identified by
ordinary unlabeled graph isomorphism, not by vertex IDs, planar embedding, or
boundary orientation. Exact BLISS canonical labeling from `python-igraph` is
used instead of a Weisfeiler–Lehman hash.

The implementation uses the repository's existing `python-igraph`, NetworkX,
SciPy, and Matplotlib installations. Linux `/proc` is used to monitor the
combined memory of the coordinator and worker processes.

This enumerates abstract triangulated-disk topologies reachable under the
reference rules. It does **not** test whether coordinates exist that realize a
topology as an ordinary geometric Delaunay triangulation.

## Reproduce the capped run

From the repository root:

```bash
python enumerate_topology/enumerate_topologies.py \
  --max-triangles 40 \
  --workers 8 \
  --wall-seconds 3600 \
  --memory-gib 20 \
  --max-states-per-level 5000000 \
  --samples-per-level 5 \
  --output-dir enumerate_topology/results
```

The run stops at the first resource cap. A level is marked `exact` only after
all parents have been expanded and all children deduplicated. If a cap is hit
mid-level, `unique_topologies` is left blank, `partial_unique_seen` is reported,
and partial motifs are discarded. Caps are checked between worker batches, so
the partial state count can exceed the configured limit by a small amount.

The exit status is `0` when `--max-triangles` is completed and `2` when a
resource cap stops the run.

## Results

- `results/level_counts.csv` contains exact counts and timing/memory data.
- `results/node_count_distribution.csv` breaks each exact level down by nodes.
- `results/run_summary.json` records configuration, versions, and termination.
- `results/topology_counts.png` plots exact counts on a logarithmic scale.
- `results/samples/` contains up to five deterministic GML/PNG examples per
  exact level, spread across the available vertex-count range. PNGs use an
  organic planar layout: a safe Tutte embedding is refined by moving both
  boundary and interior vertices, balancing edge lengths, face areas, and node
  clearance. A geometric acceptance gate prevents edge crossings and falls
  back along the safe deformation path when a force-layout proposal folds.
- `results/samples_index.csv` indexes those examples.

The completed capped run produced this exact sequence through level 16:

```text
1, 1, 1, 4, 6, 18, 41, 134, 390, 1,339, 4,429, 15,645, 55,106,
199,285, 724,817, 2,671,588
```

Level 17 stopped at the state cap after 5,001,010 partial unique states, so no
exact level-17 count is claimed. The run took about 425 seconds and its peak
aggregate process RSS was 5.65 GiB. Recent complete levels grow by roughly
3.5–3.7 times per added triangle. Extending that rate to level 40 suggests an
order of magnitude around `1e20` motifs. This is only an extrapolation, but it
makes exhaustive level-40 materialization inaccessible on this workstation.

## Tests

```bash
python -m unittest enumerate_topology.test_enumerate_topologies
```

The tests cover relabeling-invariant canonicalization, equivalence with a
literal port of the reference growth loops, the exact sequence through level
10, planar triangulated-disk invariants, clean cap handling, and GML metadata.
