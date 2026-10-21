# Selective diverse topology enumeration

This pipeline continues the triangle-topology BFS to level 40 with a retained
frontier of at most 10,000 motifs. It uses the same exact attach/close growth
operations and BLISS topology deduplication as the exhaustive enumerator.

Tri-1 through Tri-11 are globally exhaustive. Tri-12 generates 15,645 motifs,
so selection begins there. Later candidate pools contain every unique child of
the preceding retained beam; they are not the complete global BFS levels.

## Diversity policy

Selection is deterministic and proceeds in this order:

1. Greedy set cover retains at least one child from every parent motif.
2. Additional children cover both attach and close directions for each parent
   where the 10,000 limit permits.
3. Remaining capacity is balanced across structural strata, using graph and
   triangle-dual diameter, degree structure, boundary/interior structure,
   closable wedges, and triangle participation.

Within a stratum, fine degree and dual signatures are round-robined. Stable
seeded hashes only break otherwise equal choices. This is a novelty-oriented
beam, not an unbiased statistical sample.

## Run

From the repository root:

```bash
python enumerate_topology/selective_res/selective_enumerate.py \
  --max-triangles 40 \
  --beam-size 10000 \
  --workers 8 \
  --selection-seed 42 \
  --examples-per-level 10 \
  --output-dir enumerate_topology/selective_res/results
```

## Outputs

- `results/levels/tri_XXX.jsonl.gz` contains every retained topology, its
  canonical code, features, selection reason, and attach/close parent IDs.
- `results/level_summary.csv` reports pool size, retained size, timings, and
  parent/operation/signature coverage.
- `results/diversity_metrics.csv` isolates the lineage, operation, structural,
  and node-range coverage measurements for each level.
- `results/node_count_distribution.csv` compares candidate and selected node
  distributions.
- `results/samples/` contains up to ten representative GML/PNG pairs per level.
  Their organic planar layout moves boundary as well as interior vertices, so
  large motifs are not artificially forced onto a regular polygon; a geometric
  acceptance gate preserves edge planarity and node clearance.
- `results/beam_growth.png` and `results/diversity_coverage.png` summarize the
  search.

## Completed run

The included run reached Tri-40 in about 1,013 seconds with a peak aggregate
RSS of 1.23 GiB. It stores 296,364 retained topology records across all levels.

- Tri-1 through Tri-11 are exhaustive.
- Tri-12 generated 15,645 candidates and retained 10,000 with 100% parent,
  attach/close, and coarse-signature coverage.
- Tri-40 generated 284,495 candidates and retained 10,000 spanning 23–41
  vertices. It preserved 100% of parent lineages, 50.50% of parent-operation
  tokens, and 9,454 of 48,579 observed coarse structural signatures.

The percentages quantify coverage within the candidate pool generated from the
preceding retained beam; they do not imply coverage of the inaccessible full
global Tri-40 BFS level.

## Tests

```bash
python -m unittest enumerate_topology.selective_res.test_selective_enumerate
```
