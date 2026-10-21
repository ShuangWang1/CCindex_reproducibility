# Complete TRI-7 trace analysis

This project visualizes all 450 significant TRI-7 trace folders and computes the comprehensive graph-criterion set for the deduplicated union of TRI-2 through TRI-7 motifs.

## Population

Each `(triangle_level, wl_hash)` contributes once to the criterion distributions, regardless of how many target lineages contain it. `tables/trace_memberships.csv.gz` preserves the many-to-many lineage memberships.

| Level | Unique motifs | Nonsignificant | Significant |
|---:|---:|---:|---:|
| 2 | 26 | 26 | 0 |
| 3 | 44 | 44 | 0 |
| 4 | 157 | 155 | 2 |
| 5 | 420 | 405 | 15 |
| 6 | 727 | 651 | 76 |
| 7 | 450 | 0 | 450 |

Raw significance means Cox `p < 0.05`. TRI-2 and TRI-3 contain no significant motifs in this traced union; TRI-7 contains no nonsignificant motifs. Their within-level significance plots are therefore descriptive and explicitly marked not testable.

## Criteria and statistics

The pipeline computes topology, composition, contacts, typed triangles, type–topology coupling, and fixed-composition label-null Z scores using 128 deterministic permutations per motif. Four between-motif heterogeneity distances are calculated relative to the centroid within each `(TRI level, significance group)` and are labeled outcome-derived/descriptive-only.

Cross-level intrinsic comparisons use Kruskal–Wallis tests, epsilon-squared, and BH correction. Within-level intrinsic comparisons use Mann–Whitney tests, Cliff's delta, Brown–Forsythe tests, and BH correction separately by level. Constant criteria remain in figures as horizontal segments.

## Outputs

- `<target_hash>/all_trace_back_growth_transitions.png`: fixed final-TRI-7 scaffold with each growth step highlighted in gold.
- `tables/tri2_to_tri7_motif_criteria.csv.gz`: one row per deduplicated motif.
- `tables/trace_memberships.csv.gz`: motif-to-target membership table.
- `tables/across_level_comparisons.csv` and `within_level_significance_comparisons.csv`: inferential results.
- `tables/criteria_distributions.csv.gz`: descriptive summaries.
- `figures/across_levels/`: six-level violin pages with points colored by raw significance.
- `figures/by_significance/tri2` through `tri7`: significance-comparison violin pages.

The trace relations are potential labeled-topological growth relations because original search parent IDs were not saved. Gallery coordinates are deterministic graph layouts, not physical tissue coordinates.
