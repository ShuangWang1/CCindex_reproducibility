# METABRIC rank-normalized sibling topology analysis

TRI-5 descendants of TRI-1 motif `2c9dad37a530a40bfd159b028f2d8b4f` are compared within their immediate TRI-4 parents, using the expanded trace supplied in the request.

The source has 3,948 TRI-5 motifs: 613 with nominal rank-normalized binary Cox p < 0.05, 1,469 with estimated p >= 0.05, and 1,866 undefined fits excluded from both comparison classes. Zero-median motifs remain eligible. Survival results are read from the trace, not refitted.

The pipeline reuses the BZ sibling-analysis engines with an isolated METABRIC nine-cell-type configuration. Its 747 intrinsic criteria include topology, boundary structure, degree, paths, spectra, composition, cell-type contacts, typed triangles, local organization, and cell-type/topology coupling. Label-null criteria use 128 within-motif fixed-composition permutations with seed 20260916. Four additional group-derived dispersion descriptors are exported for reference compatibility but excluded from same-parent criterion tests and PCA predictors. Geometry requiring physical tissue coordinates remains unavailable.

Run from the repository root:

```bash
python breast_cancer_survival_metabric/motif_results/sibling_topology_analysis_ranking_normalized/run_analysis.py
python breast_cancer_survival_metabric/motif_results/sibling_topology_analysis_ranking_normalized/run_same_parent_comparison.py
python breast_cancer_survival_metabric/motif_results/sibling_topology_analysis_ranking_normalized/validate_results.py
```

Parent comparisons (`results/same_parent_comparsion/README.md`; generated locally) contain complete criterion rankings and the top 24 distribution plots for each eligible parent, fixed-parent child graph galleries, the clustered within-parent BH-q heatmap and parent gallery, and criterion recurrence across parents. At least three motifs with finite criterion values are required per class. Tests rank by within-parent location BH q, decreasing absolute Cliff's delta, then metric name. Brown–Forsythe dispersion p-values receive separate within-parent BH correction.

A child with multiple parents appears independently in each applicable group. Parent-child growth includes both added vertices and added edges. METABRIC transition records refer to child coordinates; gallery mappings preserve cell labels, adjacency, and the recorded operation. Coordinates are graph layouts, not tissue positions.

`results/tables/tri5_selection_audit.csv.gz` retains all TRI-5 motifs with inclusion status. The parent audit includes every TRI-4 parent, including those without TRI-5 children. `all_parent_child_memberships.csv.gz` retains undefined-fit memberships; each eligible parent also receives `criterion_eligibility.csv` distinguishing testable, constant, and insufficient-data criteria. Gallery indexes record parent-to-child node mappings and added nodes/edges.

These exploratory comparisons describe significant versus nonsignificant motifs within this lineage. They do not measure held-out predictive accuracy; motifs sharing patients or multiple parents are dependent. Criterion BH q-values do not change motif significance labels.
