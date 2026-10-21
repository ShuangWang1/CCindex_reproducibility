# Rank-normalized sibling topology analysis

This directory reproduces the full TRI-5 topology analysis in
`sibling_topology_analysis_new` using the rank-normalized binary Cox result from
the expanded lineage of TRI-1 motif `65319d6de1808abbc65d476aa3e9436e`.

Motifs are classified as significant when their estimated, finite
`p_value_rank_binary` is below 0.05. Undefined and non-estimable Cox results are
excluded from both classes. Criterion-level Benjamini-Hochberg correction is
used for exploratory criterion comparisons; it does not change motif labels.

Run from the repository root:

```bash
python breast_cancer_survival/motif_results/sibling_topology_analysis_ranking_normalized/run_analysis.py
python breast_cancer_survival/motif_results/sibling_topology_analysis_ranking_normalized/run_same_parent_comparison.py
```

The first command writes the overall distributions, comparisons, PCA figures,
and topology-matched representative pairs to `results/`. The second writes the
nested same-TRI-4-parent comparison to `results/same_parent_comparsion/`, keeping
the spelling used by the reference analysis for layout compatibility.
