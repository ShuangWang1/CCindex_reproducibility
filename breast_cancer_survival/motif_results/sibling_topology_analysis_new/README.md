# Sibling topology analysis: triangulation criteria

This project implements the criteria in
`triangulation_motif_topology_criteria.md` for the estimable TRI-5 motifs in the
example lineage. It compares motifs with raw Cox p-values below 0.05 against
same-level nonsignificant motifs.

Run from the repository root:

```bash
python breast_cancer_survival/motif_results/sibling_topology_analysis_new/run_analysis.py
```

Outputs are written under `results/tables` and `results/figures`. Geometry and
patient-level criteria that cannot be recovered from the lineage graph summary
are retained in the criterion registry with explicit unavailability reasons.

The figure `results/figures/tri5_all_criteria_pca.png` is an unsupervised PCA of
median-imputed, standardized intrinsic criteria. Group-centroid heterogeneity
criteria are excluded because they use the significance group and would leak
the visualization label into the feature matrix.

To regenerate only the violin plots from the saved tables:

```bash
python breast_cancer_survival/motif_results/sibling_topology_analysis_new/run_analysis.py --plots-only
```

The representative gallery defaults to 12 matched pairs (24 motifs). Change
the count with `--representative-pairs N`; the value must be at least six to
retain one pair for every shared unlabeled TRI-5 topology.
