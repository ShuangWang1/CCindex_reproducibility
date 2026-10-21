# Sibling topology analysis

This project compares the internal labeled topology of significant and
nonsignificant same-level motifs in the lineage rooted at TRI-1 motif
`1fded5000748985170d30bdca433a111`.

TRI-5 is the primary analysis. TRI-4, TRI-6, and TRI-7 are consistency checks;
levels without at least 20 motifs in both outcome classes are retained only in
the descriptive audit. Significance is the saved unadjusted Cox p value below
0.05. Nonestimable Cox fits are excluded rather than called nonsignificant.

Run from the repository root:

```bash
python breast_cancer_survival/motif_results/sibling_topology_analysis/run_analysis.py
```

Generated tables, figures, the machine-readable audit, and the interpretation
report are written under `results/`. The source trace is read-only.
Validated graph features are reused on later report/statistics runs; pass
`--rebuild-features` after changing feature definitions or to force validation
and extraction from the source trace again.

The analysis is exploratory. It intentionally uses no abundance adjustment,
parent matching, predictive model, permutation test, or resampling. Therefore
its association p/q values do not establish that topology causes the observed
survival signal.
