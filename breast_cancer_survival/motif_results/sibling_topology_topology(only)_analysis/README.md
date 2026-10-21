# Unlabeled sibling topology analysis

This project repeats the TRI-5 sibling comparison for lineage
`tri_1:1fded5000748985170d30bdca433a111` using only unlabeled graph topology.

It discards every cell-type-dependent feature. The retained criteria describe
graph size, degree organization, connectivity, path lengths, clustering,
triangle overlap, and graph spectra. Statistical definitions and the
significant/nonsignificant population match the labeled analysis.

Run from the repository root (quote the path because it contains parentheses):

```bash
python 'breast_cancer_survival/motif_results/sibling_topology_topology(only)_analysis/run_analysis.py'
```

The default run safely reuses the already validated label-invariant columns
from the comprehensive feature cache. Pass `--rebuild-features` to reconstruct
and validate every graph from `trace_nodes.csv`.
