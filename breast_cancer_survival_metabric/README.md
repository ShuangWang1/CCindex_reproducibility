> Workflow reference retained from the research workspace. Generated outputs and historical measured results are excluded from this release. Start with the root reproduction guides for the current execution sequence.



# METABRIC motif survival analysis

This project applies the same patient-level unadjusted motif Cox screen used for
the Basel (BZ) cohort to METABRIC TRI-2 through TRI-7 motifs.

## Applicability

The method is applicable because:

- `41586_2019_1007_MOESM7_ESM.txt` contains a unique `METABRIC.ID`, overall
  survival/follow-up time `T`, and all-cause death indicator `Death`.
- `41586_2019_1007_MOESM10_ESM.xls` defines `T` as time until death or last
  follow-up in days and `Death` as 1 for death and 0 otherwise.
- The prepared motif artifacts map every one of 515 images to a METABRIC patient.
  There are 467 unique motif-mapped patients; 454 have a matching, complete OS
  record in the clinical table (212 deaths). The 13 unmatched IDs are audited and
  excluded.

The principal difference from BZ is repeated imaging: 44 METABRIC patients have
two or three images. Motif counts are therefore summed across all images for each
patient before enrichment groups are defined.

## Model

For every motif:

1. Sum raw motif counts over all images from each patient.
2. Calculate the median across the 454-patient survival cohort.
3. Set `motif_enriched = 1` when count is strictly above the median; median ties
   are non-enriched.
4. Fit `Surv(T, Death) ~ motif_enriched` with Efron ties.
5. Report the hazard ratio, 95% CI, and two-sided Wald p-value.

This is unadjusted analysis: no age, stage, grade, treatment, or other clinical
covariate enters the model. Both within-level and global Benjamini-Hochberg
q-values are reported. Raw p-values remain useful for ranking but should not be
treated as multiplicity-controlled evidence.

## Run

```bash
python breast_cancer_survival_metabric/analyze_motif_survival.py
```

The run is checkpointed in 20,000-motif chunks. Outputs are written to `results/`:

- `motif_cox_results_sorted.csv.gz`: all levels sorted by raw p-value
- `tri_N_cox_results_sorted.csv.gz`: each level sorted by raw p-value
- `top_1000_motif_cox_results.csv`: convenient uncompressed review table
- `patient_level_survival.csv`: exact included cohort
- `patient_linkage_audit.csv`: included/excluded linkage status and reason
- `analysis_summary.json` and `REPORT.md`: cohort and significance summaries

To run only selected levels in a separate output directory:

```bash
python breast_cancer_survival_metabric/analyze_motif_survival.py \
  --levels 2 3 --outdir breast_cancer_survival_metabric/results_tri2_tri3
```

## Tests

```bash
python -m unittest discover -s breast_cancer_survival_metabric/tests -v
```


## Rank-normalized survival and expanded lineage plots (TRI-1–TRI-7)

Run the complete workflow with:

```bash
python breast_cancer_survival_metabric/run_rank_normalized_analysis.py
```

The workflow generates missing TRI-1 frequencies locally from Delaunay simplices
of the 515 existing source graphs. TRI-2–7 use the prepared METABRIC count matrix.
It ranks positive counts within each image and level, gives equal positive counts
average ranks, and assigns absent motifs the last rank M. Scores are
`(M - rank) / (M - 1)` using the complete observed level vocabulary. Each patient's
score equally averages their available images; raw counts are separately summed.
The included cohort is the same 454 patients (212 deaths) used above.

Every motif with a positive patient median score enters the primary unadjusted
Cox screen (Efron ties, two-sided Wald p). Scores strictly above the median are
high; ties are low. Targets have finite rank-normalized binary Cox p < 0.05;
there is no raw-count p-value prefilter. Within-level and global BH q-values are
reported over the primary screening family. Survival time is T **in days**.
Log-rank and continuous per-SD Cox models are secondary statistics for targets.

Outputs are under `motif_results/`:

- `normalized_by_ranking_motif_analysis/significant/`: full screening table,
  selected results, and `triN/<p-value-rank>_<hash>/` packages containing survival
  curves, patient rank distributions, colored motif topologies, patient scores,
  and statistics.
- `significant_mtf_trace_ranking_normalized/triN/<hash>/`: base structural traces.
- `significant_mtf_trace_ranking_normalized/triN/<hash>_new/`: **final expanded
  traces**, including rank-normalized analysis of every structural relative,
  even those with zero patient medians. Each folder contains
  `rank_normalized_cox_p_value_by_level.png`, `trace_nodes.csv`, `trace_edges.csv`,
  and `trace_metadata.json`.
- `REPORT.md`, `analysis_summary.json`, patient linkage/cohort audits and caches.

The relationship index is built by reversing the same one-triangle growth
operations as BZ, once per observed motif, and reused across targets. It supports
all nine METABRIC cell labels. Relationships are potential topological lineage,
not persisted original search history. All nodes remain in tables; undefined
Cox p-values retain fit statuses and are omitted from plotted points/links.
Nominal p-values remain exploratory; BH significance is reported separately.

Stages run in order: `prepare screen trace extend report`. They can be selected
with `--stages`; cached inputs and settings are validated before reuse.
`--workers` controls relationship-index processes (default 6), and `--chunk-size`
controls Cox batches (default 5000). Use `--rerun` to regenerate outputs.
Use a separate output directory for partial runs, for example:

```bash
python breast_cancer_survival_metabric/run_rank_normalized_analysis.py \
  --levels 1 2 --max-targets 1 \
  --outdir breast_cancer_survival_metabric/motif_results_smoke
```

`--target-levels` and `--target-hashes` restrict trace/package targets without
changing the primary screening family. `--skip-plots` is available for table-only
runs. Existing BZ data and METABRIC raw-count results are preserved.
## Rank-normalized sibling topology analysis

The TRI-5 descendants of TRI-1 motif `2c9dad37a530a40bfd159b028f2d8b4f` are also analyzed within their immediate TRI-4 parents. The [sibling-analysis guide](motif_results/sibling_topology_analysis_ranking_normalized/README.md) describes the 747 intrinsic criteria, parent-group comparisons, graph galleries, and reproduction commands. The results report (`motif_results/sibling_topology_analysis_ranking_normalized/results/REPORT.md`; generated locally) summarizes criterion recurrence and validation.
