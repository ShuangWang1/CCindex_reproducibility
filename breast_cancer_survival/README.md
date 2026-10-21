> Workflow reference retained from the research workspace. Generated outputs and historical measured results are excluded from this release. Start with the root reproduction guides for the current execution sequence.

# Basel Ki-67 survival study

This folder contains a reproducible, exploratory patient-level analysis of tumor-cell Ki-67 and overall survival in the Basel imaging mass cytometry cohort.

It also contains a patient-level Cox screen of TRI-2 and TRI-3 spatial motifs.

## TRI-1 Delaunay frequencies

Generate the labeled TRI-1 base motifs directly from each completed BZ
sample's Delaunay simplices with:

```bash
python breast_cancer_survival/generate_tri1_motif_frequencies.py
```

This writes `ccindex_tri_1.csv` beside the existing TRI-2–TRI-7 files under
`breast_cancer_meta/BZ_res_threshold_10_cpp_0811/sample*/`. Every Delaunay
simplex contributes exactly one count, grouped by its sorted three-cell label
combination. No frequency threshold is applied at TRI-1. Root-level
`tri_1_frequency_manifest.csv` and `tri_1_frequency_audit.json` files reconcile
the generated frequencies against the source samples.

Run the matching patient-level unadjusted Cox screen with:

```bash
python breast_cancer_survival/analyze_tri1_motif_survival.py
```

This writes `motif_results/tri_1_cox_results_sorted.csv` with the same columns
and strict-median enriched-versus-non-enriched definition as the existing
TRI-2 result. Its audit is saved separately as
`motif_results/tri_1_cox_analysis_summary.json`, so the consolidated TRI-2–7
results are not overwritten. TRI-1 multiplicity values use the 84 TRI-1 motifs
as their correction family.

## Study design

- **Population:** patients whose metadata-designated tumor core is present in `BZ.h5ad` and contains cells classified as `tumor cells`.
- **Primary endpoint:** overall survival (`OSmonth`). `death` and `death by primary disease` are events; `alive` and `alive w metastases` are censored.
- **Follow-up requirement:** patients with missing or zero `OSmonth` are excluded because they contribute no observable survival time.
- **Primary exposure:** a cell is Ki-67-high when its `1441101Er168Di Ki67` value is above the pooled tumor-cell 75th percentile. A patient's score is the fraction of their tumor cells that are Ki-67-high. Patients above the patient-score median are called **enriched**; the rest are **non-enriched**. Both thresholds are selected without using survival outcomes.
- **Primary comparison:** Kaplan-Meier curves, a two-sided log-rank test, and an unadjusted Cox proportional-hazards model.
- **Robustness analyses:** Cox models using the Ki-67 score continuously (per 1 SD), and models adjusted for prespecified clinical variables: age, grade, tumor size, nodal status, metastatic status, ER status, and HER2 status.
- **Unit of analysis:** patient. Multiple cells and cores are pooled within patient; cells are never counted as independent survival observations.

This is an exploratory association study, not an optimized search for the smallest p-value. The hazard ratio and its 95% confidence interval should be interpreted alongside the p-value. Median splits lose information, so the continuous model is important. The approximate Schoenfeld-residual diagnostics should also be reviewed before interpreting a Cox result.

## Run

From the repository root:

```bash
python breast_cancer_survival/analyze_ki67_survival.py
```

To rank TRI-2 and TRI-3 motifs by their enriched-versus-non-enriched Cox
p-values:

```bash
python breast_cancer_survival/analyze_motif_survival.py
```

For this screen, raw motif matches are summed across each patient's available
tumor cores. A patient is enriched when their count is strictly above that
motif's cohort median. The complete sorted tables, adjusted estimates, global
Benjamini-Hochberg q-values, and a concise report are written to
`motif_results/`.

The unadjusted high-level screen uses the prepared sparse motif artifacts and
a vectorized one-predictor Efron Cox solver:

```bash
python breast_cancer_survival/analyze_motif_survival_unadjusted.py
```

It analyzes TRI-4 through TRI-7 by default, checkpoints each completed level,
and writes the complete sorted results to
`motif_results_tri4_tri7_unadjusted/`.

After consolidating TRI-2 through TRI-7, generate the labeled top-30 Cox plots,
motif network galleries, and cell-type summary with:

```bash
python breast_cancer_survival/visualize_top_motif_survival.py
```

Generate enriched-versus-non-enriched Kaplan-Meier curves for those same 30
motifs per level with:

```bash
python breast_cancer_survival/plot_top30_motif_kaplan_meier.py
```

The script reconstructs each motif's patient groups from the sparse artifacts
and verifies their medians, group sizes, and event counts against the saved
top-30 results before writing any curve for that level.

To select and visualize the top 30 motifs per level after excluding zero
patient-median thresholds, run:

```bash
python breast_cancer_survival/visualize_top_motif_survival.py \
  --nonzero-threshold-only \
  --outdir breast_cancer_survival/motif_results_tri4_tri7_unadjusted/non-zero_top30_visualizations
python breast_cancer_survival/plot_top30_motif_kaplan_meier.py \
  --top30-csv breast_cancer_survival/motif_results_tri4_tri7_unadjusted/non-zero_top30_visualizations/top30_motifs_by_level.csv \
  --outdir breast_cancer_survival/motif_results_tri4_tri7_unadjusted/non-zero_top30_visualizations
```

Trace the selected tumor/stromal TRI-7 motif back through all potential
TRI-6 through TRI-2 structural parents and plot their survival curves, raw Cox
p-value lineage, and global BH q-value lineage with:

```bash
python breast_cancer_survival/trace_back_tri7_motif_survival.py
```

The outputs are written to
`motif_results_tri4_tri7_unadjusted/tri-7_trace_back/`. Because the forward
search deduplicated graphs by WL hash without saving parent IDs, this is a
branched set of potential structural parents rather than a uniquely recorded
historical search path. Motif galleries use the final TRI-7 topology as a
fixed gray scaffold, and the growth-transition galleries highlight each added
node and its incident edges in gold.

To compare the three TRI-6 siblings using labeled topology descriptors,
patient abundance contrasts, clinically adjusted Cox sensitivity models, and
patient-stratified bootstrap resampling, run:

```bash
python breast_cancer_survival/analyze_sibling_motif_topology_survival.py
```

The auditable tables, topology contrast plots, and exploratory interpretation
are written under `tri-7_trace_back/tri6_sibling_topology_analysis/`.

Build compact bidirectional lineage traces for raw-significant motifs with a
positive patient-median split threshold with:

```bash
python breast_cancer_survival/trace_significant_motifs.py --target-levels 7
```

Outputs are written under `motif_results/significant_mtf_trace/tri7/<wl_hash>/`.
Each target receives its complete motif identifier, lineage node/edge tables,
reproduction command, and a raw Cox p-value lineage plot. Significance selects
focal motifs only; structural relatives are not filtered by their p-values.
Use `--target-levels 1 2 3 4 5 6` for the later lower-level backward and forward
traces. Once all levels have been traced, run:

```bash
python breast_cancer_survival/cluster_significant_motif_trajectories.py
```

This performs deterministic branch-aware clustering of complete TRI-1–TRI-7
significance trajectories and reports incomplete forward lineages separately.

To plot the patient-level motif frequencies for every significant focal motif,
ordered by summed count and colored by the exact Cox median split, run:

```bash
python breast_cancer_survival/plot_significant_motif_frequency_histograms.py
```

Each plot is saved as `patient_frequency_histogram.png` in its existing
`significant_mtf_trace/triN/<wl_hash>/` folder. The dashed horizontal line marks
the motif-specific patient median, the dotted vertical line separates patients
at or below the median from patients strictly above it, and the reconstruction
is checked against the saved median and group sizes before plotting.

Normalize all 644 focal motifs for total cells, complete same-level motif
opportunity, and eight-part cell-type composition with:

```bash
python breast_cancer_survival/normalize_significant_motif_frequencies.py
```

Outputs are written under `motif_results/normalized_motif_analysis/`. The
primary normalized score is the observed-versus-expected Pearson residual from
a motif-specific negative-binomial model. Counts, expectations, and variances
are summed before calculating scores for patients with multiple cores.
Continuous unadjusted Cox models use standardized patient residuals; zero-split
Kaplan–Meier curves are descriptive. These results are exploratory because the
644 motifs were originally selected from their raw-count survival associations.
Within each `triN/` output, a literal `*` suffix on the motif directory denotes
a nominally significant normalized continuous Cox result (`p < 0.05`); it does
not denote BH-adjusted significance.

Alternatively, normalize those same 644 focal motifs by their within-core rank
among the complete observed motif vocabulary at each TRI level with:

```bash
python breast_cancer_survival/analyze_rank_normalized_motif_survival.py
```

Positive counts are ranked from highest to lowest, equal positive counts use
their average rank, and an absent motif receives the fixed last rank for its
TRI-level vocabulary. Core rank scores are averaged equally for patients with
two tumor cores. Aggregate tables, cell-burden diagnostics, and per-motif
Kaplan–Meier audit packages are written under
`motif_results/normalized_by_ranking_motif_analysis/triN/<wl_hash>/`.
A trailing `*` on a WL-hash directory marks a nominally significant
rank-normalized binary Cox result (`p < 0.05`), not BH-adjusted significance.
Full copies of these significant motif packages are also collected without
star suffixes under
`motif_results/normalized_by_ranking_motif_analysis/significant/triN/<wl_hash>/`.
Build that collection from every motif with a nonzero rank-score median—not
only the former 644 raw-significant motifs—with:

```bash
python breast_cancer_survival/screen_all_rank_normalized_motifs.py
```

Each qualified motif folder includes an unlabeled, crossing-free planar
Delaunay-style topology image. Node colors use the same cell-type palette in
every folder; the layout represents motif adjacency rather than physical tissue
coordinates.
Qualified folder names use
`<zero-padded rank_within_level_rank_binary>_<wl_hash>`, so normal filename
sorting orders motifs by their binary rank-normalized Cox p-value within each
TRI level. Exact p-value ties share a rank prefix.

Build bidirectional structural lineage traces for all 740 motifs selected by
that complete rank-normalized screen with:

```bash
python breast_cancer_survival/trace_rank_normalized_significant_motifs.py
```

Outputs are written under
`motif_results/significant_mtf_trace_ranking_normalized/triN/<wl_hash>/`.
Each target receives lineage node and edge tables, rank-normalized selection
metadata, and a binary Cox p-value lineage plot. Structural relatives are not
filtered by significance. Relatives outside the nonzero-median ranking screen
remain in the trace with blank rank-survival fields because no estimable
rank-normalized median split was calculated for them.

To extend the rank-normalized Cox calculation to every structural relative in
those traces, including motifs whose cohort median score is zero, run:

```bash
python breast_cancer_survival/screen_trace_rank_normalized_relatives.py
```

The original trace packages are preserved. Expanded tables and plots are
written to sibling `triN/<wl_hash>_new/` directories. These use the same
`patient_rank_score > cohort_median` grouping rule, which becomes a
present-versus-absent split when the median is zero. A p-value remains blank
when that binary Cox comparison is not estimable.

Normalize the same 644 focal motifs by their exact expectation under within-core
cell-label permutation with:

```bash
python breast_cancer_survival/analyze_exact_expectation_motif_survival.py
```

This analysis directly recounts TRI-2–TRI-7 labeled motifs and their unlabeled
shapes. It retains only vertex sets whose induced graph is exactly the motif
shape (excluding matches with extra chord edges), then evaluates
`N_S * (Aut(S) / Aut(m)) * product((n_a)_(k_a)) / (N)_k`. TRI-1 retains its
one-count-per-Delaunay-simplex definition. Observed and expected counts are
summed across a patient's cores; the primary survival comparison is the literal
`observed > expected` group versus all other patients. Aggregate tables,
permutation validation, and per-motif Kaplan–Meier packages are written under
`motif_results/normalized_by_get_expectation_motif_analysis/`.

Optional paths and the cell-level quantile can be changed with command-line arguments:

```bash
python breast_cancer_survival/analyze_ki67_survival.py \
  --h5ad breast_cancer/BZ.h5ad \
  --metadata breast_cancer/Basel_PatientMetadata.csv \
  --outdir breast_cancer_survival/results \
  --cell-quantile 0.75
```

## Outputs

- `results/ki67_kaplan_meier.png` and `.pdf`: survival curves with log-rank and unadjusted Cox results.
- `results/ki67_score_distribution.png` and `.pdf`: the patient score and grouping cutoff.
- `results/patient_level_analysis.csv`: auditable analysis data, one row per patient.
- `results/cox_coefficients.csv`: all coefficients, hazard ratios, confidence intervals, and p-values.
- `results/group_summary.csv`: group sizes and event counts.
- `results/proportional_hazards_diagnostics.csv`: approximate per-term Schoenfeld residual/time association checks.
- `results/analysis_summary.json`: cohort flow, thresholds, event count, and model sample sizes.
- `results/REPORT.md`: concise findings and interpretation generated from the current run.
- `motif_results_tri4_tri7_unadjusted/top30_visualizations/tri2/` through
  `tri7/`: 30 PNG Kaplan-Meier plots per level, named from the ordered cell-type
  sequence and annotated with the consolidated unadjusted Cox result. A
  same-named TXT file records the exact median threshold, group definitions,
  patient counts, and event counts for each plot.

## Interpretation and limitations

The analysis is observational and cannot establish that Ki-67 causes survival differences. It is vulnerable to residual clinical confounding, measurement/batch effects, core sampling heterogeneity, and cutoff instability. The adjusted model uses complete cases, so compare its sample size and event count with the unadjusted model. A confirmatory study should freeze the marker and patient cutoffs from this cohort and test them in an independent cohort rather than re-optimizing them.
