# Breast cancer survival and topology

The two case studies use the Basel/Zurich (BZ) and METABRIC cohorts. The main workflow is rank-normalized motif survival and structural analysis over TRI-1–TRI-7. Raw-count analyses are retained both as supporting analyses and as prerequisites for the existing BZ screening code.

## 1. Preprocess graphs and count motifs

Supply the [external inputs](INPUTS.md), then run:

```bash
python breast_cancer_preprocess_intra.py
python tree_search_all_level_threshold_cpp_breast_cancer_intra.py \
  --dataset both --MaxDepth 7 --threshold 10
python breast_cancer_survival/generate_tri1_motif_frequencies.py
```

The graph converter preserves each cohort's native cell labels. Mining writes per-sample TRI-2–TRI-7 count tables; the TRI-1 command reconstructs labeled Delaunay-simplex frequencies. Use exact counting for survival analysis: do not enable `--SM_stop_early`. Multiple cores/images must be linked to patients before survival inference.

## 2. Prepare sparse artifacts

The BZ preparation entry point retains its historical directory structure:

```bash
python -m pip install -r requirements-artifacts.txt
python breast_cancer_classification_workstation/classification_by_level/mtf_occurrence/mrmr_selection/parameter_tuning_bz/run_bz_parameter_tuning.py prepare
python breast_cancer_classification_workstation/metabric_workstation.py prepare
```

Provide the original grade/sample linkage files listed in the input manifest. The BZ stage reconstructs completed sample inventories from per-sample summaries. The METABRIC stage creates a combined sparse vocabulary and patient/image linkage. Neither command requires running model selection or classification tuning. Existing artifacts may be supplied at the documented default paths instead.

## 3. BZ rank-normalized survival and lineage

Generate the raw result tables needed by the BZ rank screen:

```bash
python breast_cancer_survival/analyze_motif_survival.py
python breast_cancer_survival/analyze_motif_survival_unadjusted.py \
  --outdir breast_cancer_survival/motif_results
python breast_cancer_survival/analyze_tri1_motif_survival.py
python breast_cancer_survival/screen_all_rank_normalized_motifs.py
python breast_cancer_survival/trace_rank_normalized_significant_motifs.py
python breast_cancer_survival/screen_trace_rank_normalized_relatives.py
```

The initial raw tables are inputs for motif identities and candidate eligibility; rank screening does not select candidates by a raw survival p-value. Positive counts are ranked within each core and complete level vocabulary, with average ties and absent motifs last. Patient scores equally average available tumor cores. The primary screen uses positive patient medians; expanded relatives additionally include zero-median motifs and retain undefined Cox fits with statuses. Overall survival is measured in months.

Expanded structural traces are generated beneath `breast_cancer_survival/motif_results/significant_mtf_trace_ranking_normalized/`, with final packages named `<hash>_new`. Their nodes retain level, motif identity, topology, and survival statistics.

## 4. METABRIC rank-normalized survival and lineage

```bash
python breast_cancer_survival_metabric/analyze_motif_survival.py
python breast_cancer_survival_metabric/run_rank_normalized_analysis.py
```

The staged runner performs `prepare screen trace extend report`, including TRI-1 reconstruction when required. It uses patient-level overall survival `T` in days and all-cause death `Death`; repeated images are aggregated within patient. Rank scores equally average images, while raw counts are summed. To inspect a partial run without mixing outputs:

```bash
python breast_cancer_survival_metabric/run_rank_normalized_analysis.py \
  --levels 1 2 --target-levels 1 2 --max-targets 1 \
  --outdir breast_cancer_survival_metabric/motif_results_smoke
```

Partial-run statistics use a different screening family from the full study.

## 5. Cohort-specific topology studies

Each cohort has a rank-normalized TRI-5 sibling analysis within immediate TRI-4 parents of a selected TRI-1 lineage:

```bash
python breast_cancer_survival/motif_results/sibling_topology_analysis_ranking_normalized/run_analysis.py
python breast_cancer_survival/motif_results/sibling_topology_analysis_ranking_normalized/run_same_parent_comparison.py
python breast_cancer_survival_metabric/motif_results/sibling_topology_analysis_ranking_normalized/run_analysis.py
python breast_cancer_survival_metabric/motif_results/sibling_topology_analysis_ranking_normalized/run_same_parent_comparison.py
```

The default lineage roots are BZ `65319d6de1808abbc65d476aa3e9436e` and METABRIC `2c9dad37a530a40bfd159b028f2d8b4f`. Feature engines compare topology, boundary structure, cell composition, contacts, typed triangles, and local organization. Undefined survival fits are excluded from significance classes. Criterion-level corrections do not redefine motif-level survival significance. Intrinsic features exclude outcome-derived dispersion descriptors from same-parent tests and PCA predictors.

## 6. Common motifs and topology at each level

```bash
python breast_cancer_survival_common/compare_all_observed_motifs.py
python breast_cancer_survival_common/compare_common_same_parent.py
python breast_cancer_survival_common/plot_overlapped_common_heatmaps.py
python breast_cancer_survival_common/validate_common_same_parent.py
```

Common motifs are matched across TRI-1–TRI-7 after translating shared cell-type names, with exact labeled isomorphism verification. Shared-parent comparisons use TRI-2–TRI-7 children and their common immediate preceding-level parents, separately for each cohort. They reuse intrinsic graph features and apply cohort-specific survival classes, Mann–Whitney tests, effect sizes, and within-parent BH correction. Untestable cells remain missing. A child can belong to multiple valid parents.

The default all-observed comparison has no positive-median or significance prefilter. This differs from the primary rank-normalized screen and the optional significant-only overlap export. Shared-cell-type matching preserves identity and does not merge cohort-specific stromal annotations.

## Supporting analyses and interpretation

Raw-count Cox/Kaplan–Meier, Ki-67, exact-expectation normalization, and raw-significant lineage scripts remain available in `breast_cancer_survival/`. The additional raw TRI-7 lineage topology workflow is under `motif_results/tri7_trace_analysis(all)/`. These are separate analyses, with their own selection families and output directories.

These workflows describe exploratory associations. Nominal p-values, BH-adjusted q-values, and non-estimable fits are separate outputs. Graph layouts represent adjacency rather than measured tissue geometry. Recomputed current-matcher outputs should be stored as a complete new run before comparing them with historical study artifacts.
