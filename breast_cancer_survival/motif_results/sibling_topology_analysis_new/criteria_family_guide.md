## How to read the criteria families

Each row of `tri5_motif_criteria.csv.gz` is one estimable TRI-5 motif. The
comparison groups are defined by the motif's original unadjusted Cox result:

- **significant:** raw Cox `p < 0.05`
- **nonsignificant:** raw Cox `p >= 0.05`

This grouping is not a healthy-versus-disease label. The criteria describe
motifs whose abundance was or was not associated with survival in the original
Cox screen.

Cell-type codes used in metric names are:

| Code | Cell type |
|---:|---|
| 0 | B cells |
| 1 | T and B cells |
| 2 | T cells |
| 3 | endothelial |
| 4 | macrophages |
| 5 | stromal cells |
| 6 | tumor cells |
| 7 | unknown |

### 1. Size

These columns measure how much graph structure is present:

- `size_nodes`: number of nodes, or cells, in the motif (`n`).
- `size_edges`: number of undirected cell-contact edges (`m`).
- `size_triangles`: number of graph triangles (`T`).

All motifs in this analysis are TRI-5, so `size_triangles` is fixed at five.
Node and edge counts can still vary because five triangles can be arranged in
different ways. A difference in another metric should be checked against these
size measures before being interpreted as an organizational effect.

### 2. Pure topology

These criteria use adjacency alone and ignore cell identity:

- `topology_triangles_per_node` is `T/n`. Higher values mean more triangular
  faces relative to the number of cells.
- `topology_component_count` is the number of connected components.
- `topology_euler_characteristic` is `n - m + T`, treating triangles as filled
  faces of a cell complex.
- `topology_hole_count` is `components - Euler characteristic`, truncated at
  zero.

For a connected, hole-free triangulated patch, the Euler characteristic should
be one. Hole and component criteria are often constant in a lineage of valid
connected motifs; they remain in the tables so this assumption is auditable.

### 3. Boundary and interior structure

An edge is treated as a boundary edge when it belongs to exactly one graph
triangle. A boundary node is incident to at least one such edge.

- `boundary_edge_count`: number of boundary edges.
- `boundary_node_count`: number of boundary nodes.
- `boundary_node_fraction`: boundary nodes divided by `n`.
- `interior_node_fraction`: one minus the boundary-node fraction.
- `boundary_edges_per_node`: boundary edges divided by `n`.
- `boundary_edges_per_triangle`: boundary edges divided by `T`.

High boundary fractions indicate a thin, open, or elongated triangulated patch;
lower fractions indicate that more nodes are embedded in the interior. These
measures are preferable to interpreting generic graph density alone because
Delaunay-like planar graphs are strongly constrained by their boundary.

### 4. Degree structure

Node degree is the number of graph neighbors of a cell. The family contains:

- `degree_mean`
- `degree_variance`
- `degree_sd`
- `degree_cv`, calculated as SD divided by the mean
- `degree_maximum`
- `degree_q25`, `degree_median`, and `degree_q75`

Higher mean degree means more contacts per cell. Variance and CV describe how
unevenly contacts are distributed: two motifs may have the same mean degree but
differ because one contains a highly connected hub. The pooled discrete degree
histograms are in `tri5_within_motif_distributions.csv`.

### 5. Triangle participation

For each node `v`, triangle participation `t_v` is the number of triangles
containing that node. The family summarizes the node-level values with:

- mean, variance, SD, CV and maximum;
- Q1, median and Q3.

The corresponding columns begin with `triangle_participation_`. Large
variation means triangular organization is concentrated on a subset of cells;
low variation means triangles are distributed more evenly. The full pooled
participation histograms are in `tri5_within_motif_distributions.csv`.

### 6. Path topology

Shortest paths measure how many graph steps separate cells:

- `path_diameter`: largest finite shortest-path distance.
- `path_mean_shortest`: mean distance over unordered connected node pairs.
- `path_diameter_normalized_sqrt_n`: diameter divided by `sqrt(n)`.
- `path_mean_normalized_sqrt_n`: mean path length divided by `sqrt(n)`.

The normalized versions reduce the expected dependence on the linear scale of
a two-dimensional motif. Larger values generally indicate a more elongated or
chain-like topology; smaller values indicate a more compact topology. Pooled
path-length histograms are provided in `tri5_within_motif_distributions.csv`.

### 7. Connectivity

- `connectivity_normalized_lambda2` is the second-smallest eigenvalue of the
  normalized graph Laplacian.

This is algebraic connectivity. Values near zero indicate weak connection,
fragmentation, or a narrow bottleneck. Larger values indicate that the graph is
more strongly integrated. Its scale should be compared only among similarly
sized graphs.

### 8. Spectral structure

The normalized-Laplacian spectrum supplies a compact global graph fingerprint:

- `spectral_normalized_lambda3` and `spectral_normalized_lambda4`: the third
  and fourth smallest normalized-Laplacian eigenvalues.
- `spectral_normalized_laplacian_entropy`: Shannon entropy of the positive
  eigenvalues after normalizing them to sum to one.
- `spectral_normalized_laplacian_entropy_scaled`: spectral entropy divided by
  its maximum for the observed number of positive eigenvalues.

The complete eigenvalue records, indexed by motif and eigenvalue position, are
stored in `tri5_normalized_laplacian_eigenvalues.csv.gz`. Spectral differences
indicate global shape differences but usually have less direct biological
interpretability than boundary or contact criteria.

### 9. Node composition

These criteria describe which cell types occur, without considering their
positions or contacts:

- `composition_type_richness`: number of represented cell types.
- `composition_shannon_entropy`: `-sum(p_c log p_c)`.
- `composition_shannon_entropy_normalized`: Shannon entropy divided by
  `log(min(C,n))`.
- `composition_simpson_diversity`: `1 - sum(p_c^2)`.
- `composition_hill_d0`: richness.
- `composition_hill_d1`: exponential of Shannon entropy.
- `composition_hill_d2`: inverse Simpson concentration, `1/sum(p_c^2)`.
- `composition_evenness`: Shannon entropy divided by `log(richness)`.
- `composition_dominant_type_fraction`: fraction belonging to the most common
  type.
- `composition_top2_type_fraction`: combined fraction of the two most common
  types.

Richness and Hill numbers rise with diversity. Dominant-type fraction rises
when one cell type controls the motif. These are composition criteria, not
evidence that cell types are spatially organized.

### 10. Pairwise contacts

This family contains four aggregate criteria plus three values for every one of
the 36 unordered cell-type pairs:

- `contact_homotypic_fraction`: fraction of edges joining equal cell types.
- `contact_heterotypic_fraction`: fraction joining different cell types.
- `contact_edge_type_entropy`: Shannon entropy across observed edge types.
- `contact_edge_type_entropy_normalized`: entropy scaled by the number of
  edge types present.
- `contact_A_B_count`: raw number of A–B edges.
- `contact_A_B_fraction`: A–B edges divided by all motif edges.
- `contact_A_B_permutation_z`: enrichment relative to a fixed-composition
  node-label permutation null.

For example, `contact_2_4_fraction` is the fraction of T-cell–macrophage edges.
A positive permutation Z-score means the contact occurs more often than
expected after randomly relocating the motif's existing cell labels on the
same topology. A negative score means avoidance. Counts and fractions can be
driven by composition; the Z-score is the intended organization-beyond-
composition criterion.

The null uses 128 deterministic Monte Carlo permutations per motif. Z-scores
are missing when the permutation distribution has zero variance.

### 11. Coarse triangle classes

Every triangle is assigned to one of three classes:

- `triangle_AAA_fraction`: all three cells have the same type.
- `triangle_AAB_fraction`: two cells share a type and the third differs.
- `triangle_ABC_fraction`: all three cells have different types.

The fractions sum to one. The family also includes:

- `triangle_type_entropy`: Shannon entropy across the detailed typed-triangle
  categories.
- `triangle_type_entropy_normalized`: entropy scaled by the number of observed
  triangle types.

AAA emphasizes homogeneous three-cell neighborhoods; ABC emphasizes maximally
mixed neighborhoods. AAB represents an intermediate arrangement.

### 12. Typed triangles

There are 120 possible unordered triples from eight cell types. Each triple has
three criteria, producing 360 columns:

- `typed_triangle_A_B_C_count`
- `typed_triangle_A_B_C_fraction`
- `typed_triangle_A_B_C_permutation_z`

For example, `typed_triangle_2_2_4_fraction` is the fraction of triangles made
of two T cells and one macrophage. As with contacts, raw counts and fractions
combine composition and organization. The permutation Z-score asks whether
the triangle is enriched after holding both graph topology and within-motif
cell-type counts fixed. The same 128 deterministic label permutations are used.

Many typed triangles are rare or identically zero. They remain in the
distribution tables, while only criteria with usable variation appear in the
paginated violin plots.

### 13. Type–topology coupling

For each of the eight cell types, six criteria ask whether that type occupies a
particular structural role:

- `type_*_count`: number of cells of that type.
- `type_*_fraction`: fraction of motif nodes of that type.
- `type_*_degree_mean`: mean degree among cells of that type.
- `type_*_triangle_participation_mean`: mean triangle participation among cells
  of that type.
- `type_*_boundary_enrichment_difference`: boundary fraction among that type
  minus the motif's overall boundary fraction.
- `type_*_egophily_mean`: mean fraction of same-type neighbors for cells of
  that type.

A positive boundary-enrichment difference means the type preferentially
occupies boundary positions within that motif. High egophily means that the
type tends to contact itself. Type-conditional values are missing when the cell
type is absent from a motif; consult `n` and `n_missing` before comparing them.

### 14. Local organization

For each node, neighborhood entropy is calculated from the types of its graph
neighbors. Motif-level summaries are:

- `local_neighbor_entropy_mean`
- `local_neighbor_entropy_median`
- `local_neighbor_entropy_variance`
- `local_neighbor_entropy_iqr`

High entropy means that a cell sees a diverse neighborhood. Variance and IQR
indicate whether mixing is uniform throughout the motif or localized to
particular regions. The family also contains:

- `local_egophily_mean`
- `local_egophily_variance`

Egophily is the fraction of each node's neighbors with the same type.

### 15. Between-motif heterogeneity

This family measures how far each motif lies from the centroid of its own
significance group:

- `dispersion_composition_js_to_group_centroid`: Jensen–Shannon distance of the
  node-composition vector.
- `dispersion_mixing_frobenius_to_group_centroid`: Frobenius distance of the
  normalized contact matrix.
- `dispersion_null_corrected_mixing_frobenius_to_group_centroid`: Frobenius
  distance of the contact permutation-Z matrix.
- `dispersion_triangle_js_to_group_centroid`: Jensen–Shannon distance of the
  typed-triangle spectrum.

Larger values mean the motif is less typical of its own group. Comparing these
distributions asks whether significant motifs are more or less heterogeneous
than nonsignificant motifs. These are centroid-distance summaries, not
exhaustive average pairwise distances. Because centroids are estimated
separately within each group, these metrics should be interpreted descriptively
rather than as classifiers.

## How to read the distribution tables

`tri5_criteria_distribution_long.csv` contains two rows per criterion—one for
each significance group—with:

- `n` and `n_missing`;
- `mean`, `sd`, and `variance`;
- `minimum`, `q05`, `q25`, `median`, `q75`, `q95`, and `maximum`;
- `iqr`, calculated as Q3 minus Q1;
- `zero_fraction`, the proportion of finite observations equal to zero.

`tri5_criteria_distribution_wide.csv` presents the same quantities side by
side. Family-specific subsets are in `distributions_by_family/`.

`tri5_criteria_comparisons.csv` adds:

- `mann_whitney_u` and `location_p_value`: two-sided distribution-location
  comparison;
- `location_q_bh`: Benjamini–Hochberg correction across all testable criteria;
- `cliffs_delta`: rank effect size from -1 to 1; positive values mean higher
  values in significant motifs;
- `median_difference` and `mean_difference`: significant minus nonsignificant;
- `brown_forsythe_statistic` and `dispersion_p_value`: comparison of group
  variability using absolute deviations from the group median;
- `dispersion_q_bh`: BH correction for the dispersion tests;
- `variance_ratio_significant_over_nonsignificant`: values above one mean more
  variance among significant motifs.

Multiple related columns are mathematically correlated, especially raw counts
and fractions. A small q-value therefore should not be treated as an
independent biological discovery without effect-size, redundancy, abundance,
and patient-level checks.

## How to read the violin + scatter plots

For every plotted group:

- violin width represents the estimated density of observations at that value;
- transparent jittered points show individual motifs with finite values; their
  horizontal displacement is only for visibility and has no quantitative
  meaning;
- the orange point is the median;
- the thick black vertical segment spans Q1 to Q3 and therefore contains the
  middle 50% of observations;
- a group with only one unique value is drawn as a short horizontal segment
  instead of an artificial kernel-density shape.

The plot title reports the BH-adjusted location q-value. A flat mark at zero can
still accompany a small q-value when both groups are zero-inflated but differ
in their relatively uncommon nonzero values; consult the distribution table
and `zero_fraction` in that situation.

## Criteria not available from this trace

The source contains graph adjacency and node labels, but not cell coordinates.
Consequently, physical area, perimeter, density, compactness, solidity,
anisotropy, edge-length, triangle-area, and triangle-shape criteria cannot be
computed. Graph boundary metrics are not substitutes for physical perimeter.

The trace also lacks motif-to-patient and motif-to-region membership. Therefore
patient/ROI-aware phenotype permutation, mixed-effects modeling, and overlap
correction cannot be performed here. The exact unavailable criteria and reasons
are recorded in `criteria_registry.csv`.

## PCA visualization

`tri5_all_criteria_pca.png` represents every TRI-5 motif as a vector of the
computed intrinsic criteria. For each criterion, missing values are replaced by
the across-motif median and the resulting column is standardized to mean zero
and variance one. Constant and unusable columns are removed before PCA.

The four `dispersion_*_to_group_centroid` criteria are deliberately excluded:
their values depend on whether a motif is significant or nonsignificant, so
including them would leak the displayed group label into an otherwise
unsupervised PCA. `tri5_pca_preprocessing.csv` records every inclusion,
imputation, scaling value, and exclusion reason. `tri5_pca_loadings.csv` lists
the criteria contributing to PC1 and PC2, and `tri5_pca_scores.csv.gz` contains
the motif coordinates.

Blue and red points show nonsignificant and significant motifs, respectively;
large X symbols mark group centroids and dashed curves are 95% covariance
ellipses. PCA maximizes total variance without using the significance labels.
Therefore, overlap means the first two dominant axes do not cleanly separate
the groups; it does not prove that higher-dimensional nonlinear separation is
impossible.

Four additional PCA fits isolate major criterion blocks:

- **Topology:** size, pure topology, boundary/interior, degree, unlabeled
  triangle participation, paths, connectivity, and spectral criteria.
- **Composition:** global diversity criteria plus cell-type counts and
  fractions.
- **Contacts:** pairwise-contact matrices, local entropy/egophily, and
  cell-type-specific degree, boundary enrichment, and egophily.
- **Triangles:** coarse triangle classes, the typed-triangle spectrum, and
  cell-type-specific triangle participation.

The individual figures are `tri5_pca_topology.png`,
`tri5_pca_composition.png`, `tri5_pca_contacts.png`, and
`tri5_pca_triangles.png`; `tri5_pca_by_family.png` combines them. Each block is
imputed and standardized independently and has corresponding scores, loadings,
and preprocessing tables.

All 6,099 motifs are included. Topology and composition are discrete at fixed
TRI-5 scale, so many motifs have exactly identical feature vectors and PCA
coordinates. Their plotted points receive small deterministic display-only
jitter so coincident motifs become visible. The unjittered PCA coordinates are
retained in the score tables and are used for centroids and covariance ellipses.

## Representative motif head-to-head gallery

`tri5_representative_motif_head_to_head.png` contains 12 nonsignificant versus
significant pairs, or 24 actual motifs. Selection uses the 340 usable intrinsic
criteria; the outcome-derived `dispersion_*` criteria are excluded. Every
criterion is median-imputed and standardized, and the topology, composition,
contact, and triangle blocks are weighted to contribute equally despite their
different numbers of columns.

All six unlabeled TRI-5 graph shapes are represented. Pair quotas are allocated
as 3, 3, 2, 2, 1, and 1 according to the square root of significant-motif
prevalence. Within each group and graph shape, deterministic K-means finds
feature-space clusters and the actual motif nearest each center is retained.
The selected representatives are paired by minimum family-balanced distance.

Each pair has exactly the same unlabeled graph topology. The panels use an
isomorphism-aligned, crossing-free planar layout that preserves as many cell
labels as possible, so positional color changes can be compared directly.
The original spatial coordinates are unavailable, so these are
Delaunay-compatible planar embeddings rather than the motifs' physical tissue
coordinates. Titles report Cox p-value,
hazard ratio, cluster support, and the largest standardized criterion changes.
Complete values and differences are stored in the three
`tri5_representative_*` tables. These are representative prototypes, not the
most extreme motifs and not patient-level samples.

## Interpretation boundary

This is an exploratory motif-level comparison. Raw Cox significance is used to
define the two motif groups; the topology q-values test a separate association
screen and do not replace the original Cox correction. Motifs may share
patients, cells, or regions, so the rows should not be interpreted as
independent biological replicates or as evidence that topology causes the
survival association.
