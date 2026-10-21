# Criteria for Triangulation-Motif Topology Analysis

## Scope and notation

The setting is a collection of spatial motifs derived from a two-dimensional Delaunay triangulation. Each motif is an undirected, unweighted graph with no explicit edge features, many possible node/cell types, variable motif size, and a phenotype label such as healthy or disease.

Represent one motif as

$$
M=(V,E,\tau,P),
$$

where $V$ is the node set, $E$ is the undirected edge set, $\tau(v)\in\{1,\ldots,C\}$ is the node type, and $P_v=(x_v,y_v)$ is the spatial coordinate when coordinates are retained. Record at least:

- $n=|V|$: number of nodes
- $m=|E|$: number of edges
- $T$: number of Delaunay triangles
- $A$: physical area of the motif

Motif size should not be represented by $n$ alone. Motifs with the same number of cells may occupy different physical areas or have very different shapes.

## 1. Pure triangulation topology

These criteria describe the graph structure independently of node identity.

### Basic size measures

- Number of nodes, $n$
- Number of edges, $m$
- Number of triangles, $T$
- Triangle density per node, $T/n$

Always state whether “motif size” means $n$, $T$, physical area, or another quantity.

### Boundary and interior structure

- Number of boundary edges, $b$
- Number and fraction of boundary nodes, $n_{\mathrm{boundary}}$ and $n_{\mathrm{boundary}}/n$
- Interior-node fraction, $n_{\mathrm{interior}}/n$
- Boundary-normalized quantities such as $b/n$ and $b/T$

For a connected, hole-free, fully triangulated planar patch,

$$
m=3n-b-3,\qquad T=2n-b-2.
$$

Consequently, apparent density differences at fixed $n$ can largely reflect boundary structure. For Delaunay-derived motifs, boundary-to-interior organization is often more informative than generic graph density.

### Holes and fragmentation

- Number of holes
- Number of connected components, if disconnected motifs are permitted
- Euler characteristic,

$$
\chi=|V|-|E|+|F|.
$$

For a connected planar region, $\chi=1-H_{\mathrm{holes}}$. These measures capture fragmentation and non-simply-connected tissue patches.

### Degree structure

- Mean degree, $2m/n$
- Degree variance and standard deviation
- Degree coefficient of variation, $\mathrm{CV}(d)=\sigma_d/\mu_d$
- Maximum degree
- Full degree histogram or distribution

Mean degree alone has limited discriminatory value because large interior regions of a planar Delaunay triangulation naturally approach mean degree six.

### Triangle participation

For each node $v$, define

$$
t_v=\#\{\text{triangles containing }v\}.
$$

Summarize the mean, variance, coefficient of variation, quantiles, and distribution of $t_v$. This is often more directly relevant than degree in a triangulation motif.

### Path-based topology

- Graph diameter
- Mean shortest-path length
- Path-length distribution

Because these measures strongly depend on size, normalize them for a two-dimensional patch, for example by $\sqrt n$:

$$
\mathrm{diameter}/\sqrt n,\qquad \bar L/\sqrt n.
$$

### Connectivity and bottlenecks

Using the normalized Laplacian

$$
L_{\mathrm{norm}}=I-D^{-1/2}AD^{-1/2},
$$

the second-smallest eigenvalue $\lambda_2$ measures algebraic connectivity. It can distinguish compact motifs from motifs joined by narrow bridges even when they have the same number of nodes.

### Spectral structure

- Normalized-Laplacian eigenvalue distribution
- Spectral entropy
- Selected low-frequency eigenvalues

These are useful as advanced motif fingerprints or classifier inputs, but they are lower priority than boundary, degree-variation, triangle-participation, and normalized-path measures.

## 2. Spatial geometry associated with the triangulation

Although edges have no supplied features, the spatial coordinates generate natural geometric features.

- Physical area
- Perimeter
- Cell density, $\rho=n/A$
- Compactness, $4\pi A/P^2$
- Solidity, $A/A_{\mathrm{convex\ hull}}$
- Elongation or anisotropy from coordinate PCA
- Mean, median, coefficient of variation, and quantile ratio of Delaunay edge lengths
- Mean, variance, and coefficient of variation of triangle areas
- Triangle-shape measures: minimum angle, maximum angle, aspect ratio, and circumradius-to-inradius ratio

Topology and geometry should be reported separately: identical adjacency structures can be embedded at different physical scales or with different shapes.

## 3. Node-type composition: one-body organization

Let $n_c$ be the count of type $c$ and $p_c=n_c/n$.

- Type richness, $R=|\{c:n_c>0\}|$
- Shannon entropy, $H_{\mathrm{node}}=-\sum_c p_c\log p_c$
- Size-aware normalized Shannon entropy,

$$
H_{\mathrm{node}}^{\mathrm{norm}}=
\frac{-\sum_c p_c\log p_c}{\log(\min(C,n))}
$$

- Simpson diversity, $1-\sum_c p_c^2$
- Hill numbers $D_0$, $D_1$, and $D_2$
- Evenness, such as $H_{\mathrm{node}}/\log R$
- Dominant-type fraction and top-$k$ type fraction
- Burden of biologically defined rare types

These quantify composition, not spatial organization. Richness and entropy estimates require explicit control for motif size.

## 4. Pairwise cell-contact topology: two-body organization

### Homotypic and heterotypic contacts

Define the homotypic edge fraction

$$
h=\frac{1}{m}\sum_{(u,v)\in E}\mathbf 1[\tau(u)=\tau(v)].
$$

Then $1-h$ is the heterotypic edge fraction. These capture same-type clustering versus type mixing.

### Cell-type mixing matrix

For an undirected graph,

$$
M_{ab}=\#\{(u,v)\in E:\tau(u)=a,\tau(v)=b\},\qquad M_{ab}=M_{ba}.
$$

Retain both the raw matrix and its normalized form $\widetilde M_{ab}=M_{ab}/m$. The matrix is a high-dimensional motif fingerprint of cell-cell contacts.

### Edge-type entropy

Treat each unordered type pair $(a,b)$, with $a\le b$, as an edge category. If $q_{ab}=M_{ab}/m$, compute

$$
H_{\mathrm{edge}}=-\sum_{a\le b}q_{ab}\log q_{ab}.
$$

This measures the diversity of contact types and should not be conflated with node-composition entropy.

### Contact enrichment beyond composition

Keep graph topology and within-motif type counts fixed, randomly permute node labels, and calculate

$$
Z_{ab}=\frac{M_{ab}^{\mathrm{obs}}-\mathbb E_{\mathrm{perm}}[M_{ab}]}{\mathrm{SD}_{\mathrm{perm}}[M_{ab}]}.
$$

Positive values indicate enriched contact beyond what composition alone predicts; negative values indicate avoidance.

## 5. Triangle topology: three-body organization

Triangle analysis is especially important because the motifs originate from a Delaunay triangulation and naturally encode higher-order three-cell organization.

### Coarse triangle classes

Classify each triangle as:

- AAA: all three nodes have the same type
- AAB: two nodes share a type and the third has another type
- ABC: all three nodes have different types

Record the fractions $P_{AAA}$, $P_{AAB}$, and $P_{ABC}$, which sum to one.

### Typed triangle spectrum

Count each unordered cell-type triple $(a,b,c)$ with $a\le b\le c$. The theoretical number of categories is

$$
\binom{C+2}{3}.
$$

This vector is the detailed triangle signature of a motif.

### Triangle-type entropy

If $q_{abc}=T_{abc}/T$, compute

$$
H_{\triangle}=-\sum_{a\le b\le c}q_{abc}\log q_{abc}.
$$

Together, $H_{\mathrm{node}}$, $H_{\mathrm{edge}}$, and $H_{\triangle}$ form a one-body/two-body/three-body hierarchy.

### Triangle enrichment beyond composition

Using the same within-motif label-permutation null,

$$
Z_{abc}=\frac{T_{abc}^{\mathrm{obs}}-\mathbb E_{\mathrm{perm}}[T_{abc}]}{\mathrm{SD}_{\mathrm{perm}}[T_{abc}]}.
$$

This identifies three-cell configurations that cannot be explained solely by the abundance of their constituent cell types.

## 6. Type–topology coupling

These criteria ask which cell types occupy particular structural roles.

- Mean and distribution of degree conditional on cell type
- Mean and distribution of triangle participation conditional on cell type
- Cell-type enrichment at the boundary versus the interior
- Cell-type-specific egophily: the fraction of a node's neighbors with the same type
- Node-level neighborhood entropy and its motif-level distribution

For node $v$, define the neighbor-type proportions $p_{v,c}$ and local entropy

$$
H_v=-\sum_c p_{v,c}\log p_{v,c}.
$$

Summarize $\{H_v:v\in V\}$ using its mean, median, variance, and interquantile range. The variance is particularly useful because it captures whether mixing is spatially uniform or varies strongly across regions within a motif.

## 7. Between-motif heterogeneity

For each phenotype and size, compare both the average motif value and the variability among motifs. For every major metric, report:

- Mean or median shift
- Variance or dispersion shift
- Full distributional difference

A disease group may have the same mean as the healthy group but substantially greater motif-to-motif variability.

Useful distances include:

- Jensen–Shannon divergence between node-composition vectors
- Frobenius distance between normalized mixing matrices
- Frobenius distance between null-corrected $Z_{ab}$ matrices
- Jensen–Shannon divergence between triangle spectra

Average pairwise distance within a phenotype and size stratum gives a direct measure of between-motif compositional, contact, or triangle heterogeneity.

## 8. Size and cross-scale analysis

Treat motif size as an analysis dimension rather than only a nuisance covariate.

- Plot each metric as a function of size for healthy and disease groups
- Estimate the phenotype difference $\Delta(k)$ at each size $k$
- Estimate heterogeneity growth, $g(k)=H(k+1)-H(k)$ or $dH/dk$
- Estimate a saturation scale using, for example,

$$
H(k)=H_\infty\left(1-e^{-k/\tau}\right)
$$

- Test whether phenotype-associated differences emerge only above a particular multicellular scale

Metrics strongly confounded by size include richness, numbers of edge and triangle types, diameter, triangle count, boundary fraction, and entropy estimates.

## 9. Required null models and statistical controls

### Node-label null

Fix topology and within-motif type counts; permute node labels. This tests organization beyond composition and produces statistics such as $Z_{ab}$ and $Z_{abc}$.

### Phenotype-label null

Fix the motif/data structure; permute healthy/disease labels at the appropriate biological-sample level. This tests phenotype association.

### Size-matched null

Match or reweight the size distributions across phenotypes. This tests whether an apparent phenotype effect is caused by unequal motif sizes.

Motifs from the same patient or region of interest are not independent because they may overlap, share nodes or triangles, and be spatially correlated. The experimental unit should therefore remain the patient or biological sample. Suitable methods include mixed-effects models, patient-level bootstrap, patient-level permutation, or aggregation by patient and motif size.

For discrete sizes, a useful model is

$$
z_{ijr}=\beta_0+\beta_1\mathrm{Disease}+\beta_2\mathrm{Size}
+\beta_3(\mathrm{Disease}\times\mathrm{Size})+u_j+u_{jr}+\epsilon_{ijr},
$$

where the interaction $\beta_3$ tests whether the phenotype effect changes with motif size. For a broad size range, use a nonlinear mixed model such as a generalized additive mixed model.

## 10. Recommended first-pass metric panel

| Dimension | Core criteria |
|---|---|
| Size | $n$, $T$, physical area |
| Pure topology | Boundary fraction, triangles per node, degree CV, normalized diameter, hole count |
| Geometry | Cell density, compactness, anisotropy, edge-length CV, triangle-area CV |
| Composition | Richness, normalized Shannon entropy, Hill $D_1$ and $D_2$, dominant-type fraction |
| Pairwise contacts | Homotypic and heterotypic edge fractions, normalized edge-type entropy, $Z_{ab}$ mixing matrix |
| Three-body organization | AAA/AAB/ABC fractions, triangle-type entropy, $Z_{abc}$ enrichment |
| Local organization | Mean and variance of local entropy, mean egophily |
| Between motifs | Composition JS dispersion, mixing-matrix dispersion, triangle-spectrum dispersion |

## 11. Interpretation framework

Organize each motif through three nested representations:

1. One-body composition: $\mathbf p=[p_1,\ldots,p_C]$
2. Two-body spatial interaction: $\mathbf M=[M_{ab}]$
3. Three-body spatial organization: $\mathbf T=[T_{abc}]$

At every level, analyze diversity, enrichment relative to a null model, between-motif dispersion, healthy-disease differences, and size dependence. Keep “composition” distinct from “organization beyond composition.”

A compact overall profile is

$$
\mathcal H(M)=\{H_{\mathrm{node}},H_{\mathrm{edge}},H_{\triangle},H_{\mathrm{local}},H_{\mathrm{geometry}},H_{\mathrm{topology}}\},
$$

and the primary scientific target is the conditional distribution $P(\mathcal H\mid k,Y)$, where $k$ is motif scale and $Y$ is phenotype.

## Key cautions

- Do not rely on generic graph density or mean degree alone; planar Delaunay constraints make them partly deterministic.
- Do not interpret raw cell-contact or triangle counts without controlling for cell-type abundance.
- Do not compare differently sized motifs without normalization, matching, or explicit size modeling.
- Do not treat motifs as independent observations when they come from the same patient or region.
- Do not summarize phenotype heterogeneity only through mean differences; examine dispersion and multimodality as well.
