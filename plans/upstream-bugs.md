# Upstream bugs

Found by the cluster arc (#5), the metrics leg (A2, #34), the PCA leg (A4, #47), the truncated SVD leg
(A5, #48) and the UMAP leg (B1, #50) against libcuml 26.08.00, cuVS 26.08.01 and raft
26.08. None is filed upstream (owner, 2026-10-20: log locally). Each is refused,
worked around or documented by the shim, and a test pins it; the leg's PR
packet (§4) has the detail.

## cuML

| leg | behaviour | where | rkt-cuml |
|---|---|---|---|
| 2 #16 | k-means is not reproducible run to run on one input, batch size and `random_state`: on 4096-row blobs the k-means++ n-init 3 winner changes and float32 random init flips single rows (float atomics); 5 of 32 blob cases changed labels or n-iter over 5 calls, as much at a repeated batch 32768 as at 7, 100, 4095 or 4096 | `kmeans/` | documented (`current-batch-samples`); the batch test weighs a batch against default refits |
| 3 #21 | `max_mbytes_per_batch` too small gives batch size 0, then `ceildiv(n, 0)`: SIGFPE | `dbscan/dbscan.cuh`, `dbscan/runner.cuh:121` | refused |
| 3 #21 | `metric="precomputed"` on a non-square X reads out of bounds | `dbscan/` | refused |
| 3 #21 | the cosine branch divides column j by row j's norm: wrong, order- and scale-dependent labels; out of bounds when d > n | `dbscan/vertexdeg/algo.cuh:173-184` | own unit rows on the euclidean path |
| 3 #21 | eps and min_samples unchecked: eps −1 acts as 1, min_samples ≤ 0 as 1 | `dbscan/` | refused |
| 5 #29 | spectral at n = 2 or 3 segfaults on both paths (Python requires only n ≥ 2) | spectral | refused below 4 |
| 5 #29 | n_components outside (0, n), n_clusters > n (labels out of range), n_init ≤ 0 (every label 1), n_neighbors 1, ≥ n or negative (RMM overflow) unchecked | spectral | refused |
| 5 #29 | NaN, Inf and 1e20 rows give silently wrong labels | spectral | refused |
| 5 #29 | the COO graph is unchecked: out-of-range index runs silently (10⁸ is an illegal address), lengths not compared, duplicate self-loops shrink the Laplacian below nnz, nnz = 0 fails with "invalid device ordinal" | spectral | refused |
| 5 #29 | labels renumber between otherwise identical runs | spectral | documented |
| 7 #12 | `min_cluster_size` below 2 segfaults; it is an `int` | `hdbscan/` | refused outside [2, 2³¹−1] |
| 7 #12 | `max_cluster_size` is an `int`: 2³¹ would read as unlimited | `hdbscan/` | refused past 2³¹−1 |
| 7 #12 | `cluster_selection_epsilon` −1 or NaN acts as 0 | `hdbscan/` | refused |
| 7 #12 | `alpha` 0, −1, NaN, or 0 as a float, gives all noise; below about 1e−17 too | `hdbscan/` | refused; the small range documented |
| 7 #12 | a NaN, Inf, 1e20 or 1e18 fit row becomes noise, other outputs unchanged | `hdbscan/` | refused past a squared norm of 2¹²⁵ |
| 7 #12 | a NaN or Inf query row is an illegal address (context poisoned) in `approximate_predict` and `membership_vector`; a 1e18 row predicts noise | `hdbscan/` prediction | refused past a squared norm of 2¹²⁵, as fit rows |
| 7 #12 | prediction with fewer than 2·min_samples training rows is an illegal address | `approximate_predict.cuh` | refused |
| 7 #12 | an empty query fails inside cuVS or RAFT | `hdbscan/` prediction | refused |
| 7 #12 | a query with the wrong column count gives garbage | `hdbscan/` prediction | refused |
| 7 #12 | the predict kernel indexes `selected_clusters` by caller labels | `approximate_predict.cuh` | labels are a model region |
| 7 #12 | `int` counts over edges, knn sizes and m·k memberships | `hdbscan/detail/` | refused past 2³¹−1 |
| A2 #34 | ARI sizes its table by the label range, not the label count: labels {0, 10⁵} ask for 80 GB, {0, 2⁴⁰} int64 for 17.6 TB | `metrics/adjusted_rand_index` | labels remapped to dense codes |
| A2 #34 | ARI on labels {0, 46341} is an illegal address (context poisoned): the table index is 32-bit | `metrics/adjusted_rand_index` | refused past 46340 distinct labels |
| A2 #34 | raw silhouette with labels outside [0, n_labels) is silent garbage (scores above 1) | `metrics/silhouette_score` | labels remapped to dense codes |
| A4 #47 | whitened transform and inverse scale by the call's own row count: 1 row is all zeros, 10 rows ×0.095 | `raft/linalg/detail/pca.cuh` `pca_transform`, `pca_inverse_transform` | whiten never passed; the shim scales |
| A4 #47 | noise variance averages d − k eigenvalues, null ones included, when n < d | `pca.cuh` `trunc_comp_exp_vars` | fit min(n, d) components; mean of λ[k:m] |
| A4 #47 | float32 explained variances go negative; singular values clamp, variances don't | `pca.cuh` | clamped to 0 |
| A4 #47 | covariance overflow (float32 3e38, float64 1e160) is NaN everywhere; float32 1e19 is garbage | `pca.cuh` (covariance gemm) | refused past √(max/(4nd)) |
| A4 #47 | covariance underflow: below about 1e-19 (float32) or 1e-154 (float64) XᵀX is subnormal or 0, so λ and σ are 0, the components arbitrary, and Jacobi returns NaN components (float32 1e-20, float64 1e-160) | `pca.cuh` (covariance gemm), `eigJacobi` | X fitted at a power-of-two scale, results scaled back |
| A4 #47 | n_components above n_rows accepted: arbitrary null-space components | `pca.cuh` `pca_fit` | refused past min(n, d) |
| A4 #47 | the column mean is not reproducible run to run | `raft::stats::mean` | documented; tolerance laws |
| A4 #47 | Jacobi: 0 sweeps means cusolver's default, 2³¹ wraps to int, non-convergence is silent; `syevd`'s dev_info is unread | `raft/linalg/detail/eig.cuh` | 0 and above 2³¹−1 refused; non-convergence documented |
| A4 #47 | `pcaFit` and `pcaTransform` write to their input | `pca.cuh` | the shim works on a copy |
| B1 #50 | UMAP fit uses a = b = −1 unless the caller fits them; find_ab is a different float32 gradient-descent fit, up to 1e-3 off umap-learn's | `umap/runner.cuh` `_fit`, `optimize.cuh` `find_params_ab` | Racket fits a, b (LM, normalised coordinates) |
| B1 #50 | Python's curve_fit for a, b is garbage at extreme spread: (1, 1) at 1e-6, b = −70.5 at 1e-3 | `umap.pyx` `find_ab_params` | fit in normalised coordinates; departure pinned |
| B1 #50 | nn_descent with an L1 or Linf metric raises "nn_descent does not support transform" during a fit | `knn_graph/algo.cuh:72-77` | refused by metric |
| B1 #50 | init_transform accumulates into the output (caller must zero) and indexes row·k in int | `runner.cuh:58-79` | output zeroed; refused past 2³¹−1 |
| B1 #50 | transform with n_epochs 1 or 2 runs 0 epochs: every edge trimmed, then "min_element failed to launch cub::DeviceReduce::ArgMin: cudaErrorInvalidValue" (P5) | `runner.cuh:549-559` | refused below 3 (B1-g) |
| B1 #50 | spectral init failure (incl. n ≤ n_components, and 12 identical rows with labels i mod 3, 1 component, random_state 1; k = 600 on 800 rows is now refused by f26) silently falls back to random init | `init_embed/runner.cuh` | n ≤ n-components refused; other failures documented (B1 r2 finding 1) |
| B1 #50 | n_neighbors 1 gives an empty graph (self weight 0), then "__copy:: D->H: failed: cudaErrorInvalidValue" (P6) | `fuzzy_simpl_set/naive.cuh`, `runner.cuh` `trim_graph` | refused below 2 |
| B1 #50 | smooth_knn_dist keeps max_nonzero in an int, truncating ρ when local_connectivity exceeds the non-zero count | `fuzzy_simpl_set/naive.cuh:51-75` | unreachable: local-connectivity fixed at 1 |
| B1 #50 | Python passes integer labels as float32: labels above 2²⁴ merge (P12) | `umap.pyx` `check_y` | dense codes |
| B1 #50 | nn-descent is not reproducible run to run, and gives non-finite embeddings at 10, 30 and 60 rows (P3, P7) | `knn_graph/algo.cuh`, cuvs#184 | documented; refused below 150 rows |
| B1 #50 | negative_sample_rate 0 is accepted and gives a finite, deterministic embedding (P4) | `optimize_batch_kernel` | accepted; lower bound 0 |
| B1 #50 | a fit on rows of magnitude 1e20 raises "ValueError: negative axis 0 index: -1"; NaN rows are refused only by Python's check (P8) | `umap.pyx`, `runner.cuh` | non-finite and over-large rows refused before the fit |
| B1 #50 | cosine with a zero row: the fit is finite after a spectral fallback warning, the transform gives NaN in that row (P9) | `knn_graph` cosine | zero-norm rows refused for cosine |
| B1 #50 | the optimisation diverges silently: learning-rate 1e35 gives an all-NaN (or inf) embedding (P10) | `optimize_batch_kernel` | non-finite scan after fit and transform |
| B1 #50 | transform picks its index width from the new rows alone, and init_transform reads `embeddings[indices[i+j] * n_components + d]` as an int product, so a training set with n·n-components past 2³¹−1 is read out of bounds | `umap.cu:138` `dispatch_to_uint64_t`, `runner.cuh:72` | refused past 2³¹−1 at transform (t10) |
| B1 #50 | learning-rate 1e-50 rounds `initial_alpha` to 0 in float32, and the fit silently returns the initial layout (B1 r0 p1) | `runner.cuh`, `optimize_batch_kernel` | refused below 2⁻¹²⁶ in float32 (f9) |
| B2 #51 | FFT t-SNE stops on mean \|x-step\|/2 (y ignored), checked from iteration 0: at learning rate 200, n ≥ 12000 stops at iteration 16–19, collapsed embedding (std 3e-6), kl 0 | `fft_tsne.cuh:729-760` | min_grad_norm 0 for fft (B2-b); L8 pins Python's collapse |
| B2 #51 | KL divergence computed only on the last iteration: 0 after an early stop; n_iter is one short after an early stop | `exact_tsne.cuh`, `fft_tsne.cuh` | kl #f, n-iter corrected |
| B2 #51 | Barnes-Hut spins forever once coordinates are non-finite (learning rate 1e30, early exaggeration 1e30, 1e19 rows, constant X, cosine zero row) | `barnes_hut_kernels.cuh` | not exposed (B2-a) |
| B2 #51 | NaN embedding without error (constant X, 1e19/1e20 rows, cosine zero row, n_neighbors 1, divergence, n ≤ 12), KL finite beside it | `distances.cuh` `normalize_distances`, integration | refused before (f5, f19–f22) and after (f23) |
| B2 #51 | init='pca' writes to X (≤ 9.5e-7) and builds the kNN from the written X | `tsne_runner.cuh:115` | shim works on a copy |
| B2 #51 | exact grid y = ⌈n/32⌉ exceeds 65535 past 2,097,120 rows: n = 2,097,121 raises cudaErrorInvalidValue at `exact_kernels.cuh:384` after 127 s of kNN (P5; n = 2,097,120 did not finish its kNN in 900 s) | `exact_kernels.cuh:369` | refused (f18) |
| B2 #51 | random_vector takes int size: n·2 narrows past 2³¹−1 (source) | `utils.cuh:53` | refused (f16) |
| B2 #51 | perplexity above n_neighbors is unreachable and acts as n_neighbors | `sigmas_kernel` | refused (f17) |
| B2 #51 | FFT at perplexity 5 with 90 neighbours: trustworthiness 0.507 against exact 0.967 | `fft_tsne` | n-neighbors defaults to 3·perplexity |
| B2 #51 | exact and Barnes-Hut are not reproducible for a fixed seed | atomics | Q-D oracles |
| B2 #51 | perplexity_max_iter 0 leaves P uninitialised; cufftDestroy skipped on a throw (source) | `exact_kernels.cuh:42`, `fft_tsne.cuh` | unreachable / documented |
| B2 #51 | a seeded FFT run is 1.3–1.9× slower than random_state=None (blobs 1000×10 0.35 s against 0.18 s, digits 0.355 against 0.21, 12000×50 0.705 against 0.56; P4) | `fft_tsne.cuh` deterministic path | seeds always drawn (B2-l) |
| B2 #51 | `square_distances` squares the kNN distances before normalising them, so under `L2Expanded` (sqeuclidean) it takes a fourth power, which overflows float32 to NaN once opposite rows reach a float32 squared norm of 2⁶²: 2⁶² and 2⁶²−2³⁵·⁸ give NaN, and so do two-column rows of exact squared norm 2⁶²−2³⁸−2³⁴·³ whose float32 norm rounds up to 2⁶² | `tsne_runner.cuh:210-217` | refused past 2⁶¹ under sqeuclidean (f20b) |
| B2 #51 | PCA init divides Y by its float32 standard deviation; once n·var passes float32's maximum the sum of y² is inf, exact returns an all-zero embedding without error and FFT gives NaN (16 rows at 4·10³⁸, 4000 at about 4·10³⁸; random init fits the same rows) | `tsne_runner.cuh:141-145` | refused past a total squared distance from the mean of 2¹²⁶ under pca init (f20d); constant embedding refused after the fit |
| B2 #51 | `square_distances` squares the L1 distance under manhattan, which overflows float32 for rows inside the 2¹²⁵ squared-norm bound: ±(3.5·10¹⁸)×3 gives NaN under FFT and exact | `tsne_runner.cuh:210-217` | refused past an L1 norm of 2^62.5 under manhattan (f20c) |
| B2 #51 | the squared kNN distances are scaled by 1/max, inf in float32 once the largest neighbour distance under the metric is below 2⁻⁶⁴, which gives NaN | `distances.cuh:147` | f23 after the fit; documented |
| B2 #51 | the same scaling is 0·inf = NaN when every computed kNN distance is 0: rows far from the origin, whose ‖x‖²+‖y‖²−2x·y cancels in float32 (a 1.7·10⁹ time column under euclidean), or duplicates filling every neighbourhood (40×(0,0), 40×(1,1), perplexity 5) | `distances.cuh:147` | f23 after the fit, advising "centre or rescale X, or raise n-neighbors"; documented |

## cuVS

| leg | behaviour | where | rkt-cuml |
|---|---|---|---|
| 4 #26 | l1, manhattan, cityblock and cosine fail under `pairwise` with "kmeans requires L2Expanded" after allocating the n² graph; under `knn` components join by L2 whatever the metric | `kmeans_common.cuh:343`, `cross_component_nn.cuh:336` | euclidean only |
| 4 #26 | `pairwise` computes nnz = m·m in an `int`: at n = 46341 it asks for 18 EB | `connectivities.cuh:120,183` | refused past 46340 |
| 4 #26 | d = 0 is an illegal address that poisons the CUDA context | single linkage | refused |
| 4 #26 | n_clusters = 0 wraps a size inside RMM | single linkage | refused |
| 4 #26 | NaN, Inf and overflowing rows fail as "KNN graph could not be connected" or "Number of edges found by MST is invalid" | single linkage | refused |
| 4 #26 | `build_k` sums ⌊log₂ n⌋ + c in `size_t`, so a negative c wraps to k = n | `knn_graph.cuh:24` | c clamped |
| 4 #26 | at small k the knn path is not the MST (k = 2: 20–37% off scipy), and it is nondeterministic on ties | single linkage | documented |
| 4 #26 | the knn join is capped at 10 iterations: 1024 separated groups fail at the default c | single linkage | documented |
| 5 #29 | k-means++ allocates its (2+⌈ln k⌉)·n buffer with int extents | `kmeans.cuh:105,121` | refused past 2³¹−1 |
| A2 #34 | a silhouette chunk size of 0 or −1 hangs forever in a host loop | `stats/detail/batched/silhouette_score.cuh` | refused below 1 |
| A2 #34 | silhouette with a NaN, Inf or 1e20 (float32) row gives NaN for every sample | batched silhouette | refused past a squared norm of 2¹²⁵ (float64: 2¹⁰²¹) |
| A2 #34 | silhouette under cosine with an all-zero row gives NaN for every sample; a row whose squared norm is subnormal, or whose terms are (1000 columns at 1.3e-38), gives NaN, −inf or silently wrong samples | batched silhouette | refused below a squared norm of d·2⁻¹²⁶ (float64: d·2⁻¹⁰²²) for d columns |
| A2 #34 | pairwise distances underflow at tiny scale: float32 `sqeuclidean` silhouette samples of data scaled by 1e-19 are off by 0.98, float64 by 1e-14 at 1e-155; float32 `euclidean` sums squared differences before the root, so it fails once those near the least normal float: off by 0.86 at 1e-30; trustworthiness of X scaled by 1e-23 scores 0.401 for 0.511 | pairwise distances | documented; tests pin the float32 `sqeuclidean` and `euclidean` cases |
| A2 #34 | cosine distances are not clamped to [0, 2] (sklearn's `cosine_distances` clamps), so silhouette samples of rows pointing the same way exceed 1 by rounding: 1.0000004 (float32), 1 + 7e-16 (float64) | `CosineExpanded` distance | documented |
| A2 #34 | the batched silhouette counts n·n_labels and chunk² in an `int` | batched silhouette | refused past 2³¹−1 and chunk 46340 |
| B1 #50 | spectral embedding with n < 2·nev + 2 (UMAP's spectral init asks for nev = n_components + 1) sizes its Lanczos basis at ncv = min(n − nev, max(2·nev + 1, 20)) < nev + 2, and `lanczos_smallest` writes row nev + 1 of it (`lanczos.cuh:678-679`): at n = 2·nev a plain segfault, and under compute-sanitizer either 86 to 144 errors (6/2, 8/3, 40/19, 98/48, 798/398) or a host segfault whose summary still reads "0 errors" (exit 11; 400/199, 800/399); at n = 2·nev + 1 a host segfault (exit 139 plain, 11 under memcheck with "0 errors"), measured at 7/2, 9/3, 41/19, 99/48, 401/199 and 799/398; clean from n = 2·nev + 2 (B1 r0, corrected in B1 r1). Python: `UMAP(n_components=3, init="spectral", random_state=0).fit(X)` on 9 rows of 10-column blobs segfaults; 10 rows is clean | `spectral_embedding::transform`, raft `sparse/solver/detail/lanczos.cuh` | refused unless n ≥ 2(n-components + 2) (f26) |
| B1 #50 | spectral init divides the eigenvectors by their largest absolute value with no NaN check, so a degenerate solve returns NaN without throwing and the random fallback never runs: 6 rows, 1 component, labels i mod 3, `n_epochs=20` give an all-NaN embedding in Python cuML 26.08, at `learning_rate=0.01` too | `init_embed/spectral_algo.cuh:73-91` | refused by the non-finite scan (f25), whose advice names init `'random` (f29) |
| B3 #27 | spectral embedding at n ∈ {nev+1, nev+2} raises cudaErrorInvalidValue and reads out of bounds (164 memcheck errors, 4 rows/2 components with drop_first), process usable; n ∈ {2nev, 2nev+1} segfaults as in B1 | `spectral_embedding.cuh` compute_eigenpairs, `lanczos.cuh` | refused unless n ≥ 2(nev+1) (s7, g2) |
| B3 #27 | brute-force kNN past 256 neighbours breaks k-th-distance ties differently run to run, so the dense embedding is not repeatable (n ≥ 2570 at Python's n/10 default) | all_neighbors / select_k | default capped at 256; documented (B3-d) |
| B3 #27 | n_neighbors 1, n_neighbors = n, all-equal or doubly-duplicated rows, rows far from the origin (1e9), rows below 1e-24, all-zero weights: all-zero embedding, no error | lanczos breakdown, knn ties | s2, s12, s14, g7; tiny X rescaled (B3-h) |
| B3 #27 | X scaled 1e-22/1e-23 or offset 1e5: finite wrong embedding (float32 expanded L2 underflow/cancellation) | brute-force L2SqrtExpanded | tiny X rescaled; offset documented |
| B3 #27 | rows near 3e38 squared norm: cuBLAS dot failure, then illegal address (context poisoned); at 2^127 wrong and not repeatable | knn → laplacian → lanczos | refused past 2^125 (s11) |
| B3 #27 | n_neighbors > n: cuSPARSE nnz > size or cudaErrorInvalidValue; 0 and negative fail inside cuVS/RMM | create_connectivity_graph | refused (s2, s9) |
| B3 #27 | precomputed graphs are not symmetrised (sklearn symmetrises) | transform(coo) | documented (B3-l) |
| B3 #27 | duplicate self-loops make compute_graph_laplacian under-allocate its diagonal entries | raft `laplacian.cuh` | self-loops refused (g5) |
| B3 #27 | eigen tolerance fixed at 1e-5: sklearn agreement 7e-2 on a 400-node path, 0.81 on one overlapping-blob graph | lanczos | documented; L6 on gapped graphs |
| B3 #27 | the 1e-5 tolerance is an absolute residual, so an unnormalised Laplacian with weights scaled by 2^-17 to 2^-20 gives a finite wrong embedding (gap 2.4 against sklearn), by 2^-30 or less an all-zero one, and by 2^80 or more "couldn't converge"; after that rescale, one weight pair 2^16 times the rest still gives a gap of 2.1 and from 2^23 an all-zero column (B3-u) | `lanczos.cuh` | weights rescaled to [1, 2) without norm-laplacian (B3-x); the heavy-weight case documented, its refusal advising norm-laplacian |

## raft

| leg | behaviour | where | rkt-cuml |
|---|---|---|---|
| 3 #21 | `make_monotonic` uses `int tid`: labels at index ≥ 2³¹ are never remapped | `label/classlabels.cuh:122` | refused past 2³¹−1 rows |
| 5 #29, B1 #50 | Lanczos indexes its n·ncv basis in an `int`: past 2³¹ it corrupts memory and poisons the context. UMAP's spectral init reaches it through `lanczos_smallest` with nev = n-components + 1 (15,000,000 rows at 100 components is 3.0e9 entries; from source, not run) | `sparse/solver/lanczos.cuh:111` | refused by the shared `require_basis`, in spectral clustering and in UMAP's spectral init (f27) |
| 5 #29 | Lanczos never reports non-convergence; a NaN or huge tol gives garbage | `sparse/solver/lanczos.cuh` | tol refused, drift documented |
| 5 #29 | repeated eigenvalues break the single-vector Lanczos: disconnected cliques fail to converge, weakly linked cliques give a wrong partition silently | `sparse/solver/lanczos.cuh` | documented |
| 5 #29 | the spectral degrees sum in float32, so large weights overflow into a wrong partition | spectral Laplacian | refused past FLT_MAX |
| 5 #29 | Lanczos leaks a cuSPARSE handle under LSan | `sparse/solver/lanczos.cuh` | suppressed (`leak:libcusparse.so`) |
| A2 #34 | `contingencyMatrix` sizes its square table from the first vector's min and max only: a predicted label outside that range writes out of bounds | `stats/detail/contingencyMatrix.cuh` | dense codes of both vectors, one K×K table |
| A2 #34 | trustworthiness with a NaN, Inf or 1e20 entry in X or X-embedded is an illegal address (context poisoned) | `stats/detail/trustworthiness_score.cuh` | refused past a squared norm of 2¹²⁵ |
| A2 #34 | trustworthiness at n-neighbors ≥ n is an illegal address; 2k ≥ n is outside the formula's domain | `stats/detail/trustworthiness_score.cuh` | refused at 2k ≥ n |
| A2 #34 | trustworthiness's batch loop never ends at a batch size of 0, and n·(k+1), batch·n and n·k are `int` | `stats/detail/trustworthiness_score.cuh` | refused below 1 and past 2³¹−1 |
| A5 #48 | rank-deficient X gives NaN singular values: √ of negative rounding eigenvalues | `linalg/detail/tsvd.cuh:382` `weighted_sqrt` | NaN mapped to +0 |
| A5 #48 | XᵀX overflow is a silent NaN in every output | `linalg/detail/tsvd.cuh:349` (Gram gemm) | refused past a squared Frobenius norm of 2¹²⁵ (float32) / 2¹⁰²¹ (float64) |
| A5 #48 | the ratio divides by the total variance unchecked: 0/0 is NaN | `linalg/detail/tsvd.cuh:522` | +0 when it is 0; near-constant noise documented |
| A5 #48 | n_components in (min(n, d), d] is accepted: the tail is an arbitrary null-space basis with NaN σ | `linalg/detail/tsvd.cuh:342` | refused past min(n, d) |
| A5 #48 | Gram underflow: below about 1e-19 (float32) or 1e-154 (float64) XᵀX is subnormal or 0, so σ, ev and ratio are 0 and the components arbitrary; at float32 1e-20 the subnormal total variance's reciprocal overflows and the ratio is 0 | `linalg/detail/tsvd.cuh` (Gram gemm, ratio) | X fitted at a power-of-two scale, results scaled back, as A4 #47; fit-transform's matrix is the transform of the unscaled X |
| A5 #48 | `tsvdTransform` and `tsvdInverseTransform` overflow to ±inf on finite X | `linalg/detail/tsvd.cuh` (gemm) | refused past max/(4·d) and max/(4·k) |

## Python cuML

| leg | behaviour | rkt-cuml |
|---|---|---|
| 4 #26 | `AgglomerativeClustering` converts float64 to float32 without warning, which can merge distinct points | float64 refused |
| 4 #26 | the docstring's knn metric restriction is not checked | euclidean only |
| A2 #34 | `adjusted_rand_score` truncates int64 labels to int32 without a warning: [0, 0, 2⁴⁰, 2⁴⁰] against [0, 0, 1, 1] scores 0.0, not 1.0 | int64 labels remapped |
| A2 #34 | `silhouette_score` lets a 1e20 float32 row through and returns NaN, where sklearn gives 0.593 | refused |
| A2 #34 | `trustworthiness` passes a 1e20 float32 entry into the illegal address above | refused |
| B1 #50 | transform is not bitwise repeatable where neighbour distances tie: three transforms of digits' last 360 rows differ in Python, and ours differ in 54 rows by up to 0.345; a 0.01 jitter makes both repeatable | brute-force knn tie order | documented; L6 checks the digits transform on jittered rows (B1 r0) |
| B2 #51 | learning_rate_method='adaptive' (the default) silently overrides n_neighbors and early_exaggeration | `cuml/manifold/t_sne.pyx` | 'auto' not offered (B2-c) |
| B3 #27 | default n_neighbors max(n/10, 1) is 1 below 20 rows: an all-zero embedding without error | `spectral_embedding.pyx` | default floored at 2, s14 |
| B3 #27 | no sign rule (sklearn flips each column to a positive largest entry) | `spectral_embedding.pyx` | sign rule in the shim (B3-e) |
