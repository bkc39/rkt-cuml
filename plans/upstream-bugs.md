# Upstream bugs

Found by the cluster arc (#5) and the metrics leg (A2, #34) against libcuml 26.08.00, cuVS 26.08.01 and raft
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
| A2 #34 | pairwise distances underflow at tiny scale: float32 `sqeuclidean` silhouette samples of data scaled by 1e-19 are off by 0.98, float64 by 1e-14 at 1e-155; trustworthiness of X scaled by 1e-23 scores 0.401 for 0.511 | pairwise distances | documented; a test pins the float32 case |
| A2 #34 | cosine distances are not clamped to [0, 2] (sklearn's `cosine_distances` clamps), so silhouette samples of rows pointing the same way exceed 1 by rounding: 1.0000004 (float32), 1 + 7e-16 (float64) | `CosineExpanded` distance | documented |
| A2 #34 | the batched silhouette counts n·n_labels and chunk² in an `int` | batched silhouette | refused past 2³¹−1 and chunk 46340 |

## raft

| leg | behaviour | where | rkt-cuml |
|---|---|---|---|
| 3 #21 | `make_monotonic` uses `int tid`: labels at index ≥ 2³¹ are never remapped | `label/classlabels.cuh:122` | refused past 2³¹−1 rows |
| 5 #29 | Lanczos indexes its n·ncv basis in an `int`: past 2³¹ it corrupts memory and poisons the context | `sparse/solver/lanczos.cuh:111` | refused |
| 5 #29 | Lanczos never reports non-convergence; a NaN or huge tol gives garbage | `sparse/solver/lanczos.cuh` | tol refused, drift documented |
| 5 #29 | repeated eigenvalues break the single-vector Lanczos: disconnected cliques fail to converge, weakly linked cliques give a wrong partition silently | `sparse/solver/lanczos.cuh` | documented |
| 5 #29 | the spectral degrees sum in float32, so large weights overflow into a wrong partition | spectral Laplacian | refused past FLT_MAX |
| 5 #29 | Lanczos leaks a cuSPARSE handle under LSan | `sparse/solver/lanczos.cuh` | suppressed (`leak:libcusparse.so`) |
| A2 #34 | `contingencyMatrix` sizes its square table from the first vector's min and max only: a predicted label outside that range writes out of bounds | `stats/detail/contingencyMatrix.cuh` | dense codes of both vectors, one K×K table |
| A2 #34 | trustworthiness with a NaN, Inf or 1e20 entry in X or X-embedded is an illegal address (context poisoned) | `stats/detail/trustworthiness_score.cuh` | refused past a squared norm of 2¹²⁵ |
| A2 #34 | trustworthiness at n-neighbors ≥ n is an illegal address; 2k ≥ n is outside the formula's domain | `stats/detail/trustworthiness_score.cuh` | refused at 2k ≥ n |
| A2 #34 | trustworthiness's batch loop never ends at a batch size of 0, and n·(k+1), batch·n and n·k are `int` | `stats/detail/trustworthiness_score.cuh` | refused below 1 and past 2³¹−1 |

## Python cuML

| leg | behaviour | rkt-cuml |
|---|---|---|
| 4 #26 | `AgglomerativeClustering` converts float64 to float32 without warning, which can merge distinct points | float64 refused |
| 4 #26 | the docstring's knn metric restriction is not checked | euclidean only |
| A2 #34 | `adjusted_rand_score` truncates int64 labels to int32 without a warning: [0, 0, 2⁴⁰, 2⁴⁰] against [0, 0, 1, 1] scores 0.0, not 1.0 | int64 labels remapped |
| A2 #34 | `silhouette_score` lets a 1e20 float32 row through and returns NaN, where sklearn gives 0.593 | refused |
| A2 #34 | `trustworthiness` passes a 1e20 float32 entry into the illegal address above | refused |
