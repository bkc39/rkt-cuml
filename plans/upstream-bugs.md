# Upstream bugs

Found by the cluster arc (#5) against libcuml 26.08.00, cuVS 26.08.01 and raft
26.08. None is filed upstream (owner, 2026-10-20: log locally). Each is refused,
worked around or documented by the shim, and a test pins it; the leg's PR
packet (§4) has the detail.

## cuML

| leg | behaviour | where | rkt-cuml |
|---|---|---|---|
| 3 #21 | `max_mbytes_per_batch` too small gives batch size 0, then `ceildiv(n, 0)`: SIGFPE | `dbscan/dbscan.cuh`, `dbscan/runner.cuh:121` | refused |
| 3 #21 | `metric="precomputed"` on a non-square X reads out of bounds | `dbscan/` | refused |
| 3 #21 | the cosine branch divides column j by row j's norm: wrong, order- and scale-dependent labels; out of bounds when d > n | `dbscan/vertexdeg/algo.cuh:173-184` | own unit rows on the euclidean path |
| 3 #21 | eps and min_samples unchecked: eps −1 acts as 1, min_samples ≤ 0 as 1 | `dbscan/` | refused |
| 5 #29 | spectral at n = 2 or 3 segfaults on both paths (Python requires only n ≥ 2) | spectral | refused below 4 |
| 5 #29 | n_components outside (0, n), n_clusters > n (labels out of range), n_init ≤ 0 (every label 1), n_neighbors 1, ≥ n or negative (RMM overflow) unchecked | spectral | refused |
| 5 #29 | NaN, Inf and 1e20 rows give silently wrong labels | spectral | refused |
| 5 #29 | the COO graph is unchecked: out-of-range index runs silently (10⁸ is an illegal address), lengths not compared, duplicate self-loops shrink the Laplacian below nnz, nnz = 0 fails with "invalid device ordinal" | spectral | refused |
| 5 #29 | labels renumber between otherwise identical runs | spectral | documented |

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

## raft

| leg | behaviour | where | rkt-cuml |
|---|---|---|---|
| 3 #21 | `make_monotonic` uses `int tid`: labels at index ≥ 2³¹ are never remapped | `label/classlabels.cuh:122` | refused past 2³¹−1 rows |
| 5 #29 | Lanczos indexes its n·ncv basis in an `int`: past 2³¹ it corrupts memory and poisons the context | `sparse/solver/lanczos.cuh:111` | refused |
| 5 #29 | Lanczos never reports non-convergence; a NaN or huge tol gives garbage | `sparse/solver/lanczos.cuh` | tol refused, drift documented |
| 5 #29 | repeated eigenvalues break the single-vector Lanczos: disconnected cliques fail to converge, weakly linked cliques give a wrong partition silently | `sparse/solver/lanczos.cuh` | documented |
| 5 #29 | the spectral degrees sum in float32, so large weights overflow into a wrong partition | spectral Laplacian | refused past FLT_MAX |
| 5 #29 | Lanczos leaks a cuSPARSE handle under LSan | `sparse/solver/lanczos.cuh` | suppressed (`leak:libcusparse.so`) |

## Python cuML

| leg | behaviour | rkt-cuml |
|---|---|---|
| 4 #26 | `AgglomerativeClustering` converts float64 to float32 without warning, which can merge distinct points | float64 refused |
| 4 #26 | the docstring's knn metric restriction is not checked | euclidean only |
