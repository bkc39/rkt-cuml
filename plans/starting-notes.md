# Starting notes for rkt-cuml

These notes collect what rkt-raft milestone 1 established (bkc39/rkt-raft epic #1, stack #14: PRs #11, #12, #18, #19, #20), so the cuML work starts from verified facts.

## What rkt-raft already gives us

- **A frozen downstream interface** (rkt-raft PR #20, AGENTS.md "The frozen downstream interface").
  - Racket `raft/unsafe`:
    - `with-array-views #:resources` binds each array to an `rr_view` and the resources to a `raft::handle_t*`. It keeps both alive for the native call, waits for the arrays' pending GPU work first, and refuses re-entry.
    - The status checker turns the downstream shim's last error into `exn:fail:raft`.
    - `raft-abi-pointer` exposes the ABI tag.
  - C++ helper headers from rkt-raft's flake dev output:
    - `raftrkt/view.hpp`: a view becomes a typed pointer, or a RAFT mdspan view. It also provides `handle_of` and `require_on_device`.
    - `raftrkt/error.hpp`: `translate_exceptions` and `classify`, plus `device_scope` and `sync_guard`.
    - `raftrkt/abi.h`.
  - **The rule:** synchronise the handle's stream on every exit, including when an exception is thrown.
- **One RAPIDS package set** exported by rkt-raft's flake: libraft, librmm and rapids-logger 26.8.0, libcuml 26.8.0, libcuvs 26.8.1, and nvJitLink 13.3. Build against exactly that set; RAFT has no versioned namespace, and `raft::handle_t`'s layout changed in 26.08.
- **The k-means canary,** `downstream/kmeans-canary/` in rkt-raft:
  - The C++ shim is `src/canary.cpp`. Racket binds it in `kmeans-canary/main.rkt`.
  - The Python twin is `tests/python/kmeans_twin.py`, using cuml-cu13 26.8.0 and scikit-learn.
  - The twin cases: 17, including 9 seed-sensitive ones (2000×4, std 4.0, k=8, three inits × seeds 1–3). They give ARI 1.0, the same `n_iter`, and inertia within a relative 5e-7.
  - The 1,000-fit memory test stays flat; drop counters show every buffer freed.

## The cuML C++ API we bind (checked against cuml release/26.10; re-check at v26.08.00 in ~/dev/rkt/cuml-upstream)

- Most of the public API takes `const raft::handle_t&` plus raw pointers and row/column counts (51 of 85 public headers). Only 4 headers use RAFT mdspan types.
- cuML's public headers contain no cuVS types. cuML calls cuVS internally, so no cuVS bindings are needed; `libcuvs.so` only has to be present at run time.
- cuML deleted its C API in 26.06 (NVIDIA/cuml#8011), so this repository needs its own C shim, as the canary has.
- Clustering entry points:

| Algorithm | Entry point | Notes |
|---|---|---|
| k-means | `ML::kmeans::fit`, `predict`, `fit_predict`, `transform` | float and double; `KMeansParams` with `rng_state`; X row-major |
| DBSCAN | `ML::Dbscan::fit` | labels int32; optional core-sample indices; `ML::distance::DistanceType` |
| Agglomerative | `ML::linkage::single_linkage` | float; children 2×(n−1) int32 |
| HDBSCAN | `hdbscan` in `cluster/hdbscan.hpp` | eight output arrays, `HDBSCANParams`, `hdbscan_output`; the condensed tree's size is known only after the call (rkt-raft #22) |
| Spectral clustering | `ML::SpectralClustering::fit_predict` | `raft::resources` plus `device_matrix_view` and `device_vector_view`, or a COO graph |
| Test data | `ML::Datasets::make_blobs` | int64 labels |

## Suggested order

1. Grow the canary into the k-means binding (fit, predict, fit_predict, transform; float and double).
2. DBSCAN.
3. Single-linkage agglomerative clustering.
4. HDBSCAN, after rkt-raft #22 or with a two-call workaround.
5. Spectral clustering.

Use cuML's own metrics (`cuml/metrics/metrics.hpp`), e.g. the adjusted Rand index, until rkt-raft has `raft/stats`. Twins are `cuml` Python plus scikit-learn.

## Process (as rkt-raft)

- One package, one collection, one manual.
- Legs as GitHub stacked PRs, each with its guide chapter and reference section.
- An independent adversarial review per PR, and every finding addressed.
- No contracts until a dedicated leg.
- Constructors named after their type.
- Build-memory caps on the lab host.
- Open questions inherited from rkt-raft:
  - #9, the GPU runner;
  - #15, the ffi2 port;
  - #21, atomic native calls;
  - #22, outputs of unknown size.
