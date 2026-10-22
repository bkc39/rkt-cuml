# rkt-cuml layout: the owner's decisions so far

1. **A multi-package repository, typed-racket style.** Every package uses
   `(define collection 'multi)` and adds modules to the one `cuml/`
   collection:

   | Package | Holds | deps |
   |---|---|---|
   | `cuml-core-lib` | `cuml/core`: shared infrastructure for the family packages | base, raft |
   | `cuml-<family>-lib` | `cuml/<family>` and its shim loader, e.g. `cuml-cluster-lib` | base, raft, cuml-core-lib |
   | `cuml-lib` | `cuml/main.rkt`, re-exporting `cuml/core` and every family | the -lib packages |
   | `cuml-doc` | the one manual, `cuml/scribblings/cuml.scrbl` | cuml-lib + doc build deps |
   | `cuml-test` | every test, `cuml/tests/...` | everything |
   | `cuml` | meta package | cuml-lib, cuml-doc |

   A family package never depends on `cuml-lib` or on another family, so
   requiring `cuml/cluster` loads no other family's native library.

2. **Family packages export algorithm-specific procedures** (`kmeans-fit`,
   `kmeans-predict`, `dbscan-fit`, ...) that wrap libcuml. They never export
   a bare algorithm name (`kmeans`, `dbscan`, ...): those are reserved for
   the estimator wrappers in `cuml-lib`, so `cuml/main` re-exports everything
   without a collision.

3. **A scikit-learn-like estimator interface** (`define-generics`) with
   wrappers in `cuml-lib`. The generics live in `cuml-core-lib`, so glmnet
   and other libraries can implement them. Fit is functional: `fit` returns
   a fitted estimator rather than mutating one. The generics are small
   capability generics (`fit`, `predict`, `transform`, ...), with a
   `fit-predict` fallback built from them.

4. **The opaque model-handle helper** (a native object plus its finalizer,
   keeping its resources alive) lives in `cuml-core-lib`.

5. **Scope: clustering only for now** (k-means, DBSCAN, single-linkage
   agglomerative, spectral, HDBSCAN). None needs sparse arrays. HDBSCAN's
   outputs sized by the call use raft's deposit records (rkt-raft #22,
   merged).

6. **cuML is GPU-only.** libcuml has no CPU path, so neither has cuml.

## Decisions recorded in leg 0

- **`exn:fail:cuml` subtypes `exn:fail:raft`** (raft exports the struct with
  `struct-out`), so a raft handler catches cuML's failures. `load-family`
  passes its constructor to `status-checker`, and `define-job` takes it as
  `#:exn`.
- **One library per family**, `libcumlrkt_<family>.so`, entry points
  `cuml_<family>_*`. The common entries are four explicit lines per family,
  not a macro.
- **The cluster shim proves the pipeline only**: the ABI handshake,
  `cuml_cluster_libcuml_version` ("26.08.00", formatted at compile time from
  `cuml/version_config.hpp` as `raft-version` formats RAFT's) and
  `ML::Datasets::make_blobs`, bound both as a job (`define-job`) and as a
  short call (`with-array-views #:resources` + `status-checker`), so both
  downstream paths raise `exn:fail:cuml`. `blobs` (the job) and
  `blobs/short-call` live in `cuml/cluster/private/blobs.rkt`; nothing is
  public but `exn:fail:cuml`.
- **The failing-call tests use the shim's refusals** (`n-clusters` 0, an
  int32 `out`): `make_blobs` raises for no input the shim lets through.
  Zero rows, zero features and more clusters than rows all succeed, so the
  first test that carries a message from cuML itself comes with the first
  estimator.
- **`cuml/cluster` requires its loader** although it exports nothing yet, so
  `(require cuml/cluster)` loads the shim and checks the ABI.
- **Collection-level `info.rkt` files** sit in the deepest directory that
  needs them (`cuml/scribblings`, `cuml/cluster`, `cuml/tests`), so no two
  packages ship `cuml/info.rkt`.
- **The meta package `cuml`** depends on and implies cuml-lib and cuml-doc,
  not cuml-test, as typed-racket's does.
- **The flake reuses rkt-raft's** `nix/treefmt.nix`,
  `packages.{rapids,shim,raft-dev,python-twins}` and its docs-dependency
  inputs (polars, datasets) through `rkt-raft.inputs`, so the pins are
  raft's. rkt-raft follows this flake's nixpkgs and treefmt-nix, both at
  raft's locked revisions, so its store paths are unchanged. The Racket
  tools (fmt, review, pretty-expressive, threading-lib) are pinned in this
  repo's own `nix/racket-tools.nix`.
- **Not carried over from rkt-raft yet:** the bounded death-test gate (no
  death tests).

## Decisions recorded in leg 1 (k-means, #6)

- `cuml/cluster` exports `(struct-out kmeans-result)`, `kmeans-fit`,
  `kmeans-predict`, `kmeans-transform` and `kmeans-score`, with no runtime
  contracts. The bodies live in `cuml/cluster/private/kmeans.rkt`; the three
  shim entries are jobs (`define-job ... #:exn exn:fail:cuml`), with no
  short call.
- The shim allocates every output after its checks and hands it back in a
  raft deposit record (fit: centroids and labels; predict: labels;
  transform: distances), so no refusal leaves an allocation behind; score
  is its own entry with a scratch label vector. The Racket body opens the
  deposit on the call's resources and adopts its regions. The init names
  map to the shim's codes in one `match`.
- The shim's refusals run in a fixed table order, first refusal wins, before
  the GPU is touched; an empty matrix has no layout to check.
- The decisions posted on #5, each with the recommended answer:
  - **Q-A.** Reversed in review: the shim refuses `tol` <= 0 and a negative
    `oversampling-factor` itself, before allocating, in the repo's wording
    (`tol: expected greater than 0, got 0`); cuVS refused them only after
    the outputs were allocated.
  - **Q-B.** `#:random-state` defaults to 0.
  - **Q-C.** `#:n-clusters` is inferred from a matrix `#:init`; an
    explicit mismatch is refused.
  - **Q-D.** Column-major or strided input is refused; nothing is copied.
  - **Q-E.** predict, transform and score take centroids, not a
    `kmeans-result`.
  - **Q-F.** There is no `'k-means||` alias.
  - **Q-G.** Failures before the shim (raft's view packer, define-job's
    contract errors) stay non-cuML until the contracts leg.
  - **Q-H.** All-zero, negative or NaN weights pass through, documented.
  - **Q-I.** No `kmeans-fit-predict`.
  - **Q-J.** `#:n-init 'auto` with a matrix init is 10.
  - **Q-K.** `kmeans-transform` returns squared distances, as the entry
    says.
  - **Q-L.** The float32 twin allows n_iter to differ by 1, with ARI >= 0.99
    when it does.
  - **Q-M.** `kmeans-result` is exported with `struct-out`.

## Decisions recorded in leg 2 (estimators, #7)

- `cuml/estimator` (cuml-core-lib) requires only `racket/base` and
  `racket/generic`, so requiring it loads no native library. Five generics,
  one per capability: `gen:estimator` (`fit`, `fitted?`,
  `estimator-params`, `estimator-set-params`), `gen:predictor` (`predict`),
  `gen:clusterer` (`fit-predict`, falling back to predict after fit for a
  predictor and to `raise-support-error` otherwise), `gen:transformer`
  (`transform`, `fit-transform` falling back to transform after fit) and
  `gen:scorer` (`score`).
- The `kmeans` estimator lives in `cuml-lib/cuml/private/kmeans-estimator.rkt`
  over the `/who` forms of leg 1's procedures, which
  `cuml/cluster/private/kmeans.rkt` provides and `cuml/cluster` does not
  export; `define/who` (`cuml/cluster/private/define-who.rkt`) defines each
  public procedure and its `/who` form from one header, and the estimator
  takes its hyperparameters from the same module's `kmeans-params` table
  (`define-hyperparameters`, see "Hyperparameter tables" below).
- The decisions posted on #5 (comment 6245375667), each with the recommended
  answer:
  - **Q-A.** `cuml` exports the bare `fit`, `predict`, `transform`,
    `score`, `fit-predict`, `fit-transform` and `fitted?`, and
    `gen:scorer`/`scorer?` keep that name. `(require cuml glmnet)` needs an
    `except-in` for `predict` until glmnet implements `gen:predictor`.
  - **Q-B.** `fit-predict` is on its own `gen:clusterer`, not on
    `gen:predictor`.
  - **Q-C.** `fit-predict` returns the labels only.
  - **Q-D.** Errors name the generic that was called (`fit:`), through the
    private `/who` procedures.
  - **Q-E.** Every parameter update gives an unfitted estimator; an empty
    update is a clone.
  - **Q-F.** `#:resources` is a constructor keyword, `#f` meaning the
    calling thread's `current-device-resources` at each call; the estimator
    records no resources at fit. Revised by the hyperparameter tables' Q-A:
    it is a setting, kept outside `estimator-params`.
  - **Q-G.** The fitted state is `kmeans-model`, the `kmeans-result` or
    `#f`.
  - **Q-H.** `estimator-set-params` takes a hash. Revised by the
    hyperparameter tables: it also takes each parameter as a keyword, which
    wins over the same key in the hash.
  - **Q-I.** `n-clusters` is resolved when the estimator is built.
  - **Q-K.** `cuml/estimator` stays in cuml-core-lib; a `cuml-estimator-lib`
    with deps `("base")` is a follow-up if glmnet adopts the interface.
  - **Q-L.** k-means' `fit` and `score` ignore `y`.

## Hyperparameter tables (#7, PR #16)

- `define-hyperparameters` (`cuml/cluster/private/hyperparameters.rkt`)
  defines a family's `<algorithm>-params` once: its keys in order, each
  default (which may read earlier keys, unless it is `#:fixed`) and its
  `#:fixed` keys. `define/who` takes the table as `#:hyperparameters`, and
  `define-estimator` (`cuml/private/estimator-support.rkt`) builds the
  estimator wrapper over it; the keywords, defaults, printer and
  `estimator-set-params` come from the one table.
- The owner's answers, each the recommended one:
  - **Q-A.** Execution settings (`#:resources`, k-means'
    `#:max-samples-per-batch`) stay constructor keywords, stored outside
    `estimator-params`; `#f` reads the current value at each call.
  - **Q-B.** The linkage table is `single-linkage-params`.
  - **Q-C.** A misspelt keyword at a static call of a `-params` procedure, a
    `define/who` name with a table or an estimator constructor is a syntax
    error; as values they stay procedures with Racket's runtime keyword
    error.
  - **Q-D.** Nothing is recomputed on a base: `(kmeans-params base #:init M)`
    keeps `base`'s `n-clusters`.
- **H13** (batch invariance, the go/no-go for `current-batch-samples`):
  go. On the integer grids every batch size (1, 7, 63, 64) gives
  bit-identical fits, labels, distances and scores. On 4096-row float32 and
  float64 blobs (four inits, n-init 1 and 3, with and without weights,
  batches 7, 100, 4095, 4096 against 32768) no label or n-iter changed at a
  batch size beyond what repeated calls at the default batch show: cuML's
  k-means is not reproducible run to run on these blobs (float atomics).
  Centroids and inertia are not bit-identical; the largest relative
  deviation was 8.1e-6 (float32), the same order as run to run. In one of
  four test runs, float32 random n-init 1 weighted at batch 100 departed in
  3 of 3 reruns while 3 default refits agreed; 100 consecutive calls of
  that case at each batch then gave identical labels. A rerun of 5 calls
  per batch found 5 of the 32 blob cases changing labels or n-iter, each
  as much at a repeated batch 32768 as at 7, 100, 4095 or 4096. Decided
  on #16: keep `current-batch-samples` with the relaxed H13; the fallback
  (the batch size back in `kmeans-params`) was not applied.
  - Upstream: cuML 26.08 k-means is not reproducible run to run on one
    input, batch size and `random_state` (the k-means++ n-init 3 winner,
    1-row flips with float32 random init); logged in `plans/upstream-bugs.md`,
    not filed.

## Decisions recorded in leg 3 (DBSCAN, #8)

- `dbscan-fit` (`cuml/cluster/private/dbscan.rkt`) binds cuML's brute-force
  `ML::Dbscan::fit` for float32 and float64 with int32 or int64 labels,
  through one job, `cuml_cluster_dbscan_fit_job`. The deposit helper
  (`deposit-for/who`, `cuml/cluster/private/deposit.rkt`) and the shim's argument
  checks (`shim/cluster/src/support.hpp`) are shared with k-means.
- The hyperparameters are `dbscan-params`; `dbscan-fit` is a `define/who`
  with `#:hyperparameters dbscan-params`, and the `dbscan` estimator, an
  estimator and a clusterer, is a `define-estimator` over it with the setting
  `resources`, so `estimator-params` has no `'resources`.
- The spec's decisions, each with the recommended answer:
  - **Q-A.** `#:out-dtype`, Python's name, on `dbscan-fit` and as an
    estimator parameter.
  - **Q-B.** int32 labels over more than 2^31-1 elements are refused, not
    switched to int64. More than 2^31-1 rows are refused at either
    out-dtype, since raft's `make_monotonic` indexes rows with an `int`.
  - **Q-C.** `'cosine` runs on an n×d scratch copy, so `X` is not modified.
  - **Q-D.** Under `'cosine`, a row whose elements are all zero is refused.
    cuML 26.08's cosine branch normalises by column
    (`vertexdeg/algo.cuh:173-184`), so rkt-cuml does not use it: the scratch
    copy is scaled to unit rows in `X`'s dtype (by the largest magnitude,
    then the 2-norm), and cuML runs euclidean on it with eps' the smallest
    value in `X`'s dtype whose square, rounded in that dtype, reaches
    2 eps, or 8 when eps is 2 or more, so that every pair is a neighbour
    pair there, as in sklearn, which clips cosine distances to [0, 2]. A
    pair at cosine distance exactly eps is a neighbour pair when the unit
    rows are exact in the dtype, as one-hot rows are. Otherwise, as with any
    floating-point DBSCAN, pairs whose distance is close to eps may fall on
    either side; the tolerance is relative and grows with the number of
    columns, about 1e-5 at 1024 float32 columns.
  - **Q-E.** Non-finite `X` is passed through, and its rows are noise.
  - **Q-F.** `eps` must be finite and positive, so `+inf.0` is refused, as
    are `min-samples` below 1.
  - **Q-G.** `eps` is the exact double for float64; under `'cosine` the
    unit rows and eps' are in `X`'s dtype.
  - **Q-H.** Core-sample indices are always computed.
  - **Q-I.** `'rbc` is deferred to #17.
  - **Q-J.** `dbscan-result` has no cluster count.
- A batch size that cuML would compute as 0, explicit or automatic, is
  refused, since cuML divides by it (SIGFPE).

## Decisions recorded in leg 4 (single linkage, #9)

- `single-linkage-fit` (`cuml/cluster/private/linkage.rkt`) binds
  `ML::linkage::single_linkage` for float32 under the euclidean metric, at
  both connectivities, through one job, `cuml_cluster_single_linkage_fit_job`.
  It returns the labels and the children; the shim sorts each child pair
  ascending on the device before the deposit.
- The hyperparameters are `single-linkage-params`, whose `linkage` is
  `#:fixed` at `'single`. `single-linkage-fit` is a `define/who` with
  `#:hyperparameters single-linkage-params`, so it takes no `#:linkage` and
  refuses another `linkage` in `#:params` before the GPU is touched. The
  `agglomerative-clustering` estimator, an estimator and a clusterer, is a
  `define-estimator` over it with the setting `resources` and
  `#:no-sample-weight`; its constructor keeps `#:linkage`, and the fit
  raises the refusal under the generic's name.
- The spec's decisions, each with the recommended answer:
  - **Q-A.** Euclidean only; other metrics are refused by name (#22).
    cuVS 26.08 fails every other metric under pairwise and joins knn
    components by L2 whatever the metric.
  - **Q-B.** float64 is refused, not converted.
  - **Q-C.** Each child pair is sorted ascending, as in scipy's `Z`.
  - **Q-D.** cuML's label numbering is kept: the k cluster roots, by
    descending node id.
  - **Q-E.** A row with a NaN or an infinity, or a squared norm above
    2^125, is refused with one scan of `X`, so every expanded distance
    stays finite.
  - **Q-F.** Merge distances and sizes are deferred (#23).
  - **Q-G.** One row is accepted.
  - **Q-H.** The default connectivity is `'knn`, as in Python.
  - **Q-I.** The estimator is `agglomerative-clustering`;
    `single-linkage` stays reserved.
  - **Q-J.** The estimator has a `#:linkage` that accepts only `'single`;
    since #7 the check is `single-linkage-fit`'s, over the fixed key.
  - **Q-K.** cuVS's knn joining failure passes through, with a hint in the
    manual; there is no silent retry under pairwise.
  - **Q-L.** `c` takes any int32 up to 2147483617, as Python accepts any
    int.
- Pairwise is refused above 46340 rows, where cuVS's `int` count of n^2
  entries overflows, and knn is refused when its graph of 2nk entries
  passes 2^31-1. Precomputed distances are deferred (#24).

## Decisions recorded in leg 5 (spectral clustering, #10)

- `spectral-clustering-fit` and `spectral-clustering-fit-graph`
  (`cuml/cluster/private/spectral.rkt`) bind the dense and the
  rows/cols/vals overloads of `ML::SpectralClustering::fit_predict` for
  float32, through `cuml_cluster_spectral_clustering_fit_job` and
  `cuml_cluster_spectral_clustering_fit_graph_job`. Both return a
  `spectral-clustering-result` of the labels. The `coo_matrix_view`
  overload is not bound.
- The hyperparameters are `spectral-clustering-params`, whose `affinity` is
  `#:fixed` at `'nearest-neighbors`. `spectral-clustering-fit` is a
  `define/who` with `#:hyperparameters spectral-clustering-params`, so it
  takes no `#:affinity` and refuses another `affinity` in `#:params` before
  the GPU is touched; `spectral-clustering-fit-graph` adds
  `#:ignoring (n-neighbors affinity)`. The `spectral-clustering` estimator, an
  estimator and a clusterer, is a `define-estimator` over the dense fit with
  the setting `resources` and `#:no-sample-weight`; its constructor keeps
  `#:affinity`, and the fit raises the refusal under the generic's name.
- The spec's decisions, each with the recommended answer:
  - **Q-A.** The precomputed affinity is a separate procedure over three
    vectors and `n-samples`; a dense precomputed matrix is deferred (#28).
  - **Q-B.** The estimator's `#:affinity` accepts only
    `'nearest-neighbors` and refuses `'precomputed` by name (#28).
  - **Q-C.** Self-loops in the graph are refused, with one scan and no
    allocation: cuVS sizes its Laplacian by the diagonal entries, and
    duplicates corrupt it.
  - **Q-D.** Non-symmetric graphs are documented and pinned, not
    symmetrized (#28).
  - **Q-E.** Python's clamps are mirrored: `n-neighbors` to n−1, auto
    `n-components` to min(k, ⌊(n−1)/3⌋); an explicit `n-components` above
    ⌊(n−1)/3⌋ is refused.
  - **Q-F.** Fewer than 4 samples are refused; cuVS segfaults at 2 and 3.
  - **Q-G.** The default `eigen-tol` is `'auto` (0, as in Python), with the
    run-to-run drift on overlapping data documented.
  - **Q-H.** The result is a one-field struct, with room for the embedding
    (#27).
  - **Q-I.** Negative and non-finite weights are refused.
- The graph is refused when its entries and samples pass 2^31-1, and the
  dense path when its kNN graph's n(2k+1) entries would.

## Owner decisions (2026-10-17)

- The repository is `bkc39/rkt-cuml`. Packages start at 0.1 and require
  `("raft" #:version "0.2")`.
- **One native package** carries every family's shim. A package per family
  is deferred (#2).
- **No contracts** in the initial versions of cuml.
- `threading-lib` follows upstream's latest commit. The pin in
  `nix/racket-tools.nix` is upstream HEAD (`e79cfe55`).
- No GPU runner for now (#1). The GPU suite runs locally before each merge.
- No tailnet nix cache in CI; the GitHub Actions cache only.
- The dev shell deletes stale Racket homes aggressively (AGENTS.md).
- The estimator generics live in cuml-core-lib, the wrappers in cuml-lib;
  fit is functional; capability generics with a `fit-predict` fallback
  (layout #3).
- The device-memory test helper calls NVML and the CUDA driver from Racket
  directly.
- On a MIG-partitioned GPU the per-process memory tests SKIP, and the `gpu`
  gate allows that SKIP (`scripts/allowed-skips.txt`).
- The shims build with ASAN and UBSAN as the `shim-sanitizers` flake check.
- A binding census for the clustering family, `scripts/check-bindings.rkt`.

Follow-ups are GitHub issues.
