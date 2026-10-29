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

5. **Scope: clustering** (k-means, DBSCAN, single-linkage agglomerative,
   spectral, HDBSCAN) **and metrics** (`cuml/metrics`: contingency,
   adjusted Rand, silhouette, trustworthiness; leg A2, #34), **then the
   rest of epic #53**: `cuml/preprocessing`, `cuml/decomposition` and
   `cuml/manifold`, plus the table conversion in `cuml-lib`. Neither needs sparse arrays or multi-GPU.
   HDBSCAN's outputs sized by the call use raft's deposit records (rkt-raft
   #22).

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
  export; `define/who` (`cuml/core/private/define-who.rkt`) defines each
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

- `define-hyperparameters` (`cuml/core/private/hyperparameters.rkt`)
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
  refuses another `linkage` in `#:params` before `X` is read. The
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
  `X` is read; `spectral-clustering-fit-graph` adds
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

## Decisions recorded in leg 6 (native models, #11)

- A fitted model whose state lives in C++ is a `cumlrkt::model`
  (`shim/common/include/cumlrkt/model.hpp`): an anchor (one int32 holding
  its `model_kind`) plus the model's arrays, published as one raft deposit.
  Racket wraps it in a `native-model` (`cuml/core/private/model.rkt`).
  Leg 6 has no public surface and no fitted model; leg 7's HDBSCAN is the
  first, with the member layout the spec's §8 checked against cuML 26.08
  (`CondensedHierarchy` is move-constructible, not move-assignable).
- The spec's decisions, each with the recommended answer:
  - **Q-A.** The model is a raft deposit owner: raft releases it once, on
    its reaper, with the creating handle alive, in every drop order.
  - **Q-B.** Each call runs on the caller's current resources, under a
    native shared lock that release takes exclusively. `use_model` takes
    the call's stream and synchronises it before it unlocks, so a release
    never frees memory that the call's queued kernels still read.
  - **Q-C.** Release frees the opaque members only; region arrays live
    until collected.
  - **Q-D.** Everything is private this leg; it goes public in
    `cuml/core` with docs and examples in leg 7.
  - **Q-E.** The name is `native-model` / `with-native-models`, since
    "handle" already means `raft::handle_t`.
  - **Q-F.** The test object is a test-only `libcumlrkt_probe.so`
    (`CUML_SHIM_PROBE`), built by `cumlrkt_family(... TEST_ONLY)` into the
    shim's `tests` output and never staged.
  - **Q-G.** `call-with-release` is restated in
    `cuml/core/private/release.rkt`; an rkt-raft issue asks for
    `with-release` and an explicit deposit release in `raft/unsafe`.
  - **Q-H.** A concurrent second release returns at once.
  - **Q-I.** The registry is header-only, one per family library; a model
    from another family is refused as "not a model anchor".
- A failed release (for one, under released resources) leaves the model
  live, and a later release succeeds.
- A release whose thread stops running (killed or suspended) mid-release
  is taken over by the next release; a raise rolls the claim back to live; a
  break during a release lets the release finish and is then delivered.
- `define/who`, `define-hyperparameters` and `define-codes` live in
  `cuml/core/private/`, so cuml-core-lib's exported who-first procedures
  are `define/who` forms like every family's, and the lint rule has no
  path exemption.

## Decisions recorded in leg 7 (HDBSCAN, #12)

- HDBSCAN is the first native model: `hdbscan-fit` returns an
  `hdbscan-result` whose `model` holds the condensed tree, the prediction
  data, the core distances and the labels in C++ (`shim/cluster/src/
  hdbscan_model.hpp`, kind `hdbscan` in `cluster_kinds.hpp`), released by
  `cuml_cluster_model_release_job`. `cuml/core` exports the native-model
  surface with its manual page.
- The spec's decisions, each with the recommended answer:
  - **Q-A.** Labels are int64, as cuML returns them, unlike the other
    families' int32.
  - **Q-B.** Prediction data is always built.
  - **Q-C.** `hdbscan-membership-vector` and
    `hdbscan-all-points-membership-vectors` are bound now.
  - **Q-D.** The labels are a model region, so a foreign vector never reaches
    the predict kernel.
  - **Q-E.** The training X is kept by reference, checked against the model's
    shape at each prediction.
  - **Q-F.** The `hdbscan` estimator is a clusterer, not a predictor.
  - **Q-G.** Fit rows are refused above a squared norm of 2¹²⁵.
  - **Q-H.** Empty queries are refused.
  - **Q-I.** Linkage child pairs are sorted.
  - **Q-J.** Unlimited `max-cluster-size` is spelled 0.
  - **Q-K.** n = min-samples is accepted (all noise).
  - **Q-L.** leaf selection and `max-cluster-size` are checked against the
    Python twin only.
  - **Q-N.** `inverse_label_map` is exposed as
    `hdbscan-condensed-tree-selected-clusters`.
- `min-cluster-size` and `max-cluster-size` are refused above 2147483647,
  since cuML's params hold them as `int`.
- Persistence varies in its last bits between identical fits, so the
  determinism laws compare it within 1e-5 and every other output exactly.

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

## Owner decisions for epic #53 (2026-10-24)

- **Q-A.** Four new family packages after cuML's namespaces,
  `cuml-metrics-lib`, `cuml-preprocessing-lib`, `cuml-decomposition-lib`,
  `cuml-manifold-lib`, each with its own `libcumlrkt_<family>.so` in the one
  native package; no family depends on another or on `cuml-lib`.
- **Q-B.** The `datasets` table conversion lives in the umbrella `cuml-lib`;
  neither core nor a family depends on `datasets` or `datasets-core`
  (`check-layout.rkt`).
- **Q-C.** The embeddings chapter uses sklearn's digits from `datasets-uci`.
- **Q-D.** An algorithm that is not bit-deterministic under a fixed seed is
  checked by quality metrics plus label agreement, not exact arrays.

## Decisions recorded in leg A1 (#32)

- **A1-a.** The name is `table->device-matrix`; `table` is a `dataset-table?`.
- **A1-b.** The default dtype is `'float32`, raft's and the guide's.
- **A1-c.** `#:layout` is bound, defaulting to `'row-major`.
- **A1-d.** An exact number the dtype cannot hold exactly is refused; inexact
  values are rounded.
- **A1-e.** A missing value is refused; no `#:missing` fill this leg.
- **A1-f.** NaN and infinite flonums pass; the families decide.
- **A1-g.** raft's failures after the scan (released resources, out of
  memory) pass through as `exn:fail:raft` naming raft's own procedure:
  `list*->device-matrix`, or `device-matrix` for a table with no rows; an
  rkt-raft issue asks for a `#:who` on the list uploads.
- **A1-h.** Duplicate columns are refused.
- **A1-i.** A table with no rows gives a `(0 k)` matrix.
- **A1-j.** No one-column `device-vector` companion; a label encoder is a
  follow-up for A3 or A6.
- **A1-k.** A public module `cuml/table`, re-exported by `cuml`, loading no
  cuML family or cuML native library: a re-export facade with `all-from-out` and no lint
  suppression, as `cuml/core` is.
- **A1-l.** The tests build tables by reflection, in `table-data.rkt` only;
  bkc39/datasets is asked for a public constructor.
- **A1-m.** cuml-lib's runtime dependency on datasets-core, whose licence
  covers GPL-2.0-only and CC-BY-4.0 data files, is accepted: data in a
  separate package, not linked code; cuml stays Apache-2.0.
- **A1-n.** `dataset-matrix` and `matrix-of` retire now; `standardized`,
  `standardized-matrix`, `rows`, `rows->matrix` and `dataset-columns` stay
  until A6.
- **A1-o.** datasets-core is pinned through raft's `datasets-src`, one pin.
- **A1-p.** What is numeric is decided from the values, not the metadata's
  `type`.

## Decisions recorded in leg A2 (#34)

- **A2-a.** Each label vector is remapped on the device to dense int32
  codes in ascending order; int32 and int64 are allowed independently, in
  either argument.
- **A2-b.** `contingency-matrix` keeps sklearn's name and returns a
  `contingency-table`: the int64 counts and the sorted distinct row and
  column labels, each in its argument's dtype.
- **A2-c.** The counts come from raft's `contingencyMatrix` over the dense
  codes, into a K×K scratch whose k₁×k₂ block is copied out.
- **A2-d.** ARI and contingency refuse more than 46340 distinct labels in
  either argument, where raft's 32-bit index overflows; the 17 GB scratch at
  the bound is documented.
- **A2-e.** Two procedures, `silhouette-score` and `silhouette-samples`, on
  two shim entries.
- **A2-f.** The silhouette metrics are euclidean, sqeuclidean, manhattan,
  cosine and chebyshev, without Python's aliases; the rest of Python's table
  is a follow-up.
- **A2-g.** The tile keyword is `#:chunk-size`, like `#:batch-size` and
  `#:n-neighbors`.
- **A2-h.** Trustworthiness refuses float64. Silhouette and trustworthiness
  rows past a squared norm of 2¹²⁵ (float64 silhouette: 2¹⁰²¹), NaN and Inf
  among them, are refused.
- **A2-i.** `scripts/check-bindings.rkt` takes a family list,
  `'("cluster" "metrics")`, with one `<family>-bindings.rktd` each.
- **A2-j.** The family-neutral shim helpers (`support.hpp`, the index and
  norm bounds in `limits.hpp`, the row scans in `rows.hpp`/`rows.cu`) live in
  `shim/common`, and `deposit-for/who` in `cuml/core/private/deposit.rkt`;
  cluster is rewired with no behaviour change.
- **A2-k.** No hyperparameter tables and no estimators: the metrics are
  plain `define/who` procedures.

## Decisions recorded in leg A3 (#33)

- **A3-a.** `standard-scaler-result` holds mean, var, scale, n-samples,
  n-features, with-mean and with-std: sklearn keeps `mean_` under
  `with_mean=False`, and the column check needs n-features with both flags
  off.
- **A3-b.** The statistics come from raft's column-major
  `meanvar<double>` over a float64 column-major scratch, a chunk of columns
  at a time: the mean from one pass, the variance from a second over the
  centred scratch (one pass was 8.7e-7 off sklearn on a 1e-4 spread over
  1e6). Deterministic, and within about 1e-14 of sklearn. Raft's
  row-major `meanvar` is rejected: its last bits change from run to run and
  its float32 variances are up to 8.7e-4 off sklearn's (probes F14, F15).
  The owner accepted the two-pass variance in round 1.
- **A3-c.** The scratch is capped by the private parameter
  `current-scratch-elements`, default 2²⁶ (512 MiB); chunking changes no bit.
- **A3-d.** The statistics are float64 whatever X's dtype; the transforms
  round them to X's dtype. Integer and float16 X stay refused.
- **A3-e.** NaN and infinities are refused at fit, transform and inverse,
  with the location; sklearn ignores NaN at fit and passes it through at
  transform. The fit's entry says so.
- **A3-f.** A column gets scale 1 by sklearn's `_is_constant_feature` mask,
  var ≤ n·ε·var + (n·mean·ε)² in float64 (probes F7, F18), not only at an
  exact zero.
- **A3-g.** `gen:transformer` gains an optional `inverse-transform`, whose
  fallback raises `exn:fail:support`.
- **A3-h.** `define-estimator`'s `#:labels` is optional (no `gen:clusterer`
  without it), and it takes an `[inverse-transform proc]` entry.
- **A3-i.** `cumlrkt_family` takes `NO_LIBCUML`: `libcumlrkt_preprocessing.so`
  links `raftrkt::headers` only, and loading `cuml/preprocessing` maps no
  libcuml.
- **A3-j.** Sample weights are refused (`#:no-sample-weight`) pending a
  follow-up; raft has no weighted variance.
- **A3-k.** `standard-scaler-fit-transform` returns `(values result
  transformed)`; the generic `fit-transform` returns the matrix.
- **A3-l.** The guide's chapters scale with `standard-scaler-fit-transform`;
  `standardized` and `standardized-matrix` retire.
- **A3-m.** The census finds `preprocessing` from `cuml-preprocessing-lib`, as
  A2's derivation finds every family from its package.
- **A3-n.** L6 asserts both bounds (decided in round 1): with exact host
  moments and u = 2⁻²⁴ or 2⁻⁵³, |mean t − (x̄ − T(m))/T(s)| ≤ u checks the
  transform's arithmetic, |mean t| ≤ 4u·(|m|/s + 1) checks the mean statistic,
  and |sd t − 1| ≤ 4u. Fixed bounds were unreachable for |m|/s up to 1e5.
- **A3-o.** A scale that rounds to 0 in X's dtype gives non-finite output, as
  sklearn's does, and the inverse refuses it. The transform's entry says so.
- **A3-p.** The doc link check skips a struct field's name only inside that
  struct's definition box: each `boxed RBoxed` table skips the names of its
  own field declarations (`name :`), wherever the header renders them. Any
  unlinked use of the name outside that box, a bare identifier on a code-block
  line included, is still a finding.
