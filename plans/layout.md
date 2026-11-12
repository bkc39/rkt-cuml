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

## Decisions recorded in leg A4 (#47)

- **A4-a.** PCA is libcuml's `pcaFit`, `pcaTransform` and
  `pcaInverseTransform` on transposed column-major copies, with the shim's
  post-fit and scaling kernels; `libcumlrkt_decomposition.so` links libcuml.
  The copies also keep the caller's X intact, since the entries write to
  their input.
- **A4-b.** The results are in X's dtype; float32 is not widened.
- **A4-c.** cuML's column mean is not reproducible run to run, so there is
  no bitwise law for fits: transform and inverse of one result, and fits of
  exact-sum data, are bitwise; the rest is the twin's tolerances. The manual
  says two fits may differ in the last bits.
- **A4-d.** The shim applies sklearn's sign rule exactly: the first index of
  largest magnitude is made positive. cuML 26.08 already returns these signs
  on every input probed, ties included, so the kernel is a guard against an
  upstream change; a gtest pins it on hand-built components.
- **A4-e.** The shim fits min(n, d) components and reports sklearn's
  definitions: noise is the mean of λ[k:m], λ is clamped at 0, and the ratio
  is λ/Σλ[:m], 0 when the sum is 0.
- **A4-f.** `n-components` is in [1, min(n, d)]; `#f` means min(n, d).
- **A4-g.** Whitening is the shim's, never cuML's: the transform divides by
  max(√λ, ε) of the fit, the inverse multiplies by √λ.
- **A4-h.** Elements above √(max/(4·n·d)) are refused at fit, above
  max/(4·d) at transform, times min(1, minⱼ max(√λⱼ, ε)) when whitening, and
  above max/(4·k·max(1, √λmax)) (whitened) or max/(4·k) at inverse, before
  any allocation.
- **A4-i.** A matrix whose rows are all equal is refused.
- **A4-j.** One row or one column is refused.
- **A4-k.** The Jacobi defaults are `#:tol 0.0` and `#:iterations 15`; 0
  iterations are refused. The names stay `#:solver` and `#:iterations`; the
  twin maps `svd_solver` and `iterated_power`.
- **A4-l.** Transform and inverse refuse X in another dtype than the fit's.
- **A4-m.** `pcaFitTransform` is not bound: `pca-fit-transform` is fit,
  then transform, and returns `(values result matrix)`.
- **A4-n.** `pca-result` holds components, explained-variance,
  explained-variance-ratio, singular-values, mean, noise-variance,
  n-samples and whiten.
- **A4-o.** A3's finiteness scan moves into `shim/common` as
  `first_non_finite` and `require_finite`, beside the new element-magnitude
  and distinct-row scans; preprocessing calls the shared code.
- **A4-p.** `tol` crosses as `_double*`, so an exact real such as `0` or
  `-1` reaches the shim's check (the documented contract is `real?`).
- **A4-q.** Four laws were amended against measurements (disputes D1–D4):
  L4 allows 8·d·u for Jacobi (4·d·u full); Jacobi noise and ratio take the
  Jacobi λ bound 8·d·u·λ₁; the L14 transform bound adds ‖e_μ‖₂, the mean
  bound's norm, scaled as the whitened transform scales; and the component
  check skips components with λᵢ ≤ 64·u·λ₁, which span the null space.
- **A4-r.** X whose largest magnitude is below √(least normal)/ε (2⁻⁴⁰
  float32, 2⁻⁴⁵⁹ float64) is fitted as X·2ᵉ, its largest magnitude in
  [1, 2), in the transposed copy; components and ratio are kept, λ and noise
  scaled by 2⁻²ᵉ, σ and the mean by 2⁻ᵉ, so XᵀX no longer underflows
  (amended after A5 r0 finding 1). Other X has e = 0 and bitwise the same
  fit. `scale_exponent` in `shim/common` is shared with truncated SVD.

## Decisions recorded in leg A5 (#48)

- **A5-a.** Truncated SVD is libcuml's `tsvdFitTransform`, `tsvdTransform`
  and `tsvdInverseTransform` on transposed column-major copies, with the
  shim's post-fit kernel; transform and inverse of one result are bitwise
  equal to cuML-Python's, and `tsvd-fit-transform`'s matrix is bitwise
  `tsvd-transform` of the same X.
- **A5-b.** The signs are A4-d's rule, from the shared kernel in
  `shim/decomposition/src/signs.cuh`; the fit-transform matrix negates the
  columns the rule flips.
- **A5-c.** `#:n-components` defaults to 1, cuML's.
- **A5-d.** `#:tol` defaults to 0.0 and `#:iterations` to 15, as in A4-k;
  cuML-Python's tol of 1e-7 is the one parameter the twin maps differently.
- **A5-e.** `n-components` is in [1, min(n, d)], with A4's message form.
- **A5-f.** NaN and negative-zero singular values are reported as +0.
- **A5-g.** The explained-variance ratio is +0 when the total variance is
  0; a matrix whose rows all equal the first reports +0 variances and
  ratios, since raft's mean leaves rounding noise there.
- **A5-h.** A squared Frobenius norm above 2¹²⁵ (float32) or 2¹⁰²¹
  (float64) is refused before any allocation.
- **A5-i.** Transform and inverse refuse X in another dtype than the fit's,
  as A4-l.
- **A5-j.** Rows are at most 2³¹−1 and columns at most 46340, so the d×d
  Gram matrix fits the solver's `int`, with A4's messages.
- **A5-k.** The explained variance is cuML's and sklearn's: the population
  variance of each column of the transform, the ratio over the sum of the
  columns' population variances of X.
- **A5-l.** One row or one column is refused, as A4-j; a constant matrix is
  accepted (unlike A4-i), since truncated SVD does not centre.
- **A5-m.** `tol` crosses as `_double*`, as A4-p.
- **A5-n.** X whose largest magnitude is below √(least normal)/ε is fitted
  as X·2ᵉ in the transposed copy, A4-r's `scale_exponent`; σ is scaled back
  by 2⁻ᵉ and the explained variance by 2⁻²ᵉ, components and ratio kept
  (amended after A5 r0 finding 1). The fit-transform matrix is then
  `tsvdTransform` of the unscaled X, so it keeps A5-a's bitwise equality
  with `tsvd-transform` (amended after A5 r1 finding 1). Other X has e = 0
  and bitwise the same fit, so L18 is unchanged.
- **A5-o.** Transform and inverse refuse an X element of magnitude above
  max/(4·d) and max/(4·k), A4's bound without the whitening factor
  (amended after A5 r0 finding 2).

## Decisions recorded in leg A6 (#49)

- **A6-a.** The Decomposition chapter runs on wine, as #49 says; digits waits
  for bkc39/datasets#10 and the embeddings chapter (B4).
- **A6-b.** `crosstab` and `agreement` are retired from the helpers and from
  all five chapters that printed them, with `guide-docs-test` and the
  `check-doc-links` self-test.
- **A6-c.** A table is `counts-of` (a guide helper over `contingency-matrix`)
  of the class ids against the device labels: rows the classes in order of
  first appearance, columns the clusters in the run's numbering. The tests pin
  its columns as a sorted multiset (`cluster-columns`), so a renumbering moves
  no pin.
- **A6-d.** A score is `(rounded (adjusted-rand-score ids labels))`, a string
  from the guide helper `rounded`; each is pinned against the exact rational
  ARI of its pinned table (`table-ari`). `class-ids`, `counts-of`, `rounded`,
  `leading`, `cumulative`, `components-for`, `columns-of` and `spectrum` live
  in `guide/helpers.rkt` once, and the tests import them (amended after A6 r0
  style items 1, 2, 4, 5 and 13, and A6 final style item 2).
- **A6-e.** No reconstruction section until `mean-squared-error` is bound.
- **A6-f.** float32, the guide's default. The k-means labels on the two
  components hold by a margin: every sample's nearest and second-nearest
  centroids differ by at least 0.071 (whitened 0.126), the bounds
  `guide-decomposition-test` checks.
- **A6-g.** The decomposition values are oracle-checked by an sklearn twin
  (`guide-twin-test`, `python/guide_twin.py`) on the device's float32 rows.
  The ratios sit at least 10 parity bounds (A4 L14's 64·u·λ₁ over Σλ) from a
  rounding boundary at 3 places; the singular values only 8.4 and 3.3
  bounds (σ's bound derived from L14's λ bound), so the twin asserts one
  bound for them, plus G4's 64·u·σ₁ for truncated SVD's.
- **A6-h.** Whitening is in the k-means comparison. Two components cluster
  about as well as the 13 columns (ARI 0.895 against 0.897) and whitened
  better (0.912), as in sklearn's example (0.896 / 0.912). The two clusterings
  are not the same partition (ARI 0.946 between them); no prose claims they
  are. Corrected after A6 r0 finding 1: the earlier "0.912 against 0.864, the
  same clusters" held only for seed 0 with one init.
- **A6-i.** `ex-decomposition` is the last Examples chapter, after HDBSCAN,
  so wine is introduced in `ex-kmeans` first.
- **A6-j.** Every k-means fit in the Decomposition chapter, the figure's
  included, takes `#:n-init 10`, and the tests evaluate the chapter's own
  examples, in order, in a sandbox, and pin the printed scores (amended after
  A6 r2 finding 1). Over seeds 0 to 99 the ARI ordering (whitened at least
  both others) holds for 99 seeds and fails at seed 73; in misplaced wines it
  holds for all 100: whitened misplaces no more wines than either other
  clustering, and two components are within one wine of the 13 columns.
  `guide-decomposition-test` checks the wine counts over seeds 0 to 9 and
  73. The other chapters' k-means comparisons hold for every k-means seed 0
  to 9 with one init and stay as they are.

## Decisions recorded in leg B1 (#50)

- **B1-a.** `define-estimator` gains `#:supervised`: the generic `fit`
  passes `y` as `#:y`. Existing estimators expand unchanged.
- **B1-b.** `fit-transform` and `umap-fit-transform` return the fit's own
  embedding (`#:fit-transform umap-result-embedding`), `eq?` to the
  result's, not fit then transform.
- **B1-c.** `random-state` defaults to 0; `#f` is Python's None, a drawn
  seed.
- **B1-d.** `#:a` and `#:b` are exposed; giving only one is refused.
- **B1-e.** a and b are fitted in Racket (`curve.rkt`, Levenberg–Marquardt
  in normalised coordinates), within 2.65e-6 relative of scipy's
  `curve_fit` on the 47 probed pairs; the header takes them, and
  learning-rate, as `double` by value (`_double*`).
- **B1-f.** n-neighbors above the row count is clamped to it, as Python
  does.
- **B1-g.** Transform refuses an n-epochs below 3 (P5: upstream reads an
  empty edge list).
- **B1-h.** nn-descent below 150 rows is refused (P7: non-finite at 10, 30
  and 60 rows).
- **B1-i.** A non-finite scan after fit and transform refuses a diverged
  optimisation and frees the output; the message prints the value (inf or
  nan, P10).
- **B1-j.** Metrics are euclidean, sqeuclidean, manhattan, chebyshev and
  cosine; nn-descent refuses manhattan and chebyshev.
- **B1-k.** X is float32 only.
- **B1-l.** y is int32 or int64, −1 for unlabelled, sent as dense float32
  codes, so labels past 2²⁴ stay distinct (P12).
- **B1-m.** No umap-learn twin in this leg: the oracles are Python cuML
  (bitwise, brute force), the measured floors and scipy for the curve.
- **B1-n.** `graph_` is not exposed.
- **B1-o.** The §8 upstream parameters are fixed, not keywords.
- **B1-p.** The logger level is `warn`.
- **B1-q.** A2's dense-label kernel and the normless-row scan move to
  `shim/common`; metrics and manifold call the shared code.
- **B1-r.** Brute-force fits and transforms are bitwise repeatable in and
  across processes for both inits (P1, branch D); nn-descent is not (P3)
  and is documented so. Transforms of rows whose neighbour distances tie
  (digits) are not, here or in Python cuML; documented, and L6 checks the
  digits transform on jittered rows (amended after B1 r0 finding 4).
- **B1-s.** The f22 magnitude refusal uses the common double-sum scan, so
  1e19 rows are refused although Python fits them (P8): a conservative
  departure.
- **B1-t.** The facade re-exports the private module minus the `/who`
  forms (A4's pattern), not an `only-in` list.
- **B1-u.** Given a and b are refused below 2⁻¹²⁶ on the value passed
  (R5), so a subnormal or underflowing value never reaches the shim's f11
  (B1 r0 finding 6).
- **B1-v.** A learning-rate whose float32 rounding is below 2⁻¹²⁶ is
  refused at fit and transform (f9), where cuML would run with a zero step
  (B1 r0 finding 7).
- **B1-w.** Transform scans the result's training X as fit scans X (finite,
  magnitude, cosine zero rows) before scanning the new X, at O(nd), since
  `umap-result` is exported and the X is kept by reference (B1 r0
  finding 8).
- **B1-x.** Spectral init is refused unless n ≥ 2(n-components + 2) (f26,
  fit only; transform never runs the spectral solve). cuVS's spectral
  embedding sizes its Lanczos basis at ncv = min(n − nev, max(2·nev + 1, 20))
  with nev = n-components + 1, and raft's `lanczos_smallest` writes row
  nev + 1 of that basis (`lanczos.cuh:678-679`), so it needs ncv ≥ nev + 2.
  Python cuML 26.08, 10-column blobs, with exit status: n = 2·nev
  (6/2, 8/3, 40/19, 98/48, 400/199, 798/398, 800/399) segfaults plain (139)
  and under memcheck either reports 86 to 144 errors (6/2, 8/3, 40/19,
  98/48, 798/398) or dies with SIGSEGV (exit 11) after printing "ERROR
  SUMMARY: 0 errors" (400/199, 800/399); n = 2·nev + 1 (7/2, 9/3, 41/19,
  99/48, 401/199, 799/398) segfaults plain (139) and dies with SIGSEGV under
  memcheck (exit 11) after printing "0 errors"; n = 2·nev + 2 and
  2·nev + 3 (8/2, 9/2, 10/3, 11/3, 42/19, 43/19, 100/48, 101/48, 402/199,
  800/398) exit 0 with 0 errors. Supervised fits match: 7/2, 9/3, 99/48
  crash, 8/2, 10/3, 100/48 are clean. Under a pooled allocator the stray
  accesses land in neighbouring memory, which is where B1 r0's
  non-repeatable 600-component transforms and the later
  `cudaErrorIllegalAddress` came from. Refused rather than documented: the
  region is memory-unsafe or crashes the process (amended after B1 r0
  fault; the r0 bound n > 2(n-components + 1) let n = 2·nev + 1 through,
  corrected after B1 r1 finding 1). Spectral init also refuses an n·ncv
  basis past 2³¹ − 1 elements through the shared `require_basis` (f27,
  B1 r1 finding 2).

## Decisions recorded in leg B2 (#51)

- **B2-a.** `#:method` takes `'fft` and `'exact` only; Barnes-Hut is not
  exposed (it spins forever once coordinates go non-finite).
- **B2-b.** `min_grad_norm` is 0 for `'fft` and 1e-7 for `'exact`, with no
  keyword. cuML's FFT early stop reads mean |x-step|/2 from iteration 0
  and collapses the embedding on 12000 rows at learning rate 200; L8 pins
  Python's collapse beside our full run.
- **B2-c.** learning-rate defaults to 200 with cuML's "none" rule (r
  before the exaggeration ends, 2r after); `'auto` is not offered.
- **B2-d.** `#:n-neighbors` defaults to `#f`, meaning
  min(1023, max(2, ⌈3·perplexity⌉)), clamped to the row count; above
  1023 is refused (f6).
- **B2-e.** perplexity is refused at or below 1 in float32 (f7, whose
  message says "in float32") and above min(n-neighbors, n) (f17).
- **B2-f.** `define-estimator` gains a `[fit-transform proc]` entry: the
  estimator is `transformer?`, its `fit-transform` is `proc` of the fit's
  model, and its `transform` raises "transform: tsne has no transform; use
  fit-transform". It cannot be combined with `[transform ...]` (a syntax
  error). Existing estimators expand unchanged.
- **B2-g.** distances are squared (`square_distances` true), as Python's
  default.
- **B2-h.** n-components is kept and accepts only 2 (f4).
- **B2-i.** the keyword is `#:max-iter`; `n-iter` is the result field.
- **B2-j.** `tsne-fit` returns a `tsne-result` (embedding, kl-divergence,
  n-iter, params). `tsne-fit-transform` returns `(values result
  embedding)`, as every other `X-fit-transform` (A3-k, B1-b), not the
  matrix alone as the spec proposed.
- **B2-k.** a kNN of all-zero distances without every row equal is caught
  by f23 after the fit.
- **B2-l.** a `#f` random-state draws the seed in Racket, so FFT is always
  reproducible under `(random-seed k)`; P4 measured the deterministic FFT
  path at 1.3–1.9× the time of `random_state=None`.
- **B2-m.** f23's message has B1's shape, with the advice of B2-x.
- **B2-n.** KL divergence and the iteration count come back as host
  out-parameters of the job (PCA's shape), not deposit regions. cuML
  computes KL at iteration max_iter − 1 before its break check and returns
  the last iteration's index, so n-iter = min(iter + 1, max-iter) and
  kl-divergence is `#f` unless n-iter = max-iter.
- **B2-o.** `to_columns` and `to_rows` moved to `shim/common`
  (`transpose.{hpp,cu}`, `device.cuh`), UMAP's scans to
  `manifold_support.{hpp,cpp}`; PCA, tSVD and UMAP results are bitwise
  unchanged. `bound-params`, `drawn-seed`, `finite-real?` and
  `metric-code` moved to `cuml/manifold/private/support.rkt`.
- **B2-p.** random-state is checked in the shim ("random-state: expected
  at least 0, got -1"); the job takes it as `_int64`.
- **B2-q.** L3 runs the small sweep at perplexity min(30, n), since f17
  refuses the default perplexity below 30 rows.
- **B2-r.** The manual's example runs at perplexity 5: on its 60 rows FFT
  at perplexity 10 ends at KL 5.41 and ARI 0.09 (2 and 3 are as bad,
  20 gives 0.75 with min-cluster-size 5), where exact gives 1.0 at every
  perplexity from 2 to 20.
- **B2-s.** `tsne-sklearn-digits-twin-test` takes 55 s (sklearn's exact
  t-SNE on digits, CPU-bound) and has a 180 s entry in `info.rkt`.
- **B2-t.** Under `'sqeuclidean` a row's squared norm is at most 2⁶¹
  (f20b). cuML squares the kNN distances, a
  fourth power under sqeuclidean; opposite rows give a squared distance
  of 4·|a|², at most 2⁶³ here, so its square stays at or below 2¹²⁶,
  a factor of 4 under float32's maximum, whatever the rounding of
  |a|²+|b|²−2a·b in float32. Two-column rows inside 2⁶²−2³⁸ whose float32
  norm rounds to 2⁶² diverge under both methods: (1807712000, 1159251072)
  and 7 other directions. Measured at 2⁶¹ on
  `tiny`-shaped opposite rows, both methods, all finite: those 8
  directions rescaled to just inside 2⁶¹ (16 fits), and 24 random two-
  and 24 random three-column directions at the limit, 37 of them with a
  float32 norm that rounds to 2⁶¹ (96 fits).
- **B2-u.** L6's bounds are measured, not Python's seed spread or
  upstream's digits-derived 0.05. Exact: 40 fits
  per fixture, trustworthiness at k = 10, floor = min − (max − min):
  blobs 0.97176–0.97447 (floor 0.969), overlap 0.86713–0.88865 (0.845),
  digits 0.99048–0.99208 (0.9888). Against sklearn (blobs 0.97687,
  overlap 0.91393, digits 0.99237), bound = largest |ours − sklearn| +
  (max − min): blobs 0.00512 → 0.008, overlap 0.04680 → 0.07, digits
  0.00189 → 0.004. FFT seed 0 is bitwise Python's (L5), so a check
  against Python's spread cannot fail; it gets an absolute floor, Python's
  20-seed min − δ: blobs 0.96900 − 0.00369 → 0.965, overlap
  0.84700 − 0.02085 → 0.826, digits 0.98827 − 0.00238 → 0.9858 (ours
  0.97011, 0.85080, 0.98857). The blobs and overlap quality tests do not
  call Python, and their names carry no `twin`.
- **B2-v.** Under `'pca` init the rows' squared distances from their mean,
  summed in double, are at most 2¹²⁶ (f20d). cuML rescales the PCA
  projection by its standard deviation, whose float32 sum of y² over all
  rows is bounded by that total; past float32's maximum it is inf and Y/inf
  is 0, so exact returns an all-zero embedding without error and FFT
  diverges, at any learning rate. The failure tracks the total, not the
  row count: 16 rows at 2.56·10³⁸ and 40 rows at 1.6·10³⁸ fit, 16 rows at
  4·10³⁸, 40 at 3.6·10³⁸ and 4000 at about 4·10³⁸ do not. 2¹²⁶ leaves a
  factor of 4, as f20b does. After the fit an embedding whose rows all
  equal row 0 is refused ("the embedding is constant, use init 'random");
  with f20d in place no input reaches it, so the gtest covers
  `require_unlike_first` and not this call.
- **B2-w.** Under `'manhattan` a row's L1 norm, summed in double, is at
  most 2^62.5 (f20c). cuML squares the kNN distances, so opposite rows give
  a squared L1 distance of at most 4·2¹²⁵ = 2¹²⁷. f20 does not cover it:
  (3.5·10¹⁸)×3 has a squared norm inside 2¹²⁵ and a squared L1 distance
  of 4.41·10³⁸. The bound keeps a factor of 2, so (3·10¹⁸)×3, which cuML
  fits, is refused; rows ±v×3 at the largest float32 v with 3v ≤ 2^62.5
  fit under both methods. Under `'euclidean` and `'chebyshev` the squared distance
  is at most 4·2¹²⁵ under f20 already, so they need no bound of their
  own; f20b and f20c are the two metrics where it is larger.
- **B2-x.** f23's advice is "lower learning-rate, centre or rescale X, or
  raise n-neighbors". cuML multiplies the squared kNN distances by the
  reciprocal of their largest. That is inf in float32 once the largest
  neighbour distance under the metric is below 2⁻⁶⁴ (about 5·10⁻²⁰), and
  0·inf is NaN once every computed distance is 0. Three inputs reach it:
  rows whose largest neighbour distance is below 2⁻⁶⁴; rows far from the
  origin, whose ‖x‖²+‖y‖²−2x·y cancels to 0 in float32 under `'euclidean`
  and `'sqeuclidean` (a 1.7·10⁹ time column, which fits centred); and
  duplicate rows that fill every row's neighbourhood (40×(0,0) and
  40×(1,1) at perplexity 5, which fit at n-neighbors 60; at 1023 or more
  copies of every row no n-neighbors fits, so X is deduplicated). No pre-fit check
  catches them. The shim does not centre X: that would break bitwise parity
  with Python cuML.

## Decisions recorded in leg B3 (#27)

- **B3-a.** Spectral embedding lives in `cuml/manifold` over
  `ML::SpectralEmbedding::transform`; `spectral-clustering-result` stays one
  field.
- **B3-b.** The procedures are `spectral-embedding-fit`, `-fit-graph`,
  `-fit-transform`, `-params` and the `spectral-embedding-result` struct;
  the estimator is `spectral-embedding`, whose `fit-transform` is the
  one-call form and whose `transform` raises.
- **B3-c.** Dense X takes a fixed `'nearest-neighbors` affinity (s0 refuses
  any other in `#:params`); a COO graph goes through `-fit-graph`. No
  `'rbf`, dense precomputed or precomputed-nearest-neighbors input (#28).
- **B3-d.** `#:n-neighbors #f` means min(256, max(2, ⌊n/10⌋)); the result
  records the count used and the params keep `#f`. Counts above 256 are
  accepted and documented as not repeatable.
- **B3-e.** The shim applies sklearn's sign rule: each column's first entry
  of largest magnitude is positive (`make_largest_positive`). The Python twin
  applies the same rule to cuML's output.
- **B3-f.** `#:norm-laplacian` and `#:drop-first` are keywords, both `#t`
  by default, shared with the estimator.
- **B3-g.** n-components defaults to 2.
- **B3-h.** X whose largest magnitude is below 2⁻⁴⁰ is embedded as an exact
  power-of-two-scaled copy, so the embedding is bitwise scale-invariant
  (L7).
- **B3-i.** Squared row norms past 2¹²⁵ are refused (s11).
- **B3-j.** After the solve, a non-finite entry (s13, gtest only) and an
  all-zero column (s14) are refused with the outputs freed.
- **B3-k.** Self-loops in a given graph are refused (g5), as leg 5 does.
- **B3-l.** A graph that is not symmetric is used as given and documented. A
  graph fit refused for an inaccurate or NaN eigenvector names the asymmetry
  in place of B3-v's advice; a zero column keeps B3-v's advice; cuML's failure
  to converge passes through as cuML raises it.
- **B3-m.** Disconnected graphs are not refused (L11). A graph whose
  Laplacian repeats eigenvalues (rings, grids, complete graphs, stars,
  identical pieces) can give a wrong embedding or a zero column, with either
  Laplacian, on both paths. The graph path refuses what B3-y detects; the
  dense path cannot check, and L11 pins its departure on two identical
  polygons. Neither the refusals nor the manual offer a remedy.
- **B3-n.** `#:random-state #f` draws its seed in Racket.
- **B3-o.** n-neighbors above the row count is refused (s9).
- **B3-p.** X needs at least 2 columns (s5).
- **B3-q.** A given graph keeps leg 5's bound: entries plus samples at most
  2³¹−1 (g3).
- **B3-r.** Leg 5's graph checks live in `shim/common/graph.{hpp,cu}`, the
  2(nev + 1) Lanczos bound in `require_lanczos_rows`, and the sign rule on a
  row-major matrix in `shim/common/largest_positive.{hpp,cu}`. PCA's sign
  kernel, which writes a transposed output, lives in
  `shim/decomposition/src/signs.cuh`.
- **B3-s.** Spectral clustering's result carries labels only; its
  embedding comes from `spectral-embedding-fit-graph`.
- **B3-t.** X is not centred; the manual says to centre X far from the
  origin.
- **B3-u.** The eigensolver's tolerance (1e-5) is not exposed. Without
  norm-laplacian, weights spanning orders of magnitude can give a wrong
  embedding or a zero column that B3-y does not see; the manual says to use
  norm-laplacian for such graphs, and the refusal advises it (B3-v).
- **B3-v.** Racket resolves n-neighbors and passes the count to the shim,
  which returns only the embedding. A refusal after the solve names the likely
  cause: on the dense path, duplicated or uncentred X or n-neighbors; on the
  graph path, graphs whose Laplacian repeats eigenvalues; plus samples without
  edges, with norm-laplacian when there are at least as many as eigenvectors
  asked for and without it when there is any; plus norm-laplacian when,
  without it, the summed weights span 2⁸ or more; for an inaccurate or NaN
  eigenvector of a graph that is not symmetric, only that.
- **B3-w.** B3's tests split per seam: L1-L3 and L7-L12 in `-law-test`, L4
  in `-determinism-test`, L5-L7's twins in `-twin-test`, and L13 in
  `-refusal-test` (dense X), `-graph-refusal-test` (the graph's shape) and
  `-eigen-refusal-test` (the eigensolver's refusals).
- **B3-x.** Without norm-laplacian, a given graph is embedded as an exact
  power-of-two-scaled copy whose largest summed weight is in [1, 2), so the
  embedding is bitwise invariant to the scale of the weights; with it, the
  embedding scales as 1/√s on its own.
- **B3-y.** After a graph fit, the shim refuses a column whose eigen-residual,
  over the largest degree (the normalised Laplacian's with norm-laplacian), is
  above 10⁻³ or NaN, and with drop-first a constant column. A fit displaced by
  samples without edges can be returned without an error.
- **B3-z.** Duplicate entries of a given graph add: the shim sums them on
  the device in double before the solve and rounds once, so a graph split into
  duplicates embeds bitwise as its sum. Python cuML sums them in float32 in no
  fixed order, so three or more duplicates of one entry can differ by an ulp.
  Each entry, before adding, is 0 or at least the smallest normal float32.

## Decisions recorded in leg B4 (#52)

- **B4-a.** The guide's palette has 10 label ids, 0 to 9: tab10's brown
  `#8c564b` and cyan `#17becf` join as ids 8 and 9, with the markers
  `triangleleft` and `6star`.
- **B4-b.** HDBSCAN in the Embeddings chapter takes `#:min-cluster-size 60
  #:min-samples 10`.
- **B4-c.** UMAP and t-SNE run at their defaults.
- **B4-d.** Spectral embedding is not in the chapter.
- **B4-e.** The trustworthiness line prints PCA's two components beside
  UMAP and t-SNE.
- **B4-f.** cuml-doc and cuml-test depend on the `datasets` umbrella, one
  dependency for the guide's data (`data_only` in `pkg-deps.sh`), with
  `datasets-core` at 0.2; digits comes from its UCI part. The dev shell's
  home is keyed on the datasets pin as well as rkt-raft's and the tools'.
- **B4-g.** Every printed value is pinned exactly. UMAP's, t-SNE's and
  HDBSCAN's values are checked against Python cuML's twin, and PCA's
  trustworthiness over 10 fits; seeds 0 to 9 are floors for the prose's
  comparisons.
- **B4-h.** `ex-embeddings` is the last Examples chapter, so digits is
  introduced last.
- **B4-i.** t-SNE's figure has an HDBSCAN panel beside the digit panel.
- **B4-j.** Repeatability is checked in one process; across processes it is
  B1 L6's and B2 L4's on digits.
- **B4-k.** `scored` is a guide helper in `guide/helpers.rkt`.

## Decisions recorded in leg L0 (#86)

- **L0-a.** `r2-score`, `mean-squared-error` and `accuracy-score` are the
  shim's own reductions (`shim/metrics/src/scores.cpp`, `scores_device.cu`):
  one deterministic thrust/CUB tree per sum, accumulated in double, with
  power-of-two prescaling. cuML's `r2_score_py` and `accuracy_score_py` are
  not called.
- **L0-b.** On a `y-true` that is constant over the samples of positive
  weight (exact equality with the first), r2 is 1.0 when `y-pred` equals it
  there and 0.0 otherwise, n = 1 included.
- **L0-c.** `#:sample-weight` on all three; matrix input to
  `mean-squared-error` with the uniform average over columns only.
  `multioutput`, `force_finite`, `normalize` and `squared` are not bound.
- **L0-d.** Mixed float32/float64 arguments are accepted, as the labels
  accept int32 with int64.
- **L0-e.** Negative weights are refused.
- **L0-f.** No magnitude refusal. Every scaling is by a power of two, so r2
  is exactly scale-invariant and the error is `+inf.0` only past the flonum
  range. Weights are divided by 2^e_w, e_w = ilogb(max w); a sample is
  positive when its scaled weight is (a weight at most 2^(e_w − 1075)
  counts as zero), and a weight w below 2^(e_w − 1022) is held as a
  subnormal multiple of 2^(e_w − 1074), a relative error of up to
  2^(e_w − 1075)/w in that weight. Over the positive samples only, with
  e = ilogb(max |y|, |ŷ|): r2 divides the values by 2^e, so that its
  weighted mean ȳ cannot overflow; MSE divides them by 2^e_s,
  e_s = max(0, e − 1021), only as far as the difference needs, so every
  residual of values below 2^1022 is the correctly rounded y − ŷ. With s
  that value scaling, each residual s·y − s·ŷ is divided again by 2^e_r,
  e_r = ilogb(max |s·y − s·ŷ|), and for r2 each deviation s·y − ȳ by
  2^e_d, e_d = ilogb(max |s·y − ȳ|), before squaring; both maxima are over
  the positive samples, whatever their weights. The final ratio divides
  the frexp mantissas and applies one ldexp, so it is rounded once, as a
  normal number, whatever the exponents: with R′ = m_R·2^a,
  D′ = m_D·2^b and d·W′ = m_W·2^c, MSE = ldexp(m_R/m_W,
  a − c + 2(e_s + e_r)) and r2 = 1 − ldexp(m_R/m_D, a − b + 2(e_r − e_d)).
  All-zero residuals give R′ = 0, so exactly 1.0 and 0.0. D′ can be 0
  while y-true varies, when the 2^e scaling flushes every y-true to zero
  (y-true 2^-60 and 2^-59 beside ŷ 2^1023); r2 is then −inf.0, which is
  the rounded score.
  Zero-weight samples never enter a maximum or a sum, so a large value
  behind a zero weight changes nothing. In r2 a residual more than 2^1022
  below the largest |y|, |ŷ| is rounded on the subnormal grid of the 2^e
  scaling; MSE keeps its bits. A light weight costs precision in two
  places. Its own term w′·r′² carries the weight's relative error. And
  since e_r (or e_d) ignores the weights, when a light sample holds the
  largest residual (or deviation), every other sample's scaled residual
  r′ is below 1, and the term w′·r′² of any sample whose r′² falls below
  2^-1022 (a residual more than 2^511 below the largest) is rounded on the
  subnormal grid, an absolute error of up to 2^-1075 per term, normal
  weights included. Those terms carry the score when the light sample's
  own term is small, so it is accurate only to about n·2^-1075 against
  their sum.
- **L0-g.** The Decomposition chapter gains its Reconstruction section.

## Decisions recorded in leg L1 (#87)

- **L1-a.** The family is package `cuml-linear-model-lib`, collection
  `cuml/linear-model` and native name `linear_model`: one `-`→`_` rule in
  `install-native.rkt`, `native.rkt`, `scripts/check-bindings.rkt`,
  `flake.nix` and the CMake family name.
- **L1-c.** Every column, y and the weights are scaled by powers of two before
  olsFit and the results scaled back (`shim/linear_model/src/glm_staging.cpp`,
  kernels in `glm_staging_device.cu`). Zero-weight rows are left out of the
  staged copy (when at least two rows have positive weight), on the solve
  and on the nothing-to-solve path, so a zero weight is a deleted row
  bitwise. The weights are staged first, divided by 4^k,
  the least power of 4 at or above 2^ilogb(max w); a row counts as
  positive when its staged weight is: a weight at most 2^−150·4^k (float32)
  or 2^−1075·4^k (float64) counts as zero. In float64 this is L0-f's
  cutoff; in float32 it is not (r2-score stages float32 weights in double),
  so the estimator's `score` can count a row its `fit` dropped, as the
  Scribble says. The positive-row count
  m and the wide rule, compaction, the constancy and nothing-to-solve checks,
  the exponents and the residual check all read the staged weights.
- **L1-c, exponents.** Column j's power of two is −ilogb of its weighted
  spread over the positive rows: √Σw(x−x̄)² with fit-intercept, √Σw·x²
  without. It is computed in double on the column divided by the power of
  two of its largest |x|, falls back to that power when the sum underflows,
  and is capped so that n times the staged max |x| times the staged max
  weight (under 2) stays finite (review r3 L1). The mean's Σw·x is summed in
  the pass that finds the largest |x|, carried as a significand and a power
  of two, so it needs no largest first. The spread, and the mean and
  weight total that feed it, are summed over the compacted positive rows one
  column at a time, in a tree fixed by the row count alone, so a zero weight
  is bitwise a deleted row (T6) and no other column, nor a column's position,
  changes them (E4); round 2's one reduce_by_key over all n·d entries let
  either move a spread within an ulp of a power of two across it (review r3
  M1). Every such sum, the target's and the residual check's included, goes
  through one chunked-fold kernel (`chunk_totals` in `glm_device.cuh`):
  2048-row chunks, each folded in a fixed tree over a 256-thread block with
  a column count the caller fixes, never chosen from d, then the chunk totals
  folded in a fixed tree; a single vector is the one-column case (review
  final C-1). y takes the
  power of its largest |y|. Rounds 0 to 2 took every column's from its
  largest |x|: a column with a large mean against its spread, or one whose
  variation sits on low-weight rows, then had a centred, weighted Gram
  eigenvalue under raft's absolute 1e-10 eps, and eig dropped its direction
  (review r2 H1, H2). One power of two per column, so E1 holds on non-wide
  fits: scaling a column by 2^k scales its coefficient by 2^−k exactly.
- **L1-c, against other libraries.** Python cuML does not scale, so a float32
  column at 1e-5, a float32 y at 1e37 and a constant column under qr give it
  wrong or NaN coefficients; scikit-learn returns the minimum-norm solution
  for collinear columns, where this fit can refuse (L1-e). With weights and
  an intercept, fits above a few thousand rows can differ in the last bits
  between runs, as in Python cuML. These were in the Scribble entry until
  the final style read (R10) cut it to what the fit does and refuses; the
  entry keeps the run-to-run disclosure.
- **L1-c, wide rule (V1).** With m positive-weight rows, a fit is wide when the
  kept columns d' exceed m − 1 with an intercept, or m without one. A wide fit
  stages every kept column with one shared exponent (the least of the kept
  columns' exponents, that of the widest spread) and solves with the Jacobi SVD whatever the algorithm, so the staged
  problem is a uniform rescaling of the original and the SVD's minimum-norm
  solution is scikit-learn's. In float32 with an intercept the SVD does not
  reach the minimum-norm solution on every wide X: it interpolates y but can
  differ from it by order 1 (spec F6); float64 reaches it to rounding.
  Bitwise parity with raw Python cuML holds only on already-staged data (E8);
  the staged recipe is bitwise (P1).
- **L1-d.** Columns constant over the positive-weight rows (zero, without an
  intercept) are dropped and get coefficient 0; with nothing left to solve, or
  a constant y, olsFit is not called.
- **L1-e.** A fit whose residual cosine on the staged data (the centred,
  weighted cosine between each kept staged column and the residual
  r = (y − ȳ) − Σ(v − v̄)·c, uncentred without fit-intercept, over the
  compacted positive-weight rows, in each column's staged units, since its
  power of two cancels in its cosine; v̄ is the planner's mean rounded to a
  double, less the remainder ρ = Σw(v − v̄)/Σw that the spread pass sums
  alongside the spread (and target_centre for y), so the centring carries
  ρ's rounding error rather than v̄'s: without ρ, a float64 column at
  10¹³ + U(0, 1) kept an error Σw·ε·(v − v̄) that read as a cosine of
  2.5·10⁻⁵ and refused a sound fit (review r4 M1); each of
  Σw·c², Σw·c·r and Σw(y − ȳ)² is carried as a significand and a power of two
  rescaled at each addition, so no product underflows or overflows double)
  exceeds
  2⁻⁸ (float32) or 2⁻¹⁸ (float64) is refused (V13). A kept column whose staged
  values are 0 on every positive-weight row (the shared exponent of a wide fit
  underflowed it) is skipped: it cannot correlate with the residual, and its
  coefficient is 0; any other column whose cosine is not finite is refused. The check
  does not see the intercept: centring removes any constant error, so Σ w r = 0
  is left to cuML's closed form ȳ − x̄ᵀc. A check of |Σ w r| against
  √(Σ w)·‖ỹ_c‖ was not added because it refuses sound float32 fits whose y mean
  dwarfs its spread (y = 10⁶ + N(0, 1): rounding alone gives about 6·10⁻²).
  Law T4 tests Σ w r = 0. The bound is on that cosine, not on the
  coefficients: a coefficient fixed only by rows of low weight w carries an
  error of about u·√(w_max/w) under every algorithm (normwise backward
  stable, within T1's κ̃ of the staged X), and eig ('auto) can lose it
  entirely once w/w_max is below about u²; the wrong fit's residual is then at
  rounding level, so the check cannot see it (review r3 M2).
- **L1-e, the T laws' coordinates (L3r, #111).** Every fit solves the
  staged problem: the weights times 2^(even_floor(−⌊log₂ max w⌋)) (the
  shim's `prepare_fit`, so 1/4 for the seeded weights 1 to 4); column j
  times D_j = 2^(−⌊log₂ s_j⌋), s_j² = G_jj the weighted (centred with an
  intercept) Gram's diagonal on those weights, the spread `plan_staging`
  measures; one shared D_j = min over the kept columns for a wide fit
  (L1-c); y times 2^(−⌊log₂ max|y|⌋) over the positive-weight rows. The
  staged coefficients are c̃ = 2^(e_y)·D⁻¹c, the problem's condition is
  κ̃ = κ(G̃)^½ over the kept columns of G̃ = DGD, and `staging-of` computes
  the weights' power, D, 2^(e_y), κ̃ and λ̲, the outer bracket's lower end
  of λ_min(G̃), exactly from the rationals (`spread_exponent`'s overflow
  cap binds only for a spread below 2^(−reach) of the column's largest,
  which a kept dtype column cannot have; a spread within ulps of a power of
  two can stage one column a factor 2 off the host's D, which moves κ̃ by at
  most 2). The weights' power of four moves every e_j by the same integer,
  so κ̃ does not see it; D and c̃ do, and with them T1's
  max(max|c̃*| + C1, 1).
  The laws are the earlier ones restated there, plus cuML's centring error
  (the next bullet), C1 and C3 below:
  - T1: max|c̃ − c̃*| ≤ 128·u·κ̃′²·max(max|c̃*| + C1, 1) + C1.
  - T3: the residual cosine is invariant under D, the weights' power and
    y's power, so only κ changes: ≤ 64·u·κ̃′·S + C3.
  - T2: b − b* = (b − (ȳ − x̄ᵀc)) − x̄ᵀ(c − c*) exactly, so
    |b − b*| ≤ B4/Σw + 2^(−e_y)·Σⱼ|x̄ⱼ|D_j·B1 over the kept columns (a
    dropped column's c_j and c*_j are both 0), B1 and B4 T1's and T4's
    bounds. T2 is therefore a corollary of T1 and T4: whenever both pass,
    it passes. It is kept as a check because it is cheap and reports the
    intercept on its own. The earlier 8·u·κ·max|y| left out x̄ᵀ(c − c*),
    which grows with the means where κ̃ does not.
  With κ of the unscaled X these bounds could not fail on breast-cancer raw
  float32 (κ = 7.96e5, κ̃ = 390): coefficients halved, with b consistent,
  passed all four (review r1 H1). In the staged coordinates T3 refuses them
  at 307 to 316 times its bound on raw float32 (0.9·c*: 61 to 63; c = 0: 614
  to 633), and T1 at 2.3e8 on raw float64; T1 alone does not, its κ̃² term
  being 128·2⁻²⁴·389.6² = 1.16 in float32, so the halved fit sits at 0.43
  to 0.46 of it (unchanged by C1 and C3 below: the raw columns' means are
  within a few spreads of 0).
- **L1-e, cuML's centring error (L3r, #111).** With an intercept cuML
  centres the staged problem with its own means, x̂̄ⱼ = x̄ⱼ + εⱼ and
  ŷ̄ = ȳ + ε_y, bounded as T4 bounds them: |εⱼ| ≤ ε̄ⱼ = γ_h·Σw|xⱼ|/Σw and
  |ε_y| ≤ ε̄_y = γ_h·Σw|y|/Σw, h for raft's order and the row count (next
  bullet). Rounding to the dtype makes |εⱼ| at least about u·|x̄ⱼ| in any
  order, and the staging normalises the spread, not the mean, so nothing
  in κ̃ bounds it (review r2 M1). Write X_c, y_c for the data centred
  exactly (X_cᵀW1 = 0, 1ᵀWy_c = 0), so cuML solves with X̂_c = X_c − 1εᵀ
  and ŷ_c = y_c − ε_y·1. For any c, with r = y_c − X_c·c (1ᵀWr = 0),
  X̂_cᵀW(ŷ_c − X̂_c·c) = X_cᵀWr + Σw·ε·(ε_y − εᵀc)
  exactly: the residual the host measures differs from the one the solver
  makes small by Σw·ε·(ε_y − εᵀc).
  - κ̃′: cuML solves with G̃ + Σw·Dεεᵀ D, whose top eigenvalue is at most
    λ̄ + Σw·‖Dε̄‖₂² (λ̄ the outer bracket's upper end) and whose bottom one
    is at least λ̲, so κ̃′² = (λ̄ + Σw·‖Dε̄‖₂²)/λ̲ bounds the condition of
    the problem it solves, and both solver terms use it. Without an
    intercept ε = 0 and κ̃′ = κ̃. The shift is not small: on breast-cancer
    float32 κ̃′/κ̃ is 1.008 and 1.056 at 2¹⁰ (unweighted, weighted), 1.12
    and 1.69 at 2¹²; float64 1.000.
  - C3: the solver's cosine is ≤ 64·u·κ̃′ in its own norms,
    ‖x̂_c,j‖² = ‖x_c,j‖² + Σw·εⱼ² and ‖ŷ_c‖² = ‖y_c‖² + Σw·ε_y², so
    S = √((1 + maxⱼ Σw·ε̄ⱼ²/‖x_c,j‖²)·(1 + Σw·ε̄_y²/‖y_c‖²)) and
    C3 = maxⱼ Σw·ε̄ⱼ·(ε̄_y + Σₖ ε̄ₖ|cₖ|)/(‖x_c,j‖·‖y_c‖), over the kept
    columns, norms weighted, c as returned.
  - C1: at the exact solution c̃ of cuML's problem the left side is 0, and
    at c* X_cᵀWr* = 0, so (G + Σw·εεᵀ)(c̃ − c*) = Σw·ε·(ε_y − εᵀc*).
    Staged (the weights as staged, in Σw too), with ẽ = Dε, t̃ =
    2^(e_y)·(ε_y − εᵀc*) and |t̃| ≤ τ = 2^(e_y)·(ε̄_y + Σₖ ε̄ₖ|c*ₖ|):
    - first order: G̃ + Σw·ẽẽᵀ ⪰ G̃, so ‖c̃ − c̃*‖₂ ≤ ‖Dε̄‖₂·Σw·τ/λ̲;
    - Sherman–Morrison: c̃ − c̃* = Σw·t̃·G̃⁻¹ẽ/(1 + Σw·q), q = ẽᵀG̃⁻¹ẽ, and
      ‖G̃⁻¹ẽ‖₂ ≤ q^½/λ_min^½, so ‖c̃ − c̃*‖₂ ≤ τ·√Σw/√λ̲·√(Σw·q)/(1 + Σw·q)
      ≤ τ·√Σw/(2√λ̲), as x/(1 + x²) ≤ ½. It holds for any ε in the box,
      with no smallness assumption.
    C1 is the smaller of the two, over the kept columns, a bound on
    max|c̃ − c̃*|. The solver's own error is about c̃, so T1's first term
    takes max|c̃| ≤ max|c̃*| + C1. On breast-cancer float32 the first-order
    form is 358 and 2465 times max|c̃*| at 2¹⁰ (unweighted, weighted), 5833
    and 40449 at 2¹²; the cap is 3.7, 9.6, 15 and 39 times. The discriminating
    case is two float32 columns offset by 2¹⁶ (an 8 × 8 grid, κ̃ about 2),
    their means moved by ε̄ along G̃'s bottom eigenvector and solved
    exactly (`linear-laws-centring-test`): T1 is 0.62 of its bound with C1
    and 51.9 without it.
  - On breast-cancer float32 offset by 2¹⁴, the exact solve of the problem
    centred with raft's medium-kernel means (simulated on the host,
    `linear-laws-centring-test`) has a cosine of 1.32 times 64·u·κ̃: a
    perfect solver fails T3 without C3. With C1 and C3 it meets all four
    laws. With ε̄ at its worst case the float32 T3 bound at c* is 0.042
    (2¹⁰ unweighted), 0.162 (2¹⁰ weighted), 0.665 (2¹² unweighted) and
    2.61 (2¹² weighted), and 10.8 at 2¹⁴ unweighted. A consistent b is
    exact by construction, so T4 cannot see a coefficient error, T2
    follows T1, and C1 leaves T1 loose: T3 is the only guard, and from
    2¹⁰ weighted on it passes materially wrong fits (0.9·c* at 0.62 of
    T3's bound at 2¹⁰ weighted, 0.15 at 2¹² unweighted, 0.04 at 2¹²
    weighted; review r3 M1). The device's means are far better than their
    worst case; the bound does not assume so. A library fix (centre with the shim's
    compensated v̄ + ρ, then call olsFit without an intercept) removes the
    term; it is its own issue.
- **L1-e, T4's intercept term (L3r, #111).** With an intercept every
  algorithm returns b = ȳ − Σⱼ x̄ⱼcⱼ from its own c, x̄ and ȳ the weighted
  means, so for any c the T4 sum is Σᵢ wᵢ(yᵢ − b − xᵢ·c) = Σw·(ȳ − Σⱼ x̄ⱼcⱼ − b)
  exactly: T4 sees the rounding of that formula, never c's error. cuML's
  preProcessData takes x̄ⱼ and ȳ with raft's `mean` (sum, then times
  fl(1/n)) or `weightedMean` (Σ v·w over Σw, Σw by `raft::stats::sum`), all
  through raft 26.08's coalesced reduction (`coalesced_reduction-inl.cuh`)
  over the n rows the shim passes (the positive-weight rows). Each thread
  sums its strided rows with Kahan–Babuška–Neumaier compensation (at most
  2u), the partials meet, and the compensation is added once:
  - thin (n ≤ 512, or many columns and n < 2048): a 32-lane warp tree,
    5 levels, so 8 roundings;
  - medium (below 2¹⁷ rows): a cub raking block reduce of 256, 7 serial
    adds and 5 levels, so 15;
  - thick (2¹⁷ rows or more, fewer columns than SMs): the medium count in
    each of 64 blocks per column (15), then a thin pass over the 64
    partials, a two-term KBN that is exact, 5 levels and the final add (6),
    so 21.
  The sum's count s is 15 below 2¹⁷ rows and 21 at or above, and
  |x̂ⱼ − x̄ⱼ| ≤ γ_h·Σw|xⱼ|/Σw with h = s + 2 unweighted (the reciprocal and
  the product: 17, 23) and h = 2s + 2 weighted (the products, the sum, the
  division and Σw's own s: 32, 44). postProcessData's gemm 1×d·d×1 adds
  γ_d·Σ|x̂ⱼcⱼ| in any order, and the subtraction u·|b|. T4's bound is
  therefore γ_h·Σw|y| + u·Σw·|b| + (γ_h + γ_d)·Σⱼ(Σᵢwᵢ|xᵢⱼ|)·|cⱼ|, b and c
  as returned: every term is the worst case for the device's order, so the
  bound is one model (the earlier ȳ term, 4·√n·u·‖y‖₂·max w, was
  probabilistic and about 4 times smaller on breast-cancer). The earlier
  u·Σw·(|b| + Σⱼ|x̄ⱼcⱼ|) counted one rounding per term. Breast-cancer
  offset by 2¹⁰ with means summed serially in float32 (the review r1
  t4sim case, `linear-laws-centring-test`) is 18.9 times the old bound
  and 0.63 of this one; the device's compensated reduction keeps x̂ⱼ
  near its first rounding, so the GPU offset set does not tell the two
  apart (at most 0.56 of the old bound). A serial float32 sum of 569
  values makes up to 568 roundings, so serial order is outside this
  model (h = 17 is raft's medium kernel) and the 0.63 is an empirical
  margin; the case stands as the old term's discriminator, not as a
  consequence of the model.
- **L1-e, the offset set (L3r, #111).** Breast-cancer with every column
  plus a power of two, every algorithm, only on arms where the laws
  discriminate: float32 at 2¹⁰ unweighted, and float64 at 2¹⁰, 2¹² and
  2¹⁴, unweighted and weighted (`offset-law-arms`, shared by the GPU file
  and the host mutation test). The host test takes the exact c* scaled by
  0.9 with a consistent intercept on every kept arm and requires a law to
  fail: T3 at 2.39 in float32; T1 at 4.6·10⁷ to 4.9·10⁷ and T3 at
  3.3·10¹⁰ to 3.4·10¹⁰ in float64. The other float32 arms are out:
  2¹⁰ weighted and 2¹² both weightings pass that mutation (above), so a
  law check there would be vacuous, and acceptance-only smoke is not kept.
  Float32 at 2¹⁴ is out too: unweighted, every fit is accepted under a
  bound that cannot fail, and weighted, the shim refuses every algorithm
  by V13 (the eig solve: residual cosine 7.7·10⁻³ > 2⁻⁸), a sound problem
  (κ̃ = 389) refused because cuML centres with its own float32 means. All
  of them return under the library fix above (#112), once the
  compensated centring makes ε̄ small enough for the laws to
  discriminate.
- **L1-e, ordering.** The workspace bound (V7) is checked after staging is
  planned, so it follows the compacted row index, plan_staging's d-sized
  scratch, the column profiles and, for svd, a gesvdj info handle; all are
  freed on the refusal. This departs from spec §3's "every refusal before any
  allocation", as spec §2.7 orders it. Racket refuses a non-boolean
  `fit-intercept` (V8's half; the algorithm half stays in C, after V1) and a
  non-real intercept (W7's half; a non-finite one stays in C, after W1–W6)
  before any view check, so they precede V1 and W1.
- **L1-e, intercept overflow.** A finite solve whose unscaled intercept
  overflows the dtype is refused with "y: the intercept exceeds the {dtype}
  range", after V14; it is not in the spec's V table.
- **L1-f.** `define-estimator` takes `#:needs-y`: y is positional, a missing y
  is refused, and `[score …]` passes y through.

## Decisions recorded in leg L2 (#88)

- **L2-a.** Ridge's penalty is not invariant under per-column scaling, so
  every kept column is scaled by one shared power of two 2^s and alpha by its
  square (`shim/linear_model/src/ridge_staging.cpp`). With e_j L1's capped
  per-column exponent, c_j = −ilogb(max|x_j|) + reach (L1's cap), H = 16
  (float32) or 64 (float64), F = H/2 and 4^k L1's weight power,
  s = min(min_j e_j + H, min_j c_j, max(⌊(2H − ilogb(α·4^−k))/2⌋, min_j e_j + F)),
  so the larger of the widest staged spread² and the staged α′ = (T)(α·4^s·4^−k),
  rounded once from double, sits near 2^(2H), and the widest staged spread never
  falls below 2^F: `ridge.cuh` zeroes singular values ≤ 1e-10 ≈ 2^−33, and with
  the widest at 2^F what it can drop is under 2^(−33−F) of the fit, 2^−17 u
  (float32) and 2^−12 u (float64). Without the floor, α/σ²_max past 2^98
  (float32) or 2^208 (float64) gave wrong or zero coefficients (review H1).
  Past the floor α′ grows with α; an α′ past the dtype's range is refused
  (`alpha: expected at most {bound} for this X in {dtype}`), measured from
  α/σ²_max = 2^111 to 2^112 (float32) and 2^959 to 2^960 (float64) on
  `normal-case 200 3 0` and `200 7 1` weighted; up to the edge the error equals
  the error at 2^90 (float32) or 2^200 (float64). Past the floor the staged
  coefficients b′ ≈ X′ᵀy′/α′ shrink with α′ and can go subnormal; what that
  costs is below u·max|c|, so R1 does not see it. With y's correlation to X at
  2^−20 to 2^−30 (float32) or 2^−40 to 2^−60 (float64), on dtype-rounded data,
  the fit is off by 1.6e-2 to 7.7 (float32) and 7e-6 to 8.3 (float64) of
  max|c*| (review r1 H-b), but rel·corr/u stays within 0.05 to 0.3 at every
  α from α/σ²_max = 1 (no floor) to the edge: the error is the u/corr
  cancellation in X′ᵀy′, which R1's 128·u·κ bound leaves out, not the floor.
  A column 2^−40 narrower than the widest (float32, corr 1) loses its own
  coefficient past α/σ²_max ≈ 2^100 (relative error 0.11 there, 1 at 2^110),
  an absolute error under 2^−40·max|c|; float64 keeps it to 3e-14 up to the
  edge. y keeps L1's power of two, which cancels from α′. Scaling X by 2^a
  with alpha by 4^a, y by 2^c, or the weights and alpha by 4^m moves the fit
  exactly (laws B2–B4).
- **L2-b.** The keyword is `#:solver` (`'auto`, `'eig`, `'svd`), Python
  cuML's and sklearn's name; `'auto` is `'eig`; `'lsmr` is not bound.
- **L2-c.** alpha 0 is L2-i's case α′ = 0: `linear-regression-fit`'s path,
  'eig for 'auto and 'eig, 'svd for 'svd, bit for bit (law B1), so exactly
  collinear X is refused under both solvers as L1's 'eig and 'svd refuse it.
  L1's 'svd covers every shape ridge takes: it already runs the Jacobi SVD for
  one kept column and whenever the kept columns outnumber the positive-weight
  rows, less one with an intercept (L1's `plan_fit`). −0.0 is accepted as 0.
- **L2-d.** NaN, ±inf and negative alpha are refused in C (V8), a non-real
  alpha in Racket; an exact alpha past the double range becomes +inf.0 and
  is refused.
- **L2-e.** After the solve the shim computes, on the staged data,
  g_j = |Σw′(v′_j − v̄′_j)r′ − α′b′_j| / (√Σw′(v′_j − v̄′_j)²·√Σw′(y′ − ȳ′)² + α′·max_k|b′_k|),
  each column's gradient against its cross term's size plus the largest
  penalty term, and refuses the fit when max g > 2^−8 (float32) or 2^−18
  (float64); a penalty term past the double range gives g = ∞. L1's residual
  cosine is its α′ = 0 case, computed by L1's own arithmetic, so it is L1's
  value bit for bit: one `stationarity_cosine`. Until review r3 M1′ the
  penalty term was column j's own α′|b′_j|: once α′ dominated the narrow
  columns' spreads, 'svd's normwise error of about u·max|b| in b_j gave them
  a cosine of about u·σ_wide/σ_narrow, which refused sound three-column fits
  at α from 2^−8·σ²_widest (float64 g = 25 and 30) to 2^10·σ²_widest (L2-k).
  Until review r2 H1 the denominator was √(Σw′(v′_j − v̄′_j)² + α′)·√Σw′(y′ − ȳ′)², the residual
  cosine of the augmented problem [X; √α′·I]. Once α′ ≫ σ′_j² that scales
  column j's gradient down by σ′_j/√α′, so a coefficient the solve zeroed
  passed at about corr·σ_j/√α: float32 columns 2^±44 apart at
  α = 2^30·σ²_narrow, and float64 columns 2^±100 apart at 2^40, returned the
  narrow column's coefficient as 1.7e-41 and 4.7e-92 against 1.4e4 and
  1.0e18, under both solvers. Those fits are now refused, but the same
  zeroed coefficient passes when its column carries under the threshold's
  share of y (L2-k). The check bounds the solve, not the conditioning: a
  float32 'svd fit 4.7× off on exactly collinear columns in the L2-i window
  passes (L2-i).
- **L2-f.** One kept column runs `ridgeFit`'s SVD under either solver
  (`ridgeFit` does this itself); 'svd with fewer positive-weight rows than
  kept columns runs eig, since gesvd aborts there. Both are covered by law R7.
- **L2-g.** Weighted fits with an intercept do not repeat bitwise from about
  2000 positive rows (measured: 2000 rows of `normal-case 2000 7 2` float32
  'eig differ by 1 ulp between runs; 1000 rows repeat), through raft
  `weightedMean` (L1-h); unweighted fits and fits without an intercept repeat.
  The bitwise laws B5–B8 keep weighted intercepted cases at ≤ 1000 rows, and
  both Scribble entries say so.
- **L2-h.** `define-regressor` in `cuml/private/estimator-support.rkt` is
  `define-estimator` with `#:needs-y`, the resources setting, predict and the
  weighted `r2-score` scorer from a predict `/who` (`r2-scorer`);
  `linear-regression` and `ridge` are each one `define-regressor`.
- **L2-i.** Every fit whose staged α′ ≤ u·σ′²_min runs `linear-regression-fit`'s
  path (L2-c), u = 2^−24 (float32) or 2^−53 (float64) and σ′²_min the smallest
  staged weighted (centred with an intercept) spread² among the kept columns,
  from the planner's per-column spread. α = 0 routes before any arithmetic;
  otherwise α′ ≤ 2^−digits·σ′_j² is decided per kept column from frexp
  mantissas and integer exponents (α taken exactly, not as the rounded α′),
  so no quotient can be 0/0 or ∞/∞ and no staged spread can underflow:
  dividing by an ldexp'd σ′_min, which is 0 on float64 columns 2^∓950 apart or
  on a column varying only on a row of weight 2^−1074, made α = 0 a NaN
  comparison and sent it down the ridge path, where those columns got 0
  (review r1 H-a; B1 rows on both). A kept column whose spread is 0 is never
  negligible against α > 0. The penalty is below rounding on every column's diagonal there, and
  the solver's noise eigenvalues of rank-deficient X would pass for α′ (review
  M1: float32 'eig on x₃ = x₁ + x₂ returned min-norm + 1.72·(1, 1, −1) at
  α = 1e-40). The route reuses ridge's checked inputs, staged weights and plan,
  so it costs no second scan, and gives L1's bits and L1's refusals. α′/σ′²_min
  equals α/σ²_min, so the threshold is u·min_j Σw(x_j − x̄_j)² in the caller's
  units; tests check u·σ²_min/2 (L1's bits) and 2u·σ²_min (the ridge solve, R1).
  Measured (`normal-case 200 7 0`, `200 7 1` weighted, `20 2 0`, and 200 rows
  with columns at 2^−13 … 2^13; both dtypes, solvers and intercepts): at 1/4
  and 1/2 of the threshold 64 of 64 fits are L1's bits; at 2 and 4, 64 of 64
  are not, and on the mixed-scale X 12 of 16 of those are refused (float32,
  and float64 'eig): an α just past rounding on the narrowest column is
  ill-conditioned there, where α = 0 fits. Above u·σ²_max the route never
  applies. Between u·σ²_min and u·σ²_max, on mixed-scale X with exactly
  collinear wide columns (x₁ at 2^−13 or 1, x₂ at 1, x₃, x₄, x₃ + x₄ at 2^13),
  'eig is refused at every α measured, and 'svd is accepted with a null-space
  error the gradient cosine cannot see: up to 4.3e4·max|c*| (float64, α =
  2u·σ²_min) and 4.7·max|c*| (float32, columns at 1). Every accepted fit is
  within u·κ_α (at most 1.8·u·κ_α, κ_α = (λ_max + α)/α ≥ 1/u there), so the
  bound holds but says nothing, and the Scribble discloses the window.
  Review r1's detector 2, the condition of the column-equilibrated Gram
  D⁻¹X′ᵀWX′D⁻¹ + α′D⁻² (D the staged spreads) run only in the window, is
  not adopted: measured exactly (`.scratch` probe p10, the window rows above
  and Q2, no intercept), u·κ is ≥ 8 on every window row, but 0.10 on Q2
  (float32), over any 2^−10 threshold, and 1679 on Q2 with x₁ at 2^−20, a fit
  'svd solves to 7.1e-5: a column damped far past its spread by α′ inflates
  the largest eigenvalue, so it refuses good fits. Equilibrating
  X′ᵀWX′ + α′I by √(σ′_j² + α′) instead (unit diagonal) gives u·κ ≥ 8 on
  every window row, 8.3e-8 on both Q2 rows (float32), 1.5e-16 (float64) and
  ≤ 2.9e-10 on the nearly collinear eps rows: it separates them, but is not
  the ruled detector; an owner decision.
  Columns of very different scales without collinearity are the window's
  other case (review r2 H1): two columns 2^±g apart, α = 2^k·σ²_narrow. Under
  L2-e's old normaliser both solvers returned the narrow coefficient as 0
  (relative error 1) at float32 g = 44, k = 30 and 57, g = 60, k = 57, and
  float64 g = 100, k = 40; the old check also accepted g = 44, k = 15, 20 and 40
  and g = 100, k = 57, unmeasured against the oracle there. With L2-e's
  normaliser each of these fits, y = a + b + noise, is refused, at cosine
  0.59 to 0.60. When the narrow column carries under 2^−8 (float32) or 2^−18
  (float64) of y, the same fits are accepted and wrong (L2-k). Every fit at
  float32 g = 8, 13 and 20 (k from −10 to 57) and float64 g = 30 is
  unchanged: accepted within 3e-7 (float32) and
  8e-16 (float64) of the ridge solution, or, at float32 g = 20 and k ≤ 10,
  refused as before. The collinear window rows above are unchanged by it:
  'svd is still accepted there, and 'eig still refuses the wide-at-2^13 rows
  and fits the all-at-1 rows to 1.7e-4 (float32) and 1.9e-13 (float64).
  The exact α (not α′ rounded in the dtype) moved the route where α′ is 0 or
  subnormal in T but α > u·σ²_min exactly (review r2 M3): such a fit took
  L1's path before and takes the ridge path now. On float32 columns 2^±60
  apart, α/σ²_narrow from 2^−23 to 2^56, L1's bits had been off the ridge
  solution by 9.8e-4 at 2^−10, 0.999 at 2^0 and 1e3 to 1e17 beyond; those
  fits are now refused, including 2^−23 to 2^−20, where L1's answer was
  within 8.3e-7. The rule is kept, since it refuses misfits the old one
  returned silently. Pinned by the gtest
  `AnAlphaRoundingToZeroInTheDtypeIsNegligibleOnlyUnderRounding` (staged
  spreads 2^120 apart, α′ = 0 in float32 with α at and over u·σ′²_min) and the
  float32 g = 60, k = 0 refusal rows; deciding from the rounded α′ fails both.
- **L2-j, refusal and accuracy.** A fit is refused when L2-e's max g passes
  2^−8 (float32) or 2^−18 (float64). An accepted fit's coefficients are off
  the ridge solution by about
  u·(σ²_max + α)/(σ²_min + α) + u·σ_max·‖y‖/((σ²_min + α)·max|b|), relative
  to the largest coefficient max|b|, σ² the eigenvalues of XᵀWX with X
  centred by its weighted mean with an intercept. The second term is the
  cancellation in X′ᵀy′; at κ ≈ 1 it is about u/corr(y, X), so it dominates
  when y is nearly orthogonal to X (review r2 M2; until review r3 M2′ it
  lacked the 1/max|b| and was not relative). Measured at κ ≈ 1: float32 with
  corr 2^−10 is off by 1.3e-5 (u/corr 6.1e-5, the first term 7.6e-6), corr
  2^−30 by 7.65 (u/corr 64), and float64 corr 2^−60 by 8.30 (u/corr 128). A
  narrow column's own coefficient can lose everything inside that bound:
  float32, a column 2^−40 narrower than the widest, past α/σ²_max ≈ 2^100
  (L2-a). The R1 laws run on `normal-case` data (noise 1/10, corr ≈ 1), where
  the second term is about u.
  Q2's float32 columns at 2^−13, 1, 2^13, 1 at α = 1 are past the threshold:
  'eig is refused and 'svd is within 2^−10·max|c*| of the ridge solution. On
  p3's collinear X both solvers are refused, as L1's 'eig and 'svd refuse it.
- **L2-k, the scope of the check.** Every silently wrong fit found in
  review (the L2-i window, review r2 H1, review r3 H1′) lies where
  u·κ_α ≥ 1. There the wrong answer is stationary to rounding, so L2-e's
  check is a best-effort detector, not a bound. A sound refusal needs a
  solve equilibrated per column, which `ridgeFit`'s scalar α cannot give
  (#104); no further detector is added in L2. The Scribble says the
  coefficients are accurate to about u·κ_α, and that past u·κ_α ≥ 1 a fit may
  be refused or returned wrong under either solver.
  Measured cases (review r3 probes r1, r4, r5, r6; `.scratch` runs at r4):
  - Accepted and wrong (H1′): columns 2^±g, α = 2^k·σ²_narrow,
    y = a + 2^−m·b + 2^−20·c with c independent, so the narrow column carries
    about 2^−m of y and its coefficient is still max|c*|. A solve that zeroes
    it leaves a cosine of about 2^−m. float32 g = 44, k = 30 and 57, and
    g = 60, k = 30, are accepted at m = 9, 10, 12 and 16 with relative error
    1.0 (g = 44, k = 30, m = 9: 1.4e-41 against 31.8), and refused at m = 4,
    7 and 8 (cosine 0.064, 8.1e-3, 4.0e-3). float64 g = 100, k = 40 and 57,
    and g = 300, k = 57, are accepted at m = 19, 20 and 24 (g = 100, k = 40,
    m = 19: 3.8e-92 against 2.1e12) and refused at m = 17 and 18. Both
    solvers and both intercepts agree on every row. The float32 g = 44,
    k = 30, m = 9 and float64 g = 100, k = 40, m = 19 rows are pinned as
    known accepted-wrong fits in `ridge-refusal-test`, so a future detector
    flips them consciously.
  - Refused and sound (review r3 M1′, fixed in L2-e): three correlated
    columns, one or two of them 2^±g apart (a, a + b, c + a/2 and
    reorderings), α = 2^k·σ²_widest. Under the own-coefficient penalty term,
    'svd was refused on 34 fits of probe r1 and 50 of probe r4 (float32
    g = 10 and 13, float64 g = 25 and 30), at cosines up to 0.133 and at
    κ_α ≤ 3.6 for k ≥ 0. With the normwise term every one is accepted, within 5.1e-7
    (float32) and 6.1e-15 (float64) of max|c*|. Probe r4's 240 fits match the
    pre-r2 normaliser's row for row. Probes q1, q4 and r2's edges keep every
    decision: the H1 rows stay refused at cosine 0.59 to 0.60, and r5 and r6
    are unchanged. Pinned by the float32 g = 10, k = 10 'svd row in
    `ridge-refusal-test`.
  Over the L1 and L2 test files (3209 tests) the largest accepted cosine
  outside the pinned H1′ rows is 4.0e-4 (float32, Q2 'svd) and 2.7e-15
  (float64) under either penalty term. The pinned rows sit at 2.0e-3 and
  1.9e-6, just under the threshold. The two M1′ rows fall from 0.133 and
  0.129 to 4.1e-7, and every α′ = 0 cosine (17892) is unchanged bitwise.
  Q2 is unchanged: float32 'eig refused at 0.385 and 0.203, 'svd accepted at
  4.0e-4 and 7.9e-5; float64 accepted at ≤ 1.2e-8 ('eig) and 1.7e-13 ('svd).
- **L2-k, float32 'eig on a Gram singular to the dtype (L3r, #111).**
  Breast-cancer (569×29, texture-mean on raw X, column sds 0.0026 to 569,
  intercept): XᵀWX has λ_max 2.5e8 and λ_min 4.0e-4, u·κ₀ = 3.8e4, and 19 of
  its 29 eigenvalues lie below u·λ_max ≈ 15. No other real case (diabetes,
  mtcars, wine, raw or standardised, either weighting) has any. On the ridge
  path α_k = Σw·λ_max·10^(−3k/5), float32 'eig is 4.1e-4 to 2.6e-3 off the
  exact solve relative to max|c*|, 15 to 880 times u·κ_α; 'svd is 1.4e-7 to
  7.0e-7. Entries below are error/(u·κ_α), each row of X, y and w repeated
  r times and α scaled by r, so the exact solve and κ_α are unchanged:

  | fit | r = 1 | r = 2 | r = 4 | r = 8 |
  |---|---|---|---|---|
  | 'eig k = 1 | 682 | 680 | 92 | 97 |
  | 'eig k = 3 | 211 | 211 | 29 | 30 |
  | 'eig k = 5 | 15.6 | 15.6 | 2.1 | 2.2 |
  | 'eig k = 1, weights 1–4 | 882 | 72 | 854 | 98 |
  | 'svd, any k | 0.23 | 0.49 | 0.27 | 0.80 |

  The error does not grow with the row count. Swept over α/λ_max from 1e-4
  to 1e4, 'eig stays at 3.1e-3 to 4.5e-5 while 'svd stays at 3.3e-7 to
  1.1e-7; at α = 1e4·λ_max, where κ_α = 1.000, 'eig is 4.5e-5 (750·u) off and
  'svd 1.1e-7. Below α = 1e-6·λ_max 'eig is refused. Reversing the rows leaves
  'eig at 1.81e-3. Standardised X (κ_α ≤ 15), float64 and 'svd are within
  u·κ_α. Python cuML 26.08 `Ridge(solver='eig')` on the same float32 rows
  matches rkt-cuml's error to three digits at every k in both weightings
  (4.08e-4 against 4.08e-4 at k = 1, 2.59e-3 against 2.59e-3 at weighted
  k = 5); its 'svd is 2.3e-7 to 8.4e-7. The likely mechanism, unverified:
  `ridgeEig` forms U = XVΣ⁻¹ from float32 eigenpairs of XᵀX, so rounding in
  Xv_j on the noise-floor eigenpairs is not damped by σ_j.
  **L3r-a (decision).** Disclosed, behaviour unchanged: the Scribble says
  'auto and 'eig can lose accuracy whatever α is when XᵀWX has eigenvalues
  below u·λ_max (u·κ₀ ≥ 1), as in Python cuML, and that 'svd does not. R2
  bounds every fit. Where u·κ₀ ≥ 1, by the certified lower bound on κ₀
  over the varying columns (the inner pair of `eigen-bracket`), a float32
  'auto or 'eig fit is held to two measured envelopes, not guarantees:
  R1 at 2¹⁰·u·κ_α·max|c*| in place of 128·u·κ_α·max|c*| (8 times R1, above
  the worst error/(u·κ_α) of 882 in the table), and R3 at 2¹⁰·u·κ_α in place
  of 64·u·κ_α (16 times R3, above the worst 11.58 times below). 'svd and
  float64 keep R1 at 128 and R3 at 64. Measured on the breast-cancer ridge
  path, each law's error over its normal bound (R1 over the 2¹⁰ envelope),
  raw X, float32 'eig:

  | law, weighting | k = 1 | k = 2 | k = 3 | k = 4 | k = 5 |
  |---|---|---|---|---|---|
  | R3, unweighted | 9.15 | 6.02 | 1.91 | 0.49 | 0.13 |
  | R3, weighted | 11.58 | 7.45 | 2.15 | 0.55 | 0.15 |
  | R1 / 2¹⁰, unweighted | 0.67 | 0.57 | 0.21 | 0.054 | 0.015 |
  | R1 / 2¹⁰, weighted | 0.86 | 0.75 | 0.23 | 0.061 | 0.017 |

  R2 is at most 0.024 of its bound there, and every 'svd law, raw or
  standardised, at most 0.073.
  This keeps the Python twins bitwise and needs no threshold; refusing
  float32 'eig past u·κ₀ ≥ 1 by its own eigenvalues, or routing float32
  'auto to 'svd (which breaks parity with cuML's defaults, Q-L4), are the
  owner's alternatives, and float64 refinement (#107) is the systemic fix.

## Decisions recorded in leg L3 (#89)

Probed on the 3090 Ti (libcuml 26.8.0, Python cuML 26.08.00, sklearn
1.8.0); the spec's findings F1–F21 are summarised where they bind.

- **L3-a.** `cdFit` minimises (1/2Σw)Σw(y − b₀ − x·b)² + αρ‖b‖₁ +
  ½α(1−ρ)‖b‖² (F1: 2.4e-14 against a high-precision host CD over 216
  configurations). The L1 term scales as 2^(s+t) and the L2 term as 4^s
  under X·2^s, y·2^t, and `cdFit` takes (α, ρ), so α′ and ρ′ are exact only
  with t = s. Every kept column and y are staged at one exponent
  s = min(max_j e_j, s_hi), e_j being L1's capped exponent, with
  s_hi = min(min_j c_j, ⌊(E−6−β)/2⌋ − p, ⌊(E−6−β−p−q)/2⌋, E−4−β−q)
  (E the dtype's max_exponent, β = bit_width(m), p and q the ilogb of the
  largest |x| and |y|; `cd_staging.cpp`). α′ = (T)(α·4^s) from double;
  coefficients need no shift and the intercept is b′₀·2^−s. Every kept
  column's staged spread is at least 2^−4, so `cd.cuh:68`'s 1e-5 cut never
  fires (F4: X·1e-4 at 100 rows gives rel 1.0 raw, 2.3e-4 staged), and
  within [e_max − 4, s_hi] the fit is raw `cdFit`'s bit for bit (F3: 108/144;
  the misses are raw fits under the cut). The column cap min_j c_j is never
  the binding one, since c_j − (a) = 62 (float32) or 510 (float64) for the
  widest column. A y on its own exponent (F6) removes the absolute stop but
  puts the L2 term off by up to 6.9e-4. α′·m > 2^(E−1) is refused (V14):
  cap (a) keeps `cdFit`'s Σw̃(x′ − x̄′)² ≤ Σw̃x′² < 2^(E−4), so its denominator
  Σx′² + α′(1−ρ)m stays below 2^(E−4) + 2^(E−1)(1 + 2u), and ρα′m, the soft
  threshold (which only shrinks |x′ᵀr′|) and the quotient stay finite. Only
  "α′·m finite" (round 0) let that sum reach inf and return a coefficient 0
  with no refusal (f32 columns 2^58 apart, α within 1e-4 of the old bound).
  V12 (m > 2³¹ − 1) is defensive: `require_index` already refuses
  n·d > 2³¹ − 1, so the public path cannot reach it; a gtest calls it.
- **L3-b.** `cd.cuh:271` stops when every coefficient is under tol in the
  caller's units, after one epoch (F5: rel 0.12 to 2.08 at X·1e5). The shim
  refuses such a stop when K = max_j v_j/(max|b|·D_j) exceeds B + A, with
  v_j the subgradient violation, D_j = max(σ_j² + λ₂, σ_jΣ_kσ_k),
  λ₂ = α(1−ρ), B = tol rounded to the dtype and
  A = u√n_iter·(2 + √m·σ_y·max_j(σ_j/D_j)/max|b|)
  (L3-m, round 1). A relative stop has v_j ≤ tol·max|b|·σ_jΣ_{k≠j}σ_k, so
  K < B (F10: 3372 relative stops per dtype, K/tol ≤ 0.63; in float32
  rounding puts 128 of 444 over tol, at most 0.085 of A, F11); where λ₂
  dominates, v_j/(σ_j² + λ₂) is the move the next update of j would make,
  and K ≤ tol is the relative test applied one step ahead. Below λ₂ = σ_jΣ_{k≠j}σ_k this is round 0's check,
  K_old = max_j v_j/(σ_jσ_y) against tol·max|b|Σσ/σ_y, which refused every
  ridge-heavy one-epoch stop (α(1−ρ) ≫ σ²) however accurate: 200×7,
  corr 0.9, α 1e4 and 1e6 are 4.9e-4 and 4.9e-6 of max|c*| off after one
  epoch. A's √m stays on the gradient term only: near α_max the rounding in
  K reaches 30.8·u√n·σ_y·max(σ_j/D_j)/max|b| in float64 at m = 1e6 (28.4 at
  1e4), while the coefficient term needs under 1. F12 still refuses 72/72
  bad absolute stops (err > 1e-2) in both dtypes, and of 13 large-α
  (1e4, 1e6) stops off by more than tol, 13 in both dtypes since L3-o
  (round 1 missed two in float32, at tol 1e-7 with errors 2.2e-7 and
  1.5e-6 of max|c*|: there A/tol was 17 and 15.8, so V16b now refuses them).
  An absolute stop with A > B and K > max(B, tol) is refused (V16b; L3-o,
  amended in rounds 3 and 4), after V16: there K ≤ B + A cannot tell a
  stop within tol from one many times past it. Under a heavy L2 penalty A ≈ u√m/corr(x, y), so a
  weak signal on many rows puts it past the default tol (H2: 2^18 × 4
  float32, corr(x, y) 1e-4, α 30, ρ 0 was accepted after one epoch 5.8%
  off, K 0.058; 2^15 rows with y pure noise, α 1500, 2.1e-3 off, K 1.7e-3).
  A lower tol makes CD stop on its relative test, which needs no check
  (that fit at tol 1e-7: 4 epochs, 2.4e-7 off). K itself is computed in
  double (double residuals, exact product sums, `kkt_of` in long double),
  so in float32 A, built from float32's u, bounds what CD can reach, not
  K's own error: a float32 stop with K ≤ B has passed the check. In
  float64 A is built from the check's own u, so that argument does not
  hold there and float64 rests on measurement: 2^16 × 4, lasso with
  intercept, at α_max·(1 − ε) for ε 1e-11, 1e-12, 3e-13 and 1e-13,
  against an exact-rational oracle and an exact K, has A/B from 1.6 to
  161; round 2 refused all four, and the amendment accepts them at n-iter
  1, 8.3e-6, 8.3e-5, 2.8e-4 and 8.3e-4 of max|b*| off (exact K ≤ 2.1e-4),
  with the shim's K far closer to the exact K than A (2.82e-5 against
  2.09e-5 at ε 1e-12, where A is 0.016). Round 2 refused every
  A > B, and so refused float32 stops far inside tol on ordinary data
  (2^20 rows of one column with σ 2e4 at 0.5·α_max: K 1.3e-7, A 1.2e-3;
  2^17 × 3 in units 1, 30 and 1000 at 0.1·α_max: K 3e-8, A 1.1e-2, 2.8e-8
  off, where the advised tol 1e-7 fit was 3.2e-7 off).
  What V16b refuses that V16 accepted: no law-grid or twin fit (the law
  grid makes no absolute stop); none of F10's 866 accepted large-α
  absolute stops; near α_max (1000 × 10, α_max·(1 − 2⁻¹², 1 − 2⁻²⁰, 1)),
  23 of 44 float32 absolute stops, every one more than tol off (19 at
  1 − 2⁻²⁰, 1.1% to 15%; 4 at α_max), and 5 of 49 float64 ones, all at
  α_max itself with max|b| ≤ 1.9e-15, the rounding of an exact 0. Of
  round 2's refusals the amendment accepts float32's 15 at 1 − 2⁻¹² (within
  tol, errors ≤ 5e-4) and one at 1 − 2⁻²⁰ (numpy model: K 0.68·tol,
  0.66% off), and float64's 4 at α_max with K = 0 and max|b| ≤ 4.4e-16,
  and at 2^16 × 4 the four float64 stops at α_max·(1 − ε) for ε 1e-11 to
  1e-13 above. On the shim, float32 rounding, the cost L3-o accepts,
  is F5 at 0.99999·α_max (2^16 × 8, correlation 0.99): accepted 1.14e-3
  off with K 1.4e-4, and its tol 1e-7 refit gives the same coefficients.
  F12 is unchanged (72/72 and 13/13 in both dtypes).
  The round-2 review's bad-stop grid (9 configurations × 2 dtypes,
  2^16 to 2^20 rows) accepts 24 one-epoch absolute stops per dtype, none
  with K > B (largest K 7.3e-4 in float32, 5.9e-4 in float64); float32
  refuses 2 by V16b (K 5.9e-3 and 2.0e-3, 0.59% and 0.20% off) and 6 by
  V16, where round 2 refused 24 by V16b. One accepted float32 stop is more
  than tol off: 0.99999·α_max at correlation 0.999, K 3.7e-4, 2.2e-3 off,
  which a lower tol returns unchanged.
  What it promises: a fit returned before max-iter meets the elastic-net
  optimality conditions to within what the relative test allows, as the
  `elastic-net-fit` entry says (within tol of a stop the relative test
  would accept). What it does not: the coefficients' error, bounded only
  by ‖b − b*‖₂ ≤ K·max|b|·‖D‖₂/(λ_min + λ₂) for K computed exactly, λ_min
  the least eigenvalue of XᵀWX/Σw on the centred columns (F14; 7.9e-2 of
  max|b| at correlation 0.99 is sound under it); fits at max-iter (L3-e);
  and, as for ridge (L2-k), fits where u(λ_max + λ₂)/(λ_min + λ₂) ≥ 1.
- **L3-c.** α = 0 runs CD, as Python cuML and sklearn do (F17: 7.7e-15
  against lstsq at tol 1e-14).
- **L3-d.** K is computed only when n-iter < max-iter and
  0 < max|b′| < tol′, the only stops the absolute test makes. It costs one
  device pass for the column moments: about 10 ms at 1e6 × 100 float32
  (35 against 25 ms for the unchecked one-epoch fit).
- **L3-e.** At max-iter, n-iter = max-iter is reported; no refusal and no
  warning (Python is silent; K catches only 18/27 bad cutoffs, F13).
- **L3-f.** No `#:random-state`: `initShuffle` reseeds the generator with 0
  (`shuffle.h:18`), so `'random` is one fixed order and `srand` changes
  nothing (F7). L5 should re-probe SGD, which has the same pattern.
- **L3-g.** `cdFit`'s Σw is not deterministic (F8: 2–6 distinct results at
  50000 rows), so the shim centres with L1-e's weighted means, scales each
  row by √(w′·m/W′) and calls `cdFit` unweighted without an intercept (F9:
  1 distinct result, 2.8e-7 / 4.9e-16 against weighted `cdFit`). The
  intercept is (ȳ′ − Σ_j v̄′_j b′_j)·2^−s in double, rounded once. Python
  parity for weighted fits holds on dyadic fixtures only (E11).
- **L3-h.** Two result structs, `lasso-result` and `elastic-net-result`,
  each (coef intercept n-iter), with their predicts generated by one
  `define-cd-result`; one C job, lasso passing l1-ratio 1.
- **L3-i.** NaN, ±inf, α < 0, l1-ratio ∉ [0, 1], tol < 0 and
  max-iter ∉ [1, 2³¹−1] are refused; tol 0 and α 0 are accepted.
- **L3-j.** Python cuML's quasi-Newton solver is not bound; it can come as
  `#:solver` after L4's QN core.
- **L3-k.** n-iter is 0 when nothing was solved (constant y or no varying
  column).
- **L3-l.** A column more than 4 bits too narrow to stage beside the others
  and y is refused (V13), where raw cuML would zero it silently.
- **L3-n.** What the `elastic-net-fit` entry leaves out: `'random` has no
  random state because cuML seeds the shuffle itself (L3-f); the error
  bound's looseness on correlated columns is L3-b's 8% of the largest
  coefficient at correlation 0.99; a fit at max-iter does not warn (L3-e);
  `random_state`, `positive`, `precompute` and `warm_start` are not offered,
  nor is the quasi-Newton solver (L3-j); fits repeat bit for bit and X, y and
  sample-weight are not modified, as for `linear-regression-fit` (E6).
