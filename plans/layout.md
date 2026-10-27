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
  with f20d in place no input reaches it.
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
  40×(1,1) at perplexity 5, which fit at n-neighbors 60). No pre-fit check
  catches them. The shim does not centre X: that would break bitwise parity
  with Python cuML.
