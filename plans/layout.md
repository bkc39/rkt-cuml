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
- The Racket body allocates an output on the call's resources only when its
  shape is valid, and passes `#f` otherwise, so the shim's refusal names
  the argument. The init names map to the shim's codes in one `match`.
- The shim's refusals run in a fixed table order, first refusal wins, before
  the GPU is touched; an empty matrix has no layout to check.
- The decisions posted on #5, each with the recommended answer:
  - **Q-A.** cuVS's parameter errors (tol <= 0, oversampling-factor < 0)
    pass through as they are; tests match the first line with a regexp.
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
