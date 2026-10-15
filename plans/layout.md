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

3. **A scikit-learn-like estimator interface** (`define-generics`: `fit`,
   `predict`, ...) with wrappers in `cuml-lib`. Open:
   - whether the generics themselves live in `cuml-core-lib` (recommended,
     so glmnet and other libraries can implement them);
   - functional fit: `fit` returns a fitted estimator rather than mutating;
   - small capability generics (`fit`, `predict`, `transform`, ...) with a
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
- **The flake reuses rkt-raft's** `nix/racket-tools.nix`, `nix/treefmt.nix`,
  `packages.{rapids,shim,raft-dev,python-twins}` and its docs-dependency
  inputs (polars, datasets) through `rkt-raft.inputs`, so the pins are
  raft's. rkt-raft follows this flake's nixpkgs and treefmt-nix, both at
  raft's locked revisions, so its store paths are unchanged.
- **Not carried over from rkt-raft yet:** the ASAN/UBSAN shim build (the GPU
  suite runs the gtests under compute-sanitizer memcheck), the bounded
  death-test gate (no death tests), the binding census
  (`check-bindings.rkt`).

## Open owner questions

1. Package versions start at 0.1 and require `("raft" #:version "0.2")`;
   the GitHub repository name and when to create the remote.
2. An ASAN/UBSAN build of the shims as a flake check, as rkt-raft's
   `shim-sanitizers` (a second nvcc build in CI)?
3. A per-family binding census like rkt-raft's `check-bindings.rkt`?
4. Distribution after rkt-raft #24: one natipkg per family
   (`cuml-cluster-x86_64-linux`) or one for all shims?
5. From `plans/layout.md` #3: the estimator generics in cuml-core-lib
   (recommended), functional fit, capability generics with a `fit-predict`
   fallback.
6. Contracts for cuml's procedures: none in leg 0 (the private `blobs`
   helpers have none). Adopt raft's contracts policy when the first public
   procedure lands, or settle a cuml one?

## Follow-ups (no issue tracker yet)

- rkt-raft: its docs build requires `RAFT_KMEANS_CANARY` and reads files
  outside `raft/` (rkt-raft #23), so downstream repositories install raft
  from a writable copy of the whole tree and set the canary; drop both
  workarounds in `flake.nix` when #23 lands.
