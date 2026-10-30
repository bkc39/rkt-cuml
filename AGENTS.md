# AGENTS.md

Canonical agent guidance for rkt-cuml. Keep it current in the same change as
the code.

## Project overview

Racket bindings to NVIDIA cuML (RAPIDS 26.08), built on
[rkt-raft](https://github.com/bkc39/rkt-raft): cuML runs on raft's device
arrays, resources and jobs, and its errors are raft's. cuML is GPU-only.
Scope: clustering (k-means, DBSCAN, single-linkage agglomerative,
spectral, HDBSCAN) and metrics (contingency, adjusted Rand, silhouette,
trustworthiness), preprocessing (standard scaling), decomposition
(PCA) and manifold (UMAP, t-SNE, spectral embedding); epic #53 adds the rest of decomposition and
manifold. `plans/layout.md` holds the owner's decisions;
`plans/starting-notes.md` what rkt-raft established.

## Layout

A multi-package repository in typed-racket's style: every package is
`(define collection 'multi)` and adds modules to the one `cuml/` collection.

| Path | Package | Holds |
|---|---|---|
| `cuml-core-lib/` | cuml-core-lib | `cuml/core` (`exn:fail:cuml`), `cuml/estimator` (the estimator generics, loading nothing native), `cuml/core/private/` (the shared loader and staging; `logic-error.rkt`, the one `'logic` raise; `define-hyperparameters` and `define-codes` in `hyperparameters.rkt`, `define/who` in `define-who.rkt`, `deposit-for/who` in `deposit.rkt`; `model.rkt`, the native models: `native-model`, `deposit->native-model`, `live-anchor`, `native-model-release!`, `with-native-models`; `release.rkt`, `call-with-release` and `with-release`, restated from raft) |
| `cuml-cluster-lib/` | cuml-cluster-lib | `cuml/cluster` (k-means: `kmeans-params`, `current-batch-samples`, `kmeans-fit`, `kmeans-predict`, `kmeans-transform`, `kmeans-score`; DBSCAN: `dbscan-params`, `dbscan-fit`; single linkage: `single-linkage-params`, `single-linkage-fit`; spectral: `spectral-clustering-params`, `spectral-clustering-fit`, `spectral-clustering-fit-graph`; HDBSCAN: `hdbscan-params`, `hdbscan-fit`, `hdbscan-approximate-predict`, `hdbscan-membership-vector`, `hdbscan-all-points-membership-vectors`), `cuml/cluster/private/` (the bodies, their `/who` forms for the estimator wrappers, and the loader of `libcumlrkt_cluster.so`) |
| `cuml-metrics-lib/` | cuml-metrics-lib | `cuml/metrics` (`contingency-matrix` and its `contingency-table`, `adjusted-rand-score`, `silhouette-score`, `silhouette-samples`, `trustworthiness`), `cuml/metrics/private/` (the bodies, their `/who` forms, and the loader of `libcumlrkt_metrics.so`) |
| `cuml-preprocessing-lib/` | cuml-preprocessing-lib | `cuml/preprocessing` (`standard-scaler-params`, `standard-scaler-fit`, `standard-scaler-transform`, `standard-scaler-inverse-transform`, `standard-scaler-fit-transform`, and the `standard-scaler-result` struct), `cuml/preprocessing/private/` (the bodies, their `/who` forms, the private `current-scratch-elements`, and the loader of `libcumlrkt_preprocessing.so`) |
| `cuml-decomposition-lib/` | cuml-decomposition-lib | `cuml/decomposition` (`pca-params`, `pca-fit`, `pca-transform`, `pca-inverse-transform`, `pca-fit-transform`, and the `pca-result` struct; `tsvd-params`, `tsvd-fit`, `tsvd-transform`, `tsvd-inverse-transform`, `tsvd-fit-transform`, and the `tsvd-result` struct), `cuml/decomposition/private/` (the bodies, their `/who` forms, and the loader of `libcumlrkt_decomposition.so`) |
| `cuml-manifold-lib/` | cuml-manifold-lib | `cuml/manifold` (`umap-params`, `umap-fit`, `umap-transform`, `umap-fit-transform`, the `umap-result` struct, `tsne-params`, `tsne-fit`, `tsne-fit-transform`, the `tsne-result` struct, `spectral-embedding-params`, `spectral-embedding-fit`, `spectral-embedding-fit-graph`, `spectral-embedding-fit-transform` and the `spectral-embedding-result` struct), `cuml/manifold/private/` (the bodies, their `/who` forms, `curve.rkt`'s a, b fit, `support.rkt`'s `bound-params`, `drawn-seed`, `finite-real?`, `metric-code`, `rows-of` and `flag`, shared by UMAP, t-SNE and spectral embedding, and the loader of `libcumlrkt_manifold.so`) |
| `cuml-lib/` | cuml-lib | `cuml/main.rkt`, re-exporting `cuml/core`, `cuml/estimator`, `cuml/table` and every family; `cuml/table` (`table->device-matrix`, a `datasets-core` table to a device matrix, loading no cuML family or cuML native library); `cuml/private/` (`define-estimator` in `estimator-support.rkt`, the estimator wrappers: `kmeans`, `dbscan`, `agglomerative-clustering`, `spectral-clustering`, `hdbscan`, `standard-scaler`, `pca`, `truncated-svd`, `umap`, `tsne`, `spectral-embedding`, and `table.rkt`, the conversion's body) |
| `cuml-doc/` | cuml-doc | the one manual, `cuml/scribblings/cuml.scrbl` |
| `cuml-test/` | cuml-test | every Racket test, `cuml/tests/`, and the twin harness |
| `cuml/` | cuml | the meta package: cuml-lib and cuml-doc |
| `lint/` | cuml-lint | the `raco review` plugin for the test forms, `define-generics`, `define-hyperparameters`, `define/who`, `define-estimator`, macro templates and constructor-renamed structs, and the rule that an exported procedure taking `who` first, or one that only relays to its `/who` form, is a `define/who` |
| `shim/` | | the C++ shims, one shared library per family, and the test-only `libcumlrkt_probe.so` (`shim/probe`, a `TEST_ONLY` family) |
| `scripts/` | | the gate scripts, the binding census and its plan, and the repo's Resyntax suite (`resyntax-rules.rkt`, `cuml-recommendations`) |

Why several packages: a user who needs clustering installs and loads only
the clustering shim, and each family's native dependencies stay its own. A
family package depends only on base, raft and cuml-core-lib (and what it
truly needs), never on cuml-lib or another family. cuml-lib may also depend
on datasets-core, which no family or core may. `scripts/check-layout.rkt`
and `scripts/pkg-deps.sh` (`raco setup --check-pkg-deps --unused-pkg-deps`,
red on an unused dependency too) gate it. Family packages
export algorithm-specific procedures (`kmeans-fit`); bare algorithm names
(`kmeans`) are reserved for cuml-lib's estimator wrappers, and
`check-layout.rkt` holds the reserved list and checks the public modules of
the families and of cuml-core-lib against it.

## Owner rules

The racket-dev plugin's `style-rules` skill is the full list. The ones that
bite here:

1. **No explanatory comments in source.** Usage docs live in Scribble. A
   comment survives only for an invariant the code cannot show or a
   cross-boundary gotcha.
2. **Every failure is `exn:fail:cuml`**, a subtype of `exn:fail:raft`, with
   cuML's or the shim's message prefixed by the Racket caller's name and
   raft's `kind`. Shim messages never name `cuml_` entry points.
3. **Constructors are named after the type**, never `make-`; `->` for
   conversions; `define-syntax-parse-rule`, never `define-syntax-rule`;
   scoped native resources through `with-*` forms, never raw malloc/free.
4. **Contracts, not guards**, once contracts arrive; until then raft's rule:
   the shim refuses what memory safety needs, before the GPU is touched.
5. **`(only-in ...)` requires**, alphabetized; modules and C++ files at most
   500 lines.
6. **Test programs are quoted forms**, fixtures are files (`.py` twins).
7. **Every review finding gets a regression test** that fails without its fix.
8. **No lint suppressions** (`;; noqa`, `#|review: ignore|#`) in new code:
   teach `lint/review.rkt` the form instead. `#|review: ignore|#` stays on
   `info.rkt` files and pure re-export facades; the `no-noqa` gate allowlists
   the rest.

## Architecture

### The shims (`shim/`)

One CMake project; `cumlrkt_family(<family> SOURCES ... TESTS ...)` in
`shim/cmake/Cumlrkt.cmake` builds `libcumlrkt_<family>.so` (hidden
visibility, `--exclude-libs,ALL --no-undefined`, no version script: RMM's
GNU-unique symbols stay exported, rkt-raft #10), its gtest binary, and adds
both to the `tidy` target; `NO_LIBCUML` (preprocessing) links
`raftrkt::headers` only, reading cuML's headers for the version. libcuml is an IMPORTED target (cuml's own CMake
config needs Treelite and nvForest configs the prefix lacks). Shared C++
lives in `shim/common/include/cumlrkt/` (`common.hpp`: the error slot,
`check_abi`, `libcuml_release`, `on_floats`; `api.h`: `CUMLRKT_API`;
`support.hpp`: `guarded`, `on_device`, `settle`, the shape and bound checks,
`require_basis`, the bound on raft Lanczos's n·ncv basis, and
`require_lanczos_rows`, the 2(nev + 1) rows below which cuVS's Lanczos
reads past its basis, both shared by spectral clustering, UMAP's spectral
init and spectral embedding;
`graph.hpp` with `common/src/graph.cu`, linked into cluster and manifold:
the COO graph checks of spectral clustering and spectral embedding
(`checked_graph`, `require_samples`, `scan_graph`, `fault_message`,
`require_sound_graph`);
`largest_positive.hpp` with `common/src/largest_positive.cu`, linked into
manifold: `make_largest_positive`, sklearn's sign rule on a row-major matrix
in place;
`limits.hpp`: the index and squared-norm bounds; `rows.hpp` with
`common/src/rows.cu`, linked into every family: `first_large_row`,
`first_large_l1_row`, `total_spread`, `first_zero_row`, `first_non_finite`,
`first_large_element` and `first_row_unlike_first`; `support.hpp`'s `require_finite` and
`require_magnitudes` are the element scans with their messages, and
`require_unlike_first` refuses rows that all equal row 0 (the manifold gtest
binary compiles `rows.cu` itself to call it);
`transpose.hpp` with `common/src/transpose.cu`, linked into decomposition
and manifold: `to_columns` and `to_rows`, the row-major and column-major
copies, and `scaled_copy`, a copy times a power of two; `device.cuh`: the `each` loop and its device helpers). The
manifold family's `manifold_support.{hpp,cpp}` holds the scans UMAP and
t-SNE share (`float_rows`, `require_rows_fit`, `require_scanned`,
`require_converged`); `spectral_checks.hpp` holds spectral embedding's
post-solve scans, header-only so the gtests reach them, and
`spectral_residual.{hpp,cu}` the graph path's eigen-residual and weight span
(B3-y), compiled into the gtests too.

Each family exports `cuml_<family>_last_error`, `_last_error_kind`,
`_check_abi` and `_libcuml_version`, which the shared Racket loader binds by
name, plus its entry points. Each library has its own thread-local error
slot (an inline function's static, per shared object under hidden
visibility).

### Native models

A fitted model whose state cuML keeps in C++ is a `cumlrkt::model`
(`shim/common/include/cumlrkt/model.hpp`), published with `publish_model` as
a raft deposit owner. Region 0 is its anchor, a one-int32 device array
holding its `model_kind` code; the other regions are the model's arrays.

Native side:

- Each family library defines its own `model_kind` constants, a code and a
  name (the probe's are in `shim/probe/src/probe_kinds.hpp`, the cluster
  library's `hdbscan` kind in `shim/cluster/src/cluster_kinds.hpp`). A
  library's registry sees only its own models, so an anchor from another
  library reads as another array.
- Each family library exports its own release job
  (`cuml_cluster_model_release_job` for the cluster library), which a
  `native-model`'s release procedure calls with the anchor.
- raft releases the owner once, on its reaper, with the device set and the
  creating handle alive, whichever is dropped first (`rr_resources` and
  `rr_buffer` alias the handle and declare it before the owner).
- A job pins every buffer passed as a `view`, so a call takes the anchor as
  a `view` and the model it finds outlives the call.
- `use_model(anchor, name, kind, call_stream)` refuses another array,
  another kind or a released model. The use holds a shared lock and
  synchronises `call_stream` before it unlocks, so the lock covers every
  kernel the call queued against the model.
- `release()` takes the lock exclusively and calls `drop()`, which frees the
  opaque members only, on the creating handle's stream, never a published
  region.
- The registry is never destroyed (`never_destroyed<T>()` in `common.hpp`):
  raft's reaper can delete models after static destructors run at exit.
- No pointer to caller memory survives the creating call; anything the model
  reads later is copied or owned.

Racket side (`cuml/core/private/model.rkt`, a `native-model`):

- `native-model-release!` frees at once, under any resources; a second
  release returns at once while the first runs.
- The releasing thread claims the model by storing itself in the state. A
  raise rolls the claim back to live. A release that finds the claim held by
  a thread that is no longer running (killed or suspended) takes it over and
  runs the native release again, which is idempotent.
- A break during a release lets the release finish, then is delivered.
- `with-native-models` checks every binding before it holds any model and
  releases on every exit. Control that jumps back into its body (a generator
  resume) is refused, not re-acquired.
- An unreleased model is freed by the collector once the model and every
  region taken from it are gone. Its opaque bytes are phantom bytes,
  reachable from the model and from each region, so GC pressure counts them
  until the native owner can go.

### The downstream rules, inherited from rkt-raft

rkt-raft's AGENTS.md, "The frozen downstream interface" and the decisions of
L1d, #21/#32 and #22, is binding. In short: compile against rkt-raft's
`packages.rapids` and `packages.raft-dev` only (RAFT has no versioned
namespace); check the ABI tag at load (`raftrkt::require_abi` against
`raft-abi-pointer`); run each entry inside `raftrkt::translate_exceptions`
over the library's slot; select the handle's device (`device_scope`) and
refuse arrays on another (`require_on_device`); read arrays only through
`view.hpp`; allocate outputs with raft's constructors on the call's
resources, or deposit outputs sized by the call (`raftrkt::deposit`,
`with-deposits`); synchronise the handle's stream on every exit
(`sync_guard`); run every call that can run long as a job
(`raftrkt::run_job`, bound with `define-job ... #:exn exn:fail:cuml`), and
use `status-checker` (with `with-array-views #:resources`) for short calls
only.

A short (atomic) call is for sub-millisecond work only; anything longer is a
job.

Synchronise the handle's stream before `raftrkt::deposit`, so only scalar
writes follow the publish: a fault that surfaces in a later sync raises with
the outputs already published, held by the record until a collection.

### Racket

`cuml/core/private/native.rkt`'s `load-family` loads
`<family>/native-libs/libcumlrkt_<family>.so`, binds the four common entries,
builds the family's `status-checker` with `#:exn exn:fail:cuml`, runs the
ABI check and returns a `cuml-family`. A family's
`cuml/<family>/private/native.rkt` calls it at instantiation, so requiring
`cuml/<family>` performs the handshake. The library is staged into
`cuml-<family>-lib/cuml/<family>/native-libs/`: by the dev shell, by
`nix run .#copy-native-libs`, or by the collection's pre-installer
(`cuml/<family>/info.rkt`) from `$CUML_NATIVE_LIB_PATH/lib`. Staging writes
a temp file and renames it over the old library, never in place: rewriting a
library that a live process has mapped faults that process. rkt-raft's
catalog arc (rkt-raft #24) will ship raft as a `raft-x86_64-linux` natipkg;
the families will follow once it lands.

## Adding an algorithm family

- [ ] `cuml-<family>-lib/`: `info.rkt` (`collection 'multi`, deps base, raft,
      cuml-core-lib), `cuml/<family>.rkt`, `cuml/<family>/info.rkt` with
      `pre-install-collection`, `cuml/<family>/private/install-native.rkt`
      (`native-pre-installer`) and `private/native.rkt` (`load-family`).
- [ ] `shim/<family>/{include/cumlrkt/<family>.h,src,tests}` and one
      `cumlrkt_family(<family> ...)` line in `shim/CMakeLists.txt`.
- [ ] `families` in `flake.nix` (staging, packages list).
- [ ] cuml-lib: dep, `implies`, re-export in `cuml/main.rkt`.
- [ ] The hyperparameters as one `define-hyperparameters` table
      (`<algorithm>-params`), the entry points as `define/who` with
      `#:hyperparameters <algorithm>-params`; symbol-to-code tables are
      `define-codes` (-1 for an unknown name), beside it in
      `hyperparameters.rkt`.
- [ ] The estimator wrapper in `cuml-lib/cuml/private/<algorithm>-estimator.rkt`,
      one `define-estimator` over the table and the family's `/who`
      procedures (so errors name the generic), re-exported from
      `cuml/main.rkt`, its bare name in `reserved-names` in
      `scripts/check-layout.rkt`.
- [ ] The family's package in the `pkg-deps` gate (`.racket-dev.rktd`).
- [ ] A chapter in `cuml-doc` and its examples.
- [ ] Tests in `cuml-test/cuml/tests/` (GPU smoke, memory, a Python twin in
      `python/`), gtests in `shim/<family>/tests`.
- [ ] All gates green, including `gpu` and `check`.

## Build and test

```bash
nix develop                        # the GPU shell; first entry provisions
nix flake check --max-jobs 1 --cores 4   # the CI-equivalent; GPU cases print SKIP
scripts/gpu-suite.sh               # on the GPU host: gtests (+ ASAN/UBSAN, memcheck),
                                   #   LD_BIND_NOW load, raco tests with the twin, the
                                   #   census; red on a SKIP not in allowed-skips.txt
raco test --process --timeout 45 cuml-test
scripts/render-docs.sh <dir>       # the manual, red on any warning, broken or missing link
scripts/pkg-deps.sh <pkg> ...      # raco setup's dependency check, red on an unused one too
nix run --max-jobs 1 --cores 4 .#copy-native-libs   # restage the shims after a C++ change
cmake -S shim -B shim/build -G Ninja -DBUILD_TESTING=ON   # the shims' inner loop
```

`nix develop` exports `CUML_SHIM_TESTS`, `CUML_SHIM_SANITIZED_TESTS` (the
gtests built with `-DCUMLRKT_SANITIZE=ON`), `CUML_ASAN_OPTIONS`,
`CUML_UBSAN_OPTIONS`, `CUML_CUDA_DRIVER_PATH`,
`CUML_RAPIDS_PREFIX`, `CUML_NATIVE_LIB_PATH`, `RAFT_NATIVE_LIB_PATH` and
`RAFT_SHIM_PROBE` (rkt-raft's probe library, which the memory test reads the
pool through) and `CUML_SHIM_PROBE` (the test-only `libcumlrkt_probe.so`,
installed in the shim's `tests` output, which the model tests load; unset on
a GPU host, they fail rather than skip). `PLTUSERHOME` is
`~/.cache/rkt-cuml-devshell/<hash of the checkout path>/<hash of the rkt-raft
and nix/racket-tools.nix store paths>`. Every install skips what is already
installed, so a bump of either pin enters a fresh, empty home. Every shell
entry deletes everything else under this checkout's directory (old pins'
homes), and nothing outside it, so entering a shell with a new pin deletes
the home of a shell still open on the old pin in the same checkout: close
it first, or nest the new shell inside it (a nested shell deletes nothing).
Every entry also deletes the rest of `~/.cache/rkt-cuml-devshell/`, keeping
only the directories of checkouts that still exist (each records its path in
`.checkout`), so the homes of removed worktrees and of older layouts go on
the next entry from any checkout.
If hashing the checkout or the pins fails, the entry warns and deletes
nothing.
The next shell entry copies the new pin's raft tree, links raft from it,
installs raft's docs dependencies (polars, data-frame, datasets,
datasets-core, al2-test-runner, rkt-polars' racket-deps) and the tools at
the new revisions, and reinstalls Resyntax. A change of a package's
`info.rkt` or of the shim misses the install stamp inside the same home and
re-runs the install and setup.

### Build memory

The lab host shares 62 GB with other sessions and has OOM'd under parallel
builds. Run nix as `nix … --max-jobs 1 --cores 4`; the flake caps CMake at 4
jobs, nvcc at `--threads=1` and every `raco setup` (and `raco pkg install`
that runs one) at `-j 4`; check `free -g` before a heavy build and
wait below 30 GB available. Never two shim builds at once.

### Gates

native, format, no-syntax-rule, no-raw-malloc, no-noqa, gpu-suite-signals, layout,
pkg-deps, review, compile, test, bindings, sanitizers, gpu, resyntax, docs,
check: `.racket-dev.rktd`
declares their commands for the racket-dev runner
(`racket <plugin>/hooks/gate.rkt . all`), and `push-gates` is the pre-push
subset. The manual's link check (`scripts/check-doc-links.rkt`) runs only
in `docs`, which is not in `push-gates` and runs on CI only on the GPU
runner: run `docs` before any push that touches the scribblings.
`pkg-deps` skips cuml-doc's `racket-doc` and `threading-doc` (`doc_only`:
under `--no-docs` setup cannot see the manual's links), so it never
reports either as unused; drop one by hand when the manual stops linking
into it. `bindings` runs the census `scripts/check-bindings.rkt`, per family (its `families` list):
every `cuml_<family>_` export of the staged library is bound once (a
`_job`'s body excepted), every binding is exported, and
`scripts/<family>-bindings.rktd`, keyed by issue, plans each export; a leg
flips its planned entries to due.
`sanitizers` builds the `shim-sanitizers` flake check (ASAN and UBSAN; without
a GPU only Abi and Version run, and the GPU suite runs the sanitized gtests
on the device). An LSan leak rooted only in libcuml or libcuda frames goes in
`shim/sanitizers/lsan.supp` with its reason in the commit; never a
libcumlrkt frame. When that file exists, the check's ctest and the GPU
suite's sanitized gtests run with `LSAN_OPTIONS=suppressions=<it>`, and
`gpu-suite-signals` pins that the suite passes it. `.fmt.rkt` uses fmt's internal document model (`fmt/common`,
`fmt/core`), which fmt documents as unstable: re-check `format-fun` and
`format-hash-pairs` when `nix/racket-tools.nix` moves to a new fmt.
`scripts/check-layout.rkt` uses `threading`; `nix/racket-tools.nix` pins
`threading-lib` beside fmt and review, and every Racket home the flake builds
installs it from there first. rkt-polars' `threading-doc` builds against
that pin, so the `layout` check fails unless both `threading-lib` trees
have the same version and the same sources.

`resyntax` (`scripts/resyntax.sh`) runs `scripts/resyntax-rules-test.rkt`,
then Resyntax with the `cuml-recommendations` suite (the default
recommendations and `cuml-rules`) against the merge base with the PR's base
branch, which it asks `gh` for. `RESYNTAX_BASE=<ref>` names the base
instead, for a stacked branch with no PR yet or no `gh`:
`RESYNTAX_BASE=origin/cluster/kmeans-6 scripts/resyntax.sh`. Resyntax does
not visit a module's top level as a body, so `who-pair-to-define/who` fixes
pairs in internal definition contexts and the `review` lint refuses them at
module level.

### Test support

raft has no public test support yet, so the tests import raft internals;
replace each when raft offers a public form:

| Module | Bindings | Used by |
|---|---|---|
| `raft/private/array` | `buffer->device-array`, `device-array-buffer` (a strided view) | `private/scaler-data` |
| `raft/private/resource` | `with-release` | `loader-test`, `device-memory-test`, `private/device-memory`, `private/program`, `private/python-env` |
| `raft/private/foreign/memory` | `buffer-drop-count`, `release-failure-count`, `resources-drop-count` | `model-test` (`release-failure-count`, `resources-drop-count`), `model-memory-test`, `memory-test`, `kmeans-memory-test`, `kmeans-estimator-memory-test`, `dbscan-memory-test`, `linkage-memory-test`, `spectral-memory-test`, `metrics-memory-test`, `standard-scaler-memory-test`, `pca-memory-test`, `tsvd-memory-test`, `private/process-memory` |
| `raft/private/job` | `executor-threads`, `jobs-live` | `model-memory-test`, `memory-test`, `kmeans-memory-test`, `kmeans-estimator-memory-test`, `dbscan-memory-test`, `linkage-memory-test`, `spectral-memory-test`, `metrics-memory-test`, `standard-scaler-memory-test`, `pca-memory-test`, `tsvd-memory-test`, `private/process-memory` |
| `raft/tests/private/collect` | `collect-until`, `drain-finalizers!`, `settled-count` | `model-test`, `model-memory-test`, `memory-test`, `kmeans-memory-test`, `kmeans-estimator-memory-test`, `dbscan-memory-test`, `linkage-memory-test`, `spectral-memory-test`, `metrics-memory-test`, `standard-scaler-memory-test`, `standard-scaler-refusal-test`, `pca-memory-test`, `pca-refusal-test`, `tsvd-memory-test`, `tsvd-refusal-test`, `umap-refusal-test`, `tsne-refusal-test`, `spectral-embedding-refusal-test`, `table-refusal-test`, `private/process-memory` |
| `raft/tests/private/probe` | `probe-pool-used`, `test-pools` (needs `RAFT_SHIM_PROBE`) | `private/process-memory`, `model-memory-test`, `standard-scaler-refusal-test`, `pca-refusal-test`, `tsvd-refusal-test`, `umap-refusal-test`, `tsne-refusal-test`, `spectral-embedding-refusal-test`, `table-refusal-test` |
| `raft/tests/private/deposits` | `filled` | `model-test` |
| `raft/tests/private/gpu` | `test-gpu`, `gpu-skip`, `test-unless-skipped` | `smoke-test`, `device-memory-test`, `docs-test`, `guide-docs-test`, `guide-decomposition-test`, `guide-plots-test`, `guide-twin-test`, the `kmeans-*`, `dbscan-*`, `linkage-*`, `agglomerative-*`, `spectral-*`, `hdbscan-*`, `table-*`, `metrics-*`, `silhouette-*`, `trustworthiness-*`, `standard-scaler-*`, `pca-*`, `tsvd-*`, `umap-*` and `tsne-*` tests, `preprocessing-docs-test`, `decomposition-docs-test`, `manifold-docs-test`, `private/process-memory`, `private/python-env`, `private/model-probe` |

The repo's own test support, in `cuml-test/cuml/tests/private/`:

| Module | Bindings |
|---|---|
| `cuml-error` | `check-cuml-error kind message-or-regexp thunk`, `raised` |
| `process-memory` | `pool-sample`, `baseline-sample`, `test-process-memory`, `test-rounds-and-refusals` (one fit round, its refusals, the buffers it frees), `device-slack`, ... (the memory tests' sampling) |
| `device-memory` | `process-device-bytes`, `no-process-memory-reason`, `process-memory-skip` |
| `kmeans-data` | the k-means fixtures (the grids, blobs, host conversions, weights, blank arrays) and tolerances (`tolerance`, `close?`, `scaled-close?`) |
| `dbscan-data` | the DBSCAN fixtures (the hand grid, seeded integer grids, distance matrices, `repeat-rows`, the cosine margin generator), the host oracle `dbscan-oracle`, `canonical`, `same-partition?`, `pick` (the elements at a list of indices) and `fitted-lists` (`dbscan-fit` composed with `result->lists`) |
| `linkage-data` | `connectivities`, the single-linkage fixtures (the hand grids, the tie line, `group-rows`, `two-far`), the host oracles (`linkage-oracle`: a Prim minimum spanning tree, the minimax cophenetic matrix, `oracle-bounds` and `oracle-partition` for the cuts), `children->labels`, `children-heights`, `children-cophenetic`, `valid-dendrogram-problems`, `partition-refines?` and `linked-lists` (`single-linkage-fit` composed with `result->lists`) |
| `spectral-data` | the spectral fixtures (`blobs`, `circles`, `moons` in float32, `sparse-blocks`, `cliques`, `repeat-each`, and the shared `three-blobs`, `three-blobs-matrix` and `three-blocks`), the `fit-case` struct (a named fixture with its truth, k, options and comparison) and its constructor `planted` (options default to none, comparison to `'labels`), the `graph` struct, `unit-path n`, `unit-rings count n`, `unit-rings-of sizes` and `unit-star n` (unit-weight paths, disjoint rings of equal or given sizes, and stars), and its transforms (`knn-graph`, the host kNN connectivity graph; `permute-graph`, `graph-shuffle`, `graph-map-vals`, `graph-edit` with `set-entry`, `graph-upper`, `graph-scale-upper`, `graph-dense`), `graph->vectors` and `edited-vectors` (`graph-edit`, then `graph->vectors`), `planted?`, and `labels` and `graph-labels` (the two fits composed with the labels' host list) |
| `model-probe` | the probe library (`CUML_SHIM_PROBE`) and its jobs: `probe-model` (a native model of `#:kind` with a payload and a region), `probe-touch` (a use, optionally held on a gate), `probe-release`, `probe-counts` (created, released, destroyed, stale), `with-gate`, `await-waiter`, `test-model-probe` (a missing library raises `exn:fail:cuml`, so the tests fail rather than skip) |
| `refusal-rows` | the refusal tables' `row` struct (id, message, call), `check-row name row`, and the squared-norm cases' `with-row` and `norm-message` |
| `estimator-data` | `apply-options proc options arg ...`: `proc` called with a symbol-keyed hash as keywords; `round-trips constructor changed`: each parameter of `(constructor)` built alone from `changed`, read back; `printed v` (its `~a`, `~s` and `~v` forms); `listed v` (a device vector or matrix as host lists, `#f` as is) |
| `hdbscan-data` | the HDBSCAN fixtures (`dataset` by name from `dataset-names`, the seed-0 numpy normals in `normals-0.rktd` behind b50, b10 and one-blob, `dup`), `fitted-of` and `fit-lists` (a fit as host lists in a `fitted` struct), `same-fit?` and `persistence-close?` (equal but for persistence within 1e-5), and the host oracles `structure-problems` and `descent-problems` |
| `program` | `run-program program`: `program`, a quoted module form, run in a fresh `racket`; returns its exit status, stdout and stderr (used by `program-test`, `dbscan-refusal-test`, `spectral-law-test`, `hdbscan-refusal-test`, `hdbscan-law-test`, `table-test`, `load-test`, `pca-transform-law-test`, `private/umap-determinism-data`, `private/tsne-determinism-data`) |
| `table-data` | `synthetic-table names columns` (a `dataset-table` built by reflection: datasets-core has no public constructor, so this depends on its struct staying `#:transparent`), `numeric-selections` (each bundled dataset with its all-real columns, as `selection` structs), `host-rows`, and the edge values `edge-table` (float64) and `edge-table32` |
| `metrics-data` | the metrics fixtures (seeded `labels` and `label-pair`, `labels-of`, the `overlap` blobs and their `overlap-labels`, `three-groups`, the tie-free `projection`, `grid-20`), the relabellings (`monotone-maps`, `bijection`), `permute` and `random-permutation`, and the host oracles `sorted-distinct` and `host-counts`, and `mean` and `median` |
| `pca-data` | the PCA fixtures (`fixtures`, each a `fixture` of a name and float64 rows, among them `normal-fixture`, `integer-fixture` and `wide-fixture`; `fixture-rows` and `fixture-matrix` narrow to a dtype, optionally keeping the first rows; `axis-rows`, `tie-rows`, `constant-column`, `duplicate-rows`), `fitted-of` (a result's arrays as host lists in a `fitted` struct), `dot`, `norm`, `column`, `gaps`, `first-max-entries`, `orthonormality-bound` (L4: 4·d·u full, 8·d·u Jacobi), and the host oracles `host-project`, `host-reconstruct` and `host-gram` |
| `pca-twin-data` | the PCA twin's fixture payload (`twin-rows`, at most 40k elements; `twin-key`, `twin-ks`, `twin-case`, `twin-results`), `named` (a fixture by name), `fit-ours` (a fit as an `ours` struct), `l14-problems` (the spec's L14 bounds, the failures as strings), and `check-against-sklearn` and `check-against-cuml`, one fixture's twin cases (the twin tests split by fixture to stay under the 45 s file timeout) |
| `tsvd-data` | the truncated SVD fixtures (`normal-fixture`, `offset-fixture`, `duplicate-fixture`, `wide-fixture`, `zeros-fixture`, `twos-fixture`, `tie-fixture`, `law-fixtures`, `normal-rows-of`, and `householder-fixtures` with known spectra, each an `exact`), `tsvd-of` (a result's arrays as host lists in an `svd` struct), `exact-reference` (the float64 SVD's `reference`), `accuracy-problems` (L8's bounds, the failures as strings), `sign-matched` (a reference with each component's sign matched to a fit's, for the tie fixture), and the host oracles `host-transform`, `popvars` and `squared-norm` |
| `tsvd-twin-data` | the truncated SVD twin's cases (`twin-cases`, each a `twin-case` of at most 40k elements; `twin-results`, one Python run for a list of ops), `theirs-svd` and `theirs-reference` (a twin result as an `svd` or a `reference`), `float-bits`, and `ours` (our fit, transform and inverse of a case) |
| `scaler-data` | the standard-scaler fixtures (`normal-rows`, `generated-rows`, `dyadic-rows`, `constant-rows`, `rows->matrix`, `rows-of` (a flat list in rows of a width), `strided-vector`), `ulp`, `bits-equal?`, `unit-roundoff` (and `float32-roundoff`, its float32 value), `vector-shape` and `matrix-shape`, `round-to`, `standardisation-errors`, and the host oracles `host-transform` and `host-inverse` (float32 emulated with `flsingle`) |
| `umap-data` | the UMAP fixtures (each a `fixture` of a name, float64 rows and labels: `blobs`, `far-blobs`, `overlap`, `tiny`, and the interleaved split `held-out-fit` / `held-out-new`), `digits` (sklearn's digits from the twin; twin tests only), `tail-split` (a fixture's first 80% and last 20%) and `jittered fixture seed` (each entry plus a uniform draw below 0.01, so neighbour distances do not tie), `fixture-X` (float32 matrix), `fixture-y` (int32 labels), `relabel`, `finite-rows?`, `twin-case fixture result` (a Python cuML case for `python/umap_twin.py`, from a fit's parameters and curve), `ours fixture new options` (an `outcome` of the fit, its embedding and the transform of `new`, with `options` a symbol-keyed hash through `apply-options`), `case-of` and `check-parity fit new configs [#:transform?]` (L6: each config's fit, and transform unless `#f`, bitwise equal to Python cuML's), `check-quality init fits held new` (L7: each fit's and the held-out transform's trustworthiness within Python cuML's 20-seed spread), and `check-within name ours spread` (that spread check, shared with `tsne-data`) |
| `umap-determinism-data` | L5's quoted program over `umap-data`'s fixtures: `narrow-configs metrics` (`far-blobs`, both inits, n-components 2, 3 and 16, with and without y), `wide-configs init` (600 components on `held-out-fit`, 30 epochs) `digest` (a matrix's float32 SHA-1, required by the quoted programs) and `check-repeatable configs [#:program]` (3 runs in one process, then 2 more processes; used by the five `umap-determinism-*test`s and, with its own program, by `tsne-determinism-data`) |
| `tsne-determinism-data` | L4's quoted program over `umap-data`'s `far-blobs` and `digits`: `check-tsne-repeatable name metrics` (both inits; embedding digest, KL and n-iter equal in and across processes) |
| `tsne-data` | `wide-blobs` (12000×50, five centres), `check-small-sweep method init` (L3: n = 2 to 40, finite or the divergence refusal), `embedded`, `derived-neighbours`, `tsne-case fixture result` (a case for `python/tsne_twin.py`), `check-tsne-parity fixture configs` (L5, L13: bitwise embedding, KL and n-iter against Python cuML) and `check-quality method fixtures` (L6: trustworthiness within Python cuML's 20-seed spread) and `check-sklearn fixtures` (L6: exact within 0.05 of sklearn's exact t-SNE) |
| `spectral-embedding-data` | `stochastic-blocks sizes p-in p-out seed [#:weighted?]` (a seeded planted-block `graph`, symmetric and sorted, and its truth), `stochastic-graph` (its graph at 0.3/0.01), `heavy-pair exponent [#:graph] [#:between]` (a graph plus one pair at 2^exponent), `polygons count m` (count identical regular m-gons 1000 apart, as rows), `quantised`, `rows-matrix`, `embedded` (a result's embedding as host rows), `embedded-graph g [options]` (`-fit-graph` on a `graph`, options through `apply-options`), `sign-ruled?`, `eigen-residuals g rows [#:norm-laplacian]` (per column, the shim's B3-y residual computed on the host) and `residual-bound`, `column-gaps` (per column, max \|E − S\| / max \|S\|), `cosine-gaps`, `embedding-case` and `graph-case` (cases for `python/spectral_embedding_twin.py`), `embedding-checks` and `graph-checks` (L5 entries: a label, a case and a bitwise check against Python cuML), `check-twins entries` (every entry in one Python run), and `twin` |
| `spectral-embedding-determinism-data` | L4's quoted program over `blobs`, `digits`, `sbm` and `three-blocks`: `dense-configs name` (10 and 256 neighbours, drop-first × norm-laplacian), `graph-configs name` and `check-spectral-embedding-repeatable configs` (`check-repeatable` with that program) |
| `spectral-embedding-twin-data` | `parity-configs [n-neighbors]`: n-components 2 and 8 × drop-first × norm-laplacian, shared by the L5 twin tests |
| `manifold-refusal-data` | the refusal inputs UMAP and t-SNE share: `tiny-with cells` (`tiny` with cells replaced), `nan-row`, `large-row`, `zero-row`, their messages `nan-message`, `large-message`, `cosine-message`, and `listed-X` (a host list, for the view check) |
| `blob-data` | `uniform-source seed` (a seeded uniform [0, 1) thunk on its own generator), `normal-source seed` (a seeded Box–Muller standard-normal thunk over it) and `centred-blobs centres per-group seed` (float32 Gaussian rows around each centre, and their group labels) |
| `guide-data` | `evaluated-chapter name` (a guide example chapter's `examples` forms evaluated in order in a `make-cuml-eval` sandbox that shares `cuml`, `pict` and `raft` with the test; returns the evaluator and each block's values), `example-blocks` (the forms of each `examples` block of a read document), `read-document` (a `.scrbl` file read as a datum), `flattened-text` (a file's text with each whitespace run one space), `check-scored ids labels columns text` (a clustering's sorted table, its ARI against `table-ari`, both rounded to `text`), `mismatches` (labels that differ from the renumbered classes), `scaled-matrix` (a dataset's columns standardized on the device), `table-ari` (the exact rational adjusted Rand index of a host table), `cluster-columns` (a contingency table's columns, sorted, so a renumbering moves no pin), `host` (a device vector's first n values), `boundary-margin` (a number's distance to the nearest rounding boundary at some places); the guide's own `class-ids`, `counts-of` and `rounded` come from `guide/helpers.rkt` |
| `python-env` | `run-twin script input` (in.json to out.json), `run-cases script cases` (`run-twin` of `(hasheq 'cases cases)`, its `results`), `test-twin` (probes `import cuml` on first use, not at load), `twin-params theirs unbound` (a twin's parameters under Racket names, without `verbose`, `output-type` and `unbound`) |

`hyperparameters-test` and `estimator-support-test` define their own tables
(`widget-params`, `gadget-params`) and estimators to test the macros apart
from k-means; `hyperparameters-doc-test` reads the `defproc`s of
`reference/{clustering,spectral,hdbscan,preprocessing,decomposition,manifold,estimators}.scrbl` and checks them against
the tables.

Every SKIP line the GPU suite allows matches one ERE in
`scripts/allowed-skips.txt`; a new kind of SKIP is red until a line there
names it, and `device-memory-test` pins that the MIG line matches and the
no-GPU and no-twin lines do not.

The Python twins run with `LD_LIBRARY_PATH` set to the dev shell's driver
farm only, so no host CUDA library shadows the ones the wheels were patched
against. `import cuml` is probed from the temp directory with
`PYTHONSAFEPATH`: `python3 -c` puts the working directory on `sys.path`,
where a same-named directory would shadow the wheel and answer yes with
nothing installed.

The memory test measures this process's device memory through NVML
(`nvmlDeviceGetComputeRunningProcesses_v3` for the CUDA device's UUID,
filtered by pid), not `cuMemGetInfo`, which counts every process on the GPU.
NVML reports it at once, in 2 MiB pages, with about 262 MiB of context;
the blobs loop grows it by one 32 MiB step, hence the 128 MiB slack.
Where NVML lists no device for the CUDA UUID (a MIG instance), both memory
tests SKIP. The device-memory test's holder process sets
`PR_SET_PDEATHSIG`, so it dies with the test even when `raco test --timeout`
kills the test outright.

## Documentation

- One manual, `cuml-doc/cuml/scribblings/cuml.scrbl`. It assumes a catalog
  install, shows Racket code only, and says nothing of Nix, shims, the ABI,
  legs, future work or Python.
- The landing page: what the library binds, then the bold License,
  Acknowledgements and AI Disclosure paragraphs, the install line, the table
  of contents. No empty chapters.
- Two parts, after glmnet's: the guide (`guide.scrbl`, `guide/`: Getting
  started, Concepts with every `deftech` and the hyperparameter rule in
  `concepts-parameters`, and one `ex-*` chapter per
  algorithm or family on real `datasets-core` data) and the reference
  (`reference.scrbl`, `reference/`: the entries, on small synthetic data).
  `datasets-core` is a dependency of cuml-lib.
- The guide's helpers are one module, `guide/helpers.rkt`: Getting started
  shows its `require`, Examples its definitions, every guide sandbox
  evaluates both (`utils.rkt` reads them from the file), and the guide
  tests require the module, so a helper a chapter uses is never copied
  into a test.
- The guide's figures are picts from those helpers (`plot/no-gui`'s
  `plot-pict`), drawn live by the examples: no committed images.
  `guide-plots-test.rkt` pins the colours by label id, the sizes, and every
  count a figure's prose names; `guide-twin-test.rkt` checks the
  Decomposition chapter's printed values against an sklearn twin.
- Every exported name gets a Scribble entry with live examples
  (`scribble/example`, never pasted output), each pinned in
  `cuml-test/cuml/tests/docs-test.rkt` (the reference;
  `preprocessing-docs-test.rkt` for `reference/preprocessing.scrbl`,
  `decomposition-docs-test.rkt` for PCA in `reference/decomposition.scrbl`
  and `reference/pca-estimator.scrbl`, `tsvd-docs-test.rkt` for its
  truncated SVD section and `reference/truncated-svd-estimator.scrbl`,
  `manifold-docs-test.rkt` for `reference/manifold.scrbl`) or
  `guide-docs-test.rkt` (the guide; `guide-decomposition-test.rkt` for the
  Decomposition chapter's k-means, which evaluates the chapter's own
  examples in order in a sandbox (`evaluated-chapter`) and pins their
  printed scores, as `guide-plots-test.rkt` does for its figure).

## CI

`.github/workflows/nix.yml` (lint, `nix flake check`, Resyntax) and
`gpu.yml` (the GPU suite and the manual on the self-hosted runner, gated on
`vars.GPU_RUNNER`), with the GitHub Actions nix cache; actionlint checks them
in `format`. No GPU runner is registered (#1), so the GPU suite runs locally:
before a PR merges, run the `gpu` gate on a GPU host and give its result in
the PR.
