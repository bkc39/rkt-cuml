# AGENTS.md

Canonical agent guidance for rkt-cuml. Keep it current in the same change as
the code.

## Project overview

Racket bindings to NVIDIA cuML (RAPIDS 26.08), built on
[rkt-raft](https://github.com/bkc39/rkt-raft): cuML runs on raft's device
arrays, resources and jobs, and its errors are raft's. cuML is GPU-only.
Scope for now: clustering (k-means, DBSCAN, single-linkage agglomerative,
spectral, HDBSCAN). `plans/layout.md` holds the owner's decisions;
`plans/starting-notes.md` what rkt-raft established.

## Layout

A multi-package repository in typed-racket's style: every package is
`(define collection 'multi)` and adds modules to the one `cuml/` collection.

| Path | Package | Holds |
|---|---|---|
| `cuml-core-lib/` | cuml-core-lib | `cuml/core` (`exn:fail:cuml`), `cuml/estimator` (the estimator generics, loading nothing native), `cuml/core/private/` (the shared loader and staging) |
| `cuml-cluster-lib/` | cuml-cluster-lib | `cuml/cluster` (k-means: `kmeans-fit`, `kmeans-predict`, `kmeans-transform`, `kmeans-score`), `cuml/cluster/private/` (the bodies, their `/who` forms for the estimator wrappers, and the loader of `libcumlrkt_cluster.so`) |
| `cuml-lib/` | cuml-lib | `cuml/main.rkt`, re-exporting `cuml/core`, `cuml/estimator` and every family; `cuml/private/` (the estimator wrappers: `kmeans`) |
| `cuml-doc/` | cuml-doc | the one manual, `cuml/scribblings/cuml.scrbl` |
| `cuml-test/` | cuml-test | every Racket test, `cuml/tests/`, and the twin harness |
| `cuml/` | cuml | the meta package: cuml-lib and cuml-doc |
| `lint/` | cuml-lint | the `raco review` plugin for the test forms |
| `shim/` | | the C++ shims, one shared library per family |
| `scripts/` | | the gate scripts, the binding census and its plan |

Why several packages: a user who needs clustering installs and loads only
the clustering shim, and each family's native dependencies stay its own. A
family package depends only on base, raft and cuml-core-lib (and what it
truly needs), never on cuml-lib or another family; `scripts/check-layout.rkt`
and `raco setup --check-pkg-deps --unused-pkg-deps` gate it. Family packages
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

## Architecture

### The shims (`shim/`)

One CMake project; `cumlrkt_family(<family> SOURCES ... TESTS ...)` in
`shim/cmake/Cumlrkt.cmake` builds `libcumlrkt_<family>.so` (hidden
visibility, `--exclude-libs,ALL --no-undefined`, no version script: RMM's
GNU-unique symbols stay exported, rkt-raft #10), its gtest binary, and adds
both to the `tidy` target. libcuml is an IMPORTED target (cuml's own CMake
config needs Treelite and nvForest configs the prefix lacks). Shared C++
lives in `shim/common/include/cumlrkt/` (`common.hpp`: the error slot,
`check_abi`, `libcuml_release`, `on_floats`; `api.h`: `CUMLRKT_API`).

Each family exports `cuml_<family>_last_error`, `_last_error_kind`,
`_check_abi` and `_libcuml_version`, which the shared Racket loader binds by
name, plus its entry points. Each library has its own thread-local error
slot (an inline function's static, per shared object under hidden
visibility).

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
- [ ] The estimator wrapper in `cuml-lib/cuml/private/<algorithm>-estimator.rkt`
      (the `cuml/estimator` generics over the family's `/who` procedures,
      so errors name the generic), re-exported from `cuml/main.rkt`, its
      bare name in `reserved-names` in `scripts/check-layout.rkt`.
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
scripts/render-docs.sh <dir>       # the manual, red on any warning or broken link
nix run --max-jobs 1 --cores 4 .#copy-native-libs   # restage the shims after a C++ change
cmake -S shim -B shim/build -G Ninja -DBUILD_TESTING=ON   # the shims' inner loop
```

`nix develop` exports `CUML_SHIM_TESTS`, `CUML_SHIM_SANITIZED_TESTS` (the
gtests built with `-DCUMLRKT_SANITIZE=ON`), `CUML_ASAN_OPTIONS`,
`CUML_UBSAN_OPTIONS`, `CUML_CUDA_DRIVER_PATH`,
`CUML_RAPIDS_PREFIX`, `CUML_NATIVE_LIB_PATH`, `RAFT_NATIVE_LIB_PATH` and
`RAFT_SHIM_PROBE` (rkt-raft's probe library, which the memory test reads the
pool through). `PLTUSERHOME` is
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

native, format, no-syntax-rule, no-raw-malloc, gpu-suite-signals, layout,
pkg-deps, review, compile, test, bindings, sanitizers, gpu, resyntax, docs,
check: `.racket-dev.rktd`
declares their commands for the racket-dev runner
(`racket <plugin>/hooks/gate.rkt . all`), and `push-gates` is the pre-push
subset. `bindings` runs the census `scripts/check-bindings.rkt`: every
`cuml_cluster_` export of the staged library is bound once (a `_job`'s body
excepted), every binding is exported, and `scripts/cluster-bindings.rktd`,
keyed by issue, plans each export; a leg flips its planned entries to due.
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

### Test support

raft has no public test support yet, so the tests import raft internals;
replace each when raft offers a public form:

| Module | Bindings | Used by |
|---|---|---|
| `raft/private/resource` | `with-release` | `loader-test`, `device-memory-test`, `private/device-memory`, `private/python-env` |
| `raft/private/foreign/memory` | `buffer-drop-count`, `release-failure-count` | `memory-test`, `kmeans-memory-test`, `kmeans-estimator-memory-test` |
| `raft/private/job` | `executor-threads`, `jobs-live` | `memory-test`, `kmeans-memory-test`, `kmeans-estimator-memory-test` |
| `raft/tests/private/collect` | `collect-until`, `drain-finalizers!`, `settled-count` | `memory-test`, `kmeans-memory-test`, `kmeans-estimator-memory-test`, `private/process-memory` |
| `raft/tests/private/probe` | `probe-pool-used`, `test-pools` (needs `RAFT_SHIM_PROBE`) | `private/process-memory` |
| `raft/tests/private/gpu` | `test-gpu`, `gpu-skip`, `test-unless-skipped` | `smoke-test`, `device-memory-test`, `docs-test`, the `kmeans-*` tests, `private/process-memory`, `private/python-env` |

The repo's own test support, in `cuml-test/cuml/tests/private/`:

| Module | Bindings |
|---|---|
| `cuml-error` | `check-cuml-error kind message-or-regexp thunk`, `raised` |
| `process-memory` | `pool-sample`, `baseline-sample`, `test-process-memory`, `device-slack`, ... (the memory tests' sampling) |
| `device-memory` | `process-device-bytes`, `no-process-memory-reason`, `process-memory-skip` |
| `kmeans-data` | the k-means fixtures (the grids, blobs, host conversions, weights, blank arrays) and tolerances (`tolerance`, `close?`, `scaled-close?`) |
| `python-env` | `run-twin script input` (in.json to out.json), `test-twin` |

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
- Every exported name gets a Scribble entry with live examples
  (`scribble/example`, never pasted output), each pinned in
  `cuml-test/cuml/tests/docs-test.rkt`.

## CI

`.github/workflows/nix.yml` (lint, `nix flake check`, Resyntax) and
`gpu.yml` (the GPU suite and the manual on the self-hosted runner, gated on
`vars.GPU_RUNNER`), with the GitHub Actions nix cache; actionlint checks them
in `format`. No GPU runner is registered (#1), so the GPU suite runs locally:
before a PR merges, run the `gpu` gate on a GPU host and give its result in
the PR.
