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
| `cuml-core-lib/` | cuml-core-lib | `cuml/core` (`exn:fail:cuml`), `cuml/core/private/` (the shared loader and staging) |
| `cuml-cluster-lib/` | cuml-cluster-lib | `cuml/cluster` (public, empty for now), `cuml/cluster/private/` (the loader of `libcumlrkt_cluster.so`) |
| `cuml-lib/` | cuml-lib | `cuml/main.rkt`, re-exporting `cuml/core` and every family |
| `cuml-doc/` | cuml-doc | the one manual, `cuml/scribblings/cuml.scrbl` |
| `cuml-test/` | cuml-test | every Racket test, `cuml/tests/`, and the twin harness |
| `cuml/` | cuml | the meta package: cuml-lib and cuml-doc |
| `lint/` | cuml-lint | the `raco review` plugin for the test forms |
| `shim/` | | the C++ shims, one shared library per family |
| `scripts/` | | the gate scripts |

Why several packages: a user who needs clustering installs and loads only
the clustering shim, and each family's native dependencies stay its own. A
family package depends only on base, raft and cuml-core-lib (and what it
truly needs), never on cuml-lib or another family; `scripts/check-layout.rkt`
and `raco setup --check-pkg-deps --unused-pkg-deps` gate it. Family packages
export algorithm-specific procedures (`kmeans-fit`); bare algorithm names
(`kmeans`) are reserved for cuml-lib's estimator wrappers, and
`check-layout.rkt` holds the reserved list.

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
- [ ] cuml-lib: dep, `implies`, re-export in `cuml/main.rkt`; the estimator
      wrapper later, its bare name added to `reserved-names` in
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
scripts/gpu-suite.sh               # on the GPU host: gtests (+ memcheck), LD_BIND_NOW
                                   #   load, raco tests with the twin; red on any SKIP
raco test --process --timeout 45 cuml-test
scripts/render-docs.sh <dir>       # the manual, red on any warning or broken link
nix run --max-jobs 1 --cores 4 .#copy-native-libs   # restage the shims after a C++ change
cmake -S shim -B shim/build -G Ninja -DBUILD_TESTING=ON   # the shims' inner loop
```

`nix develop` exports `CUML_SHIM_TESTS`, `CUML_CUDA_DRIVER_PATH`,
`CUML_RAPIDS_PREFIX`, `CUML_NATIVE_LIB_PATH`, `RAFT_NATIVE_LIB_PATH` and
`RAFT_SHIM_PROBE` (rkt-raft's probe library, which the memory test reads the
pool through). `PLTUSERHOME` is
`~/.cache/rkt-cuml-devshell/<hash of the checkout path and the rkt-raft store
path>`, so a bump of the rkt-raft pin enters a fresh, empty home: the next
shell entry copies the new pin's tree, links raft from it, installs raft's
docs dependencies (polars, data-frame, datasets, datasets-core,
al2-test-runner, rkt-polars' racket-deps) at the new pin's revisions, and
reinstalls the tools and Resyntax.
The old pin's home stays behind unused; deleting it only reclaims disk. A
change of a package's `info.rkt` or of the shim misses the install stamp
inside the same home and re-runs the install and setup.

### Build memory

The lab host shares 62 GB with other sessions and has OOM'd under parallel
builds. Run nix as `nix … --max-jobs 1 --cores 4`; the flake caps CMake at 4
jobs, nvcc at `--threads=1` and every `raco setup` (and `raco pkg install`
that runs one) at `-j 4`; check `free -g` before a heavy build and
wait below 30 GB available. Never two shim builds at once.

### Gates

native, format, no-syntax-rule, no-raw-malloc, gpu-suite-signals, layout,
pkg-deps, review, compile, test, gpu, resyntax, docs, check: `.racket-dev.rktd`
declares their commands for the racket-dev runner
(`racket <plugin>/hooks/gate.rkt . all`), and `push-gates` is the pre-push
subset. `.fmt.rkt` uses fmt's internal document model (`fmt/common`,
`fmt/core`), which fmt documents as unstable: re-check `format-fun` and
`format-hash-pairs` when `nix/racket-tools.nix` moves to a new fmt.
`scripts/check-layout.rkt` uses `threading`; `nix/racket-tools.nix` pins
`threading-lib` beside fmt and review, and every Racket home the flake builds
installs it from there first.

### Test support

raft has no public test support yet, so the tests import raft internals;
replace each when raft offers a public form:

| Module | Bindings | Used by |
|---|---|---|
| `raft/private/resource` | `with-release` | `loader-test`, `private/device-memory`, `private/python-env` |
| `raft/private/foreign/memory` | `buffer-drop-count`, `release-failure-count` | `memory-test` |
| `raft/private/job` | `executor-threads`, `jobs-live` | `memory-test` |
| `raft/tests/private/collect` | `collect-until`, `drain-finalizers!`, `settled-count` | `memory-test` |
| `raft/tests/private/probe` | `probe-pool-used`, `test-pools` (needs `RAFT_SHIM_PROBE`) | `memory-test` |
| `raft/tests/private/gpu` | `test-gpu`, `gpu-skip`, `test-unless-skipped` | `smoke-test`, `memory-test` (through `test-pools`), `private/python-env` |

The Python twins run with `LD_LIBRARY_PATH` set to the dev shell's driver
farm only, so no host CUDA library shadows the ones the wheels were patched
against. `import cuml` is probed from the temp directory with
`PYTHONSAFEPATH`: `python3 -c` puts the working directory on `sys.path`,
where a same-named directory would shadow the wheel and answer yes with
nothing installed.

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
`vars.GPU_RUNNER`), with rkt-raft's tailnet cache action. The repository has
no remote yet, so neither has run; actionlint checks them in `format`.
