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

### Racket

`cuml/core/private/native.rkt`'s `load-family` loads
`<family>/native-libs/libcumlrkt_<family>.so`, binds the four common entries,
builds the family's `status-checker` with `#:exn exn:fail:cuml`, runs the
ABI check and returns a `cuml-family`. A family's
`cuml/<family>/private/native.rkt` calls it at instantiation, so requiring
`cuml/<family>` performs the handshake. The library is staged into
`cuml-<family>-lib/cuml/<family>/native-libs/`: by the dev shell, by
`nix run .#copy-native-libs`, or by the collection's pre-installer
(`cuml/<family>/info.rkt`) from `$CUML_NATIVE_LIB_PATH/lib`. rkt-raft's
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

`nix develop`, from the checkout's root: filters `/usr/local/cuda*` off
`LD_LIBRARY_PATH` and links the driver into `.cuda-driver/` (rkt-raft's
hook); stages the shims; gives the checkout its own `PLTUSERHOME` under
`~/.cache/rkt-cuml-devshell/`; installs rkt-raft's docs dependencies, raft
(linked from a writable copy of the rkt-raft tree, whose manual reads files
outside `raft/`, rkt-raft #23; its docs are built, with rkt-raft's
`packages.kmeans-canary` for their examples, so the manual links to them) and the cuml packages (linked), and the pinned Resyntax, raco fmt,
raco review and cuml-lint. It exports `CUML_SHIM_TESTS`,
`CUML_CUDA_DRIVER_PATH`, `CUML_RAPIDS_PREFIX`, `CUML_NATIVE_LIB_PATH` and
`RAFT_NATIVE_LIB_PATH`, and provides rkt-raft's `python-twins` (cuml-cu13
26.8, scikit-learn). After a change of the rkt-raft pin, delete the
checkout's `PLTUSERHOME`.

### Build memory

The lab host shares 62 GB with other sessions and has OOM'd under parallel
builds. Run nix as `nix … --max-jobs 1 --cores 4`; the flake caps CMake at 4
jobs and nvcc at `--threads=1`; check `free -g` before a heavy build and
wait below 30 GB available. Never two shim builds at once.

### Gates

`.racket-dev.rktd` declares them for the racket-dev runner
(`racket <plugin>/hooks/gate.rkt . all`); `push-gates` is the pre-push
subset.

| Gate | Checks |
|---|---|
| native | stages the nix-built shims into `native-libs/` |
| format | treefmt (`nix fmt -- --ci`): raco fmt, clang-format, nixfmt, shfmt, shellcheck, actionlint, ruff |
| no-syntax-rule, no-raw-malloc | grep gates over the Racket sources |
| gpu-suite-signals | SIGINT/TERM/HUP stop the whole GPU suite and re-raise |
| layout | family deps, cuml-lib's deps, no reserved name exported |
| pkg-deps | `raco setup --check-pkg-deps --unused-pkg-deps` over every cuml package |
| review | `raco review` over every `.rkt`, then cuml-lint's tests |
| compile, test | `raco make`; `raco test --process --timeout 45` (each file bounded; `test-timeouts` in `cuml/tests/info.rkt`) |
| gpu | `scripts/gpu-suite.sh`, prints the SKIP count; red on a SKIP that needs a GPU |
| resyntax | Resyntax against the merge base with master, red on any suggestion |
| docs | the manual, red on a warning or broken link |
| check | `nix flake check`: shim (gtests via ctest, 6 s each), racket, racket-review, layout, formatting, clang-tidy, line-count, racket-version, grep gates |

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
  downstream paths raise `exn:fail:cuml`. `blobs` lives in
  `cuml/cluster/private/blobs.rkt`; nothing public but `exn:fail:cuml`.
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
