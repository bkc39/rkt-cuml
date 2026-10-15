# rkt-cuml

Racket bindings to NVIDIA cuML, built on [rkt-raft](https://github.com/bkc39/rkt-raft).
Users install the `cuml` package; the manual is `cuml-doc`. This file is the
maintainers' view; `AGENTS.md` is the full guide.

```bash
nix develop                                  # GPU shell: stages the shims, installs raft and cuml
racket ~/dev/rkt/racket-dev-plugin/plugins/racket-dev/hooks/gate.rkt . all   # every gate
scripts/gpu-suite.sh                         # gtests, raco tests and the twin, on the GPU host
nix flake check --max-jobs 1 --cores 4       # the CI-equivalent
```

- `cuml-*-lib/`, `cuml-doc/`, `cuml-test/`, `cuml/`: the Racket packages
  (one `cuml/` collection); `shim/`: one native library per algorithm family.
- cuML sources for reference: `~/dev/rkt/cuml-upstream` (tag `v26.08.00`,
  the RAPIDS release rkt-raft pins).
- `plans/layout.md`: the owner's layout decisions; `plans/starting-notes.md`:
  what rkt-raft established.
