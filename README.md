# rkt-cuml

Racket bindings to NVIDIA cuML, built on [rkt-raft](https://github.com/bkc39/rkt-raft).

Status: not started. This repository holds the starting notes for the work; see `plans/starting-notes.md`. It has no GitHub remote yet.

- cuML sources for reference: `~/dev/rkt/cuml-upstream` (NVIDIA/cuml at tag `v26.08.00`, the RAPIDS release rkt-raft pins).
- The proven starting point: rkt-raft's `downstream/kmeans-canary/`, a minimal cuML shim running `ML::kmeans::fit`/`predict` on raft's handle and arrays, checked against `cuml.cluster.KMeans`.
