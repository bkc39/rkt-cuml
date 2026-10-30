#lang racket/base

(require (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/umap-determinism-data.rkt" check-repeatable narrow-configs))

(test-gpu "L5: euclidean and sqeuclidean fits and transforms repeat bitwise in and across processes"
  (check-repeatable (narrow-configs '(euclidean sqeuclidean))))
