#lang racket/base

(require (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/umap-determinism-data.rkt" check-repeatable wide-configs))

(test-gpu "L5: 600-component random fits and transforms repeat bitwise in and across processes"
  (check-repeatable (wide-configs 'random)))
