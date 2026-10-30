#lang racket/base

(require (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/umap-determinism-data.rkt" check-repeatable wide-configs))

(test-gpu "L5: 398-component spectral fits, the most 800 rows allow, and transforms repeat bitwise"
  (check-repeatable (wide-configs 'spectral 398)))
