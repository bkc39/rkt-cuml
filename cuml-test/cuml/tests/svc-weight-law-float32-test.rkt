#lang racket/base

(require (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/svm-oracle.rkt" check-weighted-kkt))

(test-gpu "the KKT law holds with C_i = C·w_i on the float32 grid"
  (check-weighted-kkt 'float32))
