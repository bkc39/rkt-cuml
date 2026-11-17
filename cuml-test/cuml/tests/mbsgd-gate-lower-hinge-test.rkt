#lang racket/base

(require (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/mbsgd-gate-data.rkt" check-classifier-lower))

(test-gpu "a hinge classifier fit at the advised lower eta0 passes"
  (check-classifier-lower 'hinge))
