#lang racket/base

(require (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/mbsgd-gate-data.rkt" check-constant-batch-one))

(test-gpu "a constant target's batch-1 l1 fit away from the constant is refused"
  (check-constant-batch-one 'float64))
