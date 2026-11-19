#lang racket/base

(require (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/mbsgd-gate-data.rkt" check-constant-batch-one))

(test-gpu "a constant target's batch-1 oscillation is capped at a quarter of its mean square"
  (check-constant-batch-one 'float32))
