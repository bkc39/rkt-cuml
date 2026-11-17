#lang racket/base

(require (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/mbsgd-gate-data.rkt" check-classifier-rise))

(test-gpu "a hinge classifier fit ending above the zero model is refused"
  (check-classifier-rise 'hinge))
