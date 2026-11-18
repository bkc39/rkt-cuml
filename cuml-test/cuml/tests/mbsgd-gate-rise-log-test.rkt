#lang racket/base

(require (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/mbsgd-gate-data.rkt" check-classifier-lower check-classifier-rise))

(test-gpu "a log classifier fit ending above the zero model is refused, and the advised lower eta0 passes"
  (check-classifier-rise 'log)
  (check-classifier-lower 'log))
