#lang racket/base

(require (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/linear-data.rkt" law-cases)
         (only-in "private/ridge-data.rkt" check-ridge-laws))

(test-gpu "with an intercept, against the exact-rational ridge oracle"
  (check-ridge-laws (law-cases) '(#t)))
