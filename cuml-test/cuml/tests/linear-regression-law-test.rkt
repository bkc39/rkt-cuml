#lang racket/base

(require (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/linear-data.rkt" law-cases)
         (only-in "private/linear-laws.rkt" check-oracle-laws))

(test-gpu "T1-T4: with an intercept, against the exact-rational oracle"
  (check-oracle-laws (law-cases) '(#t)))
