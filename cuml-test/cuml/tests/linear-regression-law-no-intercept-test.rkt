#lang racket/base

(require (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/linear-data.rkt" law-cases)
         (only-in "private/linear-laws.rkt" check-oracle-laws check-prediction-law))

(test-gpu "without an intercept, against the exact-rational oracle"
  (check-oracle-laws (law-cases) '(#f)))

(test-gpu "predictions are the exact host value to within the dot-product bound"
  (check-prediction-law (law-cases)))
