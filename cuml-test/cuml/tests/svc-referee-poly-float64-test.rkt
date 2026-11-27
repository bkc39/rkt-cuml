#lang racket/base

(require (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/svm-oracle.rkt" check-referee))

(test-gpu "the float64 poly decisions are within B of the exact decision"
  (check-referee '(poly) #:dtypes '(float64)))
