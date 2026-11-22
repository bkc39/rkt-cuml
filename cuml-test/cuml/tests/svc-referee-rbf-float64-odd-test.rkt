#lang racket/base

(require (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/svm-oracle.rkt" check-referee))

(test-gpu "R1: the float64 rbf decisions are within B of the exact decision, odd cases"
  (check-referee '(rbf) #:dtypes '(float64) #:half 1))
