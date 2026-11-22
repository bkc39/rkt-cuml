#lang racket/base

(require (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/svm-oracle.rkt" check-referee))

(test-gpu "R1: the float32 rbf decisions are within B of the exact decision"
  (check-referee '(rbf) #:dtypes '(float32)))
