#lang racket/base

(require (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/svm-oracle.rkt" check-referee))

(test-gpu "the sigmoid decisions are within B of the exact decision"
  (check-referee '(sigmoid)))
