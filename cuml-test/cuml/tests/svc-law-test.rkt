#lang racket/base

(require (only-in rackunit test-case)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/svm-data.rkt" kkt-cases svm-case-name)
         (only-in "private/svm-oracle.rkt" check-kkt))

(test-gpu "K: the KKT law on the grid"
  (for ([c (in-list (kkt-cases))])
    (test-case (svm-case-name c)
      (check-kkt c))))
