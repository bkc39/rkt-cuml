#lang racket/base

(require (only-in rackunit test-case)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/svm-data.rkt" svm-case-dtype svm-case-name weighted-cases)
         (only-in "private/svm-oracle.rkt" check-kkt))

(test-gpu "the KKT law holds with C_i = C·w_i on the float32 grid"
  (for ([c (in-list (weighted-cases))]
        #:when (eq? (svm-case-dtype c) 'float32))
    (test-case (svm-case-name c)
      (check-kkt c))))
