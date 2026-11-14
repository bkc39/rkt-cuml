#lang racket/base

(require (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/linear-data.rkt" dtypes)
         (only-in "private/logistic-glmnet-data.rkt" check-binomial labelled-lr-case trial))

(test-gpu "standardised breast-cancer fits at C are glmnet's at lambda 1/(C N)"
  (define t
    (trial (labelled-lr-case 'breast-cancer "diagnosis" '("id") #:standardise? #t)
           '(0.1 1.0)
           1e-2
           1e-4
           #f))
  (for ([dt (in-list dtypes)])
    (check-binomial t dt)))
