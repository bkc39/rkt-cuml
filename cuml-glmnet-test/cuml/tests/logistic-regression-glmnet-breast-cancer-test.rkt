#lang racket/base

(require (only-in cuml/tests/private/linear-data dtypes)
         (only-in cuml/tests/private/real-logistic-data labelled-case labelled-set)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/glmnet-data.rkt" trial)
         (only-in "private/logistic-glmnet-data.rkt" check-binomial))

(test-gpu "standardised breast-cancer fits at C are glmnet's at lambda 1/(C N)"
  (define t
    (trial (labelled-case (labelled-set 'breast-cancer "diagnosis" '("id") #:standardise? #t))
           #:Cs '(0.1 1.0)
           #:float32 1e-2
           #:float64 1e-4))
  (for ([dt (in-list dtypes)])
    (check-binomial t dt)))
