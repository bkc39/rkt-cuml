#lang racket/base

(require (only-in cuml/tests/private/linear-data dtypes)
         (only-in cuml/tests/private/logistic-data default-tol-cap)
         (only-in cuml/tests/private/real-logistic-data labelled-case labelled-set)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/glmnet-data.rkt" trial)
         (only-in "private/logistic-glmnet-data.rkt" check-multinomial))

(test-gpu "standardised wine fits at C are glmnet's multinomial fits at lambda 1/(C N)"
  (define t
    (trial (labelled-case (labelled-set 'wine "cultivar" '() #:standardise? #t))
           #:Cs '(0.1 1.0)
           #:float32 1e-2
           #:float64 1e-5
           #:default default-tol-cap))
  (for ([dt (in-list dtypes)])
    (check-multinomial t dt)))
