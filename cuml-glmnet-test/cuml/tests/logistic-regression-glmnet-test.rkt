#lang racket/base

(require (only-in cuml/tests/private/linear-data dtypes)
         (only-in cuml/tests/private/logistic-data blob-case default-tol-cap)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/glmnet-data.rkt" trial)
         (only-in "private/logistic-glmnet-data.rkt" check-binomial))

(test-gpu "binomial fits at C are glmnet's at lambda 1/(C W), alpha the l1 share"
  (define trials
    (list (trial (blob-case "binary" 200 4 2 41)
                 #:Cs '(0.1 1.0 10.0)
                 #:float32 1e-4
                 #:float64 1e-8
                 #:default default-tol-cap)
          (trial (blob-case "binary weighted" 200 4 2 42 #:weighted? #t)
                 #:Cs '(0.1 1.0 10.0)
                 #:float32 1e-2
                 #:float64 1e-5
                 #:default default-tol-cap)))
  (for* ([dt (in-list dtypes)]
         [t (in-list trials)])
    (check-binomial t dt)))
