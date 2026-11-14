#lang racket/base

(require (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/linear-data.rkt" dtypes)
         (only-in "private/logistic-data.rkt" blob-case)
         (only-in "private/logistic-glmnet-data.rkt" check-binomial trial))

(test-gpu "binomial fits at C are glmnet's at lambda 1/(C W), alpha the l1 share"
  (define trials
    (list
     (trial (blob-case "binary" 200 4 2 41) '(0.1 1.0 10.0) 1e-2 1e-5 #t)
     (trial (blob-case "binary weighted" 200 4 2 42 #:weighted? #t) '(0.1 1.0 10.0) 1e-2 1e-5 #t)))
  (for* ([dt (in-list dtypes)]
         [t (in-list trials)])
    (check-binomial t dt)))
