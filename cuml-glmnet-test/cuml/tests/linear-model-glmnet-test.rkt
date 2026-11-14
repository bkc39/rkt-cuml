#lang racket/base

(require (only-in cuml/tests/private/linear-data dtypes)
         (only-in cuml/tests/private/real-linear-data real-lcase)
         (only-in racket/function const)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/gaussian-glmnet-data.rkt" check-gaussian law-trials)
         (only-in "private/glmnet-data.rkt" trial))

(test-gpu "lasso, elastic-net and ridge fits are glmnet's gaussian fits with the ridge part over sd(y)"
  (define trials
    (append (law-trials '((20 2) (20 7)))
            (for/list ([weights (in-list '(unweighted weighted))])
              (trial (real-lcase 'diabetes (cons 'standardised weights) 'float64)
                     #:float32 1e-4
                     #:float64 5e-8
                     #:default (const 1e-1)))))
  (for* ([dt (in-list dtypes)]
         [t (in-list trials)])
    (check-gaussian t dt)))
