#lang racket/base

(require (only-in cuml/tests/private/linear-data by-dtype dtypes)
         (only-in cuml/tests/private/logistic-data blob-case default-tol-cap)
         (only-in cuml/tests/private/real-logistic-data labelled-case labelled-set)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/glmnet-data.rkt" trial)
         (only-in "private/logistic-glmnet-data.rkt" check-multinomial))

(define (trials dt)
  (list (trial (blob-case "three blobs" 120 3 3 33)
               #:Cs '(0.1 1.0)
               #:float32 2e-3
               #:float64 1e-6
               #:default default-tol-cap)
        ;; float32 OWL-QN stops 1.8e-2 from the float64 fit on iris l1 at C 1
        (trial (labelled-case (labelled-set 'iris "species" '()))
               #:Cs (by-dtype dt '(0.1) '(0.1 1.0))
               #:float32 1e-2
               #:float64 5e-5)
        (trial (labelled-case (labelled-set 'wine "cultivar" '() #:standardise? #t))
               #:Cs '(0.1 1.0)
               #:float32 1e-2
               #:float64 1e-5
               #:default default-tol-cap)))

(test-gpu "multinomial fits at C are glmnet's at lambda 1/(C N), intercepts centred, with its probabilities"
  (for* ([dt (in-list dtypes)]
         [t (in-list (trials dt))])
    (check-multinomial t dt)))
