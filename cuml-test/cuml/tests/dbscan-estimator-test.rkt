#lang racket/base

(require (only-in cuml
                  dbscan
                  dbscan-model
                  dbscan?
                  estimator-params
                  estimator-set-params
                  fit
                  fit-predict
                  fitted?)
         (only-in cuml/cluster dbscan-fit dbscan-result-labels dbscan-result?)
         (only-in cuml/cluster/private/dbscan dbscan-defaults)
         (only-in racket/list make-list)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? check-false check-not-equal? check-true test-case)
         (only-in raft device-vector->list)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/capabilities.rkt" capabilities)
         (only-in "private/cuml-error.rkt" check-cuml-error)
         (only-in "private/dbscan-data.rkt"
                  grid-labels
                  grid-rows
                  grid-weighted-core
                  grid-weights
                  integer-grid
                  mat
                  result->lists
                  vec)
         (only-in "private/estimator-data.rkt" round-trips)
         (only-in "private/kmeans-data.rkt" dtypes))

(define changed
  (hasheq 'eps 0.3
          'min-samples 10
          'metric 'cosine
          'out-dtype 'int64
          'max-mbytes-per-batch 4
          'resources 'some-resources))

(define (printed v)
  (list (format "~a" v) (format "~s" v) (format "~v" v)))

(test-case "L19: the default parameters, and every keyword round-trips"
  (check-equal?
   dbscan-defaults
   (hasheq 'eps 0.5 'min-samples 5 'metric 'euclidean 'out-dtype 'int32 'max-mbytes-per-batch #f))
  (check-equal? (estimator-params (dbscan)) (hash-set dbscan-defaults 'resources #f))
  (check-equal? (round-trips dbscan changed) changed))

(test-case "L19: printing shows the parameters that differ from the defaults"
  (check-equal? (printed (dbscan)) (make-list 3 "#<dbscan>"))
  (check-equal? (printed (dbscan #:eps 0.3 #:min-samples 10))
                (make-list 3 "#<dbscan #:eps 0.3 #:min-samples 10>"))
  (check-equal? (printed (dbscan #:min-samples 10 #:eps 0.3))
                (make-list 3 "#<dbscan #:eps 0.3 #:min-samples 10>"))
  (check-equal? (format "~v" (dbscan #:metric 'cosine #:out-dtype 'int64 #:max-mbytes-per-batch 0))
                "#<dbscan #:metric 'cosine #:out-dtype 'int64 #:max-mbytes-per-batch 0>")
  (check-equal? (format "~v" (dbscan #:eps 0.5 #:min-samples 5)) "#<dbscan>"))

(test-case "L19: equal? compares parameters, and set-params gives an unfitted estimator"
  (check-equal? (dbscan #:eps 0.3) (dbscan #:eps 0.3))
  (check-not-equal? (dbscan #:eps 0.3) (dbscan #:eps 0.4))
  (check-equal? (equal-hash-code (dbscan #:eps 0.3)) (equal-hash-code (dbscan #:eps 0.3)))
  (check-equal? (estimator-set-params (dbscan #:eps 0.3) (hasheq)) (dbscan #:eps 0.3))
  (check-equal? (estimator-set-params (dbscan) (hasheq 'eps 0.3 'metric 'cosine))
                (dbscan #:eps 0.3 #:metric 'cosine))
  (check-cuml-error 'logic
                    "estimator-set-params: dbscan has no parameter bogus"
                    (lambda () (estimator-set-params (dbscan) (hasheq 'bogus 1))))
  (check-equal? (capabilities (dbscan) dbscan?) '(#t #f #t #f #f #t))
  (check-false (dbscan? 'dbscan))
  (check-false (dbscan-model (dbscan))))

(test-gpu "L19: fit and fit-predict are dbscan-fit's, y is ignored and the argument stays unfitted"
  (for ([dt (in-list dtypes)])
    (test-case (symbol->string dt)
      (define X (mat grid-rows dt))
      (define e (dbscan #:eps 2 #:min-samples 4))
      (define fitted-e (fit e X '(ignored)))
      (check-equal? (list (fitted? e) (fitted? fitted-e)) '(#f #t))
      (check-false (dbscan-model e))
      (check-true (dbscan-result? (dbscan-model fitted-e)))
      (check-equal? (result->lists (dbscan-model fitted-e))
                    (result->lists (dbscan-fit X #:eps 2 #:min-samples 4)))
      (check-equal? (format "~v" fitted-e) "#<dbscan fitted #:eps 2 #:min-samples 4>")
      (check-not-equal? fitted-e (fit e X))
      (check-equal? fitted-e fitted-e)
      (check-false (fitted? (estimator-set-params fitted-e (hasheq))))
      (check-equal? (device-vector->list (fit-predict e X)) grid-labels)
      (match-define (list _ weighted-core)
        (result->lists (dbscan-model (fit e X #:sample-weight (vec grid-weights dt)))))
      (check-equal? weighted-core grid-weighted-core)
      (define Y (mat (integer-grid 300 2 20 3) dt))
      (define kws '(#:eps #:max-mbytes-per-batch #:min-samples #:out-dtype))
      (define vals '(1.5 1 4 int64))
      (check-equal? (device-vector->list (fit-predict (keyword-apply dbscan kws vals '()) Y))
                    (device-vector->list (dbscan-result-labels
                                          (keyword-apply dbscan-fit kws vals (list Y))))))))

(test-gpu "L19: errors are named after the generic that was called"
  (define X (mat grid-rows 'float64))
  (check-cuml-error 'logic
                    "fit: eps: expected greater than 0, got 0"
                    (lambda () (fit (dbscan #:eps 0) X)))
  (check-cuml-error 'logic
                    "fit-predict: min-samples: expected at least 1, got 0"
                    (lambda () (fit-predict (dbscan #:min-samples 0) X)))
  (check-cuml-error 'logic
                    "fit: metric: expected euclidean, cosine or precomputed"
                    (lambda () (fit (dbscan #:metric 'l2) X))))
