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
         (only-in cuml/cluster dbscan-fit dbscan-params dbscan-result-labels dbscan-result?)
         (only-in racket/list combinations make-list)
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
                  cosine-cloud
                  distance-matrix
                  integer-grid
                  mat
                  result->lists
                  vec)
         (only-in "private/estimator-data.rkt" apply-options round-trips)
         (only-in "private/kmeans-data.rkt" dtypes))

(define changed
  (hasheq 'eps 0.3 'min-samples 10 'metric 'cosine 'out-dtype 'int64 'max-mbytes-per-batch 4))

(define (printed v)
  (list (format "~a" v) (format "~s" v) (format "~v" v)))

(test-case "L19: the default parameters, and every keyword round-trips"
  (check-equal?
   (dbscan-params)
   (hasheq 'eps 0.5 'min-samples 5 'metric 'euclidean 'out-dtype 'int32 'max-mbytes-per-batch #f))
  (check-equal? (estimator-params (dbscan)) (dbscan-params))
  (check-equal? (round-trips dbscan changed) changed)
  (check-equal? (estimator-params (dbscan (dbscan-params #:eps 0.3) #:min-samples 10))
                (dbscan-params #:eps 0.3 #:min-samples 10)))

(test-case "L19: resources is a setting, outside the parameters"
  (check-equal? (estimator-params (dbscan #:resources 'some-resources)) (dbscan-params))
  (check-equal? (format "~v" (dbscan #:eps 0.3 #:resources 'some-resources))
                "#<dbscan #:eps 0.3 #:resources 'some-resources>")
  (check-not-equal? (dbscan #:resources 'some-resources) (dbscan))
  (check-equal? (estimator-set-params (dbscan #:resources 'some-resources) #:eps 0.3)
                (dbscan #:eps 0.3 #:resources 'some-resources)))

(test-case "L19: a base with another key, or without one, is refused under the caller's name"
  (check-cuml-error 'logic
                    "dbscan-params: dbscan has no parameter colour"
                    (lambda () (dbscan-params (hasheq 'colour 'red))))
  (check-cuml-error 'logic
                    "dbscan: dbscan has no parameter colour"
                    (lambda () (dbscan (hash-set (dbscan-params) 'colour 1))))
  (check-cuml-error 'logic
                    "dbscan-fit: expected every dbscan parameter, missing min-samples"
                    (lambda () (dbscan-fit 'X #:params (hasheq 'eps 1) #:resources 'unused))))

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
  (check-equal? (estimator-set-params (dbscan) (hasheq 'eps 0.4) #:eps 0.3 #:metric 'cosine)
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
      (define options (hasheq 'eps 1.5 'max-mbytes-per-batch 1 'min-samples 4 'out-dtype 'int64))
      (check-equal? (device-vector->list (fit-predict (apply-options dbscan options) Y))
                    (device-vector->list (dbscan-result-labels
                                          (apply-options dbscan-fit options Y)))))))

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

(define identity-changes (hasheq 'eps 1.5 'min-samples 3 'out-dtype 'int64 'max-mbytes-per-batch 1))

(define (subsets keys n)
  (for*/list ([k (in-range (add1 n))]
              [subset (in-list (combinations keys k))])
    subset))

(define (identity-fixtures dt)
  (define rows (integer-grid 150 2 12 5))
  (list (list "euclidean" (mat (integer-grid 300 2 20 3) dt) (hasheq))
        (list "precomputed" (mat (distance-matrix rows) dt) (hasheq 'metric 'precomputed))
        (list "cosine"
              (mat (cosine-cloud 300 3 0.02 4) dt)
              (hasheq 'metric 'cosine 'eps 0.02 'min-samples 4))))

(define (ways X options)
  (define base (dbscan-params #:min-samples 7 #:out-dtype 'int64))
  (define (estimated e)
    (result->lists (dbscan-model (fit e X))))
  (list (result->lists (apply-options dbscan-fit options X))
        (result->lists (dbscan-fit X #:params (apply-options dbscan-params options)))
        (result->lists (apply-options dbscan-fit (hash-set options 'params base) X))
        (result->lists (dbscan-fit X #:params (apply-options dbscan-params options base)))
        (estimated (apply-options dbscan options))
        (estimated (dbscan (apply-options dbscan-params options)))
        (estimated (apply-options estimator-set-params options (dbscan)))))

(test-gpu "L19: a keyword is the same as a set with that keyword, through dbscan-fit and the estimator"
  (for* ([dt (in-list dtypes)]
         [fixture (in-list (identity-fixtures dt))]
         [keys (in-list (subsets '(eps min-samples out-dtype max-mbytes-per-batch) 2))])
    (match-define (list name X fixed) fixture)
    (define options
      (for/fold ([options fixed]) ([key (in-list keys)])
        (hash-set options key (hash-ref identity-changes key))))
    (match-define (list keyword set on-base set-on-base by-keyword by-set by-update) (ways X options))
    (define label (format "~a ~a ~a" dt name keys))
    (check-equal? set keyword label)
    (check-equal? set-on-base on-base label)
    (for ([estimated (in-list (list by-keyword by-set by-update))])
      (check-equal? estimated keyword label))))
