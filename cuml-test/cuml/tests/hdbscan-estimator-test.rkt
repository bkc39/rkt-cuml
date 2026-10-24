#lang racket/base

(require (only-in cuml
                  estimator-params
                  estimator-set-params
                  fit
                  fit-predict
                  fitted?
                  hdbscan
                  hdbscan-model
                  hdbscan?
                  native-model-released?)
         (only-in cuml/cluster
                  hdbscan-approximate-predict
                  hdbscan-fit
                  hdbscan-params
                  hdbscan-result-labels
                  hdbscan-result-model
                  hdbscan-result?)
         (only-in racket/list make-list)
         (only-in rackunit check-equal? check-false check-not-equal? check-true test-case)
         (only-in raft device-vector->list)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/capabilities.rkt" capabilities)
         (only-in "private/cuml-error.rkt" check-cuml-error)
         (only-in "private/dbscan-data.rkt" mat vec)
         (only-in "private/estimator-data.rkt" apply-options round-trips)
         (only-in "private/hdbscan-data.rkt" dataset fit-lists prediction->lists same-fit?))

(define changed
  (hasheq 'min-cluster-size
          10
          'min-samples
          3
          'cluster-selection-epsilon
          0.5
          'max-cluster-size
          40
          'alpha
          0.5
          'cluster-selection-method
          'leaf
          'allow-single-cluster
          #t
          'metric
          'euclidean
          'build-algo
          'brute-force))

(define (printed v)
  (list (format "~a" v) (format "~s" v) (format "~v" v)))

(test-case "the default parameters, and every keyword round-trips"
  (check-equal? (hdbscan-params)
                (hasheq 'min-cluster-size
                        5
                        'min-samples
                        #f
                        'cluster-selection-epsilon
                        0.0
                        'max-cluster-size
                        0
                        'metric
                        'euclidean
                        'alpha
                        1.0
                        'cluster-selection-method
                        'eom
                        'allow-single-cluster
                        #f
                        'build-algo
                        'brute-force))
  (check-equal? (estimator-params (hdbscan)) (hdbscan-params))
  (check-equal? (round-trips hdbscan changed) changed)
  (check-equal? (estimator-params (hdbscan (hdbscan-params #:alpha 0.5) #:min-cluster-size 10))
                (hdbscan-params #:alpha 0.5 #:min-cluster-size 10)))

(test-case "printing shows the parameters that differ from the defaults"
  (check-equal? (printed (hdbscan)) (make-list 3 "#<hdbscan>"))
  (check-equal? (printed (hdbscan #:min-cluster-size 10)) (make-list 3 "#<hdbscan #:min-cluster-size 10>"))
  (check-equal? (format "~v" (hdbscan #:cluster-selection-method 'leaf #:min-samples 3))
                "#<hdbscan #:min-samples 3 #:cluster-selection-method 'leaf>")
  (check-equal? (format "~v" (hdbscan #:min-cluster-size 5 #:alpha 1.0)) "#<hdbscan>"))

(test-case "equal? compares parameters, and set-params gives an unfitted estimator"
  (check-equal? (hdbscan #:alpha 0.5) (hdbscan #:alpha 0.5))
  (check-not-equal? (hdbscan #:alpha 0.5) (hdbscan #:alpha 0.25))
  (check-equal? (equal-hash-code (hdbscan #:alpha 0.5)) (equal-hash-code (hdbscan #:alpha 0.5)))
  (check-equal? (estimator-set-params (hdbscan) (hasheq 'min-cluster-size 10))
                (hdbscan #:min-cluster-size 10))
  (check-cuml-error 'logic
                    "estimator-set-params: hdbscan has no parameter eps"
                    (lambda () (estimator-set-params (hdbscan) (hasheq 'eps 1))))
  (check-equal? (capabilities (hdbscan) hdbscan?) '(#t #f #t #f #f #t))
  (check-false (hdbscan? 'hdbscan))
  (check-false (hdbscan-model (hdbscan))))

(test-gpu "fit and fit-predict are hdbscan-fit's, and the argument stays unfitted"
  (define X (mat (dataset 'b10) 'float32))
  (define e (hdbscan #:min-cluster-size 10))
  (define fitted-e (fit e X '(ignored)))
  (check-equal? (list (fitted? e) (fitted? fitted-e)) '(#f #t))
  (check-false (hdbscan-model e))
  (check-true (hdbscan-result? (hdbscan-model fitted-e)))
  (check-true (same-fit? (fit-lists (hdbscan-model fitted-e))
                         (fit-lists (hdbscan-fit X #:min-cluster-size 10))))
  (check-equal? (format "~v" fitted-e) "#<hdbscan fitted #:min-cluster-size 10>")
  (check-not-equal? fitted-e (fit e X))
  (check-false (fitted? (estimator-set-params fitted-e (hasheq))))
  (check-equal? (device-vector->list (fit-predict e X))
                (device-vector->list (hdbscan-result-labels (hdbscan-fit X #:min-cluster-size 10))))
  (check-equal? (device-vector->list (fit-predict (apply-options hdbscan changed) X))
                (device-vector->list (hdbscan-result-labels
                 (apply-options hdbscan-fit (hash-remove (hash-remove changed 'metric) 'build-algo) X)))))

(test-gpu "errors are named after the generic that was called"
  (define X (mat (dataset 'b10) 'float32))
  (check-cuml-error 'logic
                    "fit: alpha: expected greater than 0, got 0"
                    (lambda () (fit (hdbscan #:alpha 0) X)))
  (check-cuml-error 'logic
                    "fit-predict: min-samples: expected at most 1023, got 1024"
                    (lambda () (fit-predict (hdbscan #:min-samples 1024) X)))
  (check-cuml-error 'logic
                    "fit: hdbscan takes no sample-weight"
                    (lambda () (fit (hdbscan) X #:sample-weight (vec (make-list 300 1.0) 'float32)))))

(test-gpu "a refit leaves the first model live"
  (define X (mat (dataset 'b50) 'float32))
  (define e (hdbscan))
  (define first-fit (hdbscan-model (fit e X)))
  (define second-fit (hdbscan-model (fit e X)))
  (check-false (native-model-released? (hdbscan-result-model first-fit)))
  (check-equal? (prediction->lists (hdbscan-approximate-predict first-fit X))
                (prediction->lists (hdbscan-approximate-predict second-fit X))))
