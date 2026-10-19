#lang racket/base

(require (only-in cuml
                  estimator-params
                  estimator-set-params
                  fit
                  fit-predict
                  fitted?
                  spectral-clustering
                  spectral-clustering-model
                  spectral-clustering?)
         (only-in cuml/cluster
                  spectral-clustering-fit
                  spectral-clustering-result-labels
                  spectral-clustering-result?)
         (only-in cuml/cluster/private/spectral spectral-clustering-defaults)
         (only-in racket/list make-list)
         (only-in rackunit check-equal? check-false check-not-equal? check-true test-case)
         (only-in raft device-vector->list)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/capabilities.rkt" capabilities)
         (only-in "private/cuml-error.rkt" check-cuml-error)
         (only-in "private/dbscan-data.rkt" mat)
         (only-in "private/estimator-data.rkt" apply-options round-trips)
         (only-in "private/kmeans-data.rkt" blank-vector)
         (only-in "private/spectral-data.rkt" blobs labels))

(define changed
  (hasheq 'n-clusters 3
          'n-components 2
          'random-state 7
          'n-neighbors 15
          'n-init 4
          'eigen-tol 1e-5
          'affinity 'precomputed
          'resources 'some-resources))

(define (printed v)
  (list (format "~a" v) (format "~s" v) (format "~v" v)))

(define (three-blobs)
  (define-values (rows _truth) (blobs 30 3 2 50.0 0))
  (mat rows 'float32))

(test-case "the default parameters, and every keyword round-trips"
  (check-equal? spectral-clustering-defaults
                (hasheq 'n-clusters 8
                        'n-components #f
                        'random-state 0
                        'n-neighbors 10
                        'n-init 10
                        'eigen-tol 'auto))
  (check-equal? (estimator-params (spectral-clustering))
                (hash-set* spectral-clustering-defaults 'affinity 'nearest-neighbors 'resources #f))
  (check-equal? (round-trips spectral-clustering changed) changed))

(test-case "printing shows the parameters that differ from the defaults"
  (check-equal? (printed (spectral-clustering)) (make-list 3 "#<spectral-clustering>"))
  (check-equal? (printed (spectral-clustering #:n-clusters 3))
                (make-list 3 "#<spectral-clustering #:n-clusters 3>"))
  (check-equal? (format "~v" (spectral-clustering #:eigen-tol 1e-5 #:n-clusters 3 #:n-components 2))
                "#<spectral-clustering #:n-clusters 3 #:n-components 2 #:eigen-tol 1e-5>")
  (check-equal? (format "~v" (spectral-clustering #:affinity 'precomputed))
                "#<spectral-clustering #:affinity 'precomputed>")
  (check-equal? (format "~v" (spectral-clustering #:n-clusters 8 #:eigen-tol 'auto))
                "#<spectral-clustering>"))

(test-case "equal? compares parameters, and set-params gives an unfitted estimator"
  (check-equal? (spectral-clustering #:n-clusters 3) (spectral-clustering #:n-clusters 3))
  (check-not-equal? (spectral-clustering #:n-clusters 3) (spectral-clustering #:n-clusters 4))
  (check-equal? (equal-hash-code (spectral-clustering #:n-init 3))
                (equal-hash-code (spectral-clustering #:n-init 3)))
  (check-equal? (estimator-set-params (spectral-clustering #:n-init 3) (hasheq))
                (spectral-clustering #:n-init 3))
  (check-equal? (estimator-set-params (spectral-clustering) (hasheq 'n-clusters 3 'n-neighbors 5))
                (spectral-clustering #:n-clusters 3 #:n-neighbors 5))
  (check-cuml-error 'logic
                    "estimator-set-params: spectral-clustering has no parameter bogus"
                    (lambda () (estimator-set-params (spectral-clustering) (hasheq 'bogus 1))))
  (check-equal? (capabilities (spectral-clustering) spectral-clustering?) '(#t #f #t #f #f #t))
  (check-false (spectral-clustering? 'spectral-clustering))
  (check-false (spectral-clustering-model (spectral-clustering))))

(test-gpu "fit and fit-predict are spectral-clustering-fit's, y is ignored and the argument stays unfitted"
  (define X (three-blobs))
  (define e (spectral-clustering #:n-clusters 3))
  (define fitted-e (fit e X '(ignored)))
  (check-equal? (list (fitted? e) (fitted? fitted-e)) '(#f #t))
  (check-false (spectral-clustering-model e))
  (check-true (spectral-clustering-result? (spectral-clustering-model fitted-e)))
  (check-equal?
   (device-vector->list (spectral-clustering-result-labels (spectral-clustering-model fitted-e)))
   (labels X #:n-clusters 3))
  (check-equal? (format "~v" fitted-e) "#<spectral-clustering fitted #:n-clusters 3>")
  (check-not-equal? fitted-e (fit e X))
  (check-equal? fitted-e fitted-e)
  (check-false (fitted? (estimator-set-params fitted-e (hasheq))))
  (check-equal? (device-vector->list (fit-predict e X)) (labels X #:n-clusters 3))
  (define options (hasheq 'n-clusters 4 'n-components 2 'random-state 3 'n-neighbors 6 'n-init 2))
  (check-equal? (device-vector->list (fit-predict (apply-options spectral-clustering options) X))
                (device-vector->list (spectral-clustering-result-labels
                                      (apply-options spectral-clustering-fit options X)))))

(test-gpu "errors are named after the generic that was called"
  (define X (three-blobs))
  (check-cuml-error 'logic
                    "fit: n-clusters: expected at least 1, got 0"
                    (lambda () (fit (spectral-clustering #:n-clusters 0) X)))
  (check-cuml-error 'logic
                    "fit-predict: n-neighbors: expected at least 2, got 1"
                    (lambda () (fit-predict (spectral-clustering #:n-neighbors 1) X)))
  (check-cuml-error 'logic
                    "fit: affinity: expected nearest-neighbors, got precomputed"
                    (lambda () (fit (spectral-clustering #:affinity 'precomputed #:n-clusters 0) X)))
  (check-cuml-error 'logic
                    "fit-predict: affinity: expected nearest-neighbors, got rbf"
                    (lambda () (fit-predict (spectral-clustering #:affinity 'rbf) X)))
  (check-cuml-error 'logic
                    "fit: spectral-clustering takes no sample-weight"
                    (lambda ()
                      (fit (spectral-clustering) X #:sample-weight (blank-vector 90 'float32))))
  (check-cuml-error 'logic
                    "fit-predict: spectral-clustering takes no sample-weight"
                    (lambda ()
                      (fit-predict (spectral-clustering #:n-clusters 0)
                                   X
                                   #:sample-weight (blank-vector 90 'float32)))))
