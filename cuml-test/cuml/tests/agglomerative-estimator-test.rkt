#lang racket/base

(require (only-in cuml
                  agglomerative-clustering
                  agglomerative-clustering-model
                  agglomerative-clustering?
                  estimator-params
                  estimator-set-params
                  fit
                  fit-predict
                  fitted?)
         (only-in cuml/cluster single-linkage-fit single-linkage-result-labels single-linkage-result?)
         (only-in cuml/cluster/private/linkage single-linkage-defaults)
         (only-in racket/list make-list)
         (only-in rackunit check-equal? check-false check-not-equal? check-true test-case)
         (only-in raft device-vector->list)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/capabilities.rkt" capabilities)
         (only-in "private/cuml-error.rkt" check-cuml-error)
         (only-in "private/dbscan-data.rkt" integer-grid mat)
         (only-in "private/estimator-data.rkt" apply-options round-trips)
         (only-in "private/kmeans-data.rkt" blank-vector)
         (only-in "private/linkage-data.rkt" hand-children hand-labels hand-rows result->lists))

(define changed
  (hasheq 'n-clusters 3
          'metric 'l1
          'linkage 'ward
          'connectivity 'pairwise
          'c 20
          'resources 'some-resources))

(define (printed v)
  (list (format "~a" v) (format "~s" v) (format "~v" v)))

(test-case "the default parameters, and every keyword round-trips"
  (check-equal? single-linkage-defaults
                (hasheq 'n-clusters 2 'metric 'euclidean 'connectivity 'knn 'c 15))
  (check-equal? (estimator-params (agglomerative-clustering))
                (hash-set* single-linkage-defaults 'linkage 'single 'resources #f))
  (check-equal? (round-trips agglomerative-clustering changed) changed))

(test-case "printing shows the parameters that differ from the defaults"
  (check-equal? (printed (agglomerative-clustering)) (make-list 3 "#<agglomerative-clustering>"))
  (check-equal? (printed (agglomerative-clustering #:n-clusters 3))
                (make-list 3 "#<agglomerative-clustering #:n-clusters 3>"))
  (check-equal? (format "~v"
                        (agglomerative-clustering #:c 20 #:connectivity 'pairwise #:n-clusters 3))
                "#<agglomerative-clustering #:n-clusters 3 #:connectivity 'pairwise #:c 20>")
  (check-equal? (format "~v" (agglomerative-clustering #:linkage 'ward))
                "#<agglomerative-clustering #:linkage 'ward>")
  (check-equal? (format "~v" (agglomerative-clustering #:n-clusters 2 #:c 15))
                "#<agglomerative-clustering>"))

(test-case "equal? compares parameters, and set-params gives an unfitted estimator"
  (check-equal? (agglomerative-clustering #:n-clusters 3) (agglomerative-clustering #:n-clusters 3))
  (check-not-equal? (agglomerative-clustering #:n-clusters 3)
                    (agglomerative-clustering #:n-clusters 4))
  (check-equal? (equal-hash-code (agglomerative-clustering #:c 3))
                (equal-hash-code (agglomerative-clustering #:c 3)))
  (check-equal? (estimator-set-params (agglomerative-clustering #:c 3) (hasheq))
                (agglomerative-clustering #:c 3))
  (check-equal? (estimator-set-params (agglomerative-clustering)
                                      (hasheq 'n-clusters 3 'connectivity 'pairwise))
                (agglomerative-clustering #:n-clusters 3 #:connectivity 'pairwise))
  (check-cuml-error 'logic
                    "estimator-set-params: agglomerative-clustering has no parameter bogus"
                    (lambda () (estimator-set-params (agglomerative-clustering) (hasheq 'bogus 1))))
  (check-equal? (capabilities (agglomerative-clustering) agglomerative-clustering?)
                '(#t #f #t #f #f #t))
  (check-false (agglomerative-clustering? 'agglomerative-clustering))
  (check-false (agglomerative-clustering-model (agglomerative-clustering))))

(test-gpu "fit and fit-predict are single-linkage-fit's, y is ignored and the argument stays unfitted"
  (define X (mat hand-rows 'float32))
  (define e (agglomerative-clustering #:n-clusters 3))
  (define fitted-e (fit e X '(ignored)))
  (check-equal? (list (fitted? e) (fitted? fitted-e)) '(#f #t))
  (check-false (agglomerative-clustering-model e))
  (check-true (single-linkage-result? (agglomerative-clustering-model fitted-e)))
  (check-equal? (result->lists (agglomerative-clustering-model fitted-e))
                (list (list-ref hand-labels 2) hand-children))
  (check-equal? (format "~v" fitted-e) "#<agglomerative-clustering fitted #:n-clusters 3>")
  (check-not-equal? fitted-e (fit e X))
  (check-equal? fitted-e fitted-e)
  (check-false (fitted? (estimator-set-params fitted-e (hasheq))))
  (check-equal? (device-vector->list (fit-predict e X)) (list-ref hand-labels 2))
  (define Y (mat (integer-grid 300 2 20 3) 'float32))
  (define options (hasheq 'n-clusters 5 'connectivity 'pairwise 'c 3))
  (check-equal? (device-vector->list (fit-predict (apply-options agglomerative-clustering options) Y))
                (device-vector->list (single-linkage-result-labels
                                      (apply-options single-linkage-fit options Y)))))

(test-gpu "errors are named after the generic that was called"
  (define X (mat hand-rows 'float32))
  (check-cuml-error 'logic
                    "fit: n-clusters: expected at least 1, got 0"
                    (lambda () (fit (agglomerative-clustering #:n-clusters 0) X)))
  (check-cuml-error 'logic
                    "fit-predict: connectivity: expected knn or pairwise"
                    (lambda () (fit-predict (agglomerative-clustering #:connectivity 'mutual) X)))
  (check-cuml-error 'logic
                    "fit: metric: expected euclidean"
                    (lambda () (fit (agglomerative-clustering #:metric 'l1) X)))
  (check-cuml-error 'logic
                    "fit: linkage: expected single, got ward"
                    (lambda () (fit (agglomerative-clustering #:linkage 'ward #:n-clusters 0) X)))
  (check-cuml-error 'logic
                    "fit-predict: linkage: expected single, got complete"
                    (lambda () (fit-predict (agglomerative-clustering #:linkage 'complete) X)))
  (check-cuml-error 'logic
                    "fit: agglomerative-clustering takes no sample-weight"
                    (lambda ()
                      (fit (agglomerative-clustering) X #:sample-weight (blank-vector 5 'float32))))
  (check-cuml-error 'logic
                    "fit-predict: agglomerative-clustering takes no sample-weight"
                    (lambda ()
                      (fit-predict (agglomerative-clustering #:n-clusters 0)
                                   X
                                   #:sample-weight (blank-vector 5 'float32)))))
