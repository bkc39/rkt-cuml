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
         (only-in cuml/cluster
                  single-linkage-fit
                  single-linkage-params
                  single-linkage-result-labels
                  single-linkage-result?)
         (only-in racket/list combinations make-list)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? check-false check-not-equal? check-true test-case)
         (only-in raft device-vector->list)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/capabilities.rkt" capabilities)
         (only-in "private/cuml-error.rkt" check-cuml-error)
         (only-in "private/dbscan-data.rkt" integer-grid mat)
         (only-in "private/estimator-data.rkt" apply-options printed round-trips)
         (only-in "private/kmeans-data.rkt" blank-vector)
         (only-in "private/linkage-data.rkt" hand-children hand-labels hand-rows result->lists))

(define changed (hasheq 'n-clusters 3 'metric 'l1 'linkage 'ward 'connectivity 'pairwise 'c 20))

(test-case "the default parameters, and every keyword round-trips"
  (check-equal? (single-linkage-params)
                (hasheq 'n-clusters 2 'metric 'euclidean 'linkage 'single 'connectivity 'knn 'c 15))
  (check-equal? (estimator-params (agglomerative-clustering)) (single-linkage-params))
  (check-equal? (round-trips agglomerative-clustering changed) changed)
  (check-equal? (estimator-params (agglomerative-clustering (single-linkage-params #:c 20)
                                                            #:n-clusters 3))
                (single-linkage-params #:c 20 #:n-clusters 3)))

(test-case "resources is a setting, outside the parameters"
  (check-equal? (estimator-params (agglomerative-clustering #:resources 'some-resources))
                (single-linkage-params))
  (check-equal? (format "~v" (agglomerative-clustering #:c 20 #:resources 'some-resources))
                "#<agglomerative-clustering #:c 20 #:resources 'some-resources>")
  (check-not-equal? (agglomerative-clustering #:resources 'some-resources) (agglomerative-clustering))
  (check-equal? (estimator-set-params (agglomerative-clustering #:resources 'some-resources)
                                      #:linkage 'ward)
                (agglomerative-clustering #:linkage 'ward #:resources 'some-resources)))

(test-case "a base with another key, or without one, is refused under the caller's name"
  (check-cuml-error 'logic
                    "single-linkage-params: single-linkage has no parameter colour"
                    (lambda () (single-linkage-params (hasheq 'colour 'red))))
  (check-cuml-error 'logic
                    "agglomerative-clustering: agglomerative-clustering has no parameter colour"
                    (lambda ()
                      (agglomerative-clustering (hash-set (single-linkage-params) 'colour 1))))
  (check-cuml-error 'logic
                    "estimator-set-params: agglomerative-clustering has no parameter colour"
                    (lambda () (estimator-set-params (agglomerative-clustering) #:colour 1)))
  (check-cuml-error 'logic
                    "single-linkage-fit: expected every single-linkage parameter, missing metric"
                    (lambda ()
                      (single-linkage-fit 'X #:params (hasheq 'n-clusters 3) #:resources 'unused))))

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
  (check-equal? (estimator-set-params (agglomerative-clustering)
                                      (hasheq 'n-clusters 4)
                                      #:n-clusters 3
                                      #:connectivity 'pairwise)
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

(define identity-changes (hasheq 'n-clusters 4 'connectivity 'pairwise 'c 3))

(define (ways X options)
  (define base (single-linkage-params #:n-clusters 6 #:c 5))
  (define (estimated e)
    (result->lists (agglomerative-clustering-model (fit e X))))
  (list
   (result->lists (apply-options single-linkage-fit options X))
   (result->lists (single-linkage-fit X #:params (apply-options single-linkage-params options)))
   (result->lists (apply-options single-linkage-fit (hash-set options 'params base) X))
   (result->lists (single-linkage-fit X #:params (apply-options single-linkage-params options base)))
   (estimated (apply-options agglomerative-clustering options))
   (estimated (agglomerative-clustering (apply-options single-linkage-params options)))
   (estimated (apply-options estimator-set-params options (agglomerative-clustering)))))

(test-gpu "a keyword is the same as a set with that keyword, through single-linkage-fit and the estimator"
  (define X (mat (integer-grid 300 2 20 3) 'float32))
  (for* ([k (in-range 3)]
         [keys (in-list (combinations '(n-clusters connectivity c) k))])
    (define options
      (for/hasheq ([key (in-list keys)])
        (values key (hash-ref identity-changes key))))
    (match-define (list keyword set on-base set-on-base by-keyword by-set by-update) (ways X options))
    (define label (format "~a" keys))
    (check-equal? set keyword label)
    (check-equal? set-on-base on-base label)
    (for ([estimated (in-list (list by-keyword by-set by-update))])
      (check-equal? estimated keyword label))))

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
                    "fit: agglomerative-clustering takes no sample-weight"
                    (lambda ()
                      (fit (agglomerative-clustering #:linkage 'ward) 'X #:sample-weight 'w)))
  (check-cuml-error 'logic
                    "fit-predict: agglomerative-clustering takes no sample-weight"
                    (lambda ()
                      (fit-predict (agglomerative-clustering #:n-clusters 0)
                                   X
                                   #:sample-weight (blank-vector 5 'float32)))))
