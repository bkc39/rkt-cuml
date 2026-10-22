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
                  spectral-clustering-fit-graph
                  spectral-clustering-params
                  spectral-clustering-result-labels
                  spectral-clustering-result?)
         (only-in racket/list combinations make-list)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? check-false check-not-equal? check-true test-case)
         (only-in raft device-vector->list)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/capabilities.rkt" capabilities)
         (only-in "private/cuml-error.rkt" check-cuml-error)
         (only-in "private/dbscan-data.rkt" mat)
         (only-in "private/estimator-data.rkt" apply-options round-trips)
         (only-in "private/kmeans-data.rkt" blank-vector)
         (only-in "private/spectral-data.rkt"
                  blobs
                  graph-labels
                  labels
                  planted?
                  three-blobs-matrix
                  three-blocks))

(define changed
  (hasheq 'n-clusters 3
          'n-components 2
          'random-state 7
          'n-neighbors 15
          'n-init 4
          'eigen-tol 1e-5
          'affinity 'precomputed))

(define (printed v)
  (list (format "~a" v) (format "~s" v) (format "~v" v)))

(test-case "the default parameters, and every keyword round-trips"
  (check-equal? (spectral-clustering-params)
                (hasheq 'n-clusters 8
                        'n-components #f
                        'random-state 0
                        'n-neighbors 10
                        'n-init 10
                        'eigen-tol 'auto
                        'affinity 'nearest-neighbors))
  (check-equal? (estimator-params (spectral-clustering)) (spectral-clustering-params))
  (check-equal? (round-trips spectral-clustering changed) changed)
  (check-equal? (estimator-params (spectral-clustering (spectral-clustering-params #:n-init 4)
                                                       #:n-clusters 3))
                (spectral-clustering-params #:n-init 4 #:n-clusters 3)))

(test-case "resources is a setting, outside the parameters"
  (check-equal? (estimator-params (spectral-clustering #:resources 'some-resources))
                (spectral-clustering-params))
  (check-equal? (format "~v" (spectral-clustering #:n-init 4 #:resources 'some-resources))
                "#<spectral-clustering #:n-init 4 #:resources 'some-resources>")
  (check-not-equal? (spectral-clustering #:resources 'some-resources) (spectral-clustering))
  (check-equal? (estimator-set-params (spectral-clustering #:resources 'some-resources)
                                      #:affinity 'precomputed)
                (spectral-clustering #:affinity 'precomputed #:resources 'some-resources)))

(test-case "a base with another key, or without one, is refused under the caller's name"
  (check-cuml-error 'logic
                    "spectral-clustering-params: spectral-clustering has no parameter colour"
                    (lambda () (spectral-clustering-params (hasheq 'colour 'red))))
  (check-cuml-error 'logic
                    "spectral-clustering: spectral-clustering has no parameter colour"
                    (lambda ()
                      (spectral-clustering (hash-set (spectral-clustering-params) 'colour 1))))
  (check-cuml-error 'logic
                    "estimator-set-params: spectral-clustering has no parameter colour"
                    (lambda () (estimator-set-params (spectral-clustering) #:colour 1)))
  (check-cuml-error
   'logic
   "spectral-clustering-fit: expected every spectral-clustering parameter, missing n-components"
   (lambda () (spectral-clustering-fit 'X #:params (hasheq 'n-clusters 3) #:resources 'unused)))
  (check-cuml-error
   'logic
   "spectral-clustering-fit-graph: expected every spectral-clustering parameter, missing affinity"
   (lambda ()
     (spectral-clustering-fit-graph 'rows
                                    'cols
                                    'vals
                                    3
                                    #:params (hash-remove (spectral-clustering-params) 'affinity)
                                    #:resources 'unused))))

(define solver-keywords '(#:eigen-tol #:n-clusters #:n-components #:n-init))

(test-case "the fits take the table's keywords but affinity, and the graph fit no n-neighbors"
  (define-values (required accepted) (procedure-keywords spectral-clustering-fit))
  (define-values (graph-required graph-accepted) (procedure-keywords spectral-clustering-fit-graph))
  (check-equal? (list required graph-required) '(() ()))
  (check-equal? accepted
                (append solver-keywords '(#:n-neighbors #:params #:random-state #:resources)))
  (check-equal? graph-accepted (append solver-keywords '(#:params #:random-state #:resources))))

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
  (check-equal?
   (estimator-set-params (spectral-clustering) (hasheq 'n-clusters 4) #:n-clusters 3 #:n-neighbors 5)
   (spectral-clustering #:n-clusters 3 #:n-neighbors 5))
  (check-cuml-error 'logic
                    "estimator-set-params: spectral-clustering has no parameter bogus"
                    (lambda () (estimator-set-params (spectral-clustering) (hasheq 'bogus 1))))
  (check-equal? (capabilities (spectral-clustering) spectral-clustering?) '(#t #f #t #f #f #t))
  (check-false (spectral-clustering? 'spectral-clustering))
  (check-false (spectral-clustering-model (spectral-clustering))))

(test-gpu "fit and fit-predict are spectral-clustering-fit's, y is ignored and the argument stays unfitted"
  (define X (three-blobs-matrix #:n-each 30))
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

(define identity-changes (hasheq 'n-components 2 'random-state 4 'n-neighbors 6 'n-init 3))

(define (close-blobs-matrix)
  (define-values (rows _truth) (blobs 30 3 2 2.0 0))
  (mat rows 'float32))

(define (ways X options)
  (define base (spectral-clustering-params #:n-init 2 #:random-state 1))
  (define (estimated e)
    (result-labels (spectral-clustering-model (fit e X))))
  (list (result-labels (apply-options spectral-clustering-fit options X))
        (result-labels
         (spectral-clustering-fit X #:params (apply-options spectral-clustering-params options)))
        (result-labels (apply-options spectral-clustering-fit (hash-set options 'params base) X))
        (result-labels
         (spectral-clustering-fit X #:params (apply-options spectral-clustering-params options base)))
        (estimated (apply-options spectral-clustering options))
        (estimated (spectral-clustering (apply-options spectral-clustering-params options)))
        (estimated (apply-options estimator-set-params options (spectral-clustering)))))

(define (result-labels result)
  (device-vector->list (spectral-clustering-result-labels result)))

(test-gpu "each identity change alone moves the partition from the default fit's"
  (define X (close-blobs-matrix))
  (define default (result-labels (spectral-clustering-fit X #:n-clusters 3)))
  (for ([key (in-list '(n-components random-state n-neighbors n-init))])
    (test-case (symbol->string key)
      (define options (hasheq 'n-clusters 3 key (hash-ref identity-changes key)))
      (check-false (planted? (result-labels (apply-options spectral-clustering-fit options X))
                             default)))))

(test-gpu "a keyword is the same as a set with that keyword, through spectral-clustering-fit and the estimator"
  (define X (close-blobs-matrix))
  (for* ([k (in-range 3)]
         [keys (in-list (combinations '(n-components random-state n-neighbors n-init) k))])
    (define options
      (for/fold ([options (hasheq 'n-clusters 3)]) ([key (in-list keys)])
        (hash-set options key (hash-ref identity-changes key))))
    (match-define (list keyword set on-base set-on-base by-keyword by-set by-update) (ways X options))
    (define label (format "~a" keys))
    (check-true (planted? set keyword) label)
    (check-true (planted? set-on-base on-base) label)
    (for ([estimated (in-list (list by-keyword by-set by-update))])
      (check-true (planted? estimated keyword) label))))

(test-gpu "the graph fit ignores the set's n-neighbors and affinity"
  (define-values (g _truth) (three-blocks))
  (check-true (planted? (graph-labels g
                                      #:params (spectral-clustering-params #:n-clusters 3
                                                                           #:affinity 'precomputed
                                                                           #:n-neighbors 3
                                                                           #:random-state 1))
                        (graph-labels g #:n-clusters 3 #:random-state 1))))

(test-gpu "errors are named after the generic that was called"
  (define X (three-blobs-matrix #:n-each 30))
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
                    "fit: spectral-clustering takes no sample-weight"
                    (lambda () (fit (spectral-clustering #:affinity 'rbf) 'X #:sample-weight 'w)))
  (check-cuml-error 'logic
                    "fit-predict: spectral-clustering takes no sample-weight"
                    (lambda ()
                      (fit-predict (spectral-clustering #:n-clusters 0)
                                   X
                                   #:sample-weight (blank-vector 90 'float32)))))
