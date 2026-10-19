#lang racket/base

(require (only-in cuml
                  agglomerative-clustering
                  dbscan
                  dbscan-fit
                  dbscan-result
                  dbscan-result-core-sample-indices
                  dbscan-result-labels
                  estimator-params
                  estimator-set-params
                  exn:fail:cuml
                  exn:fail:cuml?
                  fit
                  fit-predict
                  fitted?
                  gen:clusterer
                  gen:estimator
                  gen:predictor
                  kmeans
                  kmeans-fit
                  kmeans-predict
                  kmeans-result
                  kmeans-result-centroids
                  kmeans-score
                  kmeans-transform
                  predict
                  score
                  single-linkage-fit
                  single-linkage-result
                  single-linkage-result-labels
                  spectral-clustering
                  spectral-clustering-fit
                  spectral-clustering-fit-graph
                  spectral-clustering-result-labels)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? check-exn check-true test-case)
         (only-in raft
                  device-matrix->list*
                  device-vector->list
                  dtype
                  exn:fail:raft
                  exn:fail:raft-kind
                  exn:fail:raft?
                  list*->device-matrix
                  list->device-vector)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in threading ~>)
         (only-in "private/cuml-error.rkt" check-cuml-error raised)
         (only-in "private/dbscan-data.rkt" mat)
         (only-in "private/linkage-data.rkt" hand-children hand-labels hand-rows)
         (only-in "private/spectral-data.rkt" planted? truth-of))

(test-case "the manual: a raft handler catches exn:fail:cuml and reads its kind"
  (check-equal? (with-handlers ([exn:fail:raft? exn:fail:raft-kind])
                  (raise (exn:fail:cuml "my-binding: n-clusters: expected at least 1, got 0"
                                        (current-continuation-marks)
                                        'logic)))
                'logic))

(test-case "the manual: exn:fail:cuml is both predicates' and keeps its kind"
  (define e (exn:fail:cuml "my-binding: out of memory" (current-continuation-marks) 'out-of-memory))
  (check-equal? (list (exn:fail:cuml? e) (exn:fail:raft? e) (exn:fail:raft-kind e))
                '(#t #t out-of-memory)))

(test-case "the manual: a raft failure is not a cuML failure"
  (check-equal? (exn:fail:cuml?
                 (exn:fail:raft "device-resources: no device 7" (current-continuation-marks) 'logic))
                #f))

(define (grid)
  (list*->device-matrix '((0 0) (0 2) (2 0) (2 2) (10 10) (10 12) (12 10) (12 12)) #:dtype 'float64))

(define (start)
  (list*->device-matrix '((1 0) (11 11)) #:dtype 'float64))

(define (chapter-fit)
  (kmeans-fit (grid) #:n-clusters 2 #:random-state 1))

(test-gpu "the manual: the chapter's fit finds the two cells of the grid"
  (match-define (kmeans-result centroids _ inertia _) (chapter-fit))
  (check-equal? inertia 16.0)
  (check-equal? (sort (device-matrix->list* centroids) < #:key car) '((1.0 1.0) (11.0 11.0))))

(test-gpu "the manual: one step from a given start is the cell means, with n-clusters inferred"
  (check-equal?
   (device-matrix->list* (kmeans-result-centroids (kmeans-fit (grid) #:init (start) #:max-iter 1)))
   '((1.0 1.0) (11.0 11.0))))

(test-gpu "the manual: #:max-iter 0 returns the start"
  (check-equal?
   (device-matrix->list* (kmeans-result-centroids (kmeans-fit (grid) #:init (start) #:max-iter 0)))
   '((1.0 0.0) (11.0 11.0))))

(test-gpu "the manual: weights pull the centroids"
  (define w (list->device-vector '(1 3 1 3 1 1 1 1) #:dtype 'float64))
  (check-equal?
   (device-matrix->list* (kmeans-result-centroids
                          (kmeans-fit (grid) #:init (start) #:max-iter 1 #:sample-weight w)))
   '((1.0 1.5) (11.0 11.0))))

(test-gpu "the manual: more clusters than samples is refused"
  (check-cuml-error 'logic
                    "kmeans-fit: X: 8 samples are fewer than the 9 clusters"
                    (lambda () (kmeans-fit (grid) #:n-clusters 9))))

(test-gpu "the manual: predict, transform and score on the cells of the grid"
  (match-define (kmeans-result _ _ inertia _) (chapter-fit))
  (define cells (kmeans-result-centroids (kmeans-fit (grid) #:init (start))))
  (check-equal? (device-vector->list
                 (kmeans-predict cells (list*->device-matrix '((1 1) (11 12)) #:dtype 'float64)))
                '(0 1))
  (match-define (list* first-row _) (device-matrix->list* (kmeans-transform cells (grid))))
  (check-equal? first-row '(2.0 242.0))
  (check-equal? (list (kmeans-score cells (grid)) (- inertia)) '(-16.0 -16.0)))

(test-gpu "the manual: an X that is not a device array raises raft's error, not cuML's"
  (define e (raised (lambda () (kmeans-fit '((0 0) (1 1)) #:n-clusters 1))))
  (check-equal? (list (exn:fail:raft? e) (exn:fail:cuml? e)) '(#t #f)))

(test-gpu "the manual: an inexact number argument raises exn:fail:contract"
  (check-exn exn:fail:contract? (lambda () (kmeans-fit (grid) #:n-clusters 2 #:n-init 2.5))))

(struct mean-split (cut)
  #:methods gen:estimator
  [(define (fit _e X [_y #f] #:sample-weight [_w #f])
     (mean-split (/ (apply + X) (length X))))
   (define (fitted? e)
     (and (mean-split-cut e) #t))
   (define (estimator-params _e)
     (hasheq))
   (define (estimator-set-params _e _params)
     (mean-split #f))]
  #:methods gen:predictor
  [(define (predict e X)
     (for/list ([x (in-list X)])
       (if (< x (mean-split-cut e)) 0 1)))]
  #:methods gen:clusterer
  [])

(test-case "the manual: mean-split clusters through the fit-predict fallback"
  (check-equal? (fit-predict (mean-split #f) '(1 2 9 10)) '(0 0 1 1))
  (check-equal? (fitted? (mean-split #f)) #f))

(test-gpu "the manual: the k-means estimator on the grid"
  (define km (kmeans #:n-clusters 2 #:random-state 1))
  (check-equal? (format "~v" km) "#<kmeans #:n-clusters 2 #:random-state 1>")
  (define fitted (fit km (grid)))
  (check-equal? (list (fitted? km) (fitted? fitted)) '(#f #t))
  (check-equal?
   (device-vector->list (predict fitted (list*->device-matrix '((1 1) (11 12)) #:dtype 'float64)))
   '(1 0))
  (check-equal? (score fitted (grid)) -16.0)
  (check-equal? (hash-ref (estimator-params km) 'n-clusters) 2)
  (check-equal? (fitted? (estimator-set-params fitted (hasheq 'max-iter 10))) #f)
  (check-cuml-error 'logic "predict: kmeans is not fitted" (lambda () (predict km (grid)))))

(define (points)
  (list*->device-matrix '((0) (1) (2) (4) (6) (7) (8) (20)) #:dtype 'float64))

(test-gpu "the manual: DBSCAN finds two clusters, sample 3 joins 0 and sample 7 is noise"
  (match-define (dbscan-result labels core) (dbscan-fit (points) #:eps 2 #:min-samples 4))
  (check-equal? (device-vector->list labels) '(0 0 0 0 1 1 1 -1))
  (check-equal? (device-vector->list core) '(2 4)))

(test-gpu "the manual: weights count towards min-samples"
  (define w (list->device-vector '(4 1 1 1 1 1 1 1) #:dtype 'float64))
  (check-equal? (~> (dbscan-fit (points) #:eps 2 #:min-samples 4 #:sample-weight w)
                    dbscan-result-core-sample-indices
                    device-vector->list)
                '(0 1 2 4)))

(test-gpu "the manual: a precomputed distance matrix gives the same labels"
  (define ps '(0 1 2 4 6 7 8 20))
  (define D
    (list*->device-matrix (for/list ([a (in-list ps)])
                            (for/list ([b (in-list ps)])
                              (abs (- a b))))
                          #:dtype 'float64))
  (check-equal? (~> (dbscan-fit D #:eps 2 #:min-samples 4 #:metric 'precomputed)
                    dbscan-result-labels
                    device-vector->list)
                '(0 0 0 0 1 1 1 -1)))

(test-gpu "the manual: the labels take the out-dtype, and eps 0 is refused"
  (check-equal?
   (dtype (dbscan-result-labels (dbscan-fit (points) #:eps 2 #:min-samples 4 #:out-dtype 'int64)))
   'int64)
  (check-cuml-error 'logic
                    "dbscan-fit: eps: expected greater than 0, got 0"
                    (lambda () (dbscan-fit (points) #:eps 0))))

(test-gpu "the manual: the DBSCAN estimator on the points"
  (define db (dbscan #:eps 2 #:min-samples 4))
  (check-equal? (format "~v" db) "#<dbscan #:eps 2 #:min-samples 4>")
  (check-equal? (device-vector->list (fit-predict db (points))) '(0 0 0 0 1 1 1 -1))
  (check-equal? (format "~v" (estimator-set-params db (hasheq 'metric 'cosine)))
                "#<dbscan #:eps 2 #:min-samples 4 #:metric 'cosine>")
  (check-cuml-error 'logic
                    "fit: eps: expected greater than 0, got 0"
                    (lambda () (fit (dbscan #:eps 0) (points)))))

(define (hand)
  (mat hand-rows 'float32))

(define (hand-labels-at k)
  (list-ref hand-labels (sub1 k)))

(test-gpu "the manual: single linkage cuts the hand grid into three clusters"
  (match-define (single-linkage-result labels children) (single-linkage-fit (hand) #:n-clusters 3))
  (check-equal? (device-vector->list labels) (hand-labels-at 3))
  (check-equal? (device-matrix->list* children) hand-children))

(test-gpu "the manual: both connectivities, one cluster per sample, and float64 refused"
  (check-equal? (~> (single-linkage-fit (hand) #:n-clusters 3 #:connectivity 'pairwise)
                    single-linkage-result-labels
                    device-vector->list)
                (hand-labels-at 3))
  (check-equal?
   (~> (single-linkage-fit (hand) #:n-clusters 5) single-linkage-result-labels device-vector->list)
   (hand-labels-at 5))
  (check-cuml-error 'logic
                    "single-linkage-fit: X: expected float32, got float64"
                    (lambda ()
                      (single-linkage-fit (list*->device-matrix '((0) (1)) #:dtype 'float64)))))

(test-gpu "the manual: the agglomerative clustering estimator on the hand grid"
  (define ac (agglomerative-clustering #:n-clusters 3))
  (check-equal? (format "~v" ac) "#<agglomerative-clustering #:n-clusters 3>")
  (check-equal? (device-vector->list (fit-predict ac (hand))) (hand-labels-at 3))
  (check-cuml-error 'logic
                    "fit: linkage: expected single, got ward"
                    (lambda () (fit (agglomerative-clustering #:linkage 'ward) (hand)))))

(define (three-groups)
  (list*->device-matrix (for*/list ([centre (in-list '(0 10 20))]
                                    [i (in-range 8)])
                          (list (+ centre (* 0.5 (quotient i 4))) (* 0.5 (remainder i 4))))
                        #:dtype 'float32))

(define group-truth (truth-of '(8 8 8)))

(test-gpu "the manual: spectral clustering finds the three groups, dense and as a graph"
  (check-true (planted? (~> (spectral-clustering-fit (three-groups) #:n-clusters 3 #:n-neighbors 5)
                            spectral-clustering-result-labels
                            device-vector->list)
                        group-truth))
  (define-values (rows cols)
    (for*/lists (_rows _cols)
                ([i (in-range 24)]
                 [j (in-range 24)]
                 #:when (and (not (= i j)) (= (quotient i 8) (quotient j 8)) (<= (abs (- i j)) 2)))
                (values i j)))
  (check-true (planted? (~> (spectral-clustering-fit-graph
                             (list->device-vector rows #:dtype 'int32)
                             (list->device-vector cols #:dtype 'int32)
                             (list->device-vector (map (lambda (_) 1.0) rows) #:dtype 'float32)
                             24
                             #:n-clusters 3)
                            spectral-clustering-result-labels
                            device-vector->list)
                        group-truth))
  (check-cuml-error
   'logic
   "spectral-clustering-fit-graph: rows: entry 1: self-loop at sample 1; leave the diagonal out"
   (lambda ()
     (spectral-clustering-fit-graph (list->device-vector '(0 1 2) #:dtype 'int32)
                                    (list->device-vector '(1 1 0) #:dtype 'int32)
                                    (list->device-vector '(1.0 1.0 1.0) #:dtype 'float32)
                                    4
                                    #:n-clusters 2))))

(test-gpu "the manual: the spectral clustering estimator on two groups"
  (define S
    (list*->device-matrix
     '((0 0) (0 1) (0 2) (1 0) (1 1) (1 2) (10 0) (10 1) (10 2) (11 0) (11 1) (11 2))
     #:dtype 'float32))
  (define sc (spectral-clustering #:n-clusters 2 #:n-neighbors 4))
  (check-equal? (format "~v" sc) "#<spectral-clustering #:n-clusters 2 #:n-neighbors 4>")
  (check-true (planted? (device-vector->list (fit-predict sc S)) (truth-of '(6 6))))
  (check-cuml-error 'logic
                    "fit: affinity: expected nearest-neighbors, got precomputed"
                    (lambda () (fit (spectral-clustering #:affinity 'precomputed) S))))
