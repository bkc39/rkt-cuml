#lang racket/base

(require (only-in cuml
                  agglomerative-clustering
                  current-batch-samples
                  dbscan
                  dbscan-fit
                  dbscan-params
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
                  hdbscan
                  hdbscan-approximate-predict
                  hdbscan-fit
                  hdbscan-prediction-labels
                  hdbscan-result-labels
                  hdbscan-result-model
                  kmeans
                  kmeans-fit
                  kmeans-params
                  kmeans-predict
                  kmeans-result
                  kmeans-result-centroids
                  kmeans-result-inertia
                  kmeans-result-labels
                  kmeans-score
                  kmeans-transform
                  native-model-release!
                  native-model-released?
                  native-model?
                  predict
                  score
                  single-linkage-fit
                  single-linkage-params
                  single-linkage-result
                  single-linkage-result-labels
                  spectral-clustering
                  spectral-clustering-fit
                  spectral-clustering-fit-graph
                  spectral-clustering-params
                  spectral-clustering-result-labels
                  with-native-models)
         (only-in racket/function const)
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
  (check-equal? (format "~v" (estimator-set-params km #:max-iter 10))
                "#<kmeans #:n-clusters 2 #:max-iter 10 #:random-state 1>")
  (check-cuml-error 'logic "predict: kmeans is not fitted" (lambda () (predict km (grid)))))

(test-case "the manual: kmeans-params overrides a base and refuses an unknown key"
  (define tight (kmeans-params #:tol 1e-6 #:n-init 3))
  (check-equal? (hash-ref (kmeans-params tight #:n-init 5) 'n-init) 5)
  (check-cuml-error 'logic
                    "kmeans-params: kmeans has no parameter colour"
                    (lambda () (kmeans-params (hasheq 'colour 'red)))))

(test-gpu "the manual: a sweep over n-clusters from one set"
  (define tight (kmeans-params #:tol 1e-6 #:n-init 3))
  (check-equal? (for/list ([k (in-range 1 4)])
                  (kmeans-result-inertia (kmeans-fit (grid) #:params tight #:n-clusters k)))
                '(416.0 16.0 13.333333333333332)))

(test-gpu "the manual: current-batch-samples changes no label"
  (define (labels)
    (device-vector->list (kmeans-result-labels (kmeans-fit (grid) #:n-clusters 2 #:random-state 1))))
  (check-equal? (parameterize ([current-batch-samples 3])
                  (labels))
                (labels)))

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
  (check-equal? (format "~v" (estimator-set-params db #:metric 'cosine))
                "#<dbscan #:eps 2 #:min-samples 4 #:metric 'cosine>")
  (check-cuml-error 'logic
                    "fit: eps: expected greater than 0, got 0"
                    (lambda () (fit (dbscan #:eps 0) (points)))))

(test-case "the manual: dbscan-params overrides a base and refuses an unknown key"
  (define dense (dbscan-params #:eps 2 #:min-samples 4))
  (check-equal? (hash-ref (dbscan-params dense #:eps 3) 'min-samples) 4)
  (check-cuml-error 'logic
                    "dbscan-params: dbscan has no parameter colour"
                    (lambda () (dbscan-params (hasheq 'colour 'red)))))

(test-gpu "the manual: dbscan-fit from a set"
  (check-equal? (~> (dbscan-fit (points) #:params (dbscan-params #:eps 2 #:min-samples 4))
                    dbscan-result-labels
                    device-vector->list)
                '(0 0 0 0 1 1 1 -1)))

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
  (check-equal? (format "~v" (estimator-set-params ac #:connectivity 'pairwise))
                "#<agglomerative-clustering #:n-clusters 3 #:connectivity 'pairwise>")
  (check-cuml-error 'logic
                    "fit: linkage: expected single, got ward"
                    (lambda () (fit (agglomerative-clustering #:linkage 'ward) (hand)))))

(test-case "the manual: single-linkage-params overrides a base, and a linkage but single is refused"
  (define coarse (single-linkage-params #:n-clusters 2 #:connectivity 'pairwise))
  (check-equal? (hash-ref (single-linkage-params coarse #:n-clusters 3) 'connectivity) 'pairwise)
  (check-cuml-error
   'logic
   "single-linkage-fit: linkage: expected single, got ward"
   (lambda ()
     (single-linkage-fit 'P #:params (single-linkage-params #:linkage 'ward) #:resources 'unused))))

(test-gpu "the manual: single-linkage-fit from a set"
  (check-equal? (~> (single-linkage-fit (hand)
                                        #:params (single-linkage-params #:n-clusters 2
                                                                        #:connectivity 'pairwise))
                    single-linkage-result-labels
                    device-vector->list)
                (hand-labels-at 2)))

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
  (check-true (planted? (~> (spectral-clustering-fit-graph (list->device-vector rows #:dtype 'int32)
                                                           (list->device-vector cols #:dtype 'int32)
                                                           (list->device-vector (map (const 1.0) rows)
                                                                                #:dtype 'float32)
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
  (check-equal? (format "~v" (estimator-set-params sc #:n-init 3))
                "#<spectral-clustering #:n-clusters 2 #:n-neighbors 4 #:n-init 3>")
  (check-cuml-error 'logic
                    "fit: affinity: expected nearest-neighbors, got precomputed"
                    (lambda () (fit (spectral-clustering #:affinity 'precomputed) S))))

(test-case "the manual: spectral-clustering-params overrides a base, and an affinity but nearest-neighbors is refused"
  (define near (spectral-clustering-params #:n-clusters 3 #:n-neighbors 5))
  (check-equal? (hash-ref (spectral-clustering-params near #:n-init 3) 'n-neighbors) 5)
  (check-cuml-error 'logic
                    "spectral-clustering-fit: affinity: expected nearest-neighbors, got rbf"
                    (lambda ()
                      (spectral-clustering-fit 'G
                                               #:params (spectral-clustering-params #:affinity 'rbf)
                                               #:resources 'unused))))

(test-gpu "the manual: spectral-clustering-fit from a set"
  (check-true (planted? (~> (spectral-clustering-fit
                             (three-groups)
                             #:params (spectral-clustering-params #:n-clusters 3 #:n-neighbors 5))
                            spectral-clustering-result-labels
                            device-vector->list)
                        group-truth)))

(define (hdbscan-groups centres per side step)
  (list*->device-matrix (for*/list ([centre (in-list centres)]
                                    [i (in-range per)])
                          (list (+ centre (* step (quotient i side))) (* step (remainder i side))))
                        #:dtype 'float32))

(test-gpu "the manual: hdbscan-fit finds three groups and places two of three new points"
  (define result (hdbscan-fit (hdbscan-groups '(0 20 40) 12 4 0.5) #:min-cluster-size 5))
  (define labels (device-vector->list (hdbscan-result-labels result)))
  (check-true (planted? labels (truth-of '(12 12 12))))
  (check-equal? (~> (hdbscan-approximate-predict result
                                                 (list*->device-matrix '((0.5 0.5) (40.5 1) (10 30))
                                                                       #:dtype 'float32))
                    hdbscan-prediction-labels
                    device-vector->list)
                (list (car labels) (list-ref labels 24) -1)))

(test-gpu "the manual: prediction needs twice min-samples training rows"
  (define few
    (hdbscan-fit (list*->device-matrix '((0 0) (0 1) (1 0) (1 1) (9 9)) #:dtype 'float32)
                 #:min-cluster-size 2
                 #:min-samples 3))
  (check-cuml-error
   'logic
   "hdbscan-approximate-predict: model: prediction needs at least 6 training rows (2 × min-samples), got 5"
   (lambda () (hdbscan-approximate-predict few (hdbscan-groups '(0 20 40) 12 4 0.5)))))

(test-gpu "the manual: a min-cluster-size above 1023 needs an explicit min-samples"
  (define X (hdbscan-groups '(0 20 40) 12 4 0.5))
  (check-cuml-error 'logic
                    "hdbscan-fit: min-samples: expected at most 1023, got 2000"
                    (lambda () (hdbscan-fit X #:min-cluster-size 2000)))
  (check-cuml-error 'logic
                    "fit: min-samples: expected at most 1023, got 2000"
                    (lambda () (fit (hdbscan #:min-cluster-size 2000) X))))

(test-gpu "the manual: a native model is live until released, and with-native-models releases it"
  (define X (hdbscan-groups '(0 20) 10 5 1))
  (define result (hdbscan-fit X))
  (define m (hdbscan-result-model result))
  (check-true (native-model? m))
  (check-equal? (format "~v" m) "#<native-model hdbscan>")
  (check-equal? (native-model-released? m) #f)
  (native-model-release! m)
  (check-true (native-model-released? m))
  (check-true (planted? (device-vector->list (hdbscan-result-labels result)) (truth-of '(10 10))))
  (check-cuml-error 'logic
                    "hdbscan-approximate-predict: model: expected a live model, got a released one"
                    (lambda () (hdbscan-approximate-predict result X)))
  (define held (hdbscan-result-model (hdbscan-fit X)))
  (check-equal? (with-native-models ([h held]) (native-model-released? h)) #f)
  (check-true (native-model-released? held)))

(test-gpu "the manual: the hdbscan estimator on two groups"
  (define H (hdbscan-groups '(0 20) 10 5 1))
  (define hd (hdbscan #:min-cluster-size 4))
  (check-equal? (format "~v" hd) "#<hdbscan #:min-cluster-size 4>")
  (check-true (planted? (device-vector->list (fit-predict hd H)) (truth-of '(10 10))))
  (check-equal? (format "~v" (estimator-set-params hd #:cluster-selection-method 'leaf))
                "#<hdbscan #:min-cluster-size 4 #:cluster-selection-method 'leaf>")
  (check-cuml-error 'logic
                    "fit: alpha: expected greater than 0, got 0"
                    (lambda () (fit (hdbscan #:alpha 0) H))))
