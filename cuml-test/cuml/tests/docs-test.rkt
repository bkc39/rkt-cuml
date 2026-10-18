#lang racket/base

(require (only-in cuml
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
                  score)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? check-exn test-case)
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
         (only-in "private/cuml-error.rkt" check-cuml-error raised))

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
