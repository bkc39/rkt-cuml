#lang racket/base

(require (only-in cuml/cluster
                  kmeans-fit
                  kmeans-result
                  kmeans-result-inertia
                  kmeans-score
                  kmeans-transform)
         (only-in racket/list argmin make-list remove-duplicates)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? check-true test-case)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/kmeans-data.rkt"
                  blob-data
                  cyclic-weights
                  dtypes
                  first-rows
                  host
                  norm2
                  scaled-close?
                  squared-distance
                  string-inits
                  tolerance
                  vec))

(define (host-inertia rows centroids weights)
  (define rescale (/ (length rows) (apply + weights)))
  (for/fold ([inertia 0.0]
             [scale 0.0])
            ([x (in-list rows)]
             [w (in-list weights)])
    (define c (argmin (lambda (centroid) (squared-distance x centroid)) centroids))
    (define w-hat (* w rescale))
    (values (+ inertia (* w-hat (squared-distance x c)))
            (+ scale (* w-hat (+ (norm2 x) (norm2 c)))))))

(define (inertia-tolerance dt)
  (if (eq? dt 'float32)
      1e-6
      (tolerance dt)))

(define (check-inertia actual rows centroids weights dt what)
  (define-values (expected scale) (host-inertia rows centroids weights))
  (check-true (scaled-close? actual expected scale (inertia-tolerance dt))
              (format "~a ~a against the host's ~a (scale ~a)" what actual expected scale)))

(test-gpu "law F: the inertia is the weighted sum of squared distances to the nearest centroid"
  (for* ([dt (in-list dtypes)]
         [seed (in-range 1 4)])
    (test-case (format "~a, seed ~a" dt seed)
      (define X (blob-data 300 3 4 (+ seed 100) dt))
      (define weights (cyclic-weights 300))
      (match-define (kmeans-result centroids _ inertia _)
        (kmeans-fit X #:n-clusters 4 #:random-state seed #:sample-weight (vec weights dt)))
      (define rows (host X))
      (define cs (host centroids))
      (check-inertia inertia rows cs weights dt "inertia")
      (check-inertia (- (kmeans-score centroids X))
                     rows
                     cs
                     (make-list 300 1)
                     dt
                     "unweighted inertia"))))

(test-gpu "law F: transform gives the squared Euclidean distances"
  (for* ([dt (in-list dtypes)]
         [seed (in-range 1 4)])
    (define X (blob-data 200 5 3 seed dt))
    (define C (first-rows (blob-data 50 5 3 (+ seed 50) dt) 3 dt))
    (define rows (host X))
    (define cs (host C))
    (for ([x (in-list rows)]
          [found (in-list (host (kmeans-transform C X)))])
      (for ([c (in-list cs)]
            [d (in-list found)])
        (define expected (squared-distance x c))
        (check-true (scaled-close? d expected (+ (norm2 x) (norm2 c)) (tolerance dt))
                    (format "~a: distance ~a against the host's ~a" dt d expected))))))

(test-gpu "law F: Lloyd's iterations never raise the inertia"
  (define X (blob-data 500 4 6 7 'float64 #:std 2.0))
  (define init (first-rows X 6 'float64))
  (define inertias
    (for/list ([m (in-range 0 6)])
      (kmeans-result-inertia (kmeans-fit X #:init init #:max-iter m))))
  (match-define (cons _ later) inertias)
  (for ([before (in-list inertias)]
        [after (in-list later)])
    (check-true (<= after (* before (+ 1 1e-12))) (format "~a after ~a" after before))))

(test-gpu "law F: the random state changes the seeding of overlapping data"
  (define X (blob-data 2000 4 8 42 'float32 #:std 4.0))
  (for ([init (in-list string-inits)])
    (define inertias
      (for/list ([seed (in-range 1 4)])
        (kmeans-result-inertia (kmeans-fit X #:n-clusters 8 #:init init #:random-state seed))))
    (check-equal? (length (remove-duplicates inertias)) 3 (format "~a: ~a" init inertias))))
