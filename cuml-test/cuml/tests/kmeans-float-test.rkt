#lang racket/base

(require (only-in cuml/cluster kmeans-fit kmeans-result kmeans-result-inertia kmeans-score kmeans-transform)
         (only-in racket/list make-list remove-duplicates)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? check-true test-case)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/kmeans-data.rkt"
                  blob-data
                  dtypes
                  first-rows
                  host
                  squared-distance
                  vec))

(define (tolerance dt)
  (if (eq? dt 'float32) 1e-5 1e-12))

(define (norm2 v)
  (for/sum ([a (in-list v)])
    (* a a)))

(define (host-inertia rows centroids weights)
  (define total (for/sum ([w (in-list weights)]) w))
  (define scale (/ (length rows) total))
  (for/sum ([x (in-list rows)]
            [w (in-list weights)])
    (* w scale (for/fold ([best +inf.0]) ([c (in-list centroids)])
                 (min best (squared-distance x c))))))

(define (relatively-close? actual expected tol)
  (<= (abs (- actual expected)) (* tol (max 1.0 (abs expected)))))

(test-gpu "law F: the inertia is the weighted sum of squared distances to the nearest centroid"
  (for* ([dt (in-list '(float64))]
         [seed (in-range 1 4)])
    (test-case (format "~a, seed ~a" dt seed)
      (define X (blob-data 300 3 4 (+ seed 100) dt))
      (define weights (for/list ([i (in-range 300)]) (add1 (modulo (* i 7) 5))))
      (define w (vec weights dt))
      (match-define (kmeans-result centroids _ inertia _)
        (kmeans-fit X #:n-clusters 4 #:random-state seed #:sample-weight w))
      (define rows (host X))
      (define cs (host centroids))
      (define expected (host-inertia rows cs weights))
      (check-true (relatively-close? inertia expected (tolerance dt))
                  (format "inertia ~a against the host's ~a" inertia expected))
      (define unweighted (- (kmeans-score centroids X)))
      (define expected-unweighted (host-inertia rows cs (make-list 300 1)))
      (check-true (relatively-close? unweighted expected-unweighted (tolerance dt))
                  (format "unweighted inertia ~a against the host's ~a" unweighted expected-unweighted)))))

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
        (check-true (<= (abs (- d expected)) (* (tolerance dt) (max 1.0 (+ (norm2 x) (norm2 c)))))
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
  (for ([init (in-list '(scalable-k-means++ k-means++ random))])
    (define inertias
      (for/list ([seed (in-range 1 4)])
        (kmeans-result-inertia (kmeans-fit X #:n-clusters 8 #:init init #:random-state seed))))
    (check-equal? (length (remove-duplicates inertias)) 3 (format "~a: ~a" init inertias))))
