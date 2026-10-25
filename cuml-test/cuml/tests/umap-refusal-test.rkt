#lang racket/base

(require (only-in cuml fit fit-transform transform umap)
         (only-in cuml/manifold
                  umap-fit
                  umap-fit-transform
                  umap-params
                  umap-result
                  umap-result-X
                  umap-result-a
                  umap-result-b
                  umap-result-embedding
                  umap-result-params
                  umap-transform)
         (only-in cuml/manifold/private/curve curve-parameters)
         (only-in racket/flonum flsingle)
         (only-in racket/match match match-define)
         (only-in rackunit check-equal? check-exn check-true test-case)
         (only-in raft list->device-vector)
         (only-in raft/tests/private/collect drain-finalizers!)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in raft/tests/private/probe probe-pool-used test-pools)
         (only-in "private/cuml-error.rkt" check-cuml-error)
         (only-in "private/estimator-data.rkt" apply-options)
         (only-in "private/kmeans-data.rkt" blank-matrix blank-vector)
         (only-in "private/refusal-rows.rkt" check-row row row-call row-id)
         (only-in "private/scaler-data.rkt" rows->matrix strided-vector)
         (only-in "private/umap-data.rkt" fixture-X fixture-rows tiny))

(define (tiny-with cells [dt 'float32] [rows (fixture-rows tiny)])
  (rows->matrix (for/list ([row (in-list rows)]
                           [r (in-naturals)])
                  (for/list ([x (in-list row)]
                             [c (in-naturals)])
                    (match (assoc (list r c) cells)
                      [(list _ value) value]
                      [#f x])))
                dt))

(define tiny-X (fixture-X tiny))

(define (labels n [dt 'int32])
  (list->device-vector (for/list ([i (in-range n)])
                         (modulo i 2))
                       #:dtype dt))

(define large (flsingle 1e20))

(define f22-message
  (format "X: row 4: expected a squared norm of at most 4.253529586511731e+37, got ~a"
          (* large large)))

(define f23-message
  (format "X: row 3: expected a squared norm of at least ~a under cosine, got 0"
          (* 3 (expt 2.0 -126))))

(define r6-message
  (let-values ([(a _b) (curve-parameters 0.1 1e30)])
    (format (string-append "spread: with min-dist 0.1, a spread of 1e+30 gives a curve parameter"
                           " a of ~a, outside float32's normal range")
            a)))

(define (parameter-rows)
  (list
   (row "R1" "min-dist: expected a finite value of at least 0, got -0.1" (hasheq 'min-dist -0.1))
   (row "R1 nan" "min-dist: expected a finite value of at least 0, got +nan.0" (hasheq 'min-dist +nan.0))
   (row "R1 inf" "min-dist: expected a finite value of at least 0, got +inf.0" (hasheq 'min-dist +inf.0))
   (row "R2" "spread: expected a finite value above 0, got 0" (hasheq 'spread 0))
   (row "R3" "min-dist: expected at most the spread 1.0, got 1.5" (hasheq 'min-dist 1.5))
   (row "R4" "b: expected a value when a is given, got #f" (hasheq 'a 1.0))
   (row "R4 mirror" "a: expected a value when b is given, got #f" (hasheq 'b 1.0))
   (row "R5"
        "a: expected a finite value above 0 and at most 3.4028234663852886e+38, got 0"
        (hasheq 'a 0 'b 1.0))
   (row "R5 b"
        "b: expected a finite value above 0 and at most 3.4028234663852886e+38, got 1e+39"
        (hasheq 'a 1.0 'b 1e39))
   (row "R6" r6-message (hasheq 'spread 1e30))
   (row "R1 before f1"
        "min-dist: expected a finite value of at least 0, got -0.1"
        (hasheq 'min-dist -0.1 'metric 'minkowski))
   (row "f1"
        "metric: expected euclidean, sqeuclidean, manhattan, chebyshev or cosine"
        (hasheq 'metric 'minkowski))
   (row "f2" "init: expected spectral or random" (hasheq 'init 'pca))
   (row "f3" "build-algo: expected brute-force or nn-descent" (hasheq 'build-algo 'auto))
   (row "f4"
        "metric: nn-descent supports euclidean, sqeuclidean and cosine, got manhattan"
        (hasheq 'build-algo 'nn-descent 'metric 'manhattan))
   (row "f5" "n-neighbors: expected at least 2, got 1" (hasheq 'n-neighbors 1))
   (row "f1 before f5"
        "metric: expected euclidean, sqeuclidean, manhattan, chebyshev or cosine"
        (hasheq 'metric 'minkowski 'n-neighbors 1))
   (row "f7" "n-components: expected at least 1, got 0" (hasheq 'n-components 0))
   (row "f7 most"
        "n-components: expected at most 2147483647, got 2147483648"
        (hasheq 'n-components (expt 2 31)))
   (row "f8" "n-epochs: expected at least 1, got 0" (hasheq 'n-epochs 0))
   (row "f8 most"
        "n-epochs: expected at most 2147483647, got 2147483648"
        (hasheq 'n-epochs (expt 2 31)))
   (row "f9" "learning-rate: expected a finite value above 0, got 0" (hasheq 'learning-rate 0))
   (row "f9 nan"
        "learning-rate: expected a finite value above 0, got nan"
        (hasheq 'learning-rate +nan.0))
   (row "f9 most"
        "learning-rate: expected at most 3.4028234663852886e+38, got 1e+39"
        (hasheq 'learning-rate 1e39))
   (row "f10" "negative-sample-rate: expected at least 0, got -1" (hasheq 'negative-sample-rate -1))))

(define (X-rows)
  (list
   (row "f12" "X: expected float32, got float64" (cons (tiny-with '() 'float64) #f))
   (row "f13" "X: expected rank 2, got rank 1" (cons (blank-vector 8 'float32) #f))
   (row "f13 layout"
        "X: expected row-major, got col-major 8x3"
        (cons (blank-matrix 8 3 'float32 #:layout 'col-major) #f))
   (row "f14" "X: expected at least 2 rows, got 1" (cons (rows->matrix '((1 2 3)) 'float32) #f))
   (row "f15" "X: expected at least 1 column, got 0" (cons (blank-matrix 8 0 'float32) #f))
   (row "f19" "X: nn-descent needs at least 150 rows, got 8" (cons tiny-X #f))
   (row "f20" "y: expected int32 or int64, got float32" (cons tiny-X (labels 8 'float32)))
   (row "f20 rank" "y: expected rank 1, got rank 2" (cons tiny-X (blank-matrix 8 1 'int32)))
   (row "f20 stride"
        "y: expected contiguous elements, got stride 2"
        (cons tiny-X (strided-vector 8 'int64)))
   (row "f20 length" "y: expected 8 elements, got 7" (cons tiny-X (labels 7)))
   (row "f21"
        "X: row 3, column 1: expected a finite value, got nan"
        (cons (tiny-with '([(3 1) +nan.0])) (labels 8)))
   (row "f21 inf"
        "X: row 0, column 2: expected a finite value, got -inf"
        (cons (tiny-with '([(0 2) -inf.0])) #f))
   (row "f22" f22-message (cons (tiny-with `([(4 0) 0.0] [(4 1) ,large] [(4 2) 0.0])) #f))
   (row "f23" f23-message (cons (tiny-with '([(3 0) 0.0] [(3 1) 0.0] [(3 2) 0.0])) #f))
   (row "f12 before f14" "X: expected float32, got float64" (cons (rows->matrix '((1 2 3)) 'float64) #f))
   (row "f14 before f17"
        "X: expected at least 2 rows, got 1"
        (cons (rows->matrix '((1 2 3)) 'float32) #f))
   (row "f20 before f21"
        "y: expected 8 elements, got 7"
        (cons (tiny-with '([(3 1) +nan.0])) (labels 7)))
   (row "f21 before f22"
        "X: row 3, column 1: expected a finite value, got nan"
        (cons (tiny-with `([(3 1) +nan.0] [(1 1) ,large])) #f))
   (row "f22 before f23"
        f22-message
        (cons (tiny-with `([(3 0) 0.0] [(3 1) 0.0] [(3 2) 0.0] [(4 0) 0.0] [(4 1) ,large] [(4 2) 0.0]))
              #f))))

(define (x-options r)
  (match (row-id r)
    ["f19" (hasheq 'build-algo 'nn-descent)]
    [(or "f23" "f22 before f23") (hasheq 'metric 'cosine)]
    ["f14 before f17" (hasheq 'n-components 8)]
    [_ (hasheq)]))

(define (fit-calls)
  (list (cons 'umap-fit (lambda (X y options) (apply-options umap-fit (hash-set options 'y y) X)))
        (cons 'umap-fit-transform
              (lambda (X y options) (apply-options umap-fit-transform (hash-set options 'y y) X)))
        (cons 'fit (lambda (X y options) (fit (apply-options umap options) X y)))
        (cons 'fit-transform (lambda (X y options) (fit-transform (apply-options umap options) X y)))))

(test-gpu "L14: the fits refuse each bad hyperparameter with its message, R before f"
  (for* ([entry (in-list (fit-calls))]
         [r (in-list (parameter-rows))])
    (match-define (cons who call) entry)
    (test-case (format "~a ~a" who (row-id r))
      (check-row who r (lambda () (call tiny-X #f (row-call r))))))
  (check-cuml-error 'logic
                    "umap-fit: n-components: spectral init needs fewer components than the 8 rows, got 8"
                    (lambda () (umap-fit tiny-X #:n-components 8)))
  (check-cuml-error 'logic
                    "umap-fit: n-neighbors: expected at least 2, got 1"
                    (lambda () (umap-fit (tiny-with '() 'float64) #:n-neighbors 1)))
  (for ([bad (in-list (list (lambda () (umap-fit tiny-X #:n-neighbors 2.5))
                            (lambda () (umap-fit tiny-X #:learning-rate 'fast))
                            (lambda () (umap-fit tiny-X #:random-state -1))))])
    (check-exn exn:fail:contract? bad)))

(test-gpu "L14: the fits refuse each bad X and y with its message, the first refusal winning"
  (for* ([entry (in-list (fit-calls))]
         [r (in-list (X-rows))])
    (match-define (cons who call) entry)
    (match-define (cons X y) (row-call r))
    (test-case (format "~a ~a" who (row-id r))
      (check-row who r (lambda () (call X y (x-options r)))))))

(test-gpu "L14: f25: a diverged embedding is refused and freed"
  (check-cuml-error
   'logic
   #rx"^umap-fit: embedding: row [0-9]+, column [0-9]+: the optimisation diverged to (nan|inf|-inf); lower learning-rate$"
   (lambda () (umap-fit tiny-X #:learning-rate 1e35 #:init 'random))))

(define (fitted)
  (umap-fit tiny-X #:init 'random))

(define (hand-built #:X [X #f] #:embedding [embedding #f] #:params [params #f])
  (define base (fitted))
  (umap-result (or embedding (umap-result-embedding base))
               (or X (umap-result-X base))
               (umap-result-a base)
               (umap-result-b base)
               (or params (umap-result-params base))))

(define (transform-rows)
  (list
   (row "t1" "X: expected float32, got int32" (cons (fitted) (blank-matrix 2 3 'int32)))
   (row "t2" "X: expected rank 2, got rank 1" (cons (fitted) (blank-vector 3 'float32)))
   (row "t3" "X: expected at least 1 row, got 0" (cons (fitted) (blank-matrix 0 3 'float32)))
   (row "t5"
        "result X: expected float32, got float64"
        (cons (hand-built #:X (tiny-with '() 'float64)) tiny-X))
   (row "t5 rows"
        "result X: expected at least 2 rows, got 1"
        (cons (hand-built #:X (rows->matrix '((1 2 3)) 'float32)) tiny-X))
   (row "t6"
        "embedding: expected float32, got float64"
        (cons (hand-built #:embedding (blank-matrix 8 2 'float64)) tiny-X))
   (row "t6 rows"
        "embedding: expected 8 rows, got 7"
        (cons (hand-built #:embedding (blank-matrix 7 2 'float32)) tiny-X))
   (row "t6 columns"
        "embedding: expected 2 columns, got 3"
        (cons (hand-built #:embedding (blank-matrix 8 3 'float32)) tiny-X))
   (row "t7" "X: expected 3 columns, got 2" (cons (fitted) (blank-matrix 2 2 'float32)))
   (row "t8"
        "metric: expected euclidean, sqeuclidean, manhattan, chebyshev or cosine"
        (cons (hand-built #:params (umap-params #:metric 'minkowski)) tiny-X))
   (row "t8 n-neighbors"
        "n-neighbors: expected at least 2, got 1"
        (cons (hand-built #:params (umap-params #:n-neighbors 1)) tiny-X))
   (row "t9"
        "n-epochs: transform runs n-epochs/3 epochs and needs at least 3, got 2"
        (cons (hand-built #:params (umap-params #:n-epochs 2)) tiny-X))
   (row "t11"
        "X: row 3, column 1: expected a finite value, got nan"
        (cons (fitted) (tiny-with '([(3 1) +nan.0]))))
   (row "t11 norm" f22-message (cons (fitted) (tiny-with `([(4 0) 0.0] [(4 1) ,large] [(4 2) 0.0]))))
   (row "t7 before t11"
        "X: expected 3 columns, got 2"
        (cons (fitted) (rows->matrix '((+nan.0 1)) 'float32)))))

(test-gpu "L14: transform refuses each bad argument with its message"
  (for ([r (in-list (transform-rows))])
    (match-define (cons result X) (row-call r))
    (test-case (format "umap-transform ~a" (row-id r))
      (check-row 'umap-transform r (lambda () (umap-transform result X)))))
  (check-cuml-error
   'logic
   #rx"^umap-transform: output: row [0-9]+, column [0-9]+: the optimisation diverged to (nan|inf|-inf); lower learning-rate$"
   (lambda () (umap-transform (hand-built #:params (umap-params #:learning-rate 3e38)) tiny-X))))

(test-gpu "L14: the generics refuse with their own names"
  (define model (fit (umap #:init 'random) tiny-X))
  (check-cuml-error 'logic
                    "transform: X: expected 3 columns, got 2"
                    (lambda () (transform model (blank-matrix 2 2 'float32))))
  (define w (list->device-vector '(1 1 1 1 1 1 1 1) #:dtype 'float32))
  (check-cuml-error 'logic "transform: umap is not fitted" (lambda () (transform (umap) tiny-X)))
  (check-cuml-error 'logic
                    "fit: umap takes no sample-weight"
                    (lambda () (fit (umap) tiny-X #:sample-weight w)))
  (check-cuml-error 'logic
                    "fit-transform: umap takes no sample-weight"
                    (lambda () (fit-transform (umap) tiny-X #:sample-weight w))))

(test-gpu "L14: after a refused NaN, large or cosine zero row, a valid call succeeds"
  (for ([r (in-list (X-rows))]
        #:when (member (row-id r) '("f21" "f22" "f23")))
    (match-define (cons X y) (row-call r))
    (check-row 'umap-fit r (lambda () (apply-options umap-fit (hash-set (x-options r) 'y y) X)))
    (check-true (and (umap-fit tiny-X #:init 'random) #t) (row-id r))))

(test-pools "L14: a refusal leaves the RMM pool where it was"
  (define device 0)
  (define (unchanged id thunk)
    (drain-finalizers!)
    (define before (probe-pool-used device))
    (thunk)
    (check-equal? (probe-pool-used device) before id))
  (for ([r (in-list (X-rows))])
    (match-define (cons X y) (row-call r))
    (unchanged (row-id r)
               (lambda ()
                 (check-row 'umap-fit r (lambda () (apply-options umap-fit (hash-set (x-options r) 'y y) X))))))
  (for ([r (in-list (transform-rows))])
    (match-define (cons result X) (row-call r))
    (unchanged (row-id r)
               (lambda () (check-row 'umap-transform r (lambda () (umap-transform result X)))))))
