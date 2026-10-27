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
         (only-in racket/list take)
         (only-in racket/match match match-define)
         (only-in rackunit check-equal? check-exn check-true fail test-case)
         (only-in raft exn:fail:raft? list->device-vector)
         (only-in raft/tests/private/collect drain-finalizers!)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in raft/tests/private/probe probe-pool-used test-pools)
         (only-in "private/cuml-error.rkt" check-cuml-error raised)
         (only-in "private/estimator-data.rkt" apply-options)
         (only-in "private/kmeans-data.rkt" blank-matrix blank-vector)
         (only-in "private/manifold-refusal-data.rkt"
                  cosine-message
                  large-message
                  large-row
                  listed-X
                  nan-message
                  nan-row
                  tiny-with
                  zero-row)
         (only-in "private/refusal-rows.rkt" check-row row row-call row-id)
         (only-in "private/scaler-data.rkt" rows->matrix strided-vector)
         (only-in "private/umap-data.rkt" fixture-X fixture-rows tiny))

(define (tiny-X)
  (fixture-X tiny))

(define (labels n [dt 'int32] [groups 2])
  (list->device-vector (for/list ([i (in-range n)])
                         (modulo i groups))
                       #:dtype dt))

(define f29-message
  (string-append "embedding: row 0, column 0: the optimisation diverged to nan;"
                 " lower learning-rate or use init 'random"))

(define r6-message
  (let-values ([(a _b) (curve-parameters 0.1 1e30)])
    (format (string-append "spread: with min-dist 0.1, a spread of 1e+30 gives a curve parameter"
                           " a of ~a, outside float32's normal range")
            a)))

(define (r5-message name v)
  (format
   "~a: expected a finite value of at least 1.1754943508222875e-38 and at most 3.4028234663852886e+38, got ~a"
   name
   v))

(define (least-rate-message v)
  (format "learning-rate: expected at least 1.1754943508222875e-38, got ~a" v))

(define (parameter-rows)
  (list
   (row "R1" "min-dist: expected a finite value of at least 0, got -0.1" (hasheq 'min-dist -0.1))
   (row "R1 nan"
        "min-dist: expected a finite value of at least 0, got +nan.0"
        (hasheq 'min-dist +nan.0))
   (row "R1 inf"
        "min-dist: expected a finite value of at least 0, got +inf.0"
        (hasheq 'min-dist +inf.0))
   (row "R2" "spread: expected a finite value above 0, got 0" (hasheq 'spread 0))
   (row "R3" "min-dist: expected at most the spread 1.0, got 1.5" (hasheq 'min-dist 1.5))
   (row "R4" "b: expected a value when a is given, got #f" (hasheq 'a 1.0))
   (row "R4 mirror" "a: expected a value when b is given, got #f" (hasheq 'b 1.0))
   (row "R5" (r5-message 'a 0) (hasheq 'a 0 'b 1.0))
   (row "R5 b" (r5-message 'b 1e39) (hasheq 'a 1.0 'b 1e39))
   (row "R5 subnormal" (r5-message 'a 1e-40) (hasheq 'a 1e-40 'b 1.0))
   (row "R5 underflow" (r5-message 'a 1e-50) (hasheq 'a 1e-50 'b 1.0))
   (row "R5 b subnormal" (r5-message 'b 1e-40) (hasheq 'a 1.0 'b 1e-40))
   (row "R5 b underflow" (r5-message 'b 1e-50) (hasheq 'a 1.0 'b 1e-50))
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
   (row "f9 subnormal" (least-rate-message 1e-40) (hasheq 'learning-rate 1e-40))
   (row "f9 underflow" (least-rate-message 1e-50) (hasheq 'learning-rate 1e-50))
   (row "f10" "negative-sample-rate: expected at least 0, got -1" (hasheq 'negative-sample-rate -1))))

(define (X-rows)
  (list
   (row "f12" "X: expected float32, got float64" (list (tiny-with '() 'float64) #f (hasheq)))
   (row "f13" "X: expected rank 2, got rank 1" (list (blank-vector 8 'float32) #f (hasheq)))
   (row "f13 layout"
        "X: expected row-major, got col-major 8x3"
        (list (blank-matrix 8 3 'float32 #:layout 'col-major) #f (hasheq)))
   (row "f14"
        "X: expected at least 2 rows, got 1"
        (list (rows->matrix '((1 2 3)) 'float32) #f (hasheq)))
   (row "f15" "X: expected at least 1 column, got 0" (list (blank-matrix 8 0 'float32) #f (hasheq)))
   (row "f19"
        "X: nn-descent needs at least 150 rows, got 8"
        (list (tiny-X) #f (hasheq 'build-algo 'nn-descent)))
   (row "f20" "y: expected int32 or int64, got float32" (list (tiny-X) (labels 8 'float32) (hasheq)))
   (row "f20 rank"
        "y: expected rank 1, got rank 2"
        (list (tiny-X) (blank-matrix 8 1 'int32) (hasheq)))
   (row "f20 stride"
        "y: expected contiguous elements, got stride 2"
        (list (tiny-X) (strided-vector 8 'int64) (hasheq)))
   (row "f20 length" "y: expected 8 elements, got 7" (list (tiny-X) (labels 7) (hasheq)))
   (row "f21" nan-message (list (tiny-with nan-row) (labels 8) (hasheq)))
   (row "f21 inf"
        "X: row 0, column 2: expected a finite value, got -inf"
        (list (tiny-with '([(0 2) -inf.0])) #f (hasheq)))
   (row "f22" large-message (list (tiny-with large-row) #f (hasheq)))
   (row "f23" cosine-message (list (tiny-with zero-row) #f (hasheq 'metric 'cosine)))
   (row "f26"
        "n-components: spectral init of 3 components needs at least 10 rows, got 8"
        (list (tiny-X) #f (hasheq 'n-components 3)))
   (row "f26 one short"
        "n-components: spectral init of 3 components needs at least 10 rows, got 9"
        (list (tiny-with '() 'float32 (cons '(5.0 5.0 5.0) (fixture-rows tiny)))
              #f
              (hasheq 'n-components 3)))
   (row "f26 defaults"
        "n-components: spectral init of 2 components needs at least 8 rows, got 7"
        (list (tiny-with '() 'float32 (cdr (fixture-rows tiny))) #f (hasheq)))
   (row "f29"
        f29-message
        (list (rows->matrix (take (fixture-rows tiny) 6) 'float32)
              (labels 6 'int32 3)
              (hasheq 'n-components 1 'n-epochs 20)))
   (row "f12 before f14"
        "X: expected float32, got float64"
        (list (rows->matrix '((1 2 3)) 'float64) #f (hasheq)))
   (row "f14 before f17"
        "X: expected at least 2 rows, got 1"
        (list (rows->matrix '((1 2 3)) 'float32) #f (hasheq 'n-components 8)))
   (row "f20 before f21"
        "y: expected 8 elements, got 7"
        (list (tiny-with nan-row) (labels 7) (hasheq)))
   (row "f21 before f22" nan-message (list (tiny-with (append nan-row '([(1 1) 1e20]))) #f (hasheq)))
   (row "f22 before f23"
        large-message
        (list (tiny-with (append zero-row large-row)) #f (hasheq 'metric 'cosine)))))

(define (fit-calls)
  (list (cons 'umap-fit (lambda (X y options) (apply-options umap-fit (hash-set options 'y y) X)))
        (cons 'umap-fit-transform
              (lambda (X y options) (apply-options umap-fit-transform (hash-set options 'y y) X)))
        (cons 'fit (lambda (X y options) (fit (apply-options umap options) X y)))
        (cons 'fit-transform
              (lambda (X y options) (fit-transform (apply-options umap options) X y)))))

(test-gpu "L14: the fits refuse each bad hyperparameter with its message, R before f"
  (for* ([entry (in-list (fit-calls))]
         [r (in-list (parameter-rows))])
    (match-define (cons who call) entry)
    (test-case (format "~a ~a" who (row-id r))
      (check-row who r (lambda () (call (tiny-X) #f (row-call r))))))
  (check-cuml-error
   'logic
   "umap-fit: n-components: spectral init needs fewer components than the 8 rows, got 8"
   (lambda () (umap-fit (tiny-X) #:n-components 8)))
  (check-cuml-error 'logic
                    "umap-fit: n-neighbors: expected at least 2, got 1"
                    (lambda () (umap-fit (tiny-with '() 'float64) #:n-neighbors 1)))
  (for ([bad (in-list (list (lambda () (umap-fit (tiny-X) #:n-neighbors 2.5))
                            (lambda () (umap-fit (tiny-X) #:learning-rate 'fast))
                            (lambda () (umap-fit (tiny-X) #:random-state -1))))])
    (check-exn exn:fail:contract? bad)))

(test-gpu "L14: the fits refuse each bad X and y with its message, the first refusal winning"
  (for* ([entry (in-list (fit-calls))]
         [r (in-list (X-rows))])
    (match-define (cons who call) entry)
    (match-define (list X y options) (row-call r))
    (test-case (format "~a ~a" who (row-id r))
      (check-row who r (lambda () (call X y options))))))

(test-gpu "L14: f25: a diverged embedding is refused and freed"
  (check-cuml-error
   'logic
   #rx"^umap-fit: embedding: row [0-9]+, column [0-9]+: the optimisation diverged to (nan|inf|-inf); lower learning-rate$"
   (lambda () (umap-fit (tiny-X) #:learning-rate 1e35 #:init 'random))))

(define (fitted)
  (umap-fit (tiny-X) #:init 'random))

(define (hand-built #:X [X #f] #:embedding [embedding #f] #:params [params #f])
  (define base (fitted))
  (umap-result (or embedding (umap-result-embedding base))
               (or X (umap-result-X base))
               (umap-result-a base)
               (umap-result-b base)
               (or params (umap-result-params base))))

(define (transform-rows)
  (list (row "t1" "X: expected float32, got int32" (cons (fitted) (blank-matrix 2 3 'int32)))
        (row "t2" "X: expected rank 2, got rank 1" (cons (fitted) (blank-vector 3 'float32)))
        (row "t3" "X: expected at least 1 row, got 0" (cons (fitted) (blank-matrix 0 3 'float32)))
        (row "t5"
             "result X: expected float32, got float64"
             (cons (hand-built #:X (tiny-with '() 'float64)) (tiny-X)))
        (row "t5 rows"
             "result X: expected at least 2 rows, got 1"
             (cons (hand-built #:X (rows->matrix '((1 2 3)) 'float32)) (tiny-X)))
        (row "t6"
             "embedding: expected float32, got float64"
             (cons (hand-built #:embedding (blank-matrix 8 2 'float64)) (tiny-X)))
        (row "t6 rows"
             "embedding: expected 8 rows, got 7"
             (cons (hand-built #:embedding (blank-matrix 7 2 'float32)) (tiny-X)))
        (row "t6 columns"
             "embedding: expected 2 columns, got 3"
             (cons (hand-built #:embedding (blank-matrix 8 3 'float32)) (tiny-X)))
        (row "t7" "X: expected 3 columns, got 2" (cons (fitted) (blank-matrix 2 2 'float32)))
        (row "t8"
             "metric: expected euclidean, sqeuclidean, manhattan, chebyshev or cosine"
             (cons (hand-built #:params (umap-params #:metric 'minkowski)) (tiny-X)))
        (row "t8 n-neighbors"
             "n-neighbors: expected at least 2, got 1"
             (cons (hand-built #:params (umap-params #:n-neighbors 1)) (tiny-X)))
        (row "t9"
             "n-epochs: transform runs n-epochs/3 epochs and needs at least 3, got 2"
             (cons (hand-built #:params (umap-params #:n-epochs 2)) (tiny-X)))
        (row "t8 learning-rate"
             (least-rate-message 1e-50)
             (cons (hand-built #:params (umap-params #:learning-rate 1e-50)) (tiny-X)))
        (row "t11 result X"
             (string-append "result " nan-message)
             (cons (hand-built #:X (tiny-with nan-row)) (tiny-X)))
        (row "t11 result X norm"
             (string-append "result " large-message)
             (cons (hand-built #:X (tiny-with large-row)) (tiny-X)))
        (row "t11 result X cosine"
             (string-append "result " cosine-message)
             (cons (hand-built #:X (tiny-with zero-row) #:params (umap-params #:metric 'cosine))
                   (tiny-X)))
        (row "t11 result X before X"
             (string-append "result " nan-message)
             (cons (hand-built #:X (tiny-with nan-row)) (tiny-with large-row)))
        (row "t11" nan-message (cons (fitted) (tiny-with nan-row)))
        (row "t11 norm" large-message (cons (fitted) (tiny-with large-row)))
        (row "t11 cosine"
             cosine-message
             (cons (umap-fit (tiny-X) #:init 'random #:metric 'cosine) (tiny-with zero-row)))
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
   (lambda () (umap-transform (hand-built #:params (umap-params #:learning-rate 3e38)) (tiny-X)))))

(test-gpu "L14: f28: a non-array X is refused by the view check, under any seed"
  (define unseeded (umap-fit (tiny-X) #:init 'random #:random-state #f))
  (for ([call (in-list
               (list (list 'umap-fit 0 (lambda () (umap-fit listed-X #:random-state 0)))
                     (list 'umap-fit #f (lambda () (umap-fit listed-X #:random-state #f)))
                     (list 'umap-transform #f (lambda () (umap-transform unseeded listed-X)))))])
    (match-define (list who seed thunk) call)
    (test-case (format "~a, random-state ~a" who seed)
      (match (raised thunk)
        [(? exn:fail:raft? e)
         (check-equal?
          (exn-message e)
          (format "~a: expected a device array or #f for a view, given ~e" who listed-X))]
        [other (fail (format "expected an exn:fail:raft, got ~e" other))]))))

(test-gpu "L14: the generics refuse with their own names"
  (define model (fit (umap #:init 'random) (tiny-X)))
  (check-cuml-error 'logic
                    "transform: X: expected 3 columns, got 2"
                    (lambda () (transform model (blank-matrix 2 2 'float32))))
  (define w (list->device-vector '(1 1 1 1 1 1 1 1) #:dtype 'float32))
  (check-cuml-error 'logic "transform: umap is not fitted" (lambda () (transform (umap) (tiny-X))))
  (check-cuml-error 'logic
                    "fit: umap takes no sample-weight"
                    (lambda () (fit (umap) (tiny-X) #:sample-weight w)))
  (check-cuml-error 'logic
                    "fit-transform: umap takes no sample-weight"
                    (lambda () (fit-transform (umap) (tiny-X) #:sample-weight w))))

(test-gpu "L14: after a refused NaN, large or cosine zero row, a valid call succeeds"
  (for ([r (in-list (X-rows))]
        #:when (member (row-id r) '("f21" "f22" "f23")))
    (match-define (list X y options) (row-call r))
    (check-row 'umap-fit r (lambda () (apply-options umap-fit (hash-set options 'y y) X)))
    (check-true (and (umap-fit (tiny-X) #:init 'random) #t) (row-id r))))

(test-pools "L14: a refusal leaves the RMM pool where it was"
  (define device 0)
  (define (unchanged id thunk)
    (drain-finalizers!)
    (define before (probe-pool-used device))
    (thunk)
    (check-equal? (probe-pool-used device) before id))
  (for ([r (in-list (X-rows))])
    (match-define (list X y options) (row-call r))
    (unchanged
     (row-id r)
     (lambda ()
       (check-row 'umap-fit r (lambda () (apply-options umap-fit (hash-set options 'y y) X))))))
  (for ([r (in-list (transform-rows))])
    (match-define (cons result X) (row-call r))
    (unchanged (row-id r)
               (lambda () (check-row 'umap-transform r (lambda () (umap-transform result X)))))))
