#lang racket/base

(require (only-in cuml fit fit-transform inverse-transform transform truncated-svd)
         (only-in cuml/decomposition
                  tsvd-fit
                  tsvd-fit-transform
                  tsvd-inverse-transform
                  tsvd-result
                  tsvd-result-components
                  tsvd-result-explained-variance
                  tsvd-result-explained-variance-ratio
                  tsvd-result-singular-values
                  tsvd-transform)
         (only-in racket/match match match-define)
         (only-in rackunit check-equal? check-exn test-case)
         (only-in raft list->device-vector)
         (only-in raft/tests/private/collect drain-finalizers!)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in raft/tests/private/probe probe-pool-used test-pools)
         (only-in "private/cuml-error.rkt" check-cuml-error)
         (only-in "private/estimator-data.rkt" apply-options)
         (only-in "private/kmeans-data.rkt" blank-matrix blank-vector)
         (only-in "private/refusal-rows.rkt" check-row row row-call row-id row-message)
         (only-in "private/scaler-data.rkt" rows->matrix))

(define grid '((1 0 2) (0 3 1) (4 1 0) (2 2 5)))

(define (grid-with cells [dt 'float32] [rows grid])
  (rows->matrix (for/list ([row (in-list rows)]
                           [r (in-naturals)])
                  (for/list ([x (in-list row)]
                             [c (in-naturals)])
                    (match (assoc (list r c) cells)
                      [(list _ value) value]
                      [#f x])))
                dt))

(define past-float32 (expt 2.0 64))
(define past-float64 (expt 2.0 511))

(define f15-message
  "X: expected a squared Frobenius norm of at most 4.253529586511731e+37, got 3.402823669209385e+38")

(define (bad-X-rows)
  (list (row "f1" "X: expected float32 or float64, got int32" (blank-matrix 4 3 'int32))
        (row "f2" "X: expected rank 2, got rank 1" (blank-vector 4 'float32))
        (row "f3"
             "X: expected row-major, got col-major 4x3"
             (blank-matrix 4 3 'float32 #:layout 'col-major))
        (row "f1 before f2" "X: expected float32 or float64, got int32" (blank-vector 4 'int32))))

(define (fit-X-rows)
  (append
   (bad-X-rows)
   (list (row "f4" "X: expected at least 2 rows, got 1" (rows->matrix '((1 2 3)) 'float32))
         (row "f4 none" "X: expected at least 2 rows, got 0" (blank-matrix 0 3 'float64))
         (row "f5" "X: expected at least 2 columns, got 1" (rows->matrix '((1) (2) (3)) 'float64))
         (row "f4 before f5" "X: expected at least 2 rows, got 1" (rows->matrix '((1)) 'float32))
         (row "f14 nan"
              "X: row 3, column 1: expected a finite value, got nan"
              (grid-with '([(3 1) +nan.0])))
         (row "f14 inf"
              "X: row 0, column 2: expected a finite value, got inf"
              (grid-with '([(0 2) +inf.0]) 'float64))
         (row "f14 -inf"
              "X: row 2, column 0: expected a finite value, got -inf"
              (grid-with '([(2 0) -inf.0])))
         (row "f14 first in row-major order"
              "X: row 1, column 2: expected a finite value, got nan"
              (grid-with `([(1 2) +nan.0] [(2 0) +inf.0])))
         (row "f15" f15-message (grid-with `([(3 1) ,past-float32])))
         (row "f15 float64"
              (string-append "X: expected a squared Frobenius norm of at most 2.247116418577895e+307,"
                             " got 4.49423283715579e+307")
              (grid-with `([(0 0) ,past-float64]) 'float64))
         (row "f14 before f15"
              "X: row 3, column 1: expected a finite value, got nan"
              (grid-with `([(0 0) ,past-float32] [(3 1) +nan.0]))))))

(define (parameter-rows)
  (list
   (row "f7" "n-components: expected at least 1, got 0" (hasheq 'n-components 0))
   (row "f7 negative" "n-components: expected at least 1, got -2" (hasheq 'n-components -2))
   (row "f8" "n-components: expected at most 3, got 4" (hasheq 'n-components 4))
   (row "f9" "solver: expected full or jacobi" (hasheq 'solver 'arpack))
   (row "f10" "tol: expected a finite value of at least 0, got -1" (hasheq 'tol -1))
   (row "f10 nan" "tol: expected a finite value of at least 0, got nan" (hasheq 'tol +nan.0))
   (row "f10 inf" "tol: expected a finite value of at least 0, got inf" (hasheq 'tol +inf.0))
   (row "f11"
        "tol: expected at most 3.4028234663852886e+38, got 1e+39"
        (hasheq 'tol 1e39 'solver 'jacobi))
   (row "f12" "iterations: expected at least 1, got 0" (hasheq 'iterations 0))
   (row "f13"
        "iterations: expected at most 2147483647, got 2147483648"
        (hasheq 'iterations (expt 2 31)))
   (row "f7 before f9"
        "n-components: expected at least 1, got 0"
        (hasheq 'solver 'arpack 'n-components 0))
   (row "f8 before f10" "n-components: expected at most 3, got 4" (hasheq 'n-components 4 'tol -1))
   (row "f12 before f14" "iterations: expected at least 1, got 0" (hasheq 'iterations 0))))

(define (fit-calls)
  (list (cons 'tsvd-fit (lambda (X options) (apply-options tsvd-fit options X)))
        (cons 'tsvd-fit-transform (lambda (X options) (apply-options tsvd-fit-transform options X)))
        (cons 'fit (lambda (X options) (fit (apply-options truncated-svd options) X)))
        (cons 'fit-transform
              (lambda (X options) (fit-transform (apply-options truncated-svd options) X)))))

(test-gpu "L17: the fits refuse each bad X with its message, the first refusal winning"
  (for* ([entry (in-list (fit-calls))]
         [r (in-list (fit-X-rows))])
    (match-define (cons who call) entry)
    (test-case (format "~a ~a" who (row-id r))
      (check-row who r (lambda () (call (row-call r) (hasheq)))))))

(test-gpu "L17: the fits refuse each bad hyperparameter with its message"
  (for* ([entry (in-list (fit-calls))]
         [r (in-list (parameter-rows))])
    (match-define (cons who call) entry)
    (define X
      (if (equal? (row-id r) "f12 before f14")
          (grid-with '([(0 0) +nan.0]))
          (grid-with '())))
    (test-case (format "~a ~a" who (row-id r))
      (check-row who r (lambda () (call X (row-call r))))))
  (check-cuml-error 'logic
                    "tsvd-fit: n-components: expected at most 3, got 4"
                    (lambda ()
                      (tsvd-fit (rows->matrix '((1 0 2 1 0) (0 3 1 2 2) (4 1 0 3 1)) 'float32)
                                #:n-components 4)))
  (check-cuml-error 'logic
                    "tsvd-fit: X: expected at least 2 rows, got 1"
                    (lambda () (tsvd-fit (rows->matrix '((1 2 3)) 'float32) #:n-components 3)))
  (for ([bad (in-list (list (lambda () (tsvd-fit (grid-with '()) #:n-components 1.5))
                            (lambda () (tsvd-fit (grid-with '()) #:iterations 2.5))
                            (lambda () (tsvd-fit (grid-with '()) #:tol 'small))))])
    (check-exn exn:fail:contract? bad)))

(define (fitted [dt 'float32])
  (tsvd-fit (grid-with '() dt) #:n-components 2))

(define (hand-built components)
  (define base (fitted))
  (tsvd-result components
               (tsvd-result-explained-variance base)
               (tsvd-result-explained-variance-ratio base)
               (tsvd-result-singular-values base)))

(define (narrowed width [cells '()] [dt 'float32])
  (grid-with cells
             dt
             (for/list ([row (in-list grid)])
               (for/list ([x (in-list row)]
                          [_ (in-range width)])
                 x))))

(define (shared-rows width)
  (define last-column (sub1 width))
  (append
   (for/list ([r (in-list (bad-X-rows))])
     (row (row-id r) (row-message r) (cons (fitted) (row-call r))))
   (list
    (row "t4" "X: expected at least 1 row, got 0" (cons (fitted) (blank-matrix 0 width 'float32)))
    (row "t5"
         "components: expected float32 or float64, got int32"
         (cons (hand-built (blank-matrix 2 3 'int32)) (narrowed width)))
    (row "t5 rank"
         "components: expected rank 2, got rank 1"
         (cons (hand-built (blank-vector 3 'float32)) (narrowed width)))
    (row "t5 layout"
         "components: expected row-major, got col-major 2x3"
         (cons (hand-built (blank-matrix 2 3 'float32 #:layout 'col-major)) (narrowed width)))
    (row "t6"
         "components: expected at least 1 row, got 0"
         (cons (hand-built (blank-matrix 0 3 'float32)) (narrowed width)))
    (row "t7"
         "components: expected at least 2 columns, got 1"
         (cons (hand-built (blank-matrix 2 1 'float32)) (narrowed width)))
    (row "L19 float64 X"
         "X: expected float32, got float64"
         (cons (fitted) (narrowed width '() 'float64)))
    (row "L19 float32 X" "X: expected float64, got float32" (cons (fitted 'float64) (narrowed width)))
    (row "t9"
         (format "X: row 0, column ~a: expected a finite value, got nan" last-column)
         (cons (fitted) (narrowed width `([(0 ,last-column) +nan.0]))))
    (row "t4 before t5"
         "X: expected at least 1 row, got 0"
         (cons (hand-built (blank-matrix 2 3 'int32)) (blank-matrix 0 width 'float32))))))

(define (transform-rows)
  (append (shared-rows 3)
          (list (row "t8" "X: expected 3 columns, got 2" (cons (fitted) (blank-matrix 4 2 'float32)))
                (row "L19 before t8"
                     "X: expected float32, got float64"
                     (cons (fitted) (blank-matrix 4 2 'float64)))
                (row "t8 before t9"
                     "X: expected 3 columns, got 2"
                     (cons (fitted) (narrowed 2 '([(0 0) +nan.0])))))))

(define (inverse-rows)
  (append (shared-rows 2)
          (list (row "t8 inverse" "X: expected 2 columns, got 3" (cons (fitted) (grid-with '()))))))

(define (applied-calls)
  (list (list 'tsvd-transform tsvd-transform transform-rows)
        (list 'tsvd-inverse-transform tsvd-inverse-transform inverse-rows)))

(test-gpu "L17, L19: transform and inverse-transform refuse each bad argument with its message"
  (for ([entry (in-list (applied-calls))])
    (match-define (list who call rows) entry)
    (for ([r (in-list (rows))])
      (match-define (cons result X) (row-call r))
      (test-case (format "~a ~a" who (row-id r))
        (check-row who r (lambda () (call result X)))))))

(test-gpu "L17: the generics refuse a bad X with their own names"
  (define fitted-estimator (fit (truncated-svd #:n-components 2) (grid-with '())))
  (for* ([entry (in-list (list (cons 'transform transform)
                               (cons 'inverse-transform inverse-transform)))]
         [r (in-list (bad-X-rows))])
    (match-define (cons who call) entry)
    (check-row who r (lambda () (call fitted-estimator (row-call r)))))
  (check-cuml-error 'logic
                    "transform: X: expected 3 columns, got 2"
                    (lambda () (transform fitted-estimator (blank-matrix 4 2 'float32))))
  (check-cuml-error 'logic
                    "inverse-transform: X: expected 2 columns, got 3"
                    (lambda () (inverse-transform fitted-estimator (grid-with '())))))

(test-gpu "L17: the estimator refuses an unfitted use and a sample weight"
  (define X (grid-with '()))
  (define w (list->device-vector '(1 1 1 1) #:dtype 'float32))
  (check-cuml-error 'logic
                    "transform: truncated-svd is not fitted"
                    (lambda () (transform (truncated-svd) X)))
  (check-cuml-error 'logic
                    "inverse-transform: truncated-svd is not fitted"
                    (lambda () (inverse-transform (truncated-svd) X)))
  (check-cuml-error 'logic
                    "fit: truncated-svd takes no sample-weight"
                    (lambda () (fit (truncated-svd) X #:sample-weight w)))
  (check-cuml-error 'logic
                    "fit-transform: truncated-svd takes no sample-weight"
                    (lambda () (fit-transform (truncated-svd) X #:sample-weight w))))

(test-pools "L17: a refusal leaves the RMM pool where it was"
  (define device 0)
  (define (unchanged id thunk)
    (drain-finalizers!)
    (define before (probe-pool-used device))
    (thunk)
    (check-equal? (probe-pool-used device) before id))
  (for ([r (in-list (fit-X-rows))])
    (define X (row-call r))
    (unchanged (row-id r) (lambda () (check-row 'tsvd-fit r (lambda () (tsvd-fit X))))))
  (for ([entry (in-list (applied-calls))])
    (match-define (list who call rows) entry)
    (for ([r (in-list (rows))])
      (match-define (cons result X) (row-call r))
      (unchanged (row-id r) (lambda () (check-row who r (lambda () (call result X))))))))
