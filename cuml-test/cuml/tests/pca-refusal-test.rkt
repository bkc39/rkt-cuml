#lang racket/base

(require (only-in cuml fit fit-transform inverse-transform pca transform)
         (only-in cuml/decomposition
                  pca-fit
                  pca-fit-transform
                  pca-inverse-transform
                  pca-result
                  pca-result-components
                  pca-result-explained-variance
                  pca-result-explained-variance-ratio
                  pca-result-mean
                  pca-result-singular-values
                  pca-transform)
         (only-in racket/list make-list take)
         (only-in racket/match match match-define)
         (only-in rackunit check-equal? check-exn check-true test-case)
         (only-in raft device-matrix->list* device-vector->list list->device-vector)
         (only-in raft/tests/private/collect drain-finalizers!)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in raft/tests/private/probe probe-pool-used test-pools)
         (only-in "private/cuml-error.rkt" check-cuml-error)
         (only-in "private/estimator-data.rkt" apply-options)
         (only-in "private/kmeans-data.rkt" blank-matrix blank-vector)
         (only-in "private/refusal-rows.rkt" check-row row row-call row-id row-message)
         (only-in "private/scaler-data.rkt" rows->matrix strided-vector))

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

(define f16-message
  "X: row 3, column 1: expected a magnitude of at most 2662558084806667776, got 1e+19")

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
         (row "f15 nan"
              "X: row 3, column 1: expected a finite value, got nan"
              (grid-with '([(3 1) +nan.0])))
         (row "f15 inf"
              "X: row 0, column 2: expected a finite value, got inf"
              (grid-with '([(0 2) +inf.0]) 'float64))
         (row "f15 -inf"
              "X: row 2, column 0: expected a finite value, got -inf"
              (grid-with '([(2 0) -inf.0])))
         (row "f15 first in row-major order"
              "X: row 1, column 2: expected a finite value, got nan"
              (grid-with '([(1 2) +nan.0] [(2 0) +inf.0])))
         (row "f16" f16-message (grid-with '([(3 1) 1e19])))
         (row "f15 before f16"
              "X: row 3, column 1: expected a finite value, got nan"
              (grid-with '([(0 0) 1e19] [(3 1) +nan.0])))
         (row "f17"
              "X: expected at least 2 distinct rows, got 1"
              (rows->matrix (make-list 5 '(1 2)) 'float64))
         (row "f16 before f17"
              "X: row 0, column 0: expected a magnitude of at most 2662558084806667776, got 1e+19"
              (rows->matrix (make-list 4 '(1e19 2.0 3.0)) 'float32)))))

(define (parameter-rows)
  (list
   (row "f8" "solver: expected full or jacobi" (hasheq 'solver 'svd))
   (row "f9" "n-components: expected at least 1, got 0" (hasheq 'n-components 0))
   (row "f9 negative" "n-components: expected at least 1, got -2" (hasheq 'n-components -2))
   (row "f10" "n-components: expected at most 3, got 4" (hasheq 'n-components 4))
   (row "f11" "tol: expected a finite value of at least 0, got -1" (hasheq 'tol -1))
   (row "f11 nan" "tol: expected a finite value of at least 0, got nan" (hasheq 'tol +nan.0))
   (row "f11 inf" "tol: expected a finite value of at least 0, got inf" (hasheq 'tol +inf.0))
   (row "f12"
        "tol: expected at most 3.4028234663852886e+38, got 1e+39"
        (hasheq 'tol 1e39 'solver 'jacobi))
   (row "f13" "iterations: expected at least 1, got 0" (hasheq 'iterations 0))
   (row "f14"
        "iterations: expected at most 2147483647, got 2147483648"
        (hasheq 'iterations (expt 2 31)))
   (row "f8 before f9" "solver: expected full or jacobi" (hasheq 'solver 'svd 'n-components 0))
   (row "f10 before f11" "n-components: expected at most 3, got 4" (hasheq 'n-components 4 'tol -1))
   (row "f13 before f15" "iterations: expected at least 1, got 0" (hasheq 'iterations 0))))

(define (fit-calls)
  (list (cons 'pca-fit (lambda (X options) (apply-options pca-fit options X)))
        (cons 'pca-fit-transform (lambda (X options) (apply-options pca-fit-transform options X)))
        (cons 'fit (lambda (X options) (fit (apply-options pca options) X)))
        (cons 'fit-transform (lambda (X options) (fit-transform (apply-options pca options) X)))))

(test-gpu "L16: the fits refuse each bad X with its message, the first refusal winning"
  (for* ([entry (in-list (fit-calls))]
         [r (in-list (fit-X-rows))])
    (match-define (cons who call) entry)
    (test-case (format "~a ~a" who (row-id r))
      (check-row who r (lambda () (call (row-call r) (hasheq)))))))

(test-gpu "L16: the fits refuse each bad hyperparameter with its message"
  (for* ([entry (in-list (fit-calls))]
         [r (in-list (parameter-rows))])
    (match-define (cons who call) entry)
    (define X
      (if (equal? (row-id r) "f13 before f15")
          (grid-with '([(0 0) +nan.0]))
          (grid-with '())))
    (test-case (format "~a ~a" who (row-id r))
      (check-row who r (lambda () (call X (row-call r))))))
  (check-cuml-error 'logic
                    "pca-fit: n-components: expected at most 3, got 4"
                    (lambda ()
                      (pca-fit (rows->matrix '((1 0 2 1 0) (0 3 1 2 2) (4 1 0 3 1)) 'float32)
                               #:n-components 4)))
  (for ([bad (in-list (list (lambda () (pca-fit (grid-with '()) #:n-components 1.5))
                            (lambda () (pca-fit (grid-with '()) #:iterations 2.5))
                            (lambda () (pca-fit (grid-with '()) #:tol 'small))))])
    (check-exn exn:fail:contract? bad)))

(define (fitted [dt 'float32])
  (pca-fit (grid-with '() dt) #:n-components 2))

(define (hand-built #:components [components #f]
                    #:mean [mean #f]
                    #:variance [variance 'fitted]
                    #:whiten [whiten #f]
                    #:dtype [dt 'float32])
  (define base (fitted dt))
  (pca-result (or components (pca-result-components base))
              (if (eq? variance 'fitted)
                  (pca-result-explained-variance base)
                  variance)
              (pca-result-explained-variance-ratio base)
              (pca-result-singular-values base)
              (or mean (pca-result-mean base))
              0.0
              4
              whiten))

(define (floats xs [dt 'float32])
  (list->device-vector xs #:dtype dt))

(define t13-message
  "X: row 0, column 2: expected a magnitude of at most 2.8356862219877405e+37, got 1e+38")

(define (whitened-fit rows dt)
  (pca-fit (rows->matrix rows dt) #:whiten #t))

(define (whitened-row id message fit-rows X-rows dt)
  (row id message (cons (whitened-fit fit-rows dt) (rows->matrix X-rows dt))))

(define (narrowed width [cells '()] [dt 'float32])
  (grid-with cells dt (map (lambda (row) (take row width)) grid)))

(define (shared-rows width)
  (define last-column (sub1 width))
  (define inf-message (format "X: row 0, column ~a: expected a finite value, got inf" last-column))
  (append
   (for/list ([r (in-list (bad-X-rows))])
     (row (row-id r) (row-message r) (cons (fitted) (row-call r))))
   (list
    (row "t4" "X: expected at least 1 row, got 0" (cons (fitted) (blank-matrix 0 width 'float32)))
    (row "t6"
         "components: expected float32 or float64, got int32"
         (cons (hand-built #:components (blank-matrix 2 3 'int32)) (narrowed width)))
    (row "t6 rank"
         "components: expected rank 2, got rank 1"
         (cons (hand-built #:components (blank-vector 3 'float32)) (narrowed width)))
    (row "t6 layout"
         "components: expected row-major, got col-major 2x3"
         (cons (hand-built #:components (blank-matrix 2 3 'float32 #:layout 'col-major))
               (narrowed width)))
    (row "t7"
         "components: expected at least 1 row, got 0"
         (cons (hand-built #:components (blank-matrix 0 3 'float32)) (narrowed width)))
    (row "t7 columns"
         "components: expected at least 2 columns, got 1"
         (cons (hand-built #:components (blank-matrix 2 1 'float32)) (narrowed width)))
    (row "t8" "X: expected float32, got float64" (cons (fitted) (narrowed width '() 'float64)))
    (row "t10"
         "mean: expected float32, got float64"
         (cons (hand-built #:mean (floats '(1 2 3) 'float64)) (narrowed width)))
    (row "t10 length"
         "mean: expected 3 elements, got 2"
         (cons (hand-built #:mean (floats '(1 2))) (narrowed width)))
    (row "t10 stride"
         "mean: expected contiguous elements, got stride 2"
         (cons (hand-built #:mean (strided-vector 3) #:dtype 'float64) (narrowed width '() 'float64)))
    (row "t11"
         "explained-variance: expected a vector, got none"
         (cons (hand-built #:variance #f #:whiten #t) (narrowed width)))
    (row "t11 dtype"
         "explained-variance: expected float32, got float64"
         (cons (hand-built #:variance (floats '(1 2) 'float64) #:whiten #t) (narrowed width)))
    (row "t11 length"
         "explained-variance: expected 2 elements, got 3"
         (cons (hand-built #:variance (floats '(1 2 3)) #:whiten #t) (narrowed width)))
    (row "t11 stride"
         "explained-variance: expected contiguous elements, got stride 2"
         (cons (hand-built #:variance (strided-vector 2) #:whiten #t #:dtype 'float64)
               (narrowed width '() 'float64)))
    (row "t12" inf-message (cons (fitted) (narrowed width `([(0 ,last-column) +inf.0]))))
    (row "t11 is read only when whitening"
         inf-message
         (cons (hand-built #:variance #f) (narrowed width `([(0 ,last-column) +inf.0])))))))

(define (transform-rows)
  (append
   (shared-rows 3)
   (list (row "t9" "X: expected 3 columns, got 2" (cons (fitted) (blank-matrix 4 2 'float32)))
         (row "t13" t13-message (cons (fitted) (grid-with '([(0 2) 1e38]))))
         (whitened-row
          "t13 whitened"
          "X: row 0, column 1: expected a magnitude of at most 4.989600773836799e+291, got 1e+300"
          '((0 3) (1 3) (2 3) (-1 3))
          '((0 1e300))
          'float64)
         (whitened-row
          "t13 whitened float32"
          "X: row 0, column 0: expected a magnitude of at most 1.736496146773559e+34, got 1e+36"
          '((0.001 0) (-0.001 0) (0 0.0005) (0 -0.0005))
          '((1e36 0))
          'float32)
         (row "t8 before t9"
              "X: expected float32, got float64"
              (cons (fitted) (blank-matrix 4 2 'float64)))
         (row "t12 before t13"
              "X: row 1, column 0: expected a finite value, got nan"
              (cons (fitted) (grid-with '([(0 2) 1e38] [(1 0) +nan.0])))))))

(define (inverse-rows)
  (append
   (shared-rows 2)
   (list (row "t9 inverse" "X: expected 2 columns, got 3" (cons (fitted) (grid-with '())))
         (row "t13 inverse"
              "X: row 0, column 1: expected a magnitude of at most 4.2535293329816107e+37, got 1e+38"
              (cons (fitted) (rows->matrix '((1 1e38) (2 3)) 'float32)))
         (whitened-row
          "t13 whitened inverse"
          "X: row 0, column 0: expected a magnitude of at most 1.7406088932282882e+307, got 1e+308"
          '((0 3) (1 3) (2 3) (-1 3))
          '((1e308 0))
          'float64)
         (row "t12 inverse"
              "X: row 1, column 0: expected a finite value, got -inf"
              (cons (fitted) (rows->matrix '((1 1) (-inf.0 3)) 'float32))))))

(define (applied-calls)
  (list (list 'pca-transform pca-transform transform-rows)
        (list 'pca-inverse-transform pca-inverse-transform inverse-rows)))

(test-gpu "L16: transform and inverse-transform refuse each bad argument with its message"
  (for ([entry (in-list (applied-calls))])
    (match-define (list who call rows) entry)
    (for ([r (in-list (rows))])
      (match-define (cons result X) (row-call r))
      (test-case (format "~a ~a" who (row-id r))
        (check-row who r (lambda () (call result X)))))))

(test-gpu "L16: the generics refuse a bad X with their own names"
  (define fitted-estimator (fit (pca #:n-components 2) (grid-with '())))
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

(test-gpu "L16: the estimator refuses an unfitted use and a sample weight"
  (define X (grid-with '()))
  (define w (list->device-vector '(1 1 1 1) #:dtype 'float32))
  (check-cuml-error 'logic "transform: pca is not fitted" (lambda () (transform (pca) X)))
  (check-cuml-error 'logic
                    "inverse-transform: pca is not fitted"
                    (lambda () (inverse-transform (pca) X)))
  (check-cuml-error 'logic
                    "fit: pca takes no sample-weight"
                    (lambda () (fit (pca) X #:sample-weight w)))
  (check-cuml-error 'logic
                    "fit-transform: pca takes no sample-weight"
                    (lambda () (fit-transform (pca) X #:sample-weight w))))

(define (one-large dt value)
  (rows->matrix (for/list ([i (in-range 50)])
                  (for/list ([j (in-range 4)])
                    (if (and (= i 7) (= j 2))
                        value
                        (exact->inexact (- (modulo (* i (add1 j)) 7) 3)))))
                dt))

(test-gpu "L17: an element whose covariance could overflow is refused, and one under the bound fits"
  (for ([c (in-list '((float32 1e19 "652190871829835648" "1e+19")
                      (float32 3e38 "652190871829835648" "3e+38")
                      (float64 1e160 "4.740375954054588e+152" "1e+160")))])
    (match-define (list dt value most got) c)
    (check-cuml-error
     'logic
     (format "pca-fit: X: row 7, column 2: expected a magnitude of at most ~a, got ~a" most got)
     (lambda () (pca-fit (one-large dt value)))))
  (define result (pca-fit (one-large 'float32 6e17)))
  (for ([v (list (pca-result-explained-variance result)
                 (pca-result-explained-variance-ratio result)
                 (pca-result-singular-values result)
                 (pca-result-mean result))])
    (check-true (andmap rational? (device-vector->list v))))
  (check-true (andmap (lambda (row) (andmap rational? row))
                      (device-matrix->list* (pca-result-components result)))))

(test-pools "L16: a refusal leaves the RMM pool where it was"
  (define device 0)
  (define (unchanged id thunk)
    (drain-finalizers!)
    (define before (probe-pool-used device))
    (thunk)
    (check-equal? (probe-pool-used device) before id))
  (for ([r (in-list (fit-X-rows))])
    (define X (row-call r))
    (unchanged (row-id r) (lambda () (check-row 'pca-fit r (lambda () (pca-fit X))))))
  (for ([entry (in-list (applied-calls))])
    (match-define (list who call rows) entry)
    (for ([r (in-list (rows))])
      (match-define (cons result X) (row-call r))
      (unchanged (row-id r) (lambda () (check-row who r (lambda () (call result X))))))))
