#lang racket/base

(require (only-in cuml fit fit-transform inverse-transform standard-scaler transform)
         (only-in cuml/preprocessing
                  standard-scaler-fit
                  standard-scaler-fit-transform
                  standard-scaler-inverse-transform
                  standard-scaler-result
                  standard-scaler-transform)
         (only-in cuml/preprocessing/private/standard-scaler current-scratch-elements)
         (only-in rackunit check-equal? check-exn test-case)
         (only-in raft list->device-vector)
         (only-in raft/tests/private/collect drain-finalizers!)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in raft/tests/private/probe probe-pool-used test-pools)
         (only-in "private/cuml-error.rkt" check-cuml-error)
         (only-in "private/kmeans-data.rkt" blank-matrix blank-vector)
         (only-in "private/refusal-rows.rkt" check-row row row-call row-id row-message)
         (only-in "private/scaler-data.rkt" rows->matrix strided-vector))

(define grid '((1 2 3) (4 5 6) (7 8 9) (10 11 12)))

(define (grid-with i j value [dt 'float32])
  (rows->matrix (for/list ([row (in-list grid)]
                           [r (in-naturals)])
                  (for/list ([x (in-list row)]
                             [c (in-naturals)])
                    (if (and (= r i) (= c j))
                        value
                        x)))
                dt))

(define (bad-X-rows)
  (list (row "f1" "X: expected float32 or float64, got int32" (blank-matrix 4 3 'int32))
        (row "f2" "X: expected rank 2, got rank 1" (blank-vector 4 'float32))
        (row "f3" "X: expected row-major, got col-major 4x3" (blank-matrix 4 3 'float32 #:layout 'col-major))
        (row "f4" "X: expected at least 1 row, got 0" (blank-matrix 0 3 'float32))
        (row "f1 before f2" "X: expected float32 or float64, got int32" (blank-vector 4 'int32))
        (row "f2 before f4" "X: expected rank 2, got rank 1" (blank-vector 0 'float64))
        (row "f8 inf"
             "X: row 2, column 1: expected a finite value, got inf"
             (grid-with 2 1 +inf.0))
        (row "f8 nan"
             "X: row 0, column 2: expected a finite value, got nan"
             (grid-with 0 2 +nan.0 'float64))
        (row "f8 -inf"
             "X: row 3, column 0: expected a finite value, got -inf"
             (grid-with 3 0 -inf.0 'float64))
        (row "f8 first in row-major order"
             "X: row 1, column 2: expected a finite value, got nan"
             (rows->matrix '((1 2 3) (4 5 +nan.0) (+inf.0 8 9)) 'float32))))

(define (fit-rows)
  (append (bad-X-rows)
          (list (row "f5" "X: expected at least 1 column, got 0" (blank-matrix 4 0 'float32))
                (row "f4 before f5" "X: expected at least 1 row, got 0" (blank-matrix 0 0 'float64))
                (row "f3 before f8"
                     "X: expected row-major, got col-major 4x3"
                     (blank-matrix 4 3 'float32 #:layout 'col-major)))))

(define (fitted)
  (standard-scaler-fit (rows->matrix grid 'float64)))

(define (float64s xs)
  (list->device-vector xs #:dtype 'float64))

(define (hand-built mean scale)
  (standard-scaler-result mean #f scale 4 3 (and mean #t) (and scale #t)))

(define (applied-rows)
  (define good (float64s '(1 2 3)))
  (append
   (map (lambda (r) (row (row-id r) (row-message r) (cons (fitted) (row-call r)))) (bad-X-rows))
   (list
    (row "t5" "X: expected 3 columns, got 2" (cons (fitted) (blank-matrix 4 2 'float32)))
    (row "t5 with both flags off"
         "X: expected 3 columns, got 2"
         (cons (standard-scaler-fit (rows->matrix grid 'float32) #:with-mean #f #:with-std #f)
               (blank-matrix 4 2 'float64)))
    (row "t6"
         "mean: expected float64, got float32"
         (cons (hand-built (list->device-vector '(1 2 3) #:dtype 'float32) good) (rows->matrix grid 'float32)))
    (row "t7" "mean: expected 3 elements, got 2" (cons (hand-built (float64s '(1 2)) good) (rows->matrix grid 'float32)))
    (row "t8"
         "mean: expected contiguous elements, got stride 2"
         (cons (hand-built (strided-vector 3) good) (rows->matrix grid 'float32)))
    (row "t9"
         "scale: expected float64, got float32"
         (cons (hand-built good (list->device-vector '(1 2 3) #:dtype 'float32)) (rows->matrix grid 'float64)))
    (row "t10" "scale: expected 3 elements, got 2" (cons (hand-built good (float64s '(1 2))) (rows->matrix grid 'float64)))
    (row "t11"
         "scale: expected contiguous elements, got stride 2"
         (cons (hand-built good (strided-vector 3)) (rows->matrix grid 'float64)))
    (row "t12"
         "X: row 0, column 2: expected a finite value, got nan"
         (cons (fitted) (grid-with 0 2 +nan.0)))
    (row "t3 before t5" "X: expected row-major, got col-major 4x2" (cons (fitted) (blank-matrix 4 2 'float32 #:layout 'col-major)))
    (row "t4 before t5" "X: expected at least 1 row, got 0" (cons (fitted) (blank-matrix 0 2 'float32)))
    (row "t5 before t6"
         "X: expected 3 columns, got 2"
         (cons (hand-built (float64s '(1 2)) good) (blank-matrix 4 2 'float32)))
    (row "t8 before t9"
         "mean: expected contiguous elements, got stride 2"
         (cons (hand-built (strided-vector 3) (float64s '(1 2))) (rows->matrix grid 'float32)))
    (row "t11 before t12"
         "scale: expected contiguous elements, got stride 2"
         (cons (hand-built good (strided-vector 3)) (grid-with 1 1 +inf.0 'float64))))))

(define (fit-calls)
  (list (cons 'standard-scaler-fit standard-scaler-fit)
        (cons 'standard-scaler-fit-transform standard-scaler-fit-transform)
        (cons 'fit (lambda (X) (fit (standard-scaler) X)))
        (cons 'fit-transform (lambda (X) (fit-transform (standard-scaler) X)))))

(define (applied-calls)
  (list (cons 'standard-scaler-transform standard-scaler-transform)
        (cons 'standard-scaler-inverse-transform standard-scaler-inverse-transform)))

(define (generic-calls)
  (define fitted-estimator (fit (standard-scaler) (rows->matrix grid 'float64)))
  (list (cons 'transform (lambda (X) (transform fitted-estimator X)))
        (cons 'inverse-transform (lambda (X) (inverse-transform fitted-estimator X)))))

(test-gpu "L13: the fits refuse each bad X with its message, the first refusal winning"
  (for* ([entry (in-list (fit-calls))]
         [r (in-list (fit-rows))])
    (test-case (format "~a ~a" (car entry) (row-id r))
      (check-row (car entry) r (lambda () ((cdr entry) (row-call r)))))))

(test-gpu "L13: transform and inverse-transform refuse each bad argument with its message"
  (for* ([entry (in-list (applied-calls))]
         [r (in-list (applied-rows))])
    (test-case (format "~a ~a" (car entry) (row-id r))
      (check-row (car entry) r (lambda () ((cdr entry) (car (row-call r)) (cdr (row-call r))))))))

(test-gpu "L13: the generics refuse a bad X with their own names"
  (for* ([entry (in-list (generic-calls))]
         [r (in-list (append (bad-X-rows)
                             (list (row "t5" "X: expected 3 columns, got 2" (blank-matrix 4 2 'float32)))))])
    (test-case (format "~a ~a" (car entry) (row-id r))
      (check-row (car entry) r (lambda () ((cdr entry) (row-call r)))))))

(test-gpu "L13: the estimator refuses an unfitted use and a sample weight"
  (define X (rows->matrix grid 'float32))
  (define w (list->device-vector '(1 1 1 1) #:dtype 'float32))
  (check-cuml-error 'logic "transform: standard-scaler is not fitted" (lambda () (transform (standard-scaler) X)))
  (check-cuml-error 'logic
                    "inverse-transform: standard-scaler is not fitted"
                    (lambda () (inverse-transform (standard-scaler) X)))
  (check-cuml-error 'logic
                    "fit: standard-scaler takes no sample-weight"
                    (lambda () (fit (standard-scaler) X #:sample-weight w)))
  (check-cuml-error 'logic
                    "fit-transform: standard-scaler takes no sample-weight"
                    (lambda () (fit-transform (standard-scaler) X #:sample-weight w))))

(test-gpu "the private scratch parameter takes only an integer from 1 to 2^62"
  (for ([bad (in-list (list 0 -1 (add1 (expt 2 62)) 2.5 'many))])
    (check-exn exn:fail:contract? (lambda () (current-scratch-elements bad))))
  (check-equal? (current-scratch-elements) (expt 2 26))
  (check-equal? (parameterize ([current-scratch-elements 1])
                  (current-scratch-elements))
                1))

(test-pools "L13: a refusal leaves the RMM pool where it was"
  (define device 0)
  (define (unchanged id thunk)
    (drain-finalizers!)
    (define before (probe-pool-used device))
    (thunk)
    (drain-finalizers!)
    (check-equal? (probe-pool-used device) before id))
  (for ([r (in-list (fit-rows))])
    (define X (row-call r))
    (unchanged (row-id r) (lambda () (check-row 'standard-scaler-fit r (lambda () (standard-scaler-fit X))))))
  (for ([r (in-list (applied-rows))])
    (define args (row-call r))
    (unchanged (row-id r)
               (lambda ()
                 (check-row 'standard-scaler-transform
                            r
                            (lambda () (standard-scaler-transform (car args) (cdr args))))))))
