#lang racket/base

(require (only-in cuml/metrics accuracy-score mean-squared-error r2-score)
         (only-in rackunit test-case)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/kmeans-data.rkt" blank-matrix blank-vector)
         (only-in "private/refusal-rows.rkt" check-row row row-id)
         (only-in "private/scaler-data.rkt" strided-vector)
         (only-in "private/scores-data.rkt" rows vec))

(define (weight-rows proc y p)
  (define (with w)
    (lambda () (proc y p #:sample-weight w)))
  (list
   (row "V9 dtype"
        "sample-weight: expected float32 or float64, got int32"
        (with (vec '(1 1 1 1) 'int32)))
   (row "V9 rank" "sample-weight: expected rank 1, got rank 2" (with (blank-matrix 4 1 'float64)))
   (row "V9 length" "sample-weight: expected 4 elements, got 3" (with (vec '(1 1 1) 'float64)))
   (row "V9 stride"
        "sample-weight: expected contiguous elements, got stride 2"
        (with (strided-vector 4 'float32)))
   (row "V10 weights"
        "sample-weight: element 0: expected a finite value, got -inf"
        (with (vec '(-inf.0 1 1 1) 'float64)))
   (row "V11"
        "sample-weight: element 1: expected at least 0, got -1"
        (with (vec '(1 -1 1 1) 'float32)))
   (row "V12" "sample-weight: expected a positive sum, got 0" (with (vec '(0 0 0 0) 'float64)))
   (row "V10 before V11"
        "sample-weight: element 1: expected a finite value, got nan"
        (with (vec '(-1 +nan.0 1 1) 'float64)))
   (row "V11 before V12"
        "sample-weight: element 1: expected at least 0, got -1"
        (with (vec '(0 -1 0 0) 'float64)))))

(define (value-rows proc)
  (define y (vec '(1 2 3 4) 'float64))
  (define p (vec '(1 2 3 5) 'float32))
  (append
   (list
    (row "V1"
         "y-true: expected float32 or float64, got int32"
         (lambda () (proc (vec '(1 2 3 4) 'int32) p)))
    (row "V3"
         "y-true: expected contiguous elements, got stride 2"
         (lambda () (proc (strided-vector 4) p)))
    (row "V4"
         "y-pred: expected float32 or float64, got int64"
         (lambda () (proc y (vec '(1 2 3 4) 'int64))))
    (row "V5" "y-pred: expected rank 1, got rank 2" (lambda () (proc y (blank-matrix 4 1 'float64))))
    (row "V6" "y-pred: expected 4 elements, got 3" (lambda () (proc y (vec '(1 2 3) 'float64))))
    (row "V7"
         "y-pred: expected contiguous elements, got stride 2"
         (lambda () (proc y (strided-vector 4 'float32))))
    (row "V8"
         "y-true: expected at least 1 element, got 0"
         (lambda () (proc (blank-vector 0 'float64) (blank-vector 0 'float32))))
    (row "V10 true"
         "y-true: element 1: expected a finite value, got nan"
         (lambda () (proc (vec '(1 +nan.0 3 4) 'float32) p)))
    (row "V10 pred"
         "y-pred: element 3: expected a finite value, got inf"
         (lambda () (proc y (vec '(1 2 3 +inf.0) 'float64))))
    (row "V1 before V6"
         "y-true: expected float32 or float64, got int32"
         (lambda () (proc (vec '(1 2 3 4) 'int32) (vec '(1) 'float64))))
    (row "V6 before V8"
         "y-pred: expected 0 elements, got 3"
         (lambda () (proc (blank-vector 0 'float64) (vec '(1 2 3) 'float64))))
    (row
     "V8 before V9"
     "y-true: expected at least 1 element, got 0"
     (lambda ()
       (proc (blank-vector 0 'float64) (blank-vector 0 'float64) #:sample-weight (vec '(1) 'int32))))
    (row "V9 before V10"
         "sample-weight: expected float32 or float64, got int32"
         (lambda () (proc (vec '(1 +nan.0 3 4) 'float64) p #:sample-weight (vec '(1 1 1 1) 'int32))))
    (row "V10 true before pred"
         "y-true: element 0: expected a finite value, got -inf"
         (lambda ()
           (proc (vec '(-inf.0 2 3 4) 'float64)
                 (vec '(+nan.0 2 3 4) 'float64)
                 #:sample-weight (vec '(+nan.0 1 1 1) 'float64))))
    (row "V10 pred before weights"
         "y-pred: element 0: expected a finite value, got nan"
         (lambda ()
           (proc y (vec '(+nan.0 2 3 4) 'float64) #:sample-weight (vec '(+nan.0 1 1 1) 'float64)))))
   (weight-rows proc y p)))

(define (r2-rows)
  (cons (row "V2"
             "y-true: expected rank 1, got rank 2"
             (lambda () (r2-score (blank-matrix 4 1 'float64) (blank-vector 4 'float64))))
        (value-rows r2-score)))

(define (matrix-rows)
  (define Y (rows '((1 2) (3 4) (5 6) (7 8)) 'float64))
  (define P (rows '((1 2) (3 4) (5 6) (7 9)) 'float32))
  (list (row "V3 matrix"
             "y-true: expected row-major, got col-major 4x2"
             (lambda () (mean-squared-error (blank-matrix 4 2 'float64 #:layout 'col-major) P)))
        (row "V5 matrix"
             "y-pred: expected rank 2, got rank 1"
             (lambda () (mean-squared-error Y (blank-vector 4 'float64))))
        (row "V6 rows"
             "y-pred: expected 4 rows, got 3"
             (lambda () (mean-squared-error Y (blank-matrix 3 2 'float64))))
        (row "V6 columns"
             "y-pred: expected 2 columns, got 3"
             (lambda () (mean-squared-error Y (blank-matrix 4 3 'float64))))
        (row "V7 matrix"
             "y-pred: expected row-major, got col-major 4x2"
             (lambda () (mean-squared-error Y (blank-matrix 4 2 'float32 #:layout 'col-major))))
        (row "V8 rows"
             "y-true: expected at least 1 row, got 0"
             (lambda () (mean-squared-error (blank-matrix 0 2 'float64) (blank-matrix 0 2 'float64))))
        (row "V8 columns"
             "y-true: expected at least 1 column, got 0"
             (lambda () (mean-squared-error (blank-matrix 4 0 'float64) (blank-matrix 4 0 'float64))))
        (row "V10 matrix"
             "y-true: row 1, column 0: expected a finite value, got inf"
             (lambda () (mean-squared-error (rows '((1 2) (+inf.0 4) (5 6) (7 8)) 'float32) P)))
        (row "V10 matrix pred"
             "y-pred: row 3, column 1: expected a finite value, got nan"
             (lambda () (mean-squared-error Y (rows '((1 2) (3 4) (5 6) (7 +nan.0)) 'float64))))
        (row "V9 matrix length"
             "sample-weight: expected 4 elements, got 3"
             (lambda () (mean-squared-error Y P #:sample-weight (vec '(1 1 1) 'float64))))))

(define (accuracy-rows)
  (define a (vec '(0 1 2 3) 'int32))
  (define b (vec '(0 1 2 4) 'int64))
  (append (list (row "V1"
                     "y-true: expected int32 or int64, got float32"
                     (lambda () (accuracy-score (blank-vector 4 'float32) b)))
                (row "V2"
                     "y-true: expected rank 1, got rank 2"
                     (lambda () (accuracy-score (blank-matrix 4 2 'int32) b)))
                (row "V3"
                     "y-true: expected contiguous elements, got stride 2"
                     (lambda () (accuracy-score (strided-vector 4 'int64) b)))
                (row "V4"
                     "y-pred: expected int32 or int64, got float64"
                     (lambda () (accuracy-score a (blank-vector 4 'float64))))
                (row "V5"
                     "y-pred: expected rank 1, got rank 2"
                     (lambda () (accuracy-score a (blank-matrix 4 1 'int64))))
                (row "V6"
                     "y-pred: expected 4 elements, got 3"
                     (lambda () (accuracy-score a (vec '(0 1 2) 'int32))))
                (row "V7"
                     "y-pred: expected contiguous elements, got stride 2"
                     (lambda () (accuracy-score a (strided-vector 4 'int32))))
                (row "V8"
                     "y-true: expected at least 1 element, got 0"
                     (lambda () (accuracy-score (blank-vector 0 'int32) (blank-vector 0 'int64))))
                (row "V1 before V6"
                     "y-true: expected int32 or int64, got float32"
                     (lambda () (accuracy-score (blank-vector 4 'float32) (vec '(1) 'int32))))
                (row "V6 before V8"
                     "y-pred: expected 0 elements, got 3"
                     (lambda () (accuracy-score (blank-vector 0 'int32) (vec '(1 2 3) 'int32))))
                (row "V8 before V9"
                     "y-true: expected at least 1 element, got 0"
                     (lambda ()
                       (accuracy-score (blank-vector 0 'int32)
                                       (blank-vector 0 'int32)
                                       #:sample-weight (vec '(1) 'int32)))))
          (weight-rows accuracy-score a b)))

(test-gpu "B1: r2-score, mean-squared-error and accuracy-score refuse each bad argument by name"
  (for* ([entry (in-list (list (cons r2-score (r2-rows))
                               (cons mean-squared-error
                                     (append (value-rows mean-squared-error) (matrix-rows)))
                               (cons accuracy-score (accuracy-rows))))]
         [r (in-list (cdr entry))])
    (define proc (car entry))
    (test-case (format "~a ~a" (object-name proc) (row-id r))
      (check-row (object-name proc) r))))
