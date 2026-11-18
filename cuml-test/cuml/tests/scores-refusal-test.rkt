#lang racket/base

(require (only-in cuml/metrics accuracy-score mean-squared-error r2-score)
         (only-in rackunit test-case)
         (only-in raft device-matrix device-vector list*->device-matrix list->device-vector)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/refusal-rows.rkt" check-row row row-id)
         (only-in "private/scaler-data.rkt" strided-vector))

(define (weight-rows proc y p)
  (define (with w)
    (lambda () (proc y p #:sample-weight w)))
  (list (row "V9 dtype"
             "sample-weight: expected float32 or float64, got int32"
             (with (list->device-vector '(1 1 1 1) #:dtype 'int32)))
        (row "V9 rank"
             "sample-weight: expected rank 1, got rank 2"
             (with (device-matrix 4 1 #:dtype 'float64)))
        (row "V9 length"
             "sample-weight: expected 4 elements, got 3"
             (with (list->device-vector '(1 1 1) #:dtype 'float64)))
        (row "V9 stride"
             "sample-weight: expected contiguous elements, got stride 2"
             (with (strided-vector 4 'float32)))
        (row "V10 weights"
             "sample-weight: element 0: expected a finite value, got -inf"
             (with (list->device-vector '(-inf.0 1 1 1) #:dtype 'float64)))
        (row "V11"
             "sample-weight: element 1: expected at least 0, got -1"
             (with (list->device-vector '(1 -1 1 1) #:dtype 'float32)))
        (row "V12"
             "sample-weight: expected a positive sum, got 0"
             (with (list->device-vector '(0 0 0 0) #:dtype 'float64)))
        (row "V10 before V11"
             "sample-weight: element 1: expected a finite value, got nan"
             (with (list->device-vector '(-1 +nan.0 1 1) #:dtype 'float64)))
        (row "V11 before V12"
             "sample-weight: element 1: expected at least 0, got -1"
             (with (list->device-vector '(0 -1 0 0) #:dtype 'float64)))))

(define (value-rows proc)
  (define y (list->device-vector '(1 2 3 4) #:dtype 'float64))
  (define p (list->device-vector '(1 2 3 5) #:dtype 'float32))
  (append
   (list
    (row "V1"
         "y-true: expected float32 or float64, got int32"
         (lambda () (proc (list->device-vector '(1 2 3 4) #:dtype 'int32) p)))
    (row "V3"
         "y-true: expected contiguous elements, got stride 2"
         (lambda () (proc (strided-vector 4) p)))
    (row "V4"
         "y-pred: expected float32 or float64, got int64"
         (lambda () (proc y (list->device-vector '(1 2 3 4) #:dtype 'int64))))
    (row "V5"
         "y-pred: expected rank 1, got rank 2"
         (lambda () (proc y (device-matrix 4 1 #:dtype 'float64))))
    (row "V6"
         "y-pred: expected 4 elements, got 3"
         (lambda () (proc y (list->device-vector '(1 2 3) #:dtype 'float64))))
    (row "V7"
         "y-pred: expected contiguous elements, got stride 2"
         (lambda () (proc y (strided-vector 4 'float32))))
    (row "V8"
         "y-true: expected at least 1 element, got 0"
         (lambda () (proc (device-vector 0 #:dtype 'float64) (device-vector 0 #:dtype 'float32))))
    (row "V10 true"
         "y-true: element 1: expected a finite value, got nan"
         (lambda () (proc (list->device-vector '(1 +nan.0 3 4) #:dtype 'float32) p)))
    (row "V10 pred"
         "y-pred: element 3: expected a finite value, got inf"
         (lambda () (proc y (list->device-vector '(1 2 3 +inf.0) #:dtype 'float64))))
    (row "V1 before V6"
         "y-true: expected float32 or float64, got int32"
         (lambda ()
           (proc (list->device-vector '(1 2 3 4) #:dtype 'int32)
                 (list->device-vector '(1) #:dtype 'float64))))
    (row "V6 before V8"
         "y-pred: expected 0 elements, got 3"
         (lambda ()
           (proc (device-vector 0 #:dtype 'float64) (list->device-vector '(1 2 3) #:dtype 'float64))))
    (row "V8 before V9"
         "y-true: expected at least 1 element, got 0"
         (lambda ()
           (proc (device-vector 0 #:dtype 'float64)
                 (device-vector 0 #:dtype 'float64)
                 #:sample-weight (list->device-vector '(1) #:dtype 'int32))))
    (row "V9 before V10"
         "sample-weight: expected float32 or float64, got int32"
         (lambda ()
           (proc (list->device-vector '(1 +nan.0 3 4) #:dtype 'float64)
                 p
                 #:sample-weight (list->device-vector '(1 1 1 1) #:dtype 'int32))))
    (row "V10 true before pred"
         "y-true: element 0: expected a finite value, got -inf"
         (lambda ()
           (proc (list->device-vector '(-inf.0 2 3 4) #:dtype 'float64)
                 (list->device-vector '(+nan.0 2 3 4) #:dtype 'float64)
                 #:sample-weight (list->device-vector '(+nan.0 1 1 1) #:dtype 'float64))))
    (row "V10 pred before weights"
         "y-pred: element 0: expected a finite value, got nan"
         (lambda ()
           (proc y
                 (list->device-vector '(+nan.0 2 3 4) #:dtype 'float64)
                 #:sample-weight (list->device-vector '(+nan.0 1 1 1) #:dtype 'float64)))))
   (weight-rows proc y p)))

(define (r2-rows)
  (cons (row "V2"
             "y-true: expected rank 1, got rank 2"
             (lambda ()
               (r2-score (device-matrix 4 1 #:dtype 'float64) (device-vector 4 #:dtype 'float64))))
        (value-rows r2-score)))

(define (matrix-rows)
  (define Y (list*->device-matrix '((1 2) (3 4) (5 6) (7 8)) #:dtype 'float64))
  (define P (list*->device-matrix '((1 2) (3 4) (5 6) (7 9)) #:dtype 'float32))
  (list
   (row "V3 matrix"
        "y-true: expected row-major, got col-major 4x2"
        (lambda () (mean-squared-error (device-matrix 4 2 #:dtype 'float64 #:layout 'col-major) P)))
   (row "V5 matrix"
        "y-pred: expected rank 2, got rank 1"
        (lambda () (mean-squared-error Y (device-vector 4 #:dtype 'float64))))
   (row "V6 rows"
        "y-pred: expected 4 rows, got 3"
        (lambda () (mean-squared-error Y (device-matrix 3 2 #:dtype 'float64))))
   (row "V6 columns"
        "y-pred: expected 2 columns, got 3"
        (lambda () (mean-squared-error Y (device-matrix 4 3 #:dtype 'float64))))
   (row "V7 matrix"
        "y-pred: expected row-major, got col-major 4x2"
        (lambda () (mean-squared-error Y (device-matrix 4 2 #:dtype 'float32 #:layout 'col-major))))
   (row "V8 rows"
        "y-true: expected at least 1 row, got 0"
        (lambda ()
          (mean-squared-error (device-matrix 0 2 #:dtype 'float64)
                              (device-matrix 0 2 #:dtype 'float64))))
   (row "V8 columns"
        "y-true: expected at least 1 column, got 0"
        (lambda ()
          (mean-squared-error (device-matrix 4 0 #:dtype 'float64)
                              (device-matrix 4 0 #:dtype 'float64))))
   (row "V10 matrix"
        "y-true: row 1, column 0: expected a finite value, got inf"
        (lambda ()
          (mean-squared-error (list*->device-matrix '((1 2) (+inf.0 4) (5 6) (7 8)) #:dtype 'float32)
                              P)))
   (row
    "V10 matrix pred"
    "y-pred: row 3, column 1: expected a finite value, got nan"
    (lambda ()
      (mean-squared-error Y (list*->device-matrix '((1 2) (3 4) (5 6) (7 +nan.0)) #:dtype 'float64))))
   (row "V9 matrix length"
        "sample-weight: expected 4 elements, got 3"
        (lambda ()
          (mean-squared-error Y P #:sample-weight (list->device-vector '(1 1 1) #:dtype 'float64))))))

(define (accuracy-rows)
  (define a (list->device-vector '(0 1 2 3) #:dtype 'int32))
  (define b (list->device-vector '(0 1 2 4) #:dtype 'int64))
  (append (list (row "V1"
                     "y-true: expected int32 or int64, got float32"
                     (lambda () (accuracy-score (device-vector 4 #:dtype 'float32) b)))
                (row "V2"
                     "y-true: expected rank 1, got rank 2"
                     (lambda () (accuracy-score (device-matrix 4 2 #:dtype 'int32) b)))
                (row "V3"
                     "y-true: expected contiguous elements, got stride 2"
                     (lambda () (accuracy-score (strided-vector 4 'int64) b)))
                (row "V4"
                     "y-pred: expected int32 or int64, got float64"
                     (lambda () (accuracy-score a (device-vector 4 #:dtype 'float64))))
                (row "V5"
                     "y-pred: expected rank 1, got rank 2"
                     (lambda () (accuracy-score a (device-matrix 4 1 #:dtype 'int64))))
                (row "V6"
                     "y-pred: expected 4 elements, got 3"
                     (lambda () (accuracy-score a (list->device-vector '(0 1 2) #:dtype 'int32))))
                (row "V7"
                     "y-pred: expected contiguous elements, got stride 2"
                     (lambda () (accuracy-score a (strided-vector 4 'int32))))
                (row "V8"
                     "y-true: expected at least 1 element, got 0"
                     (lambda ()
                       (accuracy-score (device-vector 0 #:dtype 'int32)
                                       (device-vector 0 #:dtype 'int64))))
                (row "V1 before V6"
                     "y-true: expected int32 or int64, got float32"
                     (lambda ()
                       (accuracy-score (device-vector 4 #:dtype 'float32)
                                       (list->device-vector '(1) #:dtype 'int32))))
                (row "V6 before V8"
                     "y-pred: expected 0 elements, got 3"
                     (lambda ()
                       (accuracy-score (device-vector 0 #:dtype 'int32)
                                       (list->device-vector '(1 2 3) #:dtype 'int32))))
                (row "V8 before V9"
                     "y-true: expected at least 1 element, got 0"
                     (lambda ()
                       (accuracy-score (device-vector 0 #:dtype 'int32)
                                       (device-vector 0 #:dtype 'int32)
                                       #:sample-weight (list->device-vector '(1) #:dtype 'int32)))))
          (weight-rows accuracy-score a b)))

(test-gpu "r2-score, mean-squared-error and accuracy-score refuse each bad argument by name"
  (for* ([entry (in-list (list (cons r2-score (r2-rows))
                               (cons mean-squared-error
                                     (append (value-rows mean-squared-error) (matrix-rows)))
                               (cons accuracy-score (accuracy-rows))))]
         [r (in-list (cdr entry))])
    (define proc (car entry))
    (test-case (format "~a ~a" (object-name proc) (row-id r))
      (check-row (object-name proc) r))))
