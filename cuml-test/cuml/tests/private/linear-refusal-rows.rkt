#lang racket/base

(require (only-in racket/list range)
         (only-in racket/match match-define)
         (only-in "kmeans-data.rkt" blank-matrix)
         (only-in "linear-data.rkt" [on-matrix matrix] [on-vector vector])
         (only-in "refusal-rows.rkt" row))

(provide grid
         overflow-X
         overflow-y
         shared-fit-rows
         targets
         with-cell
         with-element)

(define grid '((1 0 2) (0 3 1) (4 1 0) (2 2 5) (3 1 1)))
(define targets '(1 2 3 4 5))

(define (with-cell rows i j value)
  (for/list ([row (in-list rows)]
             [r (in-naturals)])
    (for/list ([x (in-list row)]
               [c (in-naturals)])
      (if (and (= r i) (= c j)) value x))))

(define (with-element xs i value)
  (for/list ([x (in-list xs)]
             [k (in-naturals)])
    (if (= k i) value x)))

(define (overflow-X)
  (matrix (for/list ([r (in-list grid)])
            (match-define (list a b c) r)
            (list a b (* c (expt 2 -100))))
          'float32))

(define (overflow-y)
  (vector (for/list ([r (in-list grid)]
                     [t (in-list targets)])
            (match-define (list _ _ c) r)
            (* (expt 2 60) (+ t c)))
          'float32))

(define (shared-fit-rows go)
  (define X (matrix grid))
  (define y (vector targets))
  (list (row "X dtype"
             "X: expected float32 or float64, got int32"
             (lambda () (go (blank-matrix 5 3 'int32) y)))
        (row "X rank" "X: expected rank 2, got rank 1" (lambda () (go y y)))
        (row "y dtype" "y: expected float32, got float64" (lambda () (go (matrix grid 'float32) y)))
        (row "y length"
             "y: expected 10 elements, got 9"
             (lambda () (go (matrix (append grid grid)) (vector (range 9)))))
        (row "weight length"
             "sample-weight: expected 5 elements, got 4"
             (lambda () (go X y #:sample-weight (vector '(1 1 1 1)))))
        (row "one row"
             "X: expected at least 2 rows, got 1"
             (lambda () (go (matrix '((1 2 3))) (vector '(1)))))
        (row "no column"
             "X: expected at least 1 column, got 0"
             (lambda () (go (blank-matrix 5 0 'float64) y)))
        (row "non-finite X"
             "X: row 3, column 1: expected a finite value, got nan"
             (lambda () (go (matrix (with-cell grid 3 1 +nan.0)) y)))
        (row "non-finite y"
             "y: element 4: expected a finite value, got inf"
             (lambda () (go X (vector (with-element targets 4 +inf.0)))))
        (row "negative weight"
             "sample-weight: element 1: expected at least 0, got -1"
             (lambda () (go X y #:sample-weight (vector '(1 -1 1 1 1)))))
        (row "zero weight sum"
             "sample-weight: expected a positive sum, got 0"
             (lambda () (go X y #:sample-weight (vector '(0 0 0 0 0)))))
        (row "intercept overflow"
             "y: the intercept exceeds the float32 range"
             (lambda ()
               (go (matrix '((1000) (1001) (1002)) 'float32)
                   (vector '(3e38 2.9e38 2.8e38) 'float32))))))
