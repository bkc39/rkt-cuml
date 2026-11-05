#lang racket/base

(require (only-in cuml/linear-model linear-regression-fit linear-regression-predict)
         (only-in racket/list make-list)
         (only-in "private/linear-data.rkt" lcase-X lcase-w lcase-y normal-case on-matrix on-vector)
         (only-in "private/process-memory.rkt" test-fit-memory))

(define (data)
  (define c (normal-case 2000 7 0 #:weighted? #t))
  (list (on-matrix (lcase-X c) 'float32)
        (on-vector (lcase-y c) 'float32)
        (on-vector (lcase-w c) 'float32)
        (on-matrix '((1 2) (3 4)) 'float32)))

(define (one-round X y w _)
  (linear-regression-predict (linear-regression-fit X y #:sample-weight w #:algorithm 'qr) X))

(define (refusals X y _ X2)
  (define nan-X (on-matrix (cons '(+nan.0 0 0 0 0 0 0) (make-list 1999 '(1 1 1 1 1 1 1))) 'float32))
  (define no-weight (on-vector (make-list 2000 0) 'float32))
  (define fitted (linear-regression-fit X y))
  (list (list "linear-regression-fit: X: row 0, column 0: expected a finite value, got nan"
              (lambda () (linear-regression-fit nan-X y)))
        (list "linear-regression-fit: sample-weight: expected a positive sum, got 0"
              (lambda () (linear-regression-fit X y #:sample-weight no-weight)))
        (list "linear-regression-predict: X: expected 7 columns, got 2"
              (lambda () (linear-regression-predict fitted X2)))))

(test-fit-memory "linear-regression"
                 #:rounds 50
                 #:sample-every 10
                 #:data data
                 #:round one-round
                 #:refusals refusals)
