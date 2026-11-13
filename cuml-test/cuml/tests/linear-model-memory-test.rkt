#lang racket/base

(require (only-in cuml/linear-model
                  linear-regression-fit
                  linear-regression-predict
                  logistic-regression-fit
                  logistic-regression-predict
                  logistic-regression-predict-proba)
         (only-in racket/list make-list)
         (only-in "private/linear-data.rkt" lcase-X lcase-w lcase-y normal-case on-matrix on-vector)
         (only-in "private/logistic-data.rkt" blob-case lr-case-w lr-case-X lr-case-y)
         (only-in "private/process-memory.rkt" test-fit-memory))

(define (setup)
  (define c (normal-case 2000 7 0 #:weighted? #t))
  (define X (on-matrix (lcase-X c) 'float32))
  (define y (on-vector (lcase-y c) 'float32))
  (define w (on-vector (lcase-w c) 'float32))
  (define X2 (on-matrix '((1 2) (3 4)) 'float32))
  (define nan-X (on-matrix (cons '(+nan.0 0 0 0 0 0 0) (make-list 1999 '(1 1 1 1 1 1 1))) 'float32))
  (define no-weight (on-vector (make-list 2000 0) 'float32))
  (define fitted (linear-regression-fit X y))
  (values (lambda ()
            (linear-regression-predict (linear-regression-fit X y #:sample-weight w #:algorithm 'qr)
                                       X))
          (list (list "linear-regression-fit: X: row 0, column 0: expected a finite value, got nan"
                      (lambda () (linear-regression-fit nan-X y)))
                (list "linear-regression-fit: sample-weight: expected a positive sum, got 0"
                      (lambda () (linear-regression-fit X y #:sample-weight no-weight)))
                (list "linear-regression-predict: X: expected 7 columns, got 2"
                      (lambda () (linear-regression-predict fitted X2))))))

(test-fit-memory "linear-regression" #:rounds 50 #:sample-every 10 #:setup setup)

(define (logistic-setup)
  (define c (blob-case "binary" 2000 7 2 61 #:weighted? #t))
  (define X (on-matrix (lr-case-X c) 'float32))
  (define y (on-vector (lr-case-y c) 'int32))
  (define w (on-vector (lr-case-w c) 'float32))
  (define nan-X (on-matrix (cons '(+nan.0 0 0 0 0 0 0) (make-list 1999 '(1 1 1 1 1 1 1))) 'float32))
  (define X2 (on-matrix '((1 2) (3 4)) 'float32))
  (define fitted (logistic-regression-fit X y))
  (values (lambda ()
            (define r (logistic-regression-fit X y #:sample-weight w))
            (logistic-regression-predict r X)
            (logistic-regression-predict-proba r X))
          (list (list "logistic-regression-fit: X: row 0, column 0: expected a finite value, got nan"
                      (lambda () (logistic-regression-fit nan-X y)))
                (list #rx"^logistic-regression-fit: the solver stopped at first-order residual"
                      (lambda () (logistic-regression-fit X y #:max-iter 1)))
                (list "logistic-regression-predict: X: expected 7 columns, got 2"
                      (lambda () (logistic-regression-predict fitted X2))))))

(test-fit-memory "logistic-regression" #:rounds 50 #:sample-every 10 #:setup logistic-setup #:buffers-per-round 6)
