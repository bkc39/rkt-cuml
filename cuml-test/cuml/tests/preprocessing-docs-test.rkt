#lang racket/base

(require (only-in cuml fit standard-scaler transform)
         (only-in cuml/preprocessing
                  standard-scaler-fit
                  standard-scaler-inverse-transform
                  standard-scaler-params
                  standard-scaler-result-mean
                  standard-scaler-result-scale
                  standard-scaler-result-var
                  standard-scaler-transform)
         (only-in rackunit check-equal? check-false)
         (only-in raft device-matrix->list* device-vector->list list*->device-matrix)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/cuml-error.rkt" check-cuml-error))

(define grid '((1 10) (3 10) (1 14) (3 14)))
(define unit '((-1.0 -1.0) (1.0 -1.0) (-1.0 1.0) (1.0 1.0)))

(define (X)
  (list*->device-matrix grid #:dtype 'float64))

(test-gpu "the manual: the standard-scaling example's statistics, transform and inverse"
  (define fitted (standard-scaler-fit (X)))
  (check-equal? (map device-vector->list
                     (list (standard-scaler-result-mean fitted)
                           (standard-scaler-result-var fitted)
                           (standard-scaler-result-scale fitted)))
                '((2.0 12.0) (1.0 4.0) (1.0 2.0)))
  (check-equal? (device-matrix->list* (standard-scaler-transform fitted (X))) unit)
  (check-equal? (device-matrix->list*
                 (standard-scaler-inverse-transform fitted (standard-scaler-transform fitted (X))))
                '((1.0 10.0) (3.0 10.0) (1.0 14.0) (3.0 14.0)))
  (check-cuml-error
   'logic
   "standard-scaler-transform: X: expected 2 columns, got 1"
   (lambda () (standard-scaler-transform fitted (list*->device-matrix '((1)) #:dtype 'float64)))))

(test-gpu "the manual: a constant column scales by 1, and inf is refused"
  (define C (list*->device-matrix '((5 1) (5 3)) #:dtype 'float32))
  (define fitted (standard-scaler-fit C))
  (check-equal? (device-vector->list (standard-scaler-result-scale fitted)) '(1.0 1.0))
  (check-equal? (device-matrix->list* (standard-scaler-transform fitted C)) '((0.0 -1.0) (0.0 1.0)))
  (check-cuml-error 'logic
                    "standard-scaler-fit: X: row 0, column 1: expected a finite value, got inf"
                    (lambda ()
                      (standard-scaler-fit (list*->device-matrix '((1 +inf.0)) #:dtype 'float64)))))

(test-gpu "the manual: standard-scaler-params and the estimator"
  (check-false (hash-ref (standard-scaler-params #:with-mean #f) 'with-mean))
  (check-cuml-error 'logic
                    "standard-scaler-params: standard-scaler has no parameter copy"
                    (lambda () (standard-scaler-params (hasheq 'copy #t))))
  (check-equal? (device-matrix->list* (transform (fit (standard-scaler) (X)) (X))) unit)
  (check-cuml-error 'logic
                    "transform: standard-scaler is not fitted"
                    (lambda () (transform (standard-scaler) (X)))))
