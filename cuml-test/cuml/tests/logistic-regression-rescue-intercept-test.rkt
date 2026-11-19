#lang racket/base

(require (only-in cuml/linear-model
                  logistic-regression-fit
                  logistic-regression-result-coef
                  logistic-regression-result-intercept
                  logistic-regression-result-n-iter)
         (only-in racket/match match-define)
         (only-in rackunit check-exn check-true)
         (only-in raft device-matrix->list* device-vector->list)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/linear-data.rkt" on-matrix)
         (only-in "private/logistic-data.rkt" lr-case-X lr-case-y)
         (only-in "private/logistic-oracle.rkt" residual residual-of)
         (only-in "private/real-logistic-data.rkt" labelled-case labelled-labels labelled-set))

(define stationary-bound 1e-7)
(define room 20000)
(define short 6000)

(define (raw-breast-cancer)
  (labelled-set 'breast-cancer "diagnosis" '("id")))

(define (fit set max-iter)
  (logistic-regression-fit (on-matrix (lr-case-X (labelled-case set)) 'float64)
                           (labelled-labels set)
                           #:penalty 'none
                           #:max-iter max-iter))

(test-gpu "raw breast-cancer under 'none with an intercept fits to the real-data stationarity bound at max-iter 20000"
  (define set (raw-breast-cancer))
  (define r (fit set room))
  (match-define (residual K F _)
    (residual-of (lr-case-X (labelled-case set))
                 (lr-case-y (labelled-case set))
                 #f
                 (device-matrix->list* (logistic-regression-result-coef r))
                 (device-vector->list (logistic-regression-result-intercept r))
                 #:penalty 'none
                 #:tol 1e-4))
  (check-true (<= K (* stationary-bound (max F 1)))
              (format "K ~a F ~a" (exact->inexact K) (exact->inexact F)))
  (check-true (<= (logistic-regression-result-n-iter r) room)
              (format "n-iter ~a" (logistic-regression-result-n-iter r))))

(test-gpu "raw breast-cancer under 'none with an intercept: a rescue that spends max-iter 6000 says raise it"
  (check-exn #rx"after 6000 iterations \\(max-iter 6000, tol 1e-04\\); raise max-iter$"
             (lambda () (fit (raw-breast-cancer) short))))
