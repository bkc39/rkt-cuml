#lang racket/base

(require (only-in "private/logistic-data.rkt" default-setting)
         (only-in "private/python-env.rkt" test-twin)
         (only-in "private/real-logistic-data.rkt" check-score check-sklearn-optimum labelled-set))

(define iris (labelled-set 'iris "species" '()))

(test-twin "iris: the fit is at sklearn's optimum"
  (check-sklearn-optimum iris 'float64 (list default-setting)))

(test-twin "iris: the estimator scores as sklearn"
  (check-score iris))
