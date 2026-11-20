#lang racket/base

(require (only-in "private/logistic-sklearn-twin.rkt" check-weighted-sklearn-twins)
         (only-in "private/python-env.rkt" test-twin))

(test-twin "float64: weighted multiclass fits near sklearn's optimum"
  (check-weighted-sklearn-twins 'float64))
