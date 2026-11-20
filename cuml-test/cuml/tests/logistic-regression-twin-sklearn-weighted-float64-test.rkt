#lang racket/base

(require (only-in "private/logistic-sklearn-twin.rkt" check-weighted-sklearn-twins)
         (only-in "private/python-env.rkt" test-twin))

(test-twin "float64: weighted multiclass fits are as near sklearn's optimum as cuML's fit of the repeated rows"
  (check-weighted-sklearn-twins 'float64))
