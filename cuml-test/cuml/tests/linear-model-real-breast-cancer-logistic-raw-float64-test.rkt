#lang racket/base

(require (only-in "private/python-env.rkt" test-twin)
         (only-in "private/real-logistic-data.rkt" check-raw-refusal labelled-set))

(test-twin "float64: raw breast-cancer is refused, and Python cuML's point there is off the optimum"
  (check-raw-refusal (labelled-set 'breast-cancer "diagnosis" '("id")) 'float64))
