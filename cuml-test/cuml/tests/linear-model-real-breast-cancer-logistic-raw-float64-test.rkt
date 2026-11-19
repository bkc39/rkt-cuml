#lang racket/base

(require (only-in "private/python-env.rkt" test-twin)
         (only-in "private/real-logistic-data.rkt" check-raw-refusal raw-breast-cancer))

(test-twin "float64 raw: refused, and Python cuML's point off the optimum"
  (check-raw-refusal raw-breast-cancer 'float64))
