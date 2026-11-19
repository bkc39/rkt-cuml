#lang racket/base

(require (only-in "private/python-env.rkt" test-twin)
         (only-in "private/raw-breast-cancer.rkt" raw-breast-cancer)
         (only-in "private/real-logistic-data.rkt" check-raw-refusal))

(test-twin "float32 raw: refused, and Python cuML's point off the optimum"
  (check-raw-refusal raw-breast-cancer 'float32))
