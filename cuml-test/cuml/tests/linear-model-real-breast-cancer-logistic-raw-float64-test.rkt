#lang racket/base

(require (only-in "private/python-env.rkt" test-twin)
         (only-in "private/real-logistic-data.rkt" check-raw-fit labelled-set))

(test-twin "float64: raw breast-cancer fits near sklearn's optimum, where Python cuML's point is off it"
  (check-raw-fit (labelled-set 'breast-cancer "diagnosis" '("id")) 'float64))
