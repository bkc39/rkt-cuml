#lang racket/base

(require (only-in "private/logistic-data.rkt" setting)
         (only-in "private/python-env.rkt" test-twin)
         (only-in "private/real-logistic-data.rkt"
                  check-raw-refusal
                  check-sklearn-optimum
                  labelled-set))

(define settings (list (setting 'l2 1.0 #f) (setting 'l1 1.0 #f) (setting 'elasticnet 1.0 0.5)))

(test-twin "float32: standardised breast-cancer fits near the float64 fit and as accurate as sklearn's"
  (check-sklearn-optimum (labelled-set 'breast-cancer "diagnosis" '("id") #:standardise? #t)
                         'float32
                         settings))

(test-twin "float32: raw breast-cancer is refused, and Python cuML's point there is off the optimum"
  (check-raw-refusal (labelled-set 'breast-cancer "diagnosis" '("id")) 'float32))
