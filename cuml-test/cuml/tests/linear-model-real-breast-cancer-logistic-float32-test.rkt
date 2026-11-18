#lang racket/base

(require (only-in "private/logistic-data.rkt" setting)
         (only-in "private/python-env.rkt" test-twin)
         (only-in "private/real-logistic-data.rkt" check-sklearn-optimum labelled-set))

(define settings (list (setting 'l2 1.0 #f) (setting 'l1 1.0 #f) (setting 'elasticnet 1.0 0.5)))

(test-twin "float32: standardised breast-cancer fits near the float64 fit and as accurate as sklearn's"
  (check-sklearn-optimum (labelled-set 'breast-cancer "diagnosis" '("id") #:standardise? #t)
                         'float32
                         settings))
