#lang racket/base

(require (only-in "private/python-env.rkt" test-twin)
         (only-in "private/real-logistic-data.rkt" check-raw-refusal check-sklearn-optimum labelled-set))

(define penalties '((l2 1.0 #f) (l1 1.0 #f) (elasticnet 1.0 0.5)))

(test-twin "float64: standardised breast-cancer fits are stationary at 128 bits, l2 within 1e-5 of sklearn's, and as accurate as sklearn's to a row (R1)"
  (check-sklearn-optimum (labelled-set 'breast-cancer "diagnosis" '("id") #:standardise? #t) 'float64 penalties))

(test-twin "float64: raw breast-cancer is refused, and Python cuML's point there is off the optimum or the bound (R2)"
  (check-raw-refusal (labelled-set 'breast-cancer "diagnosis" '("id")) 'float64))
