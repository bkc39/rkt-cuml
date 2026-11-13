#lang racket/base

(require (only-in cuml/linear-model logistic-regression-fit)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/cuml-error.rkt" check-cuml-error)
         (only-in "private/linear-data.rkt" on-matrix)
         (only-in "private/logistic-data.rkt" lr-case-X setting)
         (only-in "private/python-env.rkt" test-twin)
         (only-in "private/real-logistic-data.rkt"
                  check-score
                  check-sklearn-optimum
                  labelled-labels
                  labelled-set
                  labelled-case))

(define iris (labelled-set 'iris "species" '()))

(test-twin "iris: stationary, near sklearn's, predicting strings, and scoring as sklearn"
  (check-sklearn-optimum iris 'float64 (list (setting 'l2 1.0 #f)))
  (check-score iris))

(test-gpu "wine: balanced class weights over unequal cultivars are refused"
  (define wine (labelled-set 'wine "cultivar" '()))
  (check-cuml-error
   'logic
   "logistic-regression-fit: sample-weight: weights with 3 or more classes must be equal on every row (cuML's softmax ignores them); got unequal weights"
   (lambda ()
     (logistic-regression-fit (on-matrix (lr-case-X (labelled-case wine)))
                              (labelled-labels wine)
                              #:class-weight 'balanced))))
