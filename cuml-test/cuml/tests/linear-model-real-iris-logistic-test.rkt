#lang racket/base

(require (only-in cuml/linear-model logistic-regression-fit)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/cuml-error.rkt" check-cuml-error)
         (only-in "private/linear-data.rkt" on-matrix)
         (only-in "private/logistic-data.rkt" default-setting lr-case-X unequal-weights-message)
         (only-in "private/python-env.rkt" test-twin)
         (only-in "private/real-logistic-data.rkt"
                  check-score
                  check-sklearn-optimum
                  labelled-labels
                  labelled-set
                  labelled-case))

(define iris (labelled-set 'iris "species" '()))

(test-twin "iris: the fit is at sklearn's optimum"
  (check-sklearn-optimum iris 'float64 (list default-setting)))

(test-twin "iris: the estimator scores as sklearn"
  (check-score iris))

(test-gpu "wine: balanced class weights over unequal cultivars are refused"
  (define wine (labelled-set 'wine "cultivar" '()))
  (check-cuml-error 'logic
                    (string-append "logistic-regression-fit: " unequal-weights-message)
                    (lambda ()
                      (logistic-regression-fit (on-matrix (lr-case-X (labelled-case wine)))
                                               (labelled-labels wine)
                                               #:class-weight 'balanced))))
