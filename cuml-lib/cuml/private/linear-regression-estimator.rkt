#lang racket/base

(require (only-in cuml/linear-model/private/linear-regression
                  linear-regression-fit/who
                  linear-regression-params
                  linear-regression-predict/who)
         (only-in "estimator-support.rkt" define-predictor))

(provide linear-regression
         linear-regression-model
         linear-regression?)

(define-predictor linear-regression
  #:hyperparameters linear-regression-params
  #:fit linear-regression-fit/who
  #:predict linear-regression-predict/who)
