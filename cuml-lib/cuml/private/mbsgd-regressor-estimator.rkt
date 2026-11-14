#lang racket/base

(require (only-in cuml/linear-model/private/mbsgd
                  mbsgd-regressor-fit/who
                  mbsgd-regressor-params
                  mbsgd-regressor-predict/who)
         (only-in "estimator-support.rkt" define-predictor))

(provide mbsgd-regressor
         mbsgd-regressor-model
         mbsgd-regressor?)

(define-predictor mbsgd-regressor
  #:hyperparameters mbsgd-regressor-params
  #:fit mbsgd-regressor-fit/who
  #:predict mbsgd-regressor-predict/who
  #:no-sample-weight)
