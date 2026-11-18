#lang racket/base

(require (only-in cuml/linear-model/private/mbsgd
                  mbsgd-classifier-decision-function/who
                  mbsgd-classifier-fit/who
                  mbsgd-classifier-params
                  mbsgd-classifier-predict/who)
         (only-in "estimator-support.rkt" accuracy-scorer define-predictor))

(provide mbsgd-classifier
         mbsgd-classifier-model
         mbsgd-classifier?)

(define-predictor mbsgd-classifier
  #:hyperparameters mbsgd-classifier-params
  #:fit mbsgd-classifier-fit/who
  #:predict mbsgd-classifier-predict/who
  #:no-sample-weight
  #:scorer accuracy-scorer
  #:decision-function mbsgd-classifier-decision-function/who)
