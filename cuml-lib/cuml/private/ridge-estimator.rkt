#lang racket/base

(require (only-in cuml/linear-model/private/ridge ridge-fit/who ridge-params ridge-predict/who)
         (only-in "estimator-support.rkt" define-predictor))

(provide ridge
         ridge-model
         ridge?)

(define-predictor ridge
  #:hyperparameters ridge-params
  #:fit ridge-fit/who
  #:predict ridge-predict/who)
