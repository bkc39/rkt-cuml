#lang racket/base

(require (only-in cuml/linear-model/private/ridge ridge-fit/who ridge-params ridge-predict/who)
         (only-in "estimator-support.rkt" define-estimator r2-scorer))

(provide ridge
         ridge-model
         ridge?)

(define-estimator ridge
  #:hyperparameters ridge-params
  #:fit ridge-fit/who
  #:needs-y
  #:settings (resources)
  #:apply values
  [predict ridge-predict/who]
  [score (r2-scorer ridge-predict/who) #:sample-weight])
