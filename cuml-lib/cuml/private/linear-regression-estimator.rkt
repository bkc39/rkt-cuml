#lang racket/base

(require (only-in cuml/linear-model/private/linear-regression
                  linear-regression-fit/who
                  linear-regression-params
                  linear-regression-predict/who)
         (only-in "estimator-support.rkt" define-estimator r2-scorer))

(provide linear-regression
         linear-regression-model
         linear-regression?)

(define-estimator linear-regression
  #:hyperparameters linear-regression-params
  #:fit linear-regression-fit/who
  #:needs-y
  #:settings (resources)
  #:apply values
  [predict linear-regression-predict/who]
  [score (r2-scorer linear-regression-predict/who) #:sample-weight])
