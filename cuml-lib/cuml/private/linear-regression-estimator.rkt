#lang racket/base

(require (only-in cuml/core/private/define-who define/who)
         (only-in cuml/linear-model/private/linear-regression
                  linear-regression-fit/who
                  linear-regression-params
                  linear-regression-predict/who)
         (only-in cuml/metrics/private/metrics r2-score/who)
         (only-in raft current-device-resources)
         (only-in "estimator-support.rkt" define-estimator))

(provide linear-regression
         linear-regression-model
         linear-regression?)

(define/who (linear-regression-score who
                                     result
                                     X
                                     y
                                     #:sample-weight [sample-weight #f]
                                     #:resources [resources (current-device-resources)])
  (r2-score/who who
                y
                (linear-regression-predict/who who result X #:resources resources)
                #:sample-weight sample-weight
                #:resources resources))

(define-estimator linear-regression
  #:hyperparameters linear-regression-params
  #:fit linear-regression-fit/who
  #:needs-y
  #:settings (resources)
  #:apply values
  [predict linear-regression-predict/who]
  [score linear-regression-score/who #:sample-weight])
