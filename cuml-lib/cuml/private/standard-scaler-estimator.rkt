#lang racket/base

(require (only-in cuml/preprocessing/private/standard-scaler
                  standard-scaler-fit/who
                  standard-scaler-inverse-transform/who
                  standard-scaler-params
                  standard-scaler-transform/who)
         (only-in "estimator-support.rkt" define-estimator))

(provide standard-scaler
         standard-scaler?
         standard-scaler-model)

(define-estimator standard-scaler
  #:hyperparameters standard-scaler-params
  #:fit standard-scaler-fit/who
  #:settings (resources)
  #:no-sample-weight
  #:apply values
  [transform standard-scaler-transform/who]
  [inverse-transform standard-scaler-inverse-transform/who])
