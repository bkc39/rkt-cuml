#lang racket/base

(require (only-in cuml/decomposition/private/tsvd
                  tsvd-fit/who
                  tsvd-inverse-transform/who
                  tsvd-params
                  tsvd-transform/who)
         (only-in "estimator-support.rkt" define-estimator))

(provide truncated-svd
         truncated-svd-model
         truncated-svd?)

(define-estimator truncated-svd
  #:hyperparameters tsvd-params
  #:fit tsvd-fit/who
  #:settings (resources)
  #:no-sample-weight
  #:apply values
  [transform tsvd-transform/who]
  [inverse-transform tsvd-inverse-transform/who])
