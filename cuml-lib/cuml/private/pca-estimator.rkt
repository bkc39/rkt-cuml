#lang racket/base

(require (only-in cuml/decomposition/private/pca
                  pca-fit/who
                  pca-inverse-transform/who
                  pca-params
                  pca-transform/who)
         (only-in "estimator-support.rkt" define-estimator))

(provide pca
         pca-model
         pca?)

(define-estimator pca
  #:hyperparameters pca-params
  #:fit pca-fit/who
  #:settings (resources)
  #:no-sample-weight
  #:apply values
  [transform pca-transform/who]
  [inverse-transform pca-inverse-transform/who])
