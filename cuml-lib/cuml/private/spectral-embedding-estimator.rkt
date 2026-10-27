#lang racket/base

(require (only-in cuml/manifold/private/spectral-embedding
                  spectral-embedding-fit/who
                  spectral-embedding-params
                  spectral-embedding-result-embedding)
         (only-in "estimator-support.rkt" define-estimator))

(provide spectral-embedding
         spectral-embedding-model
         spectral-embedding?)

(define-estimator spectral-embedding
  #:hyperparameters spectral-embedding-params
  #:fit spectral-embedding-fit/who
  #:settings (resources)
  #:no-sample-weight
  #:apply values
  [fit-transform spectral-embedding-result-embedding])
