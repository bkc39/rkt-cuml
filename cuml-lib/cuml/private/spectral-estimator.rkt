#lang racket/base

(require (only-in cuml/cluster/private/spectral
                  spectral-clustering-fit/who
                  spectral-clustering-params
                  spectral-clustering-result-labels)
         (only-in "estimator-support.rkt" define-estimator))

(provide spectral-clustering
         spectral-clustering?
         spectral-clustering-model)

(define-estimator spectral-clustering
  #:hyperparameters spectral-clustering-params
  #:fit spectral-clustering-fit/who
  #:labels spectral-clustering-result-labels
  #:settings (resources)
  #:no-sample-weight)
