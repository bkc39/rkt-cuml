#lang racket/base

(require (only-in cuml/cluster/private/hdbscan hdbscan-fit/who hdbscan-params hdbscan-result-labels)
         (only-in "estimator-support.rkt" define-estimator))

(provide hdbscan
         hdbscan?
         hdbscan-model)

(define-estimator hdbscan
  #:hyperparameters hdbscan-params
  #:fit hdbscan-fit/who
  #:labels hdbscan-result-labels
  #:settings (resources)
  #:no-sample-weight)
