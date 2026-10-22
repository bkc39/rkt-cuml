#lang racket/base

(require (only-in cuml/cluster/private/dbscan dbscan-fit/who dbscan-params dbscan-result-labels)
         (only-in "estimator-support.rkt" define-estimator))

(provide dbscan
         dbscan?
         dbscan-model)

(define-estimator dbscan
  #:hyperparameters dbscan-params
  #:fit dbscan-fit/who
  #:labels dbscan-result-labels
  #:settings (resources))
