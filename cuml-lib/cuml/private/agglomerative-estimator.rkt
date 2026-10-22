#lang racket/base

(require (only-in cuml/cluster/private/linkage
                  single-linkage-fit/who
                  single-linkage-params
                  single-linkage-result-labels)
         (only-in "estimator-support.rkt" define-estimator))

(provide agglomerative-clustering
         agglomerative-clustering?
         agglomerative-clustering-model)

(define-estimator agglomerative-clustering
  #:hyperparameters single-linkage-params
  #:fit single-linkage-fit/who
  #:labels single-linkage-result-labels
  #:settings (resources)
  #:no-sample-weight)
