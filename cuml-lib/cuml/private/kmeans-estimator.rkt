#lang racket/base

(require (only-in cuml/cluster/private/kmeans
                  kmeans-fit/who
                  kmeans-params
                  kmeans-predict/who
                  kmeans-result-centroids
                  kmeans-result-labels
                  kmeans-score/who
                  kmeans-transform/who)
         (only-in "estimator-support.rkt" define-estimator))

(provide kmeans
         kmeans?
         kmeans-model)

(define-estimator kmeans
  #:hyperparameters kmeans-params
  #:fit kmeans-fit/who
  #:labels kmeans-result-labels
  #:settings (max-samples-per-batch resources)
  #:apply kmeans-result-centroids
  [predict kmeans-predict/who]
  [transform kmeans-transform/who]
  [score kmeans-score/who #:sample-weight])
