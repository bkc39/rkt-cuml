#lang racket/base

(require (only-in cuml/manifold/private/umap
                  umap-fit/who
                  umap-params
                  umap-result-embedding
                  umap-transform/who)
         (only-in "estimator-support.rkt" define-estimator))

(provide umap
         umap-model
         umap?)

(define-estimator umap
  #:hyperparameters umap-params
  #:fit umap-fit/who
  #:supervised
  #:settings (resources)
  #:no-sample-weight
  #:apply values
  [transform umap-transform/who #:fit-transform umap-result-embedding])
