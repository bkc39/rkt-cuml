#lang racket/base

(require (only-in cuml/manifold/private/tsne tsne-fit/who tsne-params tsne-result-embedding)
         (only-in "estimator-support.rkt" define-estimator))

(provide tsne
         tsne-model
         tsne?)

(define-estimator tsne
  #:hyperparameters tsne-params
  #:fit tsne-fit/who
  #:settings (resources)
  #:no-sample-weight
  #:apply values
  [fit-transform tsne-result-embedding])
