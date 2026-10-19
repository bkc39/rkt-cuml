#lang racket/base

(require (only-in cuml/cluster/private/linkage
                  single-linkage-defaults
                  single-linkage-fit/who
                  single-linkage-result-labels)
         (only-in "estimator-support.rkt" call-with-fixed-option define-constructor define-estimator))

(provide agglomerative-clustering
         agglomerative-clustering?
         agglomerative-clustering-model)

(define defaults (hash-set* single-linkage-defaults 'linkage 'single 'resources #f))

(define (fitted-model who e X sample-weight)
  (call-with-fixed-option single-linkage-fit/who
                          who
                          X
                          (agglomerative-clustering-params e)
                          #:option 'linkage
                          #:only 'single
                          #:estimator 'agglomerative-clustering
                          #:sample-weight sample-weight))

(define-estimator agglomerative-clustering
                  #:fitted fitted-model
                  #:labels single-linkage-result-labels
                  #:names parameter-names
                  #:defaults defaults)

(define-constructor
 (agglomerative-clustering #:n-clusters #:metric #:linkage #:connectivity #:c #:resources)
 #:names parameter-names
 #:defaults defaults
 agglomerative-clustering*)
