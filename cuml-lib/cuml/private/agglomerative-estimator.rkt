#lang racket/base

(require (only-in cuml/cluster/private/linkage
                  single-linkage-defaults
                  single-linkage-fit/who
                  single-linkage-result-labels)
         (only-in "estimator-support.rkt"
                  call-with-params
                  define-constructor
                  define-estimator
                  logic-error
                  params->string))

(provide agglomerative-clustering
         agglomerative-clustering?
         agglomerative-clustering-model)

(define defaults (hash-set* single-linkage-defaults 'linkage 'single 'resources #f))

(define (fitted-model who e X sample-weight)
  (define params (agglomerative-clustering-params e))
  (define linkage (hash-ref params 'linkage))
  (cond
    [(not (eq? linkage 'single)) (logic-error "~a: linkage: expected single, got ~a" who linkage)]
    [sample-weight (logic-error "~a: agglomerative-clustering takes no sample-weight" who)]
    [else (call-with-params single-linkage-fit/who who X (hash-remove params 'linkage) (hasheq))]))

(define (agglomerative-clustering->string e)
  (params->string 'agglomerative-clustering
                  parameter-names
                  (agglomerative-clustering-params e)
                  defaults
                  (and (agglomerative-clustering-model e) #t)))

(define-estimator agglomerative-clustering
                  #:fitted fitted-model
                  #:labels single-linkage-result-labels
                  #:string agglomerative-clustering->string)

(define-constructor
 (agglomerative-clustering #:n-clusters #:metric #:linkage #:connectivity #:c #:resources)
 #:names parameter-names
 #:defaults defaults
 agglomerative-clustering*)
