#lang racket/base

(require (only-in cuml/cluster/private/spectral
                  spectral-clustering-defaults
                  spectral-clustering-fit/who
                  spectral-clustering-result-labels)
         (only-in "estimator-support.rkt"
                  call-with-params
                  define-constructor
                  define-estimator
                  logic-error))

(provide spectral-clustering
         spectral-clustering?
         spectral-clustering-model)

(define defaults (hash-set* spectral-clustering-defaults 'affinity 'nearest-neighbors 'resources #f))

(define (fitted-model who e X sample-weight)
  (define params (spectral-clustering-params e))
  (define affinity (hash-ref params 'affinity))
  (cond
    [(not (eq? affinity 'nearest-neighbors))
     (logic-error "~a: affinity: expected nearest-neighbors, got ~a" who affinity)]
    [sample-weight (logic-error "~a: spectral-clustering takes no sample-weight" who)]
    [else
     (call-with-params spectral-clustering-fit/who who X (hash-remove params 'affinity) (hasheq))]))

(define-estimator spectral-clustering
                  #:fitted fitted-model
                  #:labels spectral-clustering-result-labels
                  #:names parameter-names
                  #:defaults defaults)

(define-constructor (spectral-clustering #:n-clusters
                                         #:n-components
                                         #:random-state
                                         #:n-neighbors
                                         #:n-init
                                         #:eigen-tol
                                         #:affinity
                                         #:resources)
                    #:names parameter-names
                    #:defaults defaults
                    spectral-clustering*)
