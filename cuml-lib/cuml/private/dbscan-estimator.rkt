#lang racket/base

(require (only-in cuml/cluster/private/dbscan dbscan-defaults dbscan-fit/who dbscan-result-labels)
         (only-in "estimator-support.rkt" call-with-params define-constructor define-estimator))

(provide dbscan
         dbscan?
         dbscan-model)

(define defaults (hash-set dbscan-defaults 'resources #f))

(define (fitted-model who e X sample-weight)
  (call-with-params dbscan-fit/who who X (dbscan-params e) (hasheq 'sample-weight sample-weight)))

(define-estimator dbscan
                  #:fitted fitted-model
                  #:labels dbscan-result-labels
                  #:names parameter-names
                  #:defaults defaults)

(define-constructor
 (dbscan #:eps #:min-samples #:metric #:out-dtype #:max-mbytes-per-batch #:resources)
 #:names parameter-names
 #:defaults defaults
 dbscan*)
