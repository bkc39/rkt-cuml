#lang racket/base

(require (only-in cuml/cluster/private/dbscan dbscan-defaults dbscan-fit/who dbscan-result-labels)
         (only-in "estimator-support.rkt"
                  call-with-params
                  define-constructor
                  define-estimator
                  params->string))

(provide dbscan
         dbscan?
         dbscan-model)

(define defaults (hash-set dbscan-defaults 'resources #f))

(define (fitted-model who e X sample-weight)
  (call-with-params dbscan-fit/who who X (dbscan-params e) (hasheq 'sample-weight sample-weight)))

(define (dbscan->string e)
  (params->string 'dbscan parameter-names (dbscan-params e) defaults (and (dbscan-model e) #t)))

(define-estimator dbscan #:fitted fitted-model #:labels dbscan-result-labels #:string dbscan->string)

(define-constructor
 (dbscan #:eps #:min-samples #:metric #:out-dtype #:max-mbytes-per-batch #:resources)
 #:names parameter-names
 #:defaults defaults
 dbscan*)
