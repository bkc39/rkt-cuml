#lang racket/base

(require (only-in cuml/cluster/private/kmeans
                  init-rows
                  kmeans-defaults
                  kmeans-fit/who
                  kmeans-predict/who
                  kmeans-result-centroids
                  kmeans-result-labels
                  kmeans-score/who
                  kmeans-transform/who)
         (only-in cuml/estimator gen:predictor gen:scorer gen:transformer)
         (only-in "estimator-support.rkt"
                  call-with-params
                  define-constructor
                  define-estimator
                  logic-error
                  resources-of))

(provide kmeans
         kmeans?
         kmeans-model)

(define defaults (hash-set kmeans-defaults 'resources #f))

(define (fitted-model who e X sample-weight)
  (call-with-params kmeans-fit/who who X (kmeans-params e) (hasheq 'sample-weight sample-weight)))

(define (model-of who e)
  (or (kmeans-model e) (logic-error "~a: kmeans is not fitted" who)))

(define (on-centroids proc who e model X)
  (define params (kmeans-params e))
  (proc who
        (kmeans-result-centroids model)
        X
        #:max-samples-per-batch (hash-ref params 'max-samples-per-batch)
        #:resources (resources-of params)))

(define (scored who e X sample-weight)
  (define params (kmeans-params e))
  (kmeans-score/who who
                    (kmeans-result-centroids (model-of who e))
                    X
                    #:sample-weight sample-weight
                    #:max-samples-per-batch (hash-ref params 'max-samples-per-batch)
                    #:resources (resources-of params)))

(define (shown-defaults params)
  (hash-set defaults 'n-clusters (init-rows (hash-ref params 'init) (hash-ref defaults 'n-clusters))))

(define-estimator kmeans
                  #:fitted fitted-model
                  #:labels kmeans-result-labels
                  #:names parameter-names
                  #:defaults defaults
                  #:shown-defaults shown-defaults
                  #:methods gen:predictor
                  [(define (predict e X)
                     (on-centroids kmeans-predict/who 'predict e (model-of 'predict e) X))]
                  #:methods gen:transformer
                  [(define (transform e X)
                     (on-centroids kmeans-transform/who 'transform e (model-of 'transform e) X))
                   (define (fit-transform e X [_y #f] #:sample-weight [sample-weight #f])
                     (on-centroids kmeans-transform/who
                                   'fit-transform
                                   e
                                   (fitted-model 'fit-transform e X sample-weight)
                                   X))]
                  #:methods gen:scorer
                  [(define (score e X [_y #f] #:sample-weight [sample-weight #f])
                     (scored 'score e X sample-weight))])

(define-constructor (kmeans #:init
                            #:n-clusters (init-rows init (hash-ref defaults 'n-clusters))
                            #:max-iter
                            #:tol
                            #:n-init
                            #:random-state
                            #:oversampling-factor
                            #:max-samples-per-batch
                            #:resources)
                    #:names parameter-names
                    #:defaults defaults
                    kmeans*)
