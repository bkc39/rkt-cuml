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
         (only-in cuml/estimator gen:clusterer gen:estimator gen:predictor gen:scorer gen:transformer)
         (only-in "estimator-support.rkt"
                  call-with-params
                  define-constructor
                  logic-error
                  params->string
                  resources-of
                  updated-params))

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

(define (kmeans->string e)
  (define params (kmeans-params e))
  (define shown-defaults
    (hash-set defaults
              'n-clusters
              (init-rows (hash-ref params 'init) (hash-ref defaults 'n-clusters))))
  (params->string 'kmeans parameter-names params shown-defaults (and (kmeans-model e) #t)))

(struct kmeans (params model)
  #:constructor-name kmeans*
  #:omit-define-syntaxes
  #:methods gen:estimator
  [(define (fit e X [_y #f] #:sample-weight [sample-weight #f])
     (kmeans* (kmeans-params e) (fitted-model 'fit e X sample-weight)))
   (define (fitted? e)
     (and (kmeans-model e) #t))
   (define (estimator-params e)
     (kmeans-params e))
   (define (estimator-set-params e params)
     (kmeans* (updated-params 'kmeans (kmeans-params e) params) #f))]
  #:methods gen:predictor
  [(define (predict e X)
     (on-centroids kmeans-predict/who 'predict e (model-of 'predict e) X))]
  #:methods gen:clusterer
  [(define (fit-predict e X [_y #f] #:sample-weight [sample-weight #f])
     (kmeans-result-labels (fitted-model 'fit-predict e X sample-weight)))]
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
     (scored 'score e X sample-weight))]
  #:methods gen:equal+hash
  [(define (equal-proc a b recur)
     (and (recur (kmeans-params a) (kmeans-params b)) (eq? (kmeans-model a) (kmeans-model b))))
   (define (hash-proc e recur)
     (+ (recur (kmeans-params e)) (eq-hash-code (kmeans-model e))))
   (define (hash2-proc e recur)
     (recur (kmeans-params e)))]
  #:methods gen:custom-write
  [(define (write-proc e port _mode)
     (write-string (kmeans->string e) port))])

(define-constructor (kmeans #:init init
                            #:n-clusters [n-clusters (init-rows init (hash-ref defaults 'n-clusters))]
                            #:max-iter max-iter
                            #:tol tol
                            #:n-init n-init
                            #:random-state random-state
                            #:oversampling-factor oversampling-factor
                            #:max-samples-per-batch max-samples-per-batch
                            #:resources resources)
                    #:names parameter-names
                    #:defaults defaults
                    (lambda (params) (kmeans* params #f)))
