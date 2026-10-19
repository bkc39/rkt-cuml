#lang racket/base

(require (only-in cuml/cluster/private/linkage
                  single-linkage-defaults
                  single-linkage-fit/who
                  single-linkage-result-labels)
         (only-in cuml/estimator gen:clusterer gen:estimator)
         (only-in "estimator-support.rkt"
                  call-with-params
                  define-constructor
                  logic-error
                  params->string
                  updated-params))

(provide agglomerative-clustering
         agglomerative-clustering?
         agglomerative-clustering-model)

(define defaults (hash-set* single-linkage-defaults 'linkage 'single 'resources #f))

(define (fitted-model who e X sample-weight)
  (error 'unimplemented))

(define (agglomerative-clustering->string e)
  (params->string 'agglomerative-clustering
                  parameter-names
                  (agglomerative-clustering-params e)
                  defaults
                  (and (agglomerative-clustering-model e) #t)))

(struct agglomerative-clustering (params model)
  #:constructor-name agglomerative-clustering*
  #:omit-define-syntaxes
  #:methods gen:estimator
  [(define (fit e X [_y #f] #:sample-weight [sample-weight #f])
     (agglomerative-clustering* (agglomerative-clustering-params e)
                                (fitted-model 'fit e X sample-weight)))
   (define (fitted? e)
     (and (agglomerative-clustering-model e) #t))
   (define (estimator-params e)
     (agglomerative-clustering-params e))
   (define (estimator-set-params e params)
     (agglomerative-clustering*
      (updated-params 'agglomerative-clustering (agglomerative-clustering-params e) params)
      #f))]
  #:methods gen:clusterer
  [(define (fit-predict e X [_y #f] #:sample-weight [sample-weight #f])
     (single-linkage-result-labels (fitted-model 'fit-predict e X sample-weight)))]
  #:methods gen:equal+hash
  [(define (equal-proc a b recur)
     (and (recur (agglomerative-clustering-params a) (agglomerative-clustering-params b))
          (eq? (agglomerative-clustering-model a) (agglomerative-clustering-model b))))
   (define (hash-proc e recur)
     (+ (recur (agglomerative-clustering-params e))
        (eq-hash-code (agglomerative-clustering-model e))))
   (define (hash2-proc e recur)
     (recur (agglomerative-clustering-params e)))]
  #:methods gen:custom-write
  [(define (write-proc e port _mode)
     (write-string (agglomerative-clustering->string e) port))])

(define-constructor
 (agglomerative-clustering #:n-clusters #:metric #:linkage #:connectivity #:c #:resources)
 #:names parameter-names
 #:defaults defaults
 agglomerative-clustering*)
