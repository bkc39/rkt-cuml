#lang racket/base

(require (only-in cuml/cluster/private/dbscan dbscan-defaults dbscan-fit/who dbscan-result-labels)
         (only-in cuml/estimator gen:clusterer gen:estimator)
         (only-in "estimator-support.rkt"
                  call-with-params
                  define-constructor
                  params->string
                  updated-params))

(provide dbscan
         dbscan?
         dbscan-model)

(define defaults (hash-set dbscan-defaults 'resources #f))

(define (fitted-model who e X sample-weight)
  (call-with-params dbscan-fit/who who X (dbscan-params e) (hasheq 'sample-weight sample-weight)))

(define (dbscan->string e)
  (params->string 'dbscan parameter-names (dbscan-params e) defaults (and (dbscan-model e) #t)))

(struct dbscan (params model)
  #:constructor-name dbscan*
  #:omit-define-syntaxes
  #:methods gen:estimator
  [(define (fit e X [_y #f] #:sample-weight [sample-weight #f])
     (dbscan* (dbscan-params e) (fitted-model 'fit e X sample-weight)))
   (define (fitted? e)
     (and (dbscan-model e) #t))
   (define (estimator-params e)
     (dbscan-params e))
   (define (estimator-set-params e params)
     (dbscan* (updated-params 'dbscan (dbscan-params e) params) #f))]
  #:methods gen:clusterer
  [(define (fit-predict e X [_y #f] #:sample-weight [sample-weight #f])
     (dbscan-result-labels (fitted-model 'fit-predict e X sample-weight)))]
  #:methods gen:equal+hash
  [(define (equal-proc a b recur)
     (and (recur (dbscan-params a) (dbscan-params b)) (eq? (dbscan-model a) (dbscan-model b))))
   (define (hash-proc e recur)
     (+ (recur (dbscan-params e)) (eq-hash-code (dbscan-model e))))
   (define (hash2-proc e recur)
     (recur (dbscan-params e)))]
  #:methods gen:custom-write
  [(define (write-proc e port _mode)
     (write-string (dbscan->string e) port))])

(define-constructor (dbscan #:eps eps
                            #:min-samples min-samples
                            #:metric metric
                            #:out-dtype out-dtype
                            #:max-mbytes-per-batch max-mbytes-per-batch
                            #:resources resources)
                    #:names parameter-names
                    #:defaults defaults
                    (lambda (params) (dbscan* params #f)))
