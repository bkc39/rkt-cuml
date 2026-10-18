#lang racket/base

(require (only-in cuml/cluster/private/dbscan dbscan-defaults dbscan-fit/who dbscan-result-labels)
         (only-in cuml/estimator gen:clusterer gen:estimator)
         (only-in "estimator-support.rkt" call-with-params params->string updated-params))

(provide dbscan
         dbscan?
         dbscan-model)

(define parameter-names '(eps min-samples metric out-dtype max-mbytes-per-batch resources))

(define (default name)
  (hash-ref dbscan-defaults name))

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
     (error 'unimplemented))
   (define (fitted? e)
     (and (dbscan-model e) #t))
   (define (estimator-params e)
     (dbscan-params e))
   (define (estimator-set-params e params)
     (dbscan* (updated-params 'dbscan (dbscan-params e) params) #f))]
  #:methods gen:clusterer
  [(define (fit-predict e X [_y #f] #:sample-weight [sample-weight #f])
     (error 'unimplemented))]
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

(define (dbscan #:eps [eps (default 'eps)]
                #:min-samples [min-samples (default 'min-samples)]
                #:metric [metric (default 'metric)]
                #:out-dtype [out-dtype (default 'out-dtype)]
                #:max-mbytes-per-batch [max-mbytes-per-batch (default 'max-mbytes-per-batch)]
                #:resources [resources #f])
  (error 'unimplemented))
