#lang racket/base

(require (only-in cuml/estimator gen:clusterer gen:estimator gen:predictor gen:scorer gen:transformer)
         (only-in racket/match match)
         (only-in raft device-matrix? shape))

(provide kmeans
         kmeans?
         kmeans-model)

(define (unimplemented . _)
  (error 'unimplemented))

(struct kmeans (params model)
  #:constructor-name kmeans*
  #:omit-define-syntaxes
  #:methods gen:estimator
  [(define fit (make-keyword-procedure unimplemented))
   (define fitted? unimplemented)
   (define estimator-params unimplemented)
   (define estimator-set-params unimplemented)]
  #:methods gen:predictor
  [(define predict unimplemented)]
  #:methods gen:clusterer
  [(define fit-predict (make-keyword-procedure unimplemented))]
  #:methods gen:transformer
  [(define transform unimplemented)
   (define fit-transform (make-keyword-procedure unimplemented))]
  #:methods gen:scorer
  [(define score (make-keyword-procedure unimplemented))]
  #:methods gen:equal+hash
  [(define equal-proc unimplemented)
   (define hash-proc unimplemented)
   (define hash2-proc unimplemented)]
  #:methods gen:custom-write
  [(define write-proc unimplemented)])

(define (init-rows init fallback)
  (match (and (device-matrix? init) (shape init))
    [(list rows _) rows]
    [_ fallback]))

(define (kmeans #:init [init 'scalable-k-means++]
                #:n-clusters [n-clusters (init-rows init 8)]
                #:max-iter [max-iter 300]
                #:tol [tol 1e-4]
                #:n-init [n-init 'auto]
                #:random-state [random-state 0]
                #:oversampling-factor [oversampling-factor 2.0]
                #:max-samples-per-batch [max-samples-per-batch 32768]
                #:resources [resources #f])
  (kmeans* (hasheq 'init
                   init
                   'n-clusters
                   n-clusters
                   'max-iter
                   max-iter
                   'tol
                   tol
                   'n-init
                   n-init
                   'random-state
                   random-state
                   'oversampling-factor
                   oversampling-factor
                   'max-samples-per-batch
                   max-samples-per-batch
                   'resources
                   resources)
           #f))
