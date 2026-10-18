#lang racket/base

(require (only-in racket/match match)
         (only-in raft current-device-resources device-matrix? shape))

(provide (struct-out kmeans-result)
         kmeans-fit
         kmeans-predict
         kmeans-score
         kmeans-transform)

(struct kmeans-result (centroids labels inertia n-iter))

(define (init-rows init fallback)
  (match (and (device-matrix? init) (shape init))
    [(list rows _) rows]
    [_ fallback]))

(define (kmeans-fit X
                    #:init [init 'scalable-k-means++]
                    #:n-clusters [n-clusters (init-rows init 8)]
                    #:max-iter [max-iter 300]
                    #:tol [tol 1e-4]
                    #:n-init [n-init 'auto]
                    #:random-state [random-state 0]
                    #:oversampling-factor [oversampling-factor 2.0]
                    #:max-samples-per-batch [max-samples-per-batch 32768]
                    #:sample-weight [sample-weight #f]
                    #:resources [resources (current-device-resources)])
  (error 'unimplemented))

(define (kmeans-predict centroids
                        X
                        #:max-samples-per-batch [max-samples-per-batch 32768]
                        #:resources [resources (current-device-resources)])
  (error 'unimplemented))

(define (kmeans-transform centroids
                          X
                          #:max-samples-per-batch [max-samples-per-batch 32768]
                          #:resources [resources (current-device-resources)])
  (error 'unimplemented))

(define (kmeans-score centroids
                      X
                      #:sample-weight [sample-weight #f]
                      #:max-samples-per-batch [max-samples-per-batch 32768]
                      #:resources [resources (current-device-resources)])
  (error 'unimplemented))
