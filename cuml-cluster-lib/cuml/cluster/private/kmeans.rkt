#lang racket/base

(require (only-in racket/match match match* match-define)
         (only-in raft current-device-resources device-matrix? shape)
         (only-in raft/unsafe deposit->list)
         (only-in "define-who.rkt" define/who)
         (only-in "defaults.rkt" define-defaults)
         (only-in "deposit.rkt" deposit-for)
         (only-in "native.rkt"
                  cuml-cluster-kmeans-fit-job
                  cuml-cluster-kmeans-predict-job
                  cuml-cluster-kmeans-score-job
                  cuml-cluster-kmeans-transform-job))

(provide (struct-out kmeans-result)
         init-rows
         kmeans-defaults
         kmeans-fit
         kmeans-fit/who
         kmeans-predict
         kmeans-predict/who
         kmeans-score
         kmeans-score/who
         kmeans-transform
         kmeans-transform/who)

(struct kmeans-result (centroids labels inertia n-iter))

(define-defaults kmeans-defaults
                 default
                 (init 'scalable-k-means++)
                 [n-clusters 8]
                 [max-iter 300]
                 [tol 1e-4]
                 [n-init 'auto]
                 [random-state 0]
                 [oversampling-factor 2.0]
                 [max-samples-per-batch 32768])

(define (init-rows init fallback)
  (cond
    [(device-matrix? init)
     (match-define (list rows _) (shape init))
     rows]
    [else fallback]))

(define (seeding init oversampling-factor)
  (match init
    ['scalable-k-means++ (values 0 oversampling-factor #f)]
    ['k-means++ (values 0 0.0 #f)]
    ['random (values 1 oversampling-factor #f)]
    [(? device-matrix?) (values 2 oversampling-factor init)]
    [_ (values -1 oversampling-factor #f)]))

(define (n-init-runs n-init code)
  (match* (n-init code)
    [('auto 0) 1]
    [('auto _) 10]
    [(runs _) runs]))

(define/who (kmeans-fit who
                        X
                        #:init (init (default 'init))
                        #:n-clusters [n-clusters (init-rows init (default 'n-clusters))]
                        #:max-iter [max-iter (default 'max-iter)]
                        #:tol [tol (default 'tol)]
                        #:n-init [n-init (default 'n-init)]
                        #:random-state [random-state (default 'random-state)]
                        #:oversampling-factor [oversampling-factor (default 'oversampling-factor)]
                        #:max-samples-per-batch
                        [max-samples-per-batch (default 'max-samples-per-batch)]
                        #:sample-weight [sample-weight #f]
                        #:resources [resources (current-device-resources)])
  (define-values (code factor start) (seeding init oversampling-factor))
  (define out (deposit-for who resources))
  (define-values (inertia n-iter)
    (cuml-cluster-kmeans-fit-job who
                                 resources
                                 X
                                 sample-weight
                                 start
                                 out
                                 n-clusters
                                 code
                                 max-iter
                                 tol
                                 (n-init-runs n-init code)
                                 factor
                                 max-samples-per-batch
                                 random-state))
  (match-define (list centroids labels) (deposit->list out))
  (kmeans-result centroids labels inertia n-iter))

(define (deposited job who resources centroids X max-samples-per-batch)
  (define out (deposit-for who resources))
  (job who resources centroids X out max-samples-per-batch)
  (match-define (list result) (deposit->list out))
  result)

(define/who (kmeans-predict who
                            centroids
                            X
                            #:max-samples-per-batch
                            [max-samples-per-batch (default 'max-samples-per-batch)]
                            #:resources [resources (current-device-resources)])
  (deposited cuml-cluster-kmeans-predict-job who resources centroids X max-samples-per-batch))

(define/who (kmeans-transform who
                              centroids
                              X
                              #:max-samples-per-batch
                              [max-samples-per-batch (default 'max-samples-per-batch)]
                              #:resources [resources (current-device-resources)])
  (deposited cuml-cluster-kmeans-transform-job who resources centroids X max-samples-per-batch))

(define/who (kmeans-score who
                          centroids
                          X
                          #:sample-weight [sample-weight #f]
                          #:max-samples-per-batch
                          [max-samples-per-batch (default 'max-samples-per-batch)]
                          #:resources [resources (current-device-resources)])
  (- (cuml-cluster-kmeans-score-job who resources centroids X sample-weight max-samples-per-batch)))
