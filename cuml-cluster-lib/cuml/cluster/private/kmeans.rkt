#lang racket/base

(require (only-in racket/match match match* match-define)
         (only-in raft
                  current-device-resources
                  device-matrix?
                  exn:fail:raft
                  exn:fail:raft-kind
                  exn:fail:raft?
                  shape)
         (only-in raft/unsafe deposit deposit->list)
         (only-in "define-who.rkt" define/who)
         (only-in "hyperparameters.rkt" define-hyperparameters)
         (only-in "native.rkt"
                  cuml-cluster-kmeans-fit-job
                  cuml-cluster-kmeans-predict-job
                  cuml-cluster-kmeans-score-job
                  cuml-cluster-kmeans-transform-job))

(provide (struct-out kmeans-result)
         current-batch-samples
         init-rows
         kmeans-fit
         kmeans-fit/who
         kmeans-params
         kmeans-predict
         kmeans-predict/who
         kmeans-score
         kmeans-score/who
         kmeans-transform
         kmeans-transform/who)

(struct kmeans-result (centroids labels inertia n-iter))

(define-hyperparameters kmeans-params
  (init 'scalable-k-means++)
  [n-clusters (init-rows init 8)]
  [max-iter 300]
  [tol 1e-4]
  [n-init 'auto]
  [random-state 0]
  [oversampling-factor 2.0])

(define (batch-samples v)
  (cond
    [(and (exact-integer? v) (<= 0 v 2147483647)) v]
    [else (raise-argument-error 'current-batch-samples "(integer-in 0 2147483647)" v)]))

(define current-batch-samples (make-parameter 32768 batch-samples 'current-batch-samples))

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

(define ((renamed who) e)
  (raise (exn:fail:raft (regexp-replace #rx"^deposit:" (exn-message e) (format "~a:" who))
                        (exn-continuation-marks e)
                        (exn:fail:raft-kind e))))

(define (deposit-for who resources)
  (with-handlers ([exn:fail:raft? (renamed who)])
    (deposit #:resources resources)))

(define (n-init-runs n-init code)
  (match* (n-init code)
    [('auto 0) 1]
    [('auto _) 10]
    [(runs _) runs]))

(define/who (kmeans-fit who
                        X
                        #:hyperparameters kmeans-params
                        #:sample-weight [sample-weight #f]
                        #:max-samples-per-batch [max-samples-per-batch (current-batch-samples)]
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
                            #:max-samples-per-batch [max-samples-per-batch (current-batch-samples)]
                            #:resources [resources (current-device-resources)])
  (deposited cuml-cluster-kmeans-predict-job who resources centroids X max-samples-per-batch))

(define/who (kmeans-transform who
                              centroids
                              X
                              #:max-samples-per-batch [max-samples-per-batch (current-batch-samples)]
                              #:resources [resources (current-device-resources)])
  (deposited cuml-cluster-kmeans-transform-job who resources centroids X max-samples-per-batch))

(define/who (kmeans-score who
                          centroids
                          X
                          #:sample-weight [sample-weight #f]
                          #:max-samples-per-batch [max-samples-per-batch (current-batch-samples)]
                          #:resources [resources (current-device-resources)])
  (- (cuml-cluster-kmeans-score-job who resources centroids X sample-weight max-samples-per-batch)))
