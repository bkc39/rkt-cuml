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
         (only-in "native.rkt"
                  cuml-cluster-kmeans-fit-job
                  cuml-cluster-kmeans-predict-job
                  cuml-cluster-kmeans-score-job
                  cuml-cluster-kmeans-transform-job))

(provide (struct-out kmeans-result)
         kmeans-fit
         kmeans-fit/who
         kmeans-predict
         kmeans-predict/who
         kmeans-score
         kmeans-score/who
         kmeans-transform
         kmeans-transform/who)

(struct kmeans-result (centroids labels inertia n-iter))

(define default-batch-samples 32768)

(define (init-rows init fallback)
  (match (and (device-matrix? init) (shape init))
    [(list rows _) rows]
    [_ fallback]))

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

(define (kmeans-fit/who who
                        X
                        #:init (init 'scalable-k-means++)
                        #:n-clusters [n-clusters (init-rows init 8)]
                        #:max-iter [max-iter 300]
                        #:tol [tol 1e-4]
                        #:n-init [n-init 'auto]
                        #:random-state [random-state 0]
                        #:oversampling-factor [oversampling-factor 2.0]
                        #:max-samples-per-batch [max-samples-per-batch default-batch-samples]
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

(define (kmeans-predict/who who
                            centroids
                            X
                            #:max-samples-per-batch [max-samples-per-batch default-batch-samples]
                            #:resources [resources (current-device-resources)])
  (deposited cuml-cluster-kmeans-predict-job who resources centroids X max-samples-per-batch))

(define (kmeans-transform/who who
                              centroids
                              X
                              #:max-samples-per-batch [max-samples-per-batch default-batch-samples]
                              #:resources [resources (current-device-resources)])
  (deposited cuml-cluster-kmeans-transform-job who resources centroids X max-samples-per-batch))

(define (kmeans-score/who who
                          centroids
                          X
                          #:sample-weight [sample-weight #f]
                          #:max-samples-per-batch [max-samples-per-batch default-batch-samples]
                          #:resources [resources (current-device-resources)])
  (- (cuml-cluster-kmeans-score-job who resources centroids X sample-weight max-samples-per-batch)))

(define (named proc who)
  (define-values (required accepted) (procedure-keywords proc))
  (procedure-reduce-keyword-arity-mask
   (make-keyword-procedure (lambda (kws vals . args) (keyword-apply proc kws vals who args)))
   (arithmetic-shift (procedure-arity-mask proc) -1)
   required
   accepted
   who))

(define kmeans-fit (named kmeans-fit/who 'kmeans-fit))
(define kmeans-predict (named kmeans-predict/who 'kmeans-predict))
(define kmeans-transform (named kmeans-transform/who 'kmeans-transform))
(define kmeans-score (named kmeans-score/who 'kmeans-score))
