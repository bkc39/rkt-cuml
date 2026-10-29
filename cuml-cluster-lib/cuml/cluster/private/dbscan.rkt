#lang racket/base

(require (only-in cuml/core/private/define-who define/who)
         (only-in cuml/core/private/deposit deposit-for/who)
         (only-in cuml/core/private/hyperparameters define-codes define-hyperparameters)
         (only-in raft current-device-resources)
         (only-in raft/unsafe deposit->list)
         (only-in "native.rkt" cuml-cluster-dbscan-fit-job))

(provide (struct-out dbscan-result)
         dbscan-fit
         dbscan-fit/who
         dbscan-params)

(struct dbscan-result (labels core-sample-indices))

(define-hyperparameters dbscan-params
  [eps 0.5]
  [min-samples 5]
  [metric 'euclidean]
  [out-dtype 'int32]
  [max-mbytes-per-batch #f])

(define-codes metric-code [euclidean 0] [cosine 1] [precomputed 2])

(define-codes out-dtype-code [int32 2] [int64 3])

(define/who (dbscan-fit who
                        X
                        #:hyperparameters dbscan-params
                        #:sample-weight [sample-weight #f]
                        #:resources [resources (current-device-resources)])
  (define out (deposit-for/who who resources))
  (cuml-cluster-dbscan-fit-job who
                               resources
                               X
                               sample-weight
                               out
                               eps
                               min-samples
                               (metric-code metric)
                               (out-dtype-code out-dtype)
                               (or max-mbytes-per-batch 0))
  (apply dbscan-result (deposit->list out)))
