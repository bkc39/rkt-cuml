#lang racket/base

(require (only-in raft current-device-resources)
         (only-in raft/unsafe deposit->list)
         (only-in "defaults.rkt" define-codes define-defaults)
         (only-in "define-who.rkt" define/who)
         (only-in "deposit.rkt" deposit-for)
         (only-in "native.rkt" cuml-cluster-dbscan-fit-job))

(provide (struct-out dbscan-result)
         dbscan-defaults
         dbscan-fit
         dbscan-fit/who)

(struct dbscan-result (labels core-sample-indices))

(define-defaults dbscan-defaults
                 [eps 0.5]
                 [min-samples 5]
                 [metric 'euclidean]
                 [out-dtype 'int32]
                 [max-mbytes-per-batch #f])

(define-codes metric-code [euclidean 0] [cosine 1] [precomputed 2])

(define-codes out-dtype-code [int32 2] [int64 3])

(define/who (dbscan-fit who
                        X
                        #:eps [eps (default 'eps)]
                        #:min-samples [min-samples (default 'min-samples)]
                        #:metric [metric (default 'metric)]
                        #:out-dtype [out-dtype (default 'out-dtype)]
                        #:max-mbytes-per-batch [max-mbytes-per-batch (default 'max-mbytes-per-batch)]
                        #:sample-weight [sample-weight #f]
                        #:resources [resources (current-device-resources)])
  (define out (deposit-for who resources))
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
