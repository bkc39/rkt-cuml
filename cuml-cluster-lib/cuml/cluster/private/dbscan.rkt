#lang racket/base

(require (only-in racket/match match match-define)
         (only-in raft current-device-resources)
         (only-in raft/unsafe deposit->list)
         (only-in "define-who.rkt" define/who)
         (only-in "deposit.rkt" deposit-for)
         (only-in "native.rkt" cuml-cluster-dbscan-fit-job))

(provide (struct-out dbscan-result)
         dbscan-defaults
         dbscan-fit
         dbscan-fit/who)

(struct dbscan-result (labels core-sample-indices))

(define dbscan-defaults
  (hasheq 'eps 0.5 'min-samples 5 'metric 'euclidean 'out-dtype 'int32 'max-mbytes-per-batch #f))

(define (default name)
  (hash-ref dbscan-defaults name))

(define (metric-code metric)
  (match metric
    ['euclidean 0]
    ['cosine 1]
    ['precomputed 2]
    [_ -1]))

(define (out-dtype-code out-dtype)
  (match out-dtype
    ['int32 2]
    ['int64 3]
    [_ -1]))

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
  (match-define (list labels core) (deposit->list out))
  (dbscan-result labels core))
