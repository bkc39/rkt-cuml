#lang racket/base

(require (only-in racket/match match match-define)
         (only-in raft current-device-resources)
         (only-in raft/unsafe deposit->list)
         (only-in "defaults.rkt" define-defaults)
         (only-in "define-who.rkt" define/who)
         (only-in "deposit.rkt" deposit-for)
         (only-in "native.rkt" cuml-cluster-single-linkage-fit-job))

(provide (struct-out single-linkage-result)
         single-linkage-defaults
         single-linkage-fit
         single-linkage-fit/who)

(struct single-linkage-result (labels children))

(define-defaults single-linkage-defaults
                 [n-clusters 2]
                 [metric 'euclidean]
                 [connectivity 'knn]
                 [c 15])

(define (metric-code metric)
  (match metric
    ['euclidean 0]
    [_ -1]))

(define (connectivity-code connectivity)
  (match connectivity
    ['pairwise 0]
    ['knn 1]
    [_ -1]))

(define/who (single-linkage-fit who
                                X
                                #:n-clusters [n-clusters (default 'n-clusters)]
                                #:metric [metric (default 'metric)]
                                #:connectivity [connectivity (default 'connectivity)]
                                #:c [c (default 'c)]
                                #:resources [resources (current-device-resources)])
  (define out (deposit-for who resources))
  (cuml-cluster-single-linkage-fit-job who
                                       resources
                                       X
                                       out
                                       n-clusters
                                       (metric-code metric)
                                       (connectivity-code connectivity)
                                       c)
  (match-define (list labels children) (deposit->list out))
  (single-linkage-result labels children))
