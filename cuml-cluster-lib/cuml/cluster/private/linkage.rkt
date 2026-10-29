#lang racket/base

(require (only-in cuml/core/private/define-who define/who)
         (only-in cuml/core/private/deposit deposit-for/who)
         (only-in cuml/core/private/hyperparameters define-codes define-hyperparameters)
         (only-in raft current-device-resources)
         (only-in raft/unsafe deposit->list)
         (only-in "native.rkt" cuml-cluster-single-linkage-fit-job))

(provide (struct-out single-linkage-result)
         single-linkage-fit
         single-linkage-fit/who
         single-linkage-params)

(struct single-linkage-result (labels children))

(define-hyperparameters single-linkage-params
  [n-clusters 2]
  [metric 'euclidean]
  [linkage 'single #:fixed]
  [connectivity 'knn]
  [c 15])

(define-codes metric-code [euclidean 0])

(define-codes connectivity-code [pairwise 0] [knn 1])

(define/who (single-linkage-fit who
                                X
                                #:hyperparameters single-linkage-params
                                #:resources [resources (current-device-resources)])
  (define out (deposit-for/who who resources))
  (cuml-cluster-single-linkage-fit-job who
                                       resources
                                       X
                                       out
                                       n-clusters
                                       (metric-code metric)
                                       (connectivity-code connectivity)
                                       c)
  (apply single-linkage-result (deposit->list out)))
