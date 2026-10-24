#lang racket/base

(require (only-in cuml/core/private/define-who define/who)
         (only-in cuml/core/private/hyperparameters define-codes define-hyperparameters)
         (only-in cuml/core/private/model deposit->native-model/who live-anchor/who)
         (only-in racket/match match-define match-define-values)
         (only-in raft current-device-resources)
         (only-in raft/unsafe deposit->list)
         (only-in "deposit.rkt" deposit-for/who)
         (only-in "native.rkt"
                  cuml-cluster-hdbscan-all-points-membership-vectors-job
                  cuml-cluster-hdbscan-approximate-predict-job
                  cuml-cluster-hdbscan-fit-job
                  cuml-cluster-hdbscan-membership-vector-job
                  cuml-cluster-model-release-job))

(provide (except-out (struct-out hdbscan-result) hdbscan-result hdbscan-result-X)
         (except-out (struct-out hdbscan-condensed-tree) hdbscan-condensed-tree)
         (except-out (struct-out hdbscan-linkage-tree) hdbscan-linkage-tree)
         (except-out (struct-out hdbscan-spanning-tree) hdbscan-spanning-tree)
         (except-out (struct-out hdbscan-prediction) hdbscan-prediction)
         hdbscan-all-points-membership-vectors
         hdbscan-all-points-membership-vectors/who
         hdbscan-approximate-predict
         hdbscan-approximate-predict/who
         hdbscan-fit
         hdbscan-fit/who
         hdbscan-membership-vector
         hdbscan-membership-vector/who
         hdbscan-params)

(struct hdbscan-result
        (labels probabilities
                cluster-persistence
                condensed-tree
                single-linkage-tree
                minimum-spanning-tree
                model
                X))

(struct hdbscan-condensed-tree (parents children lambdas sizes selected-clusters))

(struct hdbscan-linkage-tree (children distances sizes))

(struct hdbscan-spanning-tree (sources destinations weights))

(struct hdbscan-prediction (labels probabilities))

(define-hyperparameters hdbscan-params
  [min-cluster-size 5]
  [min-samples #f]
  [cluster-selection-epsilon 0.0]
  [max-cluster-size 0]
  [metric 'euclidean #:fixed]
  [alpha 1.0]
  [cluster-selection-method 'eom]
  [allow-single-cluster #f]
  [build-algo 'brute-force #:fixed])

(define-codes cluster-selection-method-code [eom 0] [leaf 1])

(define/who (hdbscan-fit who
                         X
                         #:hyperparameters hdbscan-params
                         #:resources [resources (current-device-resources)])
  (define model-out (deposit-for/who who resources))
  (define out (deposit-for/who who resources))
  (define opaque-bytes
    (cuml-cluster-hdbscan-fit-job who
                                  resources
                                  X
                                  model-out
                                  out
                                  min-cluster-size
                                  (or min-samples min-cluster-size)
                                  cluster-selection-epsilon
                                  max-cluster-size
                                  alpha
                                  (cluster-selection-method-code cluster-selection-method)
                                  (if allow-single-cluster 1 0)))
  (match-define-values (model (list labels parents children lambdas sizes stabilities selected))
    (deposit->native-model/who who
                               'hdbscan
                               model-out
                               cuml-cluster-model-release-job
                               #:opaque-bytes opaque-bytes))
  (match-define (list probabilities linkage distances linkage-sizes sources destinations weights)
    (deposit->list out))
  (hdbscan-result labels
                  probabilities
                  stabilities
                  (hdbscan-condensed-tree parents children lambdas sizes selected)
                  (hdbscan-linkage-tree linkage distances linkage-sizes)
                  (hdbscan-spanning-tree sources destinations weights)
                  model
                  X))

(define (anchor-of who result)
  (live-anchor/who who (hdbscan-result-model result) 'hdbscan))

(define/who (hdbscan-approximate-predict who
                                         result
                                         points
                                         #:resources [resources (current-device-resources)])
  (define anchor (anchor-of who result))
  (define out (deposit-for/who who resources))
  (cuml-cluster-hdbscan-approximate-predict-job who
                                                resources
                                                anchor
                                                (hdbscan-result-X result)
                                                points
                                                out)
  (apply hdbscan-prediction (deposit->list out)))

(define/who (hdbscan-membership-vector who
                                       result
                                       points
                                       #:batch-size [batch-size 4096]
                                       #:resources [resources (current-device-resources)])
  (define anchor (anchor-of who result))
  (define out (deposit-for/who who resources))
  (cuml-cluster-hdbscan-membership-vector-job who
                                              resources
                                              anchor
                                              (hdbscan-result-X result)
                                              points
                                              out
                                              batch-size)
  (car (deposit->list out)))

(define/who (hdbscan-all-points-membership-vectors who
                                                   result
                                                   #:batch-size [batch-size 4096]
                                                   #:resources [resources (current-device-resources)])
  (define anchor (anchor-of who result))
  (define out (deposit-for/who who resources))
  (cuml-cluster-hdbscan-all-points-membership-vectors-job who
                                                          resources
                                                          anchor
                                                          (hdbscan-result-X result)
                                                          out
                                                          batch-size)
  (car (deposit->list out)))
