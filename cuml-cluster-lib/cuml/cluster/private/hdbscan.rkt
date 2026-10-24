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

(provide hdbscan-all-points-membership-vectors
         hdbscan-all-points-membership-vectors/who
         hdbscan-approximate-predict
         hdbscan-approximate-predict/who
         hdbscan-condensed-tree-children
         hdbscan-condensed-tree-lambdas
         hdbscan-condensed-tree-parents
         hdbscan-condensed-tree-selected-clusters
         hdbscan-condensed-tree-sizes
         hdbscan-condensed-tree?
         hdbscan-fit
         hdbscan-fit/who
         hdbscan-linkage-tree-children
         hdbscan-linkage-tree-distances
         hdbscan-linkage-tree-sizes
         hdbscan-linkage-tree?
         hdbscan-membership-vector
         hdbscan-membership-vector/who
         hdbscan-params
         hdbscan-prediction-labels
         hdbscan-prediction-probabilities
         hdbscan-prediction?
         hdbscan-result-cluster-persistence
         hdbscan-result-condensed-tree
         hdbscan-result-labels
         hdbscan-result-minimum-spanning-tree
         hdbscan-result-model
         hdbscan-result-probabilities
         hdbscan-result-single-linkage-tree
         hdbscan-result?
         hdbscan-spanning-tree-destinations
         hdbscan-spanning-tree-sources
         hdbscan-spanning-tree-weights
         hdbscan-spanning-tree?)

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

(define (call-with-prediction who result resources run)
  (define anchor (live-anchor/who who (hdbscan-result-model result) 'hdbscan))
  (define out (deposit-for/who who resources))
  (run anchor (hdbscan-result-X result) out)
  (deposit->list out))

(define/who (hdbscan-approximate-predict who
                                         result
                                         points
                                         #:resources [resources (current-device-resources)])
  (match-define (list labels probabilities)
    (call-with-prediction
     who
     result
     resources
     (lambda (anchor X out)
       (cuml-cluster-hdbscan-approximate-predict-job who resources anchor X points out))))
  (hdbscan-prediction labels probabilities))

(define/who (hdbscan-membership-vector who
                                       result
                                       points
                                       #:batch-size [batch-size 4096]
                                       #:resources [resources (current-device-resources)])
  (match-define (list memberships)
    (call-with-prediction
     who
     result
     resources
     (lambda (anchor X out)
       (cuml-cluster-hdbscan-membership-vector-job who resources anchor X points out batch-size))))
  memberships)

(define/who (hdbscan-all-points-membership-vectors who
                                                   result
                                                   #:batch-size [batch-size 4096]
                                                   #:resources [resources (current-device-resources)])
  (match-define (list memberships)
    (call-with-prediction who
                          result
                          resources
                          (lambda (anchor X out)
                            (cuml-cluster-hdbscan-all-points-membership-vectors-job who
                                                                                    resources
                                                                                    anchor
                                                                                    X
                                                                                    out
                                                                                    batch-size))))
  memberships)
