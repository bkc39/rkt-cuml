#lang racket/base

(require (only-in cuml/core/private/define-who define/who)
         (only-in cuml/core/private/hyperparameters define-codes define-hyperparameters)
         (only-in raft current-device-resources))

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
  (error 'unimplemented))

(define/who (hdbscan-approximate-predict who
                                         result
                                         points
                                         #:resources [resources (current-device-resources)])
  (error 'unimplemented))

(define/who (hdbscan-membership-vector who
                                       result
                                       points
                                       #:batch-size [batch-size 4096]
                                       #:resources [resources (current-device-resources)])
  (error 'unimplemented))

(define/who (hdbscan-all-points-membership-vectors who
                                                   result
                                                   #:batch-size [batch-size 4096]
                                                   #:resources
                                                   [resources (current-device-resources)])
  (error 'unimplemented))
