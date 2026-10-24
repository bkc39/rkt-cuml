#lang racket/base

(require (only-in cuml/core/private/define-who define/who)
         (only-in cuml/core/private/hyperparameters define-hyperparameters)
         (only-in raft current-device-resources)
         (only-in raft/unsafe deposit->list)
         (only-in "deposit.rkt" deposit-for/who)
         (only-in "native.rkt"
                  cuml-cluster-spectral-clustering-fit-graph-job
                  cuml-cluster-spectral-clustering-fit-job))

(provide (struct-out spectral-clustering-result)
         spectral-clustering-fit
         spectral-clustering-fit/who
         spectral-clustering-fit-graph
         spectral-clustering-fit-graph/who
         spectral-clustering-params)

(struct spectral-clustering-result (labels))

(define-hyperparameters spectral-clustering-params
  [n-clusters 8]
  [n-components #f]
  [random-state 0]
  [n-neighbors 10]
  [n-init 10]
  [eigen-tol 'auto]
  [affinity 'nearest-neighbors #:fixed])

(define (native-solver-options n-components eigen-tol)
  (values (or n-components 0) (if n-components 0 1) (if (eq? eigen-tol 'auto) 0.0 eigen-tol)))

(define/who (spectral-clustering-fit who
                                     X
                                     #:hyperparameters spectral-clustering-params
                                     #:resources [resources (current-device-resources)])
  (define out (deposit-for/who who resources))
  (define-values (fixed-components auto-components tolerance)
    (native-solver-options n-components eigen-tol))
  (cuml-cluster-spectral-clustering-fit-job who
                                            resources
                                            X
                                            out
                                            n-clusters
                                            fixed-components
                                            auto-components
                                            n-neighbors
                                            n-init
                                            tolerance
                                            random-state)
  (apply spectral-clustering-result (deposit->list out)))

(define/who (spectral-clustering-fit-graph who
                                           rows
                                           cols
                                           vals
                                           n-samples
                                           #:hyperparameters spectral-clustering-params
                                           #:ignoring (n-neighbors affinity)
                                           #:resources [resources (current-device-resources)])
  (define out (deposit-for/who who resources))
  (define-values (fixed-components auto-components tolerance)
    (native-solver-options n-components eigen-tol))
  (cuml-cluster-spectral-clustering-fit-graph-job who
                                                  resources
                                                  rows
                                                  cols
                                                  vals
                                                  n-samples
                                                  out
                                                  n-clusters
                                                  fixed-components
                                                  auto-components
                                                  n-init
                                                  tolerance
                                                  random-state)
  (apply spectral-clustering-result (deposit->list out)))
