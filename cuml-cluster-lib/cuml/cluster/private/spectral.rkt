#lang racket/base

(require (only-in raft current-device-resources)
         (only-in raft/unsafe deposit->list)
         (only-in "defaults.rkt" define-defaults)
         (only-in "define-who.rkt" define/who)
         (only-in "deposit.rkt" deposit-for)
         (only-in "native.rkt"
                  cuml-cluster-spectral-clustering-fit-graph-job
                  cuml-cluster-spectral-clustering-fit-job))

(provide (struct-out spectral-clustering-result)
         spectral-clustering-defaults
         spectral-clustering-fit
         spectral-clustering-fit/who
         spectral-clustering-fit-graph
         spectral-clustering-fit-graph/who)

(struct spectral-clustering-result (labels))

(define-defaults spectral-clustering-defaults
                 [n-clusters 8]
                 [n-components #f]
                 [random-state 0]
                 [n-neighbors 10]
                 [n-init 10]
                 [eigen-tol 'auto])

(define (components n-components)
  (if n-components
      (values n-components 0)
      (values 0 1)))

(define (tolerance eigen-tol)
  (if (eq? eigen-tol 'auto) 0.0 eigen-tol))

(define/who (spectral-clustering-fit who
                                     X
                                     #:n-clusters [n-clusters (default 'n-clusters)]
                                     #:n-components [n-components (default 'n-components)]
                                     #:random-state [random-state (default 'random-state)]
                                     #:n-neighbors [n-neighbors (default 'n-neighbors)]
                                     #:n-init [n-init (default 'n-init)]
                                     #:eigen-tol [eigen-tol (default 'eigen-tol)]
                                     #:resources [resources (current-device-resources)])
  (define out (deposit-for who resources))
  (define-values (nc auto) (components n-components))
  (cuml-cluster-spectral-clustering-fit-job who
                                            resources
                                            X
                                            out
                                            n-clusters
                                            nc
                                            auto
                                            n-neighbors
                                            n-init
                                            (tolerance eigen-tol)
                                            random-state)
  (apply spectral-clustering-result (deposit->list out)))

(define/who (spectral-clustering-fit-graph who
                                           rows
                                           cols
                                           vals
                                           n-samples
                                           #:n-clusters [n-clusters (default 'n-clusters)]
                                           #:n-components [n-components (default 'n-components)]
                                           #:random-state [random-state (default 'random-state)]
                                           #:n-init [n-init (default 'n-init)]
                                           #:eigen-tol [eigen-tol (default 'eigen-tol)]
                                           #:resources [resources (current-device-resources)])
  (define out (deposit-for who resources))
  (define-values (nc auto) (components n-components))
  (cuml-cluster-spectral-clustering-fit-graph-job who
                                                  resources
                                                  rows
                                                  cols
                                                  vals
                                                  n-samples
                                                  out
                                                  n-clusters
                                                  nc
                                                  auto
                                                  n-init
                                                  (tolerance eigen-tol)
                                                  random-state)
  (apply spectral-clustering-result (deposit->list out)))
