#lang racket/base

(require (only-in raft current-device-resources)
         (only-in "defaults.rkt" define-defaults)
         (only-in "define-who.rkt" define/who))

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

(define/who (spectral-clustering-fit who
                                     X
                                     #:n-clusters [n-clusters (default 'n-clusters)]
                                     #:n-components [n-components (default 'n-components)]
                                     #:random-state [random-state (default 'random-state)]
                                     #:n-neighbors [n-neighbors (default 'n-neighbors)]
                                     #:n-init [n-init (default 'n-init)]
                                     #:eigen-tol [eigen-tol (default 'eigen-tol)]
                                     #:resources [resources (current-device-resources)])
  (error 'unimplemented))

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
  (error 'unimplemented))
