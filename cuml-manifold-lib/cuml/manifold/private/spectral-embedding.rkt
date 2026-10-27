#lang racket/base

(require (only-in cuml/core/private/define-who define/who)
         (only-in cuml/core/private/deposit deposit-for/who)
         (only-in cuml/core/private/hyperparameters define-hyperparameters)
         (only-in racket/match match-define)
         (only-in raft current-device-resources)
         (only-in raft/unsafe deposit->list)
         (only-in "native.rkt"
                  cuml-manifold-spectral-embedding-fit-graph-job
                  cuml-manifold-spectral-embedding-fit-job)
         (only-in "support.rkt" bound-params drawn-seed flag rows-of))

(provide (struct-out spectral-embedding-result)
         spectral-embedding-fit
         spectral-embedding-fit-graph
         spectral-embedding-fit-graph/who
         spectral-embedding-fit-transform
         spectral-embedding-fit-transform/who
         spectral-embedding-fit/who
         spectral-embedding-params)

(struct spectral-embedding-result (embedding n-neighbors params))

(define-hyperparameters spectral-embedding-params
  [n-components 2]
  [n-neighbors #f]
  [norm-laplacian #t]
  [drop-first #t]
  [random-state 0]
  [affinity 'nearest-neighbors #:fixed])

(define most-default-neighbors 256)

(define (neighbours n-neighbors X)
  (or n-neighbors (min most-default-neighbors (max 2 (quotient (rows-of X) 10)))))

(define (fitted who X params resources)
  (match-define (hash-table ['n-components n-components]
                            ['n-neighbors n-neighbors]
                            ['norm-laplacian norm-laplacian]
                            ['drop-first drop-first]
                            ['random-state random-state])
    params)
  (define out (deposit-for/who who resources))
  (define used (neighbours n-neighbors X))
  (cuml-manifold-spectral-embedding-fit-job who
                                            resources
                                            X
                                            used
                                            out
                                            n-components
                                            (flag norm-laplacian)
                                            (flag drop-first)
                                            (or random-state (drawn-seed)))
  (match-define (list embedding) (deposit->list out))
  (spectral-embedding-result embedding used params))

(define/who (spectral-embedding-fit who
                                    X
                                    #:hyperparameters spectral-embedding-params
                                    #:resources [resources (current-device-resources)])
  (fitted who X (bound-params spectral-embedding-params) resources))

(define/who (spectral-embedding-fit-graph who
                                          rows
                                          cols
                                          vals
                                          n-samples
                                          #:hyperparameters spectral-embedding-params
                                          #:ignoring (n-neighbors affinity)
                                          #:resources [resources (current-device-resources)])
  (define out (deposit-for/who who resources))
  (cuml-manifold-spectral-embedding-fit-graph-job who
                                                  resources
                                                  rows
                                                  cols
                                                  vals
                                                  n-samples
                                                  out
                                                  n-components
                                                  (flag norm-laplacian)
                                                  (flag drop-first)
                                                  (or random-state (drawn-seed)))
  (match-define (list embedding) (deposit->list out))
  (spectral-embedding-result embedding
                             #f
                             (spectral-embedding-params #:n-components n-components
                                                        #:norm-laplacian norm-laplacian
                                                        #:drop-first drop-first
                                                        #:random-state random-state)))

(define/who (spectral-embedding-fit-transform who
                                              X
                                              #:hyperparameters spectral-embedding-params
                                              #:resources [resources (current-device-resources)])
  (define result (fitted who X (bound-params spectral-embedding-params) resources))
  (values result (spectral-embedding-result-embedding result)))
