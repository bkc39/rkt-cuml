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
         (only-in "support.rkt" bound-params drawn-seed flag rows-or-zero))

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

(define most-derived-neighbors 256)

(define (derived-neighbors X)
  (min most-derived-neighbors (max 2 (quotient (rows-or-zero X) 10))))

(define (embedded who resources params n-neighbors job inputs)
  (match-define (hash-table ['n-components n-components]
                            ['norm-laplacian norm-laplacian]
                            ['drop-first drop-first]
                            ['random-state random-state])
    params)
  (define out (deposit-for/who who resources))
  (apply
   job
   who
   resources
   (append
    inputs
    (list out n-components (flag norm-laplacian) (flag drop-first) (or random-state (drawn-seed)))))
  (match-define (list embedding) (deposit->list out))
  (spectral-embedding-result embedding n-neighbors params))

(define (fitted who X params resources)
  (define n-neighbors (or (hash-ref params 'n-neighbors) (derived-neighbors X)))
  (embedded who
            resources
            params
            n-neighbors
            cuml-manifold-spectral-embedding-fit-job
            (list X n-neighbors)))

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
  (embedded who
            resources
            (bound-params spectral-embedding-params #:ignoring (n-neighbors affinity))
            #f
            cuml-manifold-spectral-embedding-fit-graph-job
            (list rows cols vals n-samples)))

(define/who (spectral-embedding-fit-transform who
                                              X
                                              #:hyperparameters spectral-embedding-params
                                              #:resources [resources (current-device-resources)])
  (define result (fitted who X (bound-params spectral-embedding-params) resources))
  (values result (spectral-embedding-result-embedding result)))
