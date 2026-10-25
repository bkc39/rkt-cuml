#lang racket/base

(require (only-in cuml/core/private/define-who define/who)
         (only-in cuml/core/private/deposit deposit-for/who)
         (only-in cuml/core/private/hyperparameters define-codes define-hyperparameters)
         (only-in cuml/core/private/logic-error logic-error)
         (only-in racket/flonum flsingle)
         (only-in racket/match match match-define)
         (only-in racket/math infinite? nan?)
         (only-in raft current-device-resources device-array? shape)
         (only-in raft/unsafe deposit->list)
         (only-in "curve.rkt" curve-parameters)
         (only-in "native.rkt" cuml-manifold-umap-fit-job cuml-manifold-umap-transform-job))

(provide (struct-out umap-result)
         umap-fit
         umap-fit-transform
         umap-fit-transform/who
         umap-fit/who
         umap-params
         umap-transform
         umap-transform/who)

(struct umap-result (embedding X a b params))

(define-hyperparameters umap-params
  [n-neighbors 15]
  [n-components 2]
  [metric 'euclidean]
  [n-epochs #f]
  [learning-rate 1.0]
  [min-dist 0.1]
  [spread 1.0]
  [a #f]
  [b #f]
  [init 'spectral]
  [negative-sample-rate 5]
  [random-state 0]
  [build-algo 'brute-force])

(define-codes metric-code [euclidean 0] [sqeuclidean 1] [manhattan 2] [chebyshev 3] [cosine 4])
(define-codes init-code [random 0] [spectral 1])
(define-codes build-algo-code [brute-force 0] [nn-descent 1])

(define most-float 3.4028234663852886e+38)
(define least-normal-float (expt 2.0 -126))
(define deterministic-below 300)
(define seed-half 65536)

(define (finite-real? v)
  (and (real? v) (not (nan? v)) (not (infinite? v))))

(define (float-positive? v)
  (and (finite-real? v) (> v 0) (<= v most-float)))

(define (single v)
  (flsingle (exact->inexact v)))

(define (fitted-curve who min-dist spread)
  (define-values (a b) (curve-parameters min-dist spread))
  (cond
    [(<= least-normal-float a most-float) (values (single a) (single b))]
    [else
     (logic-error
      "~a: spread: with min-dist ~a, a spread of ~a gives a curve parameter a of ~a, outside float32's normal range"
      who
      min-dist
      spread
      a)]))

(define (curve who min-dist spread a b)
  (cond
    [(not (and (finite-real? min-dist) (>= min-dist 0)))
     (logic-error "~a: min-dist: expected a finite value of at least 0, got ~a" who min-dist)]
    [(not (and (finite-real? spread) (> spread 0)))
     (logic-error "~a: spread: expected a finite value above 0, got ~a" who spread)]
    [(> min-dist spread)
     (logic-error "~a: min-dist: expected at most the spread ~a, got ~a" who spread min-dist)]
    [(and a (not b)) (logic-error "~a: b: expected a value when a is given, got #f" who)]
    [(and b (not a)) (logic-error "~a: a: expected a value when b is given, got #f" who)]
    [(and a (not (float-positive? a))) (refuse-curve-value who 'a a)]
    [(and b (not (float-positive? b))) (refuse-curve-value who 'b b)]
    [a (values (single a) (single b))]
    [else (fitted-curve who min-dist spread)]))

(define (refuse-curve-value who name v)
  (logic-error "~a: ~a: expected a finite value above 0 and at most ~a, got ~a"
               who
               name
               most-float
               v))

(define (drawn-seed)
  (+ (* (random seed-half) seed-half) (random seed-half)))

(define (rows-of a)
  (match (and (device-array? a) (shape a))
    [(list n _) n]
    [_ 0]))

(define (flag v)
  (if v 1 0))

(define (fitted who X y params resources)
  (match-define (hash-table ['n-neighbors n-neighbors]
                            ['n-components n-components]
                            ['metric metric]
                            ['n-epochs n-epochs]
                            ['learning-rate learning-rate]
                            ['min-dist min-dist]
                            ['spread spread]
                            ['a given-a]
                            ['b given-b]
                            ['init init]
                            ['negative-sample-rate negative-sample-rate]
                            ['random-state random-state]
                            ['build-algo build-algo])
    params)
  (define-values (a b) (curve who min-dist spread given-a given-b))
  (define out (deposit-for/who who resources))
  (cuml-manifold-umap-fit-job who
                              resources
                              X
                              y
                              out
                              n-neighbors
                              n-components
                              (flag n-epochs)
                              (or n-epochs 0)
                              learning-rate
                              a
                              b
                              (init-code init)
                              negative-sample-rate
                              (or random-state (drawn-seed))
                              (flag (or random-state (< (rows-of X) deterministic-below)))
                              (build-algo-code build-algo)
                              (metric-code metric))
  (match-define (list embedding) (deposit->list out))
  (umap-result embedding X a b params))

(define (params-of n-neighbors
                   n-components
                   metric
                   n-epochs
                   learning-rate
                   min-dist
                   spread
                   a
                   b
                   init
                   negative-sample-rate
                   random-state
                   build-algo)
  (umap-params #:n-neighbors n-neighbors
               #:n-components n-components
               #:metric metric
               #:n-epochs n-epochs
               #:learning-rate learning-rate
               #:min-dist min-dist
               #:spread spread
               #:a a
               #:b b
               #:init init
               #:negative-sample-rate negative-sample-rate
               #:random-state random-state
               #:build-algo build-algo))

(define/who (umap-fit who
                      X
                      #:y [y #f]
                      #:hyperparameters umap-params
                      #:resources [resources (current-device-resources)])
  (fitted who
          X
          y
          (params-of n-neighbors
                     n-components
                     metric
                     n-epochs
                     learning-rate
                     min-dist
                     spread
                     a
                     b
                     init
                     negative-sample-rate
                     random-state
                     build-algo)
          resources))

(define/who (umap-fit-transform who
                                X
                                #:y [y #f]
                                #:hyperparameters umap-params
                                #:resources [resources (current-device-resources)])
  (define result
    (fitted who
            X
            y
            (params-of n-neighbors
                       n-components
                       metric
                       n-epochs
                       learning-rate
                       min-dist
                       spread
                       a
                       b
                       init
                       negative-sample-rate
                       random-state
                       build-algo)
            resources))
  (values result (umap-result-embedding result)))

(define/who (umap-transform who result X #:resources [resources (current-device-resources)])
  (match-define (umap-result embedding train-X a b params) result)
  (match-define (hash-table ['n-neighbors n-neighbors]
                            ['n-components n-components]
                            ['metric metric]
                            ['n-epochs n-epochs]
                            ['learning-rate learning-rate]
                            ['negative-sample-rate negative-sample-rate]
                            ['random-state random-state])
    params)
  (define out (deposit-for/who who resources))
  (cuml-manifold-umap-transform-job
   who
   resources
   X
   train-X
   embedding
   out
   n-neighbors
   n-components
   (flag n-epochs)
   (or n-epochs 0)
   learning-rate
   a
   b
   negative-sample-rate
   (or random-state (drawn-seed))
   (flag (or random-state (< (max (rows-of train-X) (rows-of X)) deterministic-below)))
   (metric-code metric))
  (match-define (list transformed) (deposit->list out))
  transformed)
