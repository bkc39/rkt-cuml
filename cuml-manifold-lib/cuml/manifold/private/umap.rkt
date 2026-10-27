#lang racket/base

(require (only-in cuml/core/private/define-who define/who)
         (only-in cuml/core/private/deposit deposit-for/who)
         (only-in cuml/core/private/hyperparameters define-codes define-hyperparameters)
         (only-in cuml/core/private/logic-error logic-error)
         (only-in racket/flonum flsingle)
         (only-in racket/match match-define)
         (only-in raft current-device-resources)
         (only-in raft/unsafe deposit->list)
         (only-in "curve.rkt" curve-parameters)
         (only-in "native.rkt" cuml-manifold-umap-fit-job cuml-manifold-umap-transform-job)
         (only-in "support.rkt" bound-params drawn-seed finite-real? flag metric-code rows-of))

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

(define-codes init-code [random 0] [spectral 1])
(define-codes build-algo-code [brute-force 0] [nn-descent 1])

(define most-float 3.4028234663852886e+38)
(define least-normal-float (expt 2.0 -126))
(define deterministic-below 300)

(define (float-normal? v)
  (and (finite-real? v) (<= least-normal-float v most-float)))

(define (fitted-curve who min-dist spread)
  (define-values (a b) (curve-parameters min-dist spread))
  (cond
    [(float-normal? a) (values (flsingle a) (flsingle b))]
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
    [(and a (not (float-normal? a))) (refuse-curve-value who 'a a)]
    [(and b (not (float-normal? b))) (refuse-curve-value who 'b b)]
    [a (values (flsingle (real->double-flonum a)) (flsingle (real->double-flonum b)))]
    [else (fitted-curve who min-dist spread)]))

(define (refuse-curve-value who name v)
  (logic-error "~a: ~a: expected a finite value of at least ~a and at most ~a, got ~a"
               who
               name
               least-normal-float
               most-float
               v))

(define (schedule n-epochs random-state rows)
  (values (flag n-epochs)
          (or n-epochs 0)
          (or random-state (drawn-seed))
          (flag (or random-state (< rows deterministic-below)))))

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
  (define-values (epochs-given epochs seed deterministic)
    (schedule n-epochs random-state (rows-of X)))
  (define out (deposit-for/who who resources))
  (cuml-manifold-umap-fit-job who
                              resources
                              X
                              y
                              out
                              n-neighbors
                              n-components
                              epochs-given
                              epochs
                              learning-rate
                              a
                              b
                              (init-code init)
                              negative-sample-rate
                              seed
                              deterministic
                              (build-algo-code build-algo)
                              (metric-code metric))
  (match-define (list embedding) (deposit->list out))
  (umap-result embedding X a b params))

(define/who (umap-fit who
                      X
                      #:y [y #f]
                      #:hyperparameters umap-params
                      #:resources [resources (current-device-resources)])
  (fitted who X y (bound-params umap-params) resources))

(define/who (umap-fit-transform who
                                X
                                #:y [y #f]
                                #:hyperparameters umap-params
                                #:resources [resources (current-device-resources)])
  (define result (fitted who X y (bound-params umap-params) resources))
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
  (define-values (epochs-given epochs seed deterministic)
    (schedule n-epochs random-state (max (rows-of train-X) (rows-of X))))
  (define out (deposit-for/who who resources))
  (cuml-manifold-umap-transform-job who
                                    resources
                                    X
                                    train-X
                                    embedding
                                    out
                                    n-neighbors
                                    n-components
                                    epochs-given
                                    epochs
                                    learning-rate
                                    a
                                    b
                                    negative-sample-rate
                                    seed
                                    deterministic
                                    (metric-code metric))
  (match-define (list transformed) (deposit->list out))
  transformed)
