#lang racket/base

(require (only-in cuml/core/private/define-who define/who)
         (only-in cuml/core/private/deposit deposit-for/who)
         (only-in cuml/core/private/hyperparameters define-codes define-hyperparameters)
         (only-in racket/match match-define)
         (only-in racket/math exact-ceiling)
         (only-in raft current-device-resources)
         (only-in raft/unsafe deposit->list)
         (only-in "native.rkt" cuml-manifold-tsne-fit-job)
         (only-in "support.rkt" bound-params drawn-seed finite-real? metric-code))

(provide (struct-out tsne-result)
         tsne-fit
         tsne-fit-transform
         tsne-fit-transform/who
         tsne-fit/who
         tsne-params)

(struct tsne-result (embedding kl-divergence n-iter params))

(define-hyperparameters tsne-params
  [n-components 2]
  [perplexity 30.0]
  [early-exaggeration 12.0]
  [learning-rate 200.0]
  [max-iter 1000]
  [n-neighbors #f]
  [method 'fft]
  [metric 'euclidean]
  [init 'random]
  [random-state 0])

(define-codes method-code [fft 0] [exact 1])
(define-codes init-code [random 0] [pca 1])

(define most-neighbors 1023)

(define (neighbours n-neighbors perplexity)
  (cond
    [n-neighbors n-neighbors]
    [(finite-real? perplexity) (min most-neighbors (max 2 (exact-ceiling (* 3 perplexity))))]
    [else 2]))

(define (fitted who X params resources)
  (match-define (hash-table ['n-components n-components]
                            ['perplexity perplexity]
                            ['early-exaggeration early-exaggeration]
                            ['learning-rate learning-rate]
                            ['max-iter max-iter]
                            ['n-neighbors n-neighbors]
                            ['method method]
                            ['metric metric]
                            ['init init]
                            ['random-state random-state])
    params)
  (define out (deposit-for/who who resources))
  (define-values (kl-divergence iterations)
    (cuml-manifold-tsne-fit-job who
                                resources
                                X
                                out
                                n-components
                                perplexity
                                early-exaggeration
                                learning-rate
                                max-iter
                                (neighbours n-neighbors perplexity)
                                (method-code method)
                                (metric-code metric)
                                (init-code init)
                                (or random-state (drawn-seed))))
  (match-define (list embedding) (deposit->list out))
  (define n-iter (min (add1 iterations) max-iter))
  (tsne-result embedding (and (= n-iter max-iter) kl-divergence) n-iter params))

(define/who (tsne-fit who
                      X
                      #:hyperparameters tsne-params
                      #:resources [resources (current-device-resources)])
  (fitted who X (bound-params tsne-params) resources))

(define/who (tsne-fit-transform who
                                X
                                #:hyperparameters tsne-params
                                #:resources [resources (current-device-resources)])
  (define result (fitted who X (bound-params tsne-params) resources))
  (values result (tsne-result-embedding result)))
