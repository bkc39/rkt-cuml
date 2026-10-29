#lang racket/base

(require (only-in cuml/core/private/define-who define/who)
         (only-in cuml/core/private/deposit deposit-for/who)
         (only-in cuml/core/private/hyperparameters define-hyperparameters)
         (only-in racket/match match-define)
         (only-in raft current-device-resources)
         (only-in raft/unsafe deposit->list)
         (only-in "native.rkt"
                  cuml-decomposition-pca-fit-job
                  cuml-decomposition-pca-inverse-transform-job
                  cuml-decomposition-pca-transform-job)
         (only-in "solver.rkt" solver-code))

(provide (struct-out pca-result)
         pca-fit
         pca-fit-transform
         pca-fit-transform/who
         pca-fit/who
         pca-inverse-transform
         pca-inverse-transform/who
         pca-params
         pca-transform
         pca-transform/who)

(struct pca-result
        (components explained-variance
                    explained-variance-ratio
                    singular-values
                    mean
                    noise-variance
                    n-samples
                    whiten))

(define-hyperparameters pca-params
  [n-components #f]
  [whiten #f]
  [solver 'full]
  [tol 0.0]
  [iterations 15])

(define (fitted who X n-components whiten solver tol iterations resources)
  (define out (deposit-for/who who resources))
  (define-values (noise-variance n-samples)
    (cuml-decomposition-pca-fit-job who
                                    resources
                                    X
                                    out
                                    (if n-components 0 1)
                                    (or n-components 0)
                                    (solver-code solver)
                                    tol
                                    iterations))
  (match-define (list components variance ratio singular mean) (deposit->list out))
  (pca-result components variance ratio singular mean noise-variance n-samples (and whiten #t)))

(define (applied job who result X resources)
  (match-define (pca-result components variance _ _ mean _ _ whiten) result)
  (define out (deposit-for/who who resources))
  (job who resources X components mean (and whiten variance) out (if whiten 1 0))
  (match-define (list applied-X) (deposit->list out))
  applied-X)

(define/who (pca-fit who
                     X
                     #:hyperparameters pca-params
                     #:resources [resources (current-device-resources)])
  (fitted who X n-components whiten solver tol iterations resources))

(define/who (pca-transform who result X #:resources [resources (current-device-resources)])
  (applied cuml-decomposition-pca-transform-job who result X resources))

(define/who (pca-inverse-transform who result X #:resources [resources (current-device-resources)])
  (applied cuml-decomposition-pca-inverse-transform-job who result X resources))

(define/who (pca-fit-transform who
                               X
                               #:hyperparameters pca-params
                               #:resources [resources (current-device-resources)])
  (define result (fitted who X n-components whiten solver tol iterations resources))
  (values result (applied cuml-decomposition-pca-transform-job who result X resources)))
