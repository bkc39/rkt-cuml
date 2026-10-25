#lang racket/base

(require (only-in cuml/core/private/define-who define/who)
         (only-in cuml/core/private/deposit deposit-for/who)
         (only-in cuml/core/private/hyperparameters define-hyperparameters)
         (only-in racket/match match match-define)
         (only-in raft current-device-resources)
         (only-in raft/unsafe deposit->list)
         (only-in "native.rkt"
                  cuml-decomposition-tsvd-fit-job
                  cuml-decomposition-tsvd-inverse-transform-job
                  cuml-decomposition-tsvd-transform-job)
         (only-in "solver.rkt" solver-code))

(provide (struct-out tsvd-result)
         tsvd-fit
         tsvd-fit-transform
         tsvd-fit-transform/who
         tsvd-fit/who
         tsvd-inverse-transform
         tsvd-inverse-transform/who
         tsvd-params
         tsvd-transform
         tsvd-transform/who)

(struct tsvd-result (components explained-variance explained-variance-ratio singular-values))

(define-hyperparameters tsvd-params
  [n-components 1]
  [solver 'full]
  [tol 0.0]
  [iterations 15])

(define (fitted who X n-components solver tol iterations with-transform resources)
  (define out (deposit-for/who who resources))
  (cuml-decomposition-tsvd-fit-job who
                                   resources
                                   X
                                   out
                                   n-components
                                   (solver-code solver)
                                   tol
                                   iterations
                                   (if with-transform 1 0))
  (match (deposit->list out)
    [(list components singular variance ratio transformed ...)
     (apply values (tsvd-result components variance ratio singular) transformed)]))

(define (applied job who result X resources)
  (define out (deposit-for/who who resources))
  (job who resources X (tsvd-result-components result) out)
  (match-define (list applied-X) (deposit->list out))
  applied-X)

(define/who (tsvd-fit who
                      X
                      #:hyperparameters tsvd-params
                      #:resources [resources (current-device-resources)])
  (fitted who X n-components solver tol iterations #f resources))

(define/who (tsvd-transform who result X #:resources [resources (current-device-resources)])
  (applied cuml-decomposition-tsvd-transform-job who result X resources))

(define/who (tsvd-inverse-transform who result X #:resources [resources (current-device-resources)])
  (applied cuml-decomposition-tsvd-inverse-transform-job who result X resources))

(define/who (tsvd-fit-transform who
                                X
                                #:hyperparameters tsvd-params
                                #:resources [resources (current-device-resources)])
  (fitted who X n-components solver tol iterations #t resources))
