#lang racket/base

(require (only-in cuml/core/private/define-who define/who)
         (only-in cuml/core/private/deposit deposit-for/who)
         (only-in cuml/core/private/hyperparameters define-codes define-hyperparameters)
         (only-in racket/match match match-define)
         (only-in raft current-device-resources)
         (only-in raft/unsafe deposit->list)
         (only-in "glm.rkt" flag-code glm-predict/who)
         (only-in "native.rkt" cuml-linear-model-linear-regression-fit-job))

(provide (struct-out linear-regression-result)
         linear-regression-fit
         linear-regression-fit/who
         linear-regression-params
         linear-regression-predict
         linear-regression-predict/who)

(struct linear-regression-result (coef intercept))

(define-hyperparameters linear-regression-params
  [algorithm 'auto]
  [fit-intercept #t])

(define-codes algorithm-code [auto 1] [eig 1] [svd 0] [svd-jacobi 0] [qr 2] [svd-qr 3])

(define/who (linear-regression-fit who
                                   X
                                   y
                                   #:sample-weight [sample-weight #f]
                                   #:hyperparameters linear-regression-params
                                   #:resources [resources (current-device-resources)])
  (define out (deposit-for/who who resources))
  (define intercept
    (cuml-linear-model-linear-regression-fit-job who
                                                 resources
                                                 X
                                                 y
                                                 sample-weight
                                                 out
                                                 (flag-code fit-intercept)
                                                 (algorithm-code algorithm)))
  (match-define (list coef) (deposit->list out))
  (linear-regression-result coef intercept))

(define/who (linear-regression-predict who
                                       result
                                       X
                                       #:resources [resources (current-device-resources)])
  (match result
    [(linear-regression-result coef intercept) (glm-predict/who who coef intercept X resources)]
    [_ (raise-argument-error who "linear-regression-result?" result)]))
