#lang racket/base

(require (only-in cuml/core/private/deposit deposit-for/who)
         (only-in racket/match match-define)
         (only-in raft/unsafe deposit->list)
         (only-in "native.rkt" cuml-linear-model-gemm-predict-job))

(provide flag-code
         glm-predict)

(define (flag-code v)
  (case v
    [(#t) 1]
    [(#f) 0]
    [else -1]))

(define (glm-predict who coef intercept X resources)
  (define out (deposit-for/who who resources))
  (cuml-linear-model-gemm-predict-job who
                                      resources
                                      X
                                      coef
                                      (if (real? intercept) (real->double-flonum intercept) +nan.0)
                                      out)
  (match-define (list predictions) (deposit->list out))
  predictions)
