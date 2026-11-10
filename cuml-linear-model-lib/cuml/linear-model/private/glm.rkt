#lang racket/base

(require (only-in cuml/core/private/define-who define/who)
         (only-in cuml/core/private/deposit deposit-for/who)
         (only-in cuml/core/private/logic-error logic-error)
         (only-in racket/match match match-define)
         (only-in raft/unsafe deposit->list)
         (only-in "native.rkt" cuml-linear-model-gemm-predict-job))

(provide alpha-value/who
         glm-predict/who
         intercept-flag/who
         iteration-count/who
         real-value/who)

(define/who (intercept-flag who fit-intercept)
  (match fit-intercept
    [#t 1]
    [#f 0]
    [v (logic-error "~a: fit-intercept: expected #t or #f, got ~e" who v)]))

(define/who (real-value who key v)
  (if (real? v)
      (real->double-flonum v)
      (logic-error "~a: ~a: expected a real number, got ~e" who key v)))

(define/who (alpha-value who alpha) (real-value/who who 'alpha alpha))

(define int32-max (sub1 (expt 2 31)))

(define/who (iteration-count who n)
  (if (and (exact-integer? n) (<= 1 n int32-max))
      n
      (logic-error "~a: max-iter: expected an exact integer from 1 to ~a, got ~e" who int32-max n)))

(define (intercept-value who intercept)
  (if (real? intercept)
      (real->double-flonum intercept)
      (logic-error "~a: intercept: expected a real number, got ~e" who intercept)))

(define/who (glm-predict who coef intercept X resources)
  (define b (intercept-value who intercept))
  (define out (deposit-for/who who resources))
  (cuml-linear-model-gemm-predict-job who resources X coef b out)
  (match-define (list predictions) (deposit->list out))
  predictions)
