#lang racket/base

(require (only-in cuml/core/private/define-who define/who)
         (only-in cuml/core/private/deposit deposit-for/who)
         (only-in cuml/core/private/hyperparameters define-codes)
         (only-in cuml/core/private/logic-error logic-error)
         (only-in racket/list last)
         (only-in racket/match match match-define)
         (only-in raft device-matrix? dtype shape)
         (only-in raft/unsafe deposit->list)
         (only-in "native.rkt" cuml-linear-model-gemm-predict-job))

(provide alpha-value/who
         bounded-count/who
         checked-coef/who
         flag/who
         glm-predict/who
         intercept-flag/who
         nonnegative-value/who
         penalty-code
         positive-value/who
         real-value/who)

(define-codes penalty-code [none 0] [l1 1] [l2 2] [elasticnet 3])

(define/who (flag who key v)
  (match v
    [#t 1]
    [#f 0]
    [_ (logic-error "~a: ~a: expected #t or #f, got ~e" who key v)]))

(define/who (intercept-flag who fit-intercept) (flag/who who 'fit-intercept fit-intercept))

(define/who (real-value who key v)
  (if (real? v)
      (real->double-flonum v)
      (logic-error "~a: ~a: expected a real number, got ~e" who key v)))

(define/who (alpha-value who alpha) (real-value/who who 'alpha alpha))

(define (flonum-of v)
  (and (real? v) (real->double-flonum v)))

(define/who (positive-value who key v)
  (define f (flonum-of v))
  (cond
    [(not (and f (positive? v) (< f +inf.0)))
     (logic-error "~a: ~a: expected a positive finite real, got ~e" who key v)]
    [(zero? f)
     (logic-error "~a: ~a: expected a positive real representable as a nonzero flonum, got ~e"
                  who
                  key
                  v)]
    [else f]))

(define/who (nonnegative-value who key v)
  (define f (flonum-of v))
  (if (and f (>= v 0) (< f +inf.0))
      f
      (logic-error "~a: ~a: expected a nonnegative finite real, got ~e" who key v)))

(define int32-max (sub1 (expt 2 31)))

(define/who (bounded-count who key n)
  (if (and (exact-integer? n) (<= 1 n int32-max))
      n
      (logic-error "~a: ~a: expected an exact integer from 1 to ~a, got ~e" who key int32-max n)))

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

(define/who (checked-coef who result X result? coef-of)
  (unless (result? result)
    (raise-argument-error who (symbol->string (object-name result?)) result))
  (unless (device-matrix? X)
    (logic-error "~a: X: expected a device-matrix, got ~e" who X))
  (define coef (coef-of result))
  (define p (last (shape coef)))
  (match-define (list _ d) (shape X))
  (unless (= p d)
    (logic-error "~a: X: expected ~a columns, got ~a" who p d))
  (unless (eq? (dtype coef) (dtype X))
    (logic-error "~a: X: expected dtype ~a (the fit's), got ~a" who (dtype coef) (dtype X)))
  coef)
