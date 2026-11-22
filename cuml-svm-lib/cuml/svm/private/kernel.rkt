#lang racket/base

(require (only-in cuml/core/private/hyperparameters define-codes)
         (only-in cuml/core/private/logic-error logic-error))

(provide (struct-out svm-kernel)
         (struct-out svm-solver)
         gamma-mode
         kernel-code
         kernel-settings/who
         solver-settings/who)

(struct svm-kernel (code degree gamma-mode gamma coef0 cache-size))

(struct svm-solver (C tol max-iter nochange-steps))

(define-codes kernel-code [linear 0] [poly 1] [rbf 2] [sigmoid 3])

(define-codes gamma-mode [number 0] [scale 1] [auto 2])

(define most-int32 2147483647)

(define (finite? v)
  (and (real? v) (< -inf.0 v +inf.0)))

(define (kernel-value who kernel)
  (case kernel
    [(linear poly rbf sigmoid) (kernel-code kernel)]
    [(precomputed)
     (logic-error
      "~a: kernel: 'precomputed is not supported yet; pass X with 'linear, 'poly, 'rbf or 'sigmoid"
      who)]
    [else (logic-error "~a: kernel: expected 'linear, 'poly, 'rbf or 'sigmoid, got ~e" who kernel)]))

(define (count-value who key v least)
  (if (and (exact-integer? v) (<= least v most-int32))
      v
      (logic-error "~a: ~a: expected an exact integer from ~a to 2147483647, got ~e" who key least v)))

(define (gamma-value who gamma)
  (cond
    [(memq gamma '(scale auto)) (values (gamma-mode gamma) 0.0)]
    [(and (finite? gamma) (>= gamma 0)) (values (gamma-mode 'number) (real->double-flonum gamma))]
    [else (logic-error "~a: gamma: expected 'scale, 'auto or a finite real at least 0, got ~e" who gamma)]))

(define (finite-value who key v)
  (if (finite? v)
      (real->double-flonum v)
      (logic-error "~a: ~a: expected a finite real, got ~e" who key v)))

(define (positive-value who key v)
  (if (and (finite? v) (positive? v))
      (real->double-flonum v)
      (logic-error "~a: ~a: expected a positive finite real, got ~e" who key v)))

(define (max-iter-value who v)
  (cond
    [(not v) -1]
    [(and (exact-integer? v) (<= 1 v most-int32)) v]
    [else
     (logic-error "~a: max-iter: expected #f or an exact integer from 1 to 2147483647, got ~e" who v)]))

(define (kernel-settings/who who params)
  (define (ref key)
    (hash-ref params key))
  (define code (kernel-value who (ref 'kernel)))
  (define degree (count-value who 'degree (ref 'degree) 0))
  (define-values (mode gamma) (gamma-value who (ref 'gamma)))
  (define coef0 (finite-value who 'coef0 (ref 'coef0)))
  (svm-kernel code degree mode gamma coef0 (positive-value who 'cache-size (ref 'cache-size))))

(define (solver-settings/who who params)
  (define (ref key)
    (hash-ref params key))
  (svm-solver (positive-value who 'C (ref 'C))
              (positive-value who 'tol (ref 'tol))
              (max-iter-value who (ref 'max-iter))
              (count-value who 'nochange-steps (ref 'nochange-steps) 0)))
