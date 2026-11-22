#lang racket/base

(require (only-in cuml/core/private/define-who define/who)
         (only-in cuml/core/private/hyperparameters define-codes)
         (only-in cuml/core/private/logic-error logic-error)
         (only-in cuml/core/private/scalars
                  bounded-count/who
                  finite-value/who
                  in-dtype/who
                  int32-max
                  positive-value/who))

(provide (struct-out svm-kernel)
         (struct-out svm-solver)
         gamma-mode
         kernel-code
         svm-settings/who)

(struct svm-kernel (code degree gamma-mode gamma coef0 cache-size))

(struct svm-solver (C tol max-iter nochange-steps))

(define-codes kernel-code [linear 0] [poly 1] [rbf 2] [sigmoid 3])

(define-codes gamma-mode [number 0] [scale 1] [auto 2])

(define/who (kernel-value who kernel)
  (case kernel
    [(linear poly rbf sigmoid) (kernel-code kernel)]
    [(precomputed)
     (logic-error
      "~a: kernel: 'precomputed is not supported; pass X with 'linear, 'poly, 'rbf or 'sigmoid"
      who)]
    [else (logic-error "~a: kernel: expected 'linear, 'poly, 'rbf or 'sigmoid, got ~e" who kernel)]))

(define/who (gamma-value who gamma dt)
  (define f (and (real? gamma) (real->double-flonum gamma)))
  (cond
    [(memq gamma '(scale auto)) (values (gamma-mode gamma) 0.0)]
    [(and f (>= gamma 0) (< f +inf.0)) (values (gamma-mode 'number) (in-dtype/who who 'gamma f dt))]
    [else
     (logic-error "~a: gamma: expected 'scale, 'auto or a finite real at least 0, got ~e"
                  who
                  gamma)]))

(define/who (max-iter-value who v)
  (cond
    [(not v) -1]
    [(and (exact-integer? v) (<= 1 v int32-max)) v]
    [else
     (logic-error "~a: max-iter: expected #f or an exact integer from 1 to ~a, got ~e"
                  who
                  int32-max
                  v)]))

(define/who (svm-settings who params dt)
  (define (ref key)
    (hash-ref params key))
  (define (narrowed key v #:positive [positive #f])
    (in-dtype/who who key v dt #:positive positive))
  (define code (kernel-value/who who (ref 'kernel)))
  (define degree (bounded-count/who who 'degree (ref 'degree) #:least 0))
  (define-values (mode gamma) (gamma-value/who who (ref 'gamma) dt))
  (define coef0 (narrowed 'coef0 (finite-value/who who 'coef0 (ref 'coef0))))
  (define C (narrowed 'C (positive-value/who who 'C (ref 'C)) #:positive #t))
  (define tol (narrowed 'tol (positive-value/who who 'tol (ref 'tol)) #:positive #t))
  (define cache-size (positive-value/who who 'cache-size (ref 'cache-size)))
  (values (svm-kernel code degree mode gamma coef0 cache-size)
          (svm-solver C
                      tol
                      (max-iter-value/who who (ref 'max-iter))
                      (bounded-count/who who 'nochange-steps (ref 'nochange-steps) #:least 0))))
