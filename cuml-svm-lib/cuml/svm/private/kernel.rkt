#lang racket/base

(require (only-in cuml/core/private/hyperparameters define-codes))

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

(define (kernel-settings/who who params)
  (error 'unimplemented))

(define (solver-settings/who who params)
  (error 'unimplemented))
