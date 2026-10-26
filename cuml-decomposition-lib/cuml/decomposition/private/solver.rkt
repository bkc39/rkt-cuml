#lang racket/base

(require (only-in cuml/core/private/hyperparameters define-codes))

(provide solver-code)

(define-codes solver-code [full 0] [jacobi 1])
