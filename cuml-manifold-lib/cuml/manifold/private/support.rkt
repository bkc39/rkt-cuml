#lang racket/base

(require (only-in cuml/core/private/hyperparameters define-codes)
         (only-in racket/match match)
         (only-in racket/math infinite? nan?)
         (only-in raft device-array? shape))

(provide drawn-seed
         finite-real?
         flag
         metric-code
         rows-or-zero)

(define-codes metric-code [euclidean 0] [sqeuclidean 1] [manhattan 2] [chebyshev 3] [cosine 4])

(define seed-half 65536)

(define (drawn-seed)
  (+ (* (random seed-half) seed-half) (random seed-half)))

(define (finite-real? v)
  (and (real? v) (not (nan? v)) (not (infinite? v))))

(define (rows-or-zero a)
  (match (and (device-array? a) (shape a))
    [(cons n _) n]
    [_ 0]))

(define (flag v)
  (if v 1 0))
