#lang racket/base

(require (for-syntax racket/base
                     (only-in syntax/parse/pre attribute))
         (only-in cuml/core/private/hyperparameters define-codes key->keyword table-id)
         (only-in racket/match match)
         (only-in racket/math infinite? nan?)
         (only-in raft device-array? shape)
         (only-in syntax/parse/define define-syntax-parse-rule))

(provide bound-params
         drawn-seed
         finite-real?
         flag
         metric-code
         rows-of)

(define-codes metric-code [euclidean 0] [sqeuclidean 1] [manhattan 2] [chebyshev 3] [cosine 4])

(define seed-half 65536)

(define (drawn-seed)
  (+ (* (random seed-half) seed-half) (random seed-half)))

(define (finite-real? v)
  (and (real? v) (not (nan? v)) (not (infinite? v))))

(define (rows-of a)
  (match (and (device-array? a) (shape a))
    [(cons n _) n]
    [_ 0]))

(define (flag v)
  (if v 1 0))

(define-syntax-parse-rule (bound-params table:table-id)
  #:with (key ...) (for/list ([key (in-list (attribute table.keys))])
                     (datum->syntax #'table key #'table))
  #:with (keyword ...) (map key->keyword (attribute table.keys))
  (table (~@ keyword key) ...))
