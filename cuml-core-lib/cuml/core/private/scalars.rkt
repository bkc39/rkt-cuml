#lang racket/base

(require (only-in racket/flonum flsingle)
         (only-in "define-who.rkt" define/who)
         (only-in "logic-error.rkt" logic-error))

(provide bounded-count/who
         finite-value/who
         in-dtype/who
         int32-max
         nonnegative-value/who
         positive-value/who)

(define int32-max (sub1 (expt 2 31)))

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

(define/who (finite-value who key v)
  (define f (flonum-of v))
  (if (and f (< -inf.0 f +inf.0))
      f
      (logic-error "~a: ~a: expected a finite real, got ~e" who key v)))

(define/who (bounded-count who key n #:least [least 1])
  (if (and (exact-integer? n) (<= least n int32-max))
      n
      (logic-error "~a: ~a: expected an exact integer from ~a to ~a, got ~e" who key least int32-max n)))

(define/who (in-dtype who key f dt #:positive [positive #f])
  (define held
    (if (eq? dt 'float32)
        (flsingle f)
        f))
  (cond
    [(= (abs held) +inf.0) (logic-error "~a: ~a: ~a overflows ~a" who key f dt)]
    [(and positive (zero? held)) (logic-error "~a: ~a: ~a rounds to 0 in ~a" who key f dt)]
    [else f]))
