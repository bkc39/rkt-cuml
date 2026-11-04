#lang racket/base

(require (only-in racket/match match-define)
         (only-in racket/math exact-floor)
         (only-in raft list*->device-matrix list->device-vector)
         (only-in "blob-data.rkt" uniform-source))

(provide integer-weights
         label-cases
         on
         rows
         vec
         weights-of)

(define (vec xs dt)
  (list->device-vector xs #:dtype dt))

(define (rows xs dt)
  (list*->device-matrix xs #:dtype dt))

(define (on xs dt)
  (if (list? (car xs))
      (rows xs dt)
      (vec xs dt)))

(define (weights-of w [dt 'float64])
  (and w (vec w dt)))

(define int32-palette '(-1 0 1 2 7 2147483647 -2147483647 -2147483648))

(define int64-palette
  (append int32-palette
          (list (expt 2 40) (- (expt 2 40)) (expt 2 32) (expt 2 53) (add1 (expt 2 53)))))

(define (palette dt)
  (if (eq? dt 'int32) int32-palette int64-palette))

(define (label-cases #:seed seed #:count count #:most most)
  (define next (uniform-source seed))
  (define (pick xs)
    (list-ref xs (exact-floor (* (next) (length xs)))))
  (for/list ([i (in-range count)]
             [dtypes (in-cycle '((int32 int32) (int32 int64) (int64 int64)))])
    (match-define (list at bt) dtypes)
    (define n (add1 (exact-floor (* (next) most))))
    (define truth
      (for/list ([_ (in-range n)])
        (pick (palette at))))
    (define found
      (for/list ([t (in-list truth)])
        (define u (next))
        (cond
          [(and (eq? bt 'int64) (eqv? t 0) (< u 0.2)) (expt 2 32)]
          [(< u 0.7) t]
          [else (pick (palette bt))])))
    (list i at bt truth found)))

(define (integer-weights n seed)
  (define next (uniform-source seed))
  (define w
    (for/list ([_ (in-range n)])
      (exact->inexact (exact-floor (* 9 (next))))))
  (if (andmap zero? w)
      (cons 1.0 (cdr w))
      w))
