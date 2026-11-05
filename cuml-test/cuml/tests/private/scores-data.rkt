#lang racket/base

(require (only-in cuml/metrics accuracy-score mean-squared-error r2-score)
         (only-in racket/match match match-define)
         (only-in racket/math exact-floor)
         (only-in raft list*->device-matrix list->device-vector)
         (only-in "blob-data.rkt" uniform-source))

(provide accuracy
         integer-weights
         label-cases
         mse
         on
         r2
         rows
         vec
         weights-of)

(define (vec xs dt)
  (list->device-vector xs #:dtype dt))

(define (rows xs dt)
  (list*->device-matrix xs #:dtype dt))

(define (on xs dt)
  (match xs
    [(cons (? list?) _) (rows xs dt)]
    [_ (vec xs dt)]))

(define (weights-of w [dt 'float64])
  (and w (vec w dt)))

(define (r2 y p [w #f] #:dtype [dt 'float64] #:pred [pt dt] #:weight [wt 'float64])
  (r2-score (vec y dt) (vec p pt) #:sample-weight (weights-of w wt)))

(define (mse y p [w #f] #:dtype [dt 'float64] #:pred [pt dt] #:weight [wt 'float64])
  (mean-squared-error (on y dt) (on p pt) #:sample-weight (weights-of w wt)))

(define (accuracy a b [w #f] #:true [at 'int32] #:pred [bt at] #:weight [wt 'float64])
  (accuracy-score (vec a at) (vec b bt) #:sample-weight (weights-of w wt)))

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
  (match w
    [(cons _ rest)
     #:when (andmap zero? w)
     (cons 1.0 rest)]
    [_ w]))
