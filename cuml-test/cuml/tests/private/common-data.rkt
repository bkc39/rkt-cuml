#lang racket/base

(require (only-in racket/math pi sqr)
         (only-in raft device-vector->list device-vector? list->device-vector))

(provide columns-of
         block-starts
         flonum-exact
         gamma
         gaussian
         host
         kept-rows
         least-magnitude
         mean
         noise
         on-vector
         pick
         relative
         repeat-each
         scaled-rows
         squared-distance
         unit-roundoff
         unscaled
         worst)

(define (kept-rows xs weights)
  (for/list ([x (in-list xs)]
             [w (in-list weights)]
             #:unless (zero? w))
    x))

(define (columns-of rows)
  (apply map list rows))

(define (block-starts weights)
  (for/fold ([at 0]
             [firsts '()]
             #:result (reverse firsts))
            ([w (in-list weights)])
    (values (+ at w) (cons at firsts))))

(define (flonum-exact v)
  (inexact->exact (real->double-flonum v)))

(define (gamma k u)
  (/ (* k u) (- 1 (* k u))))

(define (gaussian)
  (* (sqrt (* -2.0 (log (- 1.0 (random))))) (cos (* 2.0 pi (random)))))

(define (host v)
  (if (device-vector? v)
      (device-vector->list v)
      v))

(define (least-magnitude dt)
  (if (eq? dt 'float32)
      (expt 2 -149)
      (expt 2 -1074)))

(define (mean xs)
  (/ (apply + xs) (length xs)))

(define (noise labels)
  (for/list ([l (in-list labels)]
             [i (in-naturals)]
             #:when (= l -1))
    i))

(define (pick xs indices)
  (define at (list->vector xs))
  (for/list ([i (in-list indices)])
    (vector-ref at i)))

(define (relative ours theirs)
  (/ (abs (- ours theirs)) (abs theirs)))

(define (repeat-each rows copies)
  (for*/list ([r (in-list rows)]
              [_ (in-range copies)])
    r))

(define (scaled-rows rows factor)
  (for/list ([row (in-list rows)])
    (map (lambda (v) (* v factor)) row)))

(define (squared-distance a b)
  (for/sum ([x (in-list a)] [y (in-list b)]) (sqr (- x y))))

(define (unit-roundoff dtype)
  (if (eq? dtype 'float32)
      (expt 2.0 -24)
      (expt 2.0 -53)))

(define (unscaled x by power)
  (exact->inexact (/ (inexact->exact x) (expt (inexact->exact by) power))))

(define (worst xs ys)
  (for/fold ([w 0.0])
            ([x (in-list xs)]
             [y (in-list ys)])
    (max w (abs (- x y)))))

(define (on-vector xs [dtype 'float64])
  (and xs (list->device-vector xs #:dtype dtype)))
