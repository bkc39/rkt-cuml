#lang racket/base

(require (only-in racket/match match-define)
         (only-in rackunit check-=)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/linear-data.rkt" law-case)
         (only-in "private/mbsgd-data.rkt" normal-rows)
         (only-in "private/mbsgd-gate-data.rkt" constant-rows loss-r2 lost-at-eta0 regressor-fit))

(define noise (map car (normal-rows 300 1 9)))

(define (near-floor-targets s)
  (for/list ([r (in-list constant-rows)]
             [e (in-list noise)])
    (+ 1e6 (* s (car r)) (* 0.2 s e))))

(define near-floor-losses '((0.5 1.0 0.05 -0.14) (0.5 2.0 0.02 -0.097)))

(test-gpu "a float32 lasso a few floors above round-off whose loss rises past the zero model's is refused"
  (for ([r (in-list near-floor-losses)])
    (match-define (list s alpha row-eta0 r2) r)
    (define y (near-floor-targets s))
    (define (fit eta0)
      (regressor-fit constant-rows y 'float32 #:penalty 'l1 #:alpha alpha #:eta0 eta0))
    (law-case (format "y = 1e6 + ~a·x₁ + ~a·e, l1 alpha ~a" s (* 0.2 s) alpha)
      (lost-at-eta0 fit row-eta0)
      (check-= (loss-r2 (fit row-eta0)) r2 0.01))))
