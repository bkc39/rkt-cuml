#lang racket/base

(require (only-in racket/match match-define)
         (only-in rackunit check-=)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/linear-data.rkt" law-case)
         (only-in "private/mbsgd-gate-data.rkt"
                  at-eta0
                  constant-rows
                  linear-targets
                  loss-r2
                  lost-at-eta0))

(define near-floor-losses '((0.5 1.0 0.05 -0.14) (0.5 2.0 0.02 -0.097)))

(test-gpu "a float32 lasso a few floors above round-off whose loss rises past the zero model's is refused"
  (for ([r (in-list near-floor-losses)])
    (match-define (list s alpha row-eta0 r2) r)
    (define fit
      (at-eta0 constant-rows (linear-targets 1e6 s (* 0.2 s)) 'float32 #:penalty 'l1 #:alpha alpha))
    (law-case (format "y = 1e6 + ~a·x₁ + ~a·e, l1 alpha ~a" s (* 0.2 s) alpha)
      (lost-at-eta0 fit row-eta0)
      (check-= (loss-r2 (fit row-eta0)) r2 0.01))))
