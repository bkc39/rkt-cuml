#lang racket/base

(require (only-in racket/match match-define)
         (only-in rackunit check-=)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/linear-data.rkt" law-case)
         (only-in "private/mbsgd-gate-data.rkt"
                  loss-r2
                  lost-at-eta0
                  refused-at-eta0
                  regressor-fit
                  wide-problem))

(define-values (twelve twelve-targets) (wide-problem 12 8000))
(define-values (ten ten-targets) (wide-problem 10 8000))

(define ((wide-lasso X y dt alpha) eta0)
  (regressor-fit X y dt #:penalty 'l1 #:alpha alpha #:eta0 eta0 #:random-state 1))

(define worse-than-zero
  `((float32 ,twelve ,twelve-targets 12 0.6 -0.394) (float64 ,twelve ,twelve-targets 12 0.6 -0.383)
                                                    (float32 ,ten ,ten-targets 10 0.3 -1.18)))

(test-gpu "a wide lasso inside the widened cap whose loss rises past the zero model's is refused"
  (for ([r (in-list worse-than-zero)])
    (match-define (list dt X y n alpha r2) r)
    (law-case (format "~a ~a rows, 8000 columns, l1 alpha ~a" dt n alpha)
      (lost-at-eta0 (wide-lasso X y dt alpha) 1e-3)
      (check-= (loss-r2 ((wide-lasso X y dt alpha) 1e-3)) r2 0.01))))

(test-gpu "the same lasso in float64, past the widened cap, is refused by its objective"
  (law-case "float64 10 rows, 8000 columns, l1 alpha 0.3"
    (refused-at-eta0 (wide-lasso ten ten-targets 'float64 0.3) 1e-3)))
