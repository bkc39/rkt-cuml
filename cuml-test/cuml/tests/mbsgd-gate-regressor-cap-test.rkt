#lang racket/base

(require (only-in racket/match match-define)
         (only-in rackunit check-not-exn)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/linear-data.rkt" dtypes law-case)
         (only-in "private/mbsgd-data.rkt" normal-rows reg-target)
         (only-in "private/mbsgd-gate-data.rkt"
                  grid
                  grid-y
                  refused-at-eta0
                  regressor-fit
                  wide-problem))

(define normal (normal-rows 512 6 3))
(define normal-targets (reg-target normal 3 #:noise 0.5))
(define small (normal-rows 512 6 5))
(define small-targets
  (for/list ([t (in-list (reg-target small 5 #:noise 0.5))])
    (* 1e-3 t)))

(define (one-epoch X y dt alpha)
  (lambda (eta0)
    (regressor-fit X
                   y
                   dt
                   #:penalty 'elasticnet
                   #:l1-ratio 0.5
                   #:alpha alpha
                   #:eta0 eta0
                   #:batch-size 512
                   #:epochs 1)))

(define cap-rows
  `((good ,normal ,normal-targets 10.0 0.01 0.073) (small-y ,small ,small-targets 0.001 1.0 0.61)))

(test-gpu "a fit needing less than a quarter of the zero model's objective passes"
  (for* ([dt (in-list dtypes)]
         [r (in-list cap-rows)]
         #:when (eq? (car r) 'good))
    (match-define (list name X y alpha eta0 _) r)
    (law-case (format "~a ~a" dt name)
      (check-not-exn ((one-epoch X y dt alpha) eta0)))))

(test-gpu "a fit needing more than a quarter of the zero model's objective is refused"
  (for* ([dt (in-list dtypes)]
         [r (in-list cap-rows)]
         #:unless (eq? (car r) 'good))
    (match-define (list name X y alpha eta0 _) r)
    (law-case (format "~a ~a" dt name)
      (refused-at-eta0 (one-epoch X y dt alpha) eta0))))

(define-values (wide wide-targets) (wide-problem 200 4000))

(test-gpu "a wide lasso whose oscillation outgrows a quarter of the zero model's objective passes"
  (for ([dt (in-list dtypes)])
    (law-case (format "~a 200 rows, 4000 columns, l1 alpha 1" dt)
      (check-not-exn (regressor-fit wide wide-targets dt #:penalty 'l1 #:alpha 1)))))

(define wide-rises '((float32 0.582 3.09) (float64 0.592 3.04)))

(test-gpu "a wide lasso rising past the widened cap is refused although it beats the zero model"
  (for ([r (in-list wide-rises)])
    (match-define (list dt _ _) r)
    (law-case (format "~a 200 rows, 4000 columns, l1 alpha 1" dt)
      (refused-at-eta0 (lambda (eta0)
                         (regressor-fit wide wide-targets dt #:penalty 'l1 #:alpha 1 #:eta0 eta0))
                       3e-3))))

(test-gpu "an alpha whose uncapped allowance overflows double is still gated"
  (law-case "l1 alpha 1e200"
    (refused-at-eta0
     (lambda (eta0)
       (regressor-fit grid grid-y 'float64 #:penalty 'l1 #:alpha 1e200 #:eta0 eta0 #:epochs 1))
     1e-3)))
