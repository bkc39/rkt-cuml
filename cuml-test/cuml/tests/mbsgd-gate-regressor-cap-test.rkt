#lang racket/base

(require (only-in cuml/linear-model mbsgd-regressor-fit)
         (only-in rackunit check-not-exn)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/linear-data.rkt" dtypes law-case on-matrix on-vector)
         (only-in "private/mbsgd-data.rkt" normal-rows reg-target)
         (only-in "private/mbsgd-gate-data.rkt" rose)
         (only-in "private/refusal-rows.rkt" check-row row))

(define normal (normal-rows 512 6 3))
(define normal-targets (reg-target normal 3 #:noise 0.5))
(define small (normal-rows 512 6 5))
(define small-targets
  (for/list ([t (in-list (reg-target small 5 #:noise 0.5))])
    (* 1e-3 t)))

(define grid '((1 0 2) (0 3 1) (4 1 0) (2 2 5) (3 1 1) (0 0 1)))
(define grid-y '(1 2 3 4 5 6))

(define (one-epoch X y dt alpha eta0)
  (lambda ()
    (mbsgd-regressor-fit (on-matrix X dt)
                         (on-vector y dt)
                         #:penalty 'elasticnet
                         #:l1-ratio 0.5
                         #:alpha alpha
                         #:eta0 eta0
                         #:batch-size 512
                         #:epochs 1)))

(test-gpu "the regressor's allowance is capped at a quarter of the zero model's objective"
  (for ([dt (in-list dtypes)])
    (law-case (format "~a good fit needing 0.073 of it passes" dt)
      (check-not-exn (one-epoch normal normal-targets dt 10.0 0.01)))
    (law-case (format "~a small-y fit needing 0.61 of it is refused" dt)
      (check-row "mbsgd-regressor-fit"
                 (row "eta0 1" (rose "1") (one-epoch small small-targets dt 0.001 1.0))))))

(test-gpu "an alpha whose uncapped allowance overflows double is still gated"
  (law-case "l1 alpha 1e200"
    (check-row "mbsgd-regressor-fit"
               (row "eta0 0.001"
                    (rose "0.001")
                    (lambda ()
                      (mbsgd-regressor-fit (on-matrix grid)
                                           (on-vector grid-y)
                                           #:penalty 'l1
                                           #:alpha 1e200
                                           #:epochs 1))))))
