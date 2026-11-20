#lang racket/base

(require (only-in cuml/linear-model mbsgd-regressor-fit)
         (only-in racket/list make-list)
         (only-in rackunit check-not-exn)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/common-data.rkt" on-vector)
         (only-in "private/linear-data.rkt" dtypes law-case on-matrix)
         (only-in "private/mbsgd-data.rkt" normal-rows)
         (only-in "private/mbsgd-gate-data.rkt" rose)
         (only-in "private/refusal-rows.rkt" check-row row))

(define constant-rows (normal-rows 300 4 3))

(test-gpu "a constant target passes: its zero-model objective is round-off"
  (for* ([dt (in-list dtypes)]
         [value (in-list '(0.3 1.7 1e6 -1e6 3e7 1e9))])
    (law-case (format "~a y = ~a" dt value)
      (check-not-exn (lambda ()
                       (mbsgd-regressor-fit (on-matrix constant-rows dt)
                                            (on-vector (make-list 300 value) dt)))))))

(define offset-targets
  (for/list ([r (in-list constant-rows)]
             [e (in-list (normal-rows 300 1 9))])
    (+ 1e6 (* 0.05 (car r)) (* 0.05 (car e)))))

(define (offset-fit dt eta0)
  (lambda ()
    (mbsgd-regressor-fit (on-matrix constant-rows dt)
                         (on-vector offset-targets dt)
                         #:batch-size 1
                         #:eta0 eta0)))

(test-gpu "a rise above y's round-off is refused however large y's offset"
  (for ([eta0 (in-list '(0.1 0.2))])
    (law-case (format "float64 eta0 ~a" eta0)
      (check-row
       "mbsgd-regressor-fit"
       (row "y = 1e6 + .05·x + .05·e" (rose (number->string eta0)) (offset-fit 'float64 eta0)))
      (check-not-exn (offset-fit 'float64 (/ eta0 10)))))
  (law-case "float32 eta0 0.2, whose residuals are float32 round-off of 1e6"
    (check-not-exn (offset-fit 'float32 0.2))))
