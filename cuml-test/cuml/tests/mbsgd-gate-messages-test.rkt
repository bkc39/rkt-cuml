#lang racket/base

(require (only-in racket/list make-list)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/linear-data.rkt" law-case)
         (only-in "private/mbsgd-gate-data.rkt"
                  constant-rows
                  linear-targets
                  regressor-fit
                  wide-problem
                  zero-mean-rows)
         (only-in "private/refusal-rows.rkt" check-row row row-id))

(define-values (twelve twelve-targets) (wide-problem 12 8000))

(define constant (make-list 300 1.7))

(define message-rows
  (list
   (row "objective: float32 l1 alpha 100, eta0 0.1"
        "the objective rose from 2.34519 to 1979.87; lower #:eta0 (now 0.1)"
        (regressor-fit constant-rows
                       (linear-targets 0 1.5 0.5)
                       'float32
                       #:penalty 'l1
                       #:alpha 100
                       #:eta0 0.1
                       #:random-state 1))
   (row "loss: float32 12 rows, 8000 columns, l1 alpha 0.6"
        "the loss rose from 1.51981 to 2.11811; lower #:eta0 (now 0.001)"
        (regressor-fit twelve
                       twelve-targets
                       'float32
                       #:penalty 'l1
                       #:alpha 0.6
                       #:eta0 1e-3
                       #:random-state 1))
   (row
    "intercept: float64 y = 1.7, l1 alpha 0.51, eta0 0.3"
    "y is constant to round-off and the fit's intercept is off by 0.00370349; lower #:eta0 or #:alpha"
    (regressor-fit constant-rows
                   constant
                   'float64
                   #:penalty 'l1
                   #:alpha 0.51
                   #:eta0 0.3
                   #:random-state 1))
   (row
    "coefficient: float64 y = 1.7 on zero-mean X, l1 alpha 0.51, eta0 0.72, one batch"
    "y is constant to round-off and a coefficient of the fit reaches 0.750447; lower #:eta0 or #:alpha"
    (regressor-fit zero-mean-rows
                   constant
                   'float64
                   #:penalty 'l1
                   #:alpha 0.51
                   #:eta0 0.72
                   #:batch-size 300))))

(test-gpu "each regressor gate refusal names its values in full"
  (for ([r (in-list message-rows)])
    (law-case (row-id r)
      (check-row "mbsgd-regressor-fit" r))))
