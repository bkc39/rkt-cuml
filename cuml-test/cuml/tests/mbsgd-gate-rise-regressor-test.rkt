#lang racket/base

(require (only-in cuml/linear-model mbsgd-regressor-fit)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/linear-data.rkt" law-case on-matrix on-vector)
         (only-in "private/mbsgd-data.rkt" normal-rows reg-target)
         (only-in "private/mbsgd-gate-data.rkt" rose)
         (only-in "private/refusal-rows.rkt" check-row row))

(define near-zero (normal-rows 512 6 42))
(define near-zero-targets (reg-target near-zero 42))

(test-gpu "a regressor fit ending above the zero model is refused"
  (define X (on-matrix near-zero))
  (define y (on-vector near-zero-targets))
  (law-case "regressor"
    (check-row "mbsgd-regressor-fit"
               (row "eta0 0.05"
                    (rose "0.05")
                    (lambda () (mbsgd-regressor-fit X y #:batch-size 1 #:alpha 10 #:eta0 0.05))))))
