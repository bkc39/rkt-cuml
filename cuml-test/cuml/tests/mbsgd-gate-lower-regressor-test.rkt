#lang racket/base

(require (only-in cuml/linear-model mbsgd-regressor-fit)
         (only-in rackunit check-not-exn)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/linear-data.rkt" law-case on-matrix on-vector)
         (only-in "private/mbsgd-data.rkt" normal-rows reg-target))

(define near-zero (normal-rows 512 6 42))
(define near-zero-targets (reg-target near-zero 42))

(test-gpu "a regressor fit at the advised lower eta0 passes"
  (define X (on-matrix near-zero))
  (define y (on-vector near-zero-targets))
  (law-case "regressor"
    (check-not-exn (lambda () (mbsgd-regressor-fit X y #:batch-size 1 #:alpha 10 #:eta0 0.005)))))
