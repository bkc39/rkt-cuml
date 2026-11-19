#lang racket/base

(require (only-in racket/list make-list)
         (only-in rackunit check-not-exn)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/linear-data.rkt" dtypes law-case)
         (only-in "private/mbsgd-data.rkt" normal-rows)
         (only-in "private/mbsgd-gate-data.rkt" refused-at-eta0 regressor-fit))

(define constant-rows (normal-rows 300 4 3))

(test-gpu "a constant target passes at the defaults under l1 and elastic net"
  (for* ([dt (in-list dtypes)]
         [penalty (in-list '(l1 elasticnet))]
         [value (in-list '(0.3 1.7 1e6))])
    (law-case (format "~a ~a y = ~a" dt penalty value)
      (check-not-exn (regressor-fit constant-rows (make-list 300 value) dt #:penalty penalty)))))

(define near-constant-targets
  (for/list ([e (in-list (normal-rows 300 1 9))])
    (+ 1.7 (* 1e-6 (car e)))))

(test-gpu "a float32 target constant to within 1e-6 passes at the defaults under l1 and elastic net"
  (for ([penalty (in-list '(l1 elasticnet))])
    (law-case (format "float32 ~a" penalty)
      (check-not-exn
       (regressor-fit constant-rows near-constant-targets 'float32 #:penalty penalty)))))

(test-gpu "a float64 target constant to within 1e-6 is refused at the defaults under l1 and elastic net"
  (for ([penalty (in-list '(l1 elasticnet))])
    (law-case (format "float64 ~a" penalty)
      (refused-at-eta0
       (lambda (eta0)
         (regressor-fit constant-rows near-constant-targets 'float64 #:penalty penalty #:eta0 eta0))
       1e-3))))
