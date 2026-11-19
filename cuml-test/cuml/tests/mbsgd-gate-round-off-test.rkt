#lang racket/base

(require (only-in cuml/linear-model mbsgd-regressor-result-coef mbsgd-regressor-result-intercept)
         (only-in racket/list make-list)
         (only-in racket/match match-define)
         (only-in rackunit check-not-exn check-true)
         (only-in raft device-vector->list)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/linear-data.rkt" dtypes law-case)
         (only-in "private/mbsgd-data.rkt" normal-rows)
         (only-in "private/mbsgd-gate-data.rkt"
                  constant-rows
                  refused-at-eta0
                  regressor-fit
                  round-off-coef-steps
                  round-off-intercept-share))

(define (fits-the-constant? fit c share)
  (and (<= (abs (- (mbsgd-regressor-result-intercept fit) c)) (* round-off-intercept-share (abs c)))
       (for/and ([w (in-list (device-vector->list (mbsgd-regressor-result-coef fit)))])
         (<= (abs w) (* round-off-coef-steps 1e-3 1e-4 share)))))

(test-gpu "a constant target passes at the defaults under l1 and elastic net, fitting the constant"
  (for* ([dt (in-list dtypes)]
         [penalty (in-list '((l1 1) (elasticnet 0.15)))]
         [value (in-list '(0.3 1.7 1e6))])
    (match-define (list name share) penalty)
    (law-case (format "~a ~a y = ~a" dt name value)
      (check-true (fits-the-constant?
                   ((regressor-fit constant-rows (make-list 300 value) dt #:penalty name))
                   value
                   share)))))

(define offset-noise (map car (normal-rows 300 1 9)))

(define near-constant-targets
  (for/list ([e (in-list offset-noise)])
    (+ 1.7 (* 1e-6 e))))

(test-gpu "a float32 target constant to within 1e-6 passes at the defaults under elastic net"
  (check-not-exn (regressor-fit constant-rows near-constant-targets 'float32 #:penalty 'elasticnet)))

(define near-constant-refusals
  '((float32 l1 33.2 -0.078) (float64 l1 33.4 -0.054) (float64 elasticnet 0.42 -0.003)))

(test-gpu "a target constant to within 1e-6 is refused at the defaults where its rise exceeds the cap"
  (for ([r (in-list near-constant-refusals)])
    (match-define (list dt penalty _ _) r)
    (law-case (format "~a ~a" dt penalty)
      (refused-at-eta0
       (lambda (eta0)
         (regressor-fit constant-rows near-constant-targets dt #:penalty penalty #:eta0 eta0))
       1e-3))))
