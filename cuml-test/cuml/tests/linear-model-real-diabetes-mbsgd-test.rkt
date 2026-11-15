#lang racket/base

(require (only-in racket/match match-define)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/linear-data.rkt" dtypes law-case)
         (only-in "private/python-env.rkt" test-twin)
         (only-in "private/real-mbsgd-data.rkt" check-diabetes-gap check-diabetes-twin diabetes))

(test-twin "diabetes as shipped, defaults: Python cuML's fit, gated through, far from the optimum"
  (define-values (X y) (diabetes))
  (for ([dt (in-list dtypes)])
    (law-case (format "~a" dt)
      (check-diabetes-twin X y dt)
      (check-diabetes-gap X y dt (hash) 0.6 0.8))))

(test-gpu "diabetes standardised: the fit is near the ridge optimum"
  (define-values (X y) (diabetes #:standardise? #t))
  (for* ([dt (in-list dtypes)]
         [arm (in-list (list (list "eta0 .01" (hash '#:eta0 0.01) 5e-4)
                             (list "defaults" (hash) 1.5e-2)
                             (list "adaptive eta0 .01" (hash '#:learning-rate 'adaptive '#:eta0 0.01) 5e-4)))])
    (match-define (list name settings bound) arm)
    (law-case (format "~a ~a" dt name)
      (check-diabetes-gap X y dt settings 0.0 bound))))
