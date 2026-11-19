#lang racket/base

(require (only-in racket/list make-list)
         (only-in racket/match match-define)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/linear-data.rkt" law-case)
         (only-in "private/mbsgd-data.rkt" normal-rows)
         (only-in "private/mbsgd-gate-data.rkt" constant-rows refused-at-eta0 regressor-fit))

(define offset-noise (map car (normal-rows 300 1 9)))

(define constant-rises
  '((float64 l1 100.0 0.1 constant 32 10.9) (float64 l1 1000.0 0.1 constant 32 109.0)
                                            (float64 l1 1e6 1e-3 constant 32 984.0)
                                            (float64 l1 1e6 1e-3 invscaling 32 20.4)
                                            (float64 l1 1e10 1e-3 constant 32 9.84e6)
                                            (float64 l1 1e100 1e-3 constant 32 9.84e96)
                                            (float64 l1 1e150 1e-3 constant 32 9.84e146)
                                            (float32 l1 100.0 0.1 constant 32 10.9)
                                            (float32 l1 1000.0 0.1 constant 32 109.0)
                                            (float32 l1 1e6 1e-3 constant 32 984.0)
                                            (float32 l1 1e6 1e-3 invscaling 32 9.99)
                                            (float32 l1 1e10 1e-3 constant 32 9.84e6)
                                            (float32 l1 1e30 1e-3 constant 32 9.76e26)))

(test-gpu "a constant target's oscillation is capped at a quarter of its mean square"
  (for ([r (in-list constant-rises)])
    (match-define (list dt penalty alpha row-eta0 schedule batch _) r)
    (law-case (format "~a y = 1.7 ~a alpha ~a ~a batch ~a" dt penalty alpha schedule batch)
      (refused-at-eta0 (lambda (eta0)
                         (regressor-fit constant-rows
                                        (make-list 300 1.7)
                                        dt
                                        #:penalty penalty
                                        #:alpha alpha
                                        #:eta0 eta0
                                        #:learning-rate schedule
                                        #:batch-size batch))
                       row-eta0))))

(define (offset-targets offset)
  (for/list ([r (in-list constant-rows)]
             [e (in-list offset-noise)])
    (+ offset (* 1.5 (car r)) (* 0.5 e))))

(define offset-rises
  '((10.0 0.01 constant 0.044) (100.0 0.1 constant -70.8)
                               (1e4 0.1 constant -7.16e5)
                               (100.0 1.0 invscaling -1.85)))

(test-gpu "a float32 target with a real spread far from 0 is gated as it is at 0"
  (for* ([r (in-list offset-rises)]
         [offset (in-list '(0.0 1e6))])
    (match-define (list alpha row-eta0 schedule _) r)
    (law-case (format "offset ~a l1 alpha ~a eta0 ~a ~a" offset alpha row-eta0 schedule)
      (refused-at-eta0 (lambda (eta0)
                         (regressor-fit constant-rows
                                        (offset-targets offset)
                                        'float32
                                        #:penalty 'l1
                                        #:alpha alpha
                                        #:eta0 eta0
                                        #:learning-rate schedule))
                       row-eta0))))
