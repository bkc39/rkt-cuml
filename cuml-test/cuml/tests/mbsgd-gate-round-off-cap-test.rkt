#lang racket/base

(require (only-in racket/list make-list)
         (only-in racket/match match-define)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/linear-data.rkt" law-case)
         (only-in "private/mbsgd-gate-data.rkt"
                  at-eta0
                  constant-rows
                  linear-targets
                  off-constant-at-eta0
                  refused-at-eta0))

(define constant-rises
  '((float64 100.0 0.1 constant 32) (float64 1000.0 0.1 constant 32)
                                    (float64 1e6 1e-3 constant 32)
                                    (float64 1e6 1e-3 invscaling 32)
                                    (float64 1e10 1e-3 constant 32)
                                    (float64 1e100 1e-3 constant 32)
                                    (float64 1e150 1e-3 constant 32)
                                    (float32 100.0 0.1 constant 32)
                                    (float32 1000.0 0.1 constant 32)
                                    (float32 1e6 1e-3 constant 32)
                                    (float32 1e6 1e-3 invscaling 32)
                                    (float32 1e10 1e-3 constant 32)
                                    (float32 1e30 1e-3 constant 32)
                                    (float64 0.51 0.3 constant 32)
                                    (float64 0.51 0.1 constant 32)
                                    (float32 0.51 0.3 constant 32)
                                    (float32 0.51 0.1 constant 32)))

(test-gpu "a constant target's l1 fit away from the constant is refused"
  (for ([r (in-list constant-rises)])
    (match-define (list dt alpha row-eta0 schedule batch) r)
    (law-case (format "~a y = 1.7 l1 alpha ~a ~a batch ~a" dt alpha schedule batch)
      (off-constant-at-eta0 (at-eta0 constant-rows
                                     (make-list 300 1.7)
                                     dt
                                     #:penalty 'l1
                                     #:alpha alpha
                                     #:learning-rate schedule
                                     #:batch-size batch)
                            row-eta0))))

(define offset-rises
  '((10.0 0.01 constant) (100.0 0.1 constant) (1e4 0.1 constant) (100.0 1.0 invscaling)))

(test-gpu "a float32 target with a real spread far from 0 is gated as it is at 0"
  (for* ([r (in-list offset-rises)]
         [offset (in-list '(0.0 1e6))])
    (match-define (list alpha row-eta0 schedule) r)
    (law-case (format "offset ~a l1 alpha ~a eta0 ~a ~a" offset alpha row-eta0 schedule)
      (refused-at-eta0 (at-eta0 constant-rows
                                (linear-targets offset 1.5 0.5)
                                'float32
                                #:penalty 'l1
                                #:alpha alpha
                                #:learning-rate schedule)
                       row-eta0))))
