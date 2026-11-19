#lang racket/base

(require (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/raw-breast-cancer.rkt" raw-breast-cancer)
         (only-in "private/real-logistic-data.rkt" check-raw-rescued))

(define float64-l1-cap 1.5e-1)
(define float32-l1-cap 2.7e-1)

(define capped
  `((float64 (l1 1.0 #f) 4000 accepted ,float64-l1-cap)
    (float32 (l1 1.0 #f) 4000 either ,float32-l1-cap)))

(test-gpu "raw l1 at max-iter 4000: within the cap, or refused"
  (check-raw-rescued raw-breast-cancer capped '()))
