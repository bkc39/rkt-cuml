#lang racket/base

(require (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/real-logistic-data.rkt" check-raw-rescued raw-breast-cancer))

(define float64-elasticnet-cap 1e-1)

(define capped
  `((float64 (elasticnet 1.0 0.5) 4000 accepted ,float64-elasticnet-cap)
    (float32 (elasticnet 1.0 0.5) 4000 either #f)))

(test-gpu "raw elastic net at max-iter 4000: within the cap, or refused"
  (check-raw-rescued raw-breast-cancer capped '()))
