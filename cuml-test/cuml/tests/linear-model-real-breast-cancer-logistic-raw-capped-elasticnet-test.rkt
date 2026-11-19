#lang racket/base

(require (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/real-logistic-data.rkt" check-raw-rescued labelled-set))

(define float64-elasticnet-cap 1e-1)

(define capped
  `((float64 elasticnet 1.0 0.5 4000 accepted ,float64-elasticnet-cap)
    (float32 elasticnet 1.0 0.5 4000 either #f)))

(test-gpu "raw breast-cancer elastic net at max-iter 4000: float64 within its cap, float32 refused or inside the check's bound"
  (check-raw-rescued (labelled-set 'breast-cancer "diagnosis" '("id")) capped '()))
