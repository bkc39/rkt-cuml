#lang racket/base

(require (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/real-logistic-data.rkt" check-raw-rescued labelled-set))

(define float64-elasticnet-cap 1e-1)
(define float32-elasticnet-cap 1.7e-1)

(define capped
  `((float64 elasticnet 1.0 0.5 4000 ,float64-elasticnet-cap)
    (float32 elasticnet 1.0 0.5 4000 ,float32-elasticnet-cap)))

(test-gpu "raw breast-cancer: an elastic-net fit at max-iter 4000 is refused or within its cap"
  (check-raw-rescued (labelled-set 'breast-cancer "diagnosis" '("id")) capped '()))
