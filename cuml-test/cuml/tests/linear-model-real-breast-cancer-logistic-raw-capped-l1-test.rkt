#lang racket/base

(require (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/real-logistic-data.rkt" check-raw-rescued labelled-set))

(define float64-l1-cap 1.5e-1)
(define float32-l1-cap 2.7e-1)

(define capped `((float64 l1 1.0 #f 4000 ,float64-l1-cap) (float32 l1 1.0 #f 4000 ,float32-l1-cap)))

(test-gpu "raw breast-cancer: an l1 fit at max-iter 4000 is refused or within its cap"
  (check-raw-rescued (labelled-set 'breast-cancer "diagnosis" '("id")) capped '()))
