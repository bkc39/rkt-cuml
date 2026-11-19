#lang racket/base

(require (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/real-logistic-data.rkt" check-raw-rescued labelled-set))

(define float32-l2-1000-cap 1.7e-3)
(define float64-l2-4000-cap 9e-4)
(define float32-l2-4000-cap 3.6e-3)

(define capped
  `((float32 l2 0.01 #f 1000 ,float32-l2-1000-cap) (float64 l2 0.01 #f 4000 ,float64-l2-4000-cap)
                                                   (float32 l2 0.01 #f 4000 ,float32-l2-4000-cap)))

(test-gpu "raw breast-cancer: an l2 fit at C 0.01 is refused or within its cap"
  (check-raw-rescued (labelled-set 'breast-cancer "diagnosis" '("id")) capped '()))
