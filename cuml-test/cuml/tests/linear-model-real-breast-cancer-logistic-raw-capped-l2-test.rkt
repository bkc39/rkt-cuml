#lang racket/base

(require (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/real-logistic-data.rkt" check-raw-rescued labelled-set))

(define capped '((float32 l2 0.01 #f 1000) (float64 l2 0.01 #f 4000) (float32 l2 0.01 #f 4000)))

(test-gpu "raw breast-cancer: an l2 fit the rescue accepts is within S4's cap of the optimum"
  (check-raw-rescued (labelled-set 'breast-cancer "diagnosis" '("id")) capped '()))
