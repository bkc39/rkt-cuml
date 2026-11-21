#lang racket/base

(require (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/raw-breast-cancer.rkt" raw-breast-cancer)
         (only-in "private/real-logistic-data.rkt" check-raw-rescued))

(define refused
  '((float64 (l2 0.01 #f) 1000) (float64 (l1 1.0 #f) 1000)
                                (float64 (elasticnet 1.0 0.5) 1000)
                                (float32 (l1 1.0 #f) 1000)
                                (float32 (elasticnet 1.0 0.5) 1000)))

(test-gpu "raw penalised fits at the default max-iter: refused"
  (check-raw-rescued raw-breast-cancer '() refused))
