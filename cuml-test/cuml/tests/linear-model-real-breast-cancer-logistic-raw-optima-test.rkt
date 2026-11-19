#lang racket/base

(require (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/real-logistic-data.rkt" check-raw-optima labelled-set))

(test-gpu "raw breast-cancer: each pinned float64 optimum is stationary"
  (check-raw-optima (labelled-set 'breast-cancer "diagnosis" '("id"))))
