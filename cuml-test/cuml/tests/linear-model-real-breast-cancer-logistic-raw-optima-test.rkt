#lang racket/base

(require (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/raw-breast-cancer.rkt" raw-breast-cancer)
         (only-in "private/real-logistic-data.rkt" check-raw-optima))

(test-gpu "each pinned raw optimum is stationary"
  (check-raw-optima raw-breast-cancer))
