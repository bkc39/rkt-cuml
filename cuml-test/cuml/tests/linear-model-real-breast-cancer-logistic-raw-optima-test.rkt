#lang racket/base

(require (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/real-logistic-data.rkt" check-raw-optima raw-breast-cancer))

(test-gpu "each pinned raw optimum is stationary"
  (check-raw-optima raw-breast-cancer))
