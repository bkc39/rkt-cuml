#lang racket/base

(require (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/elastic-net-data.rkt" check-elastic-net-laws correlated-case))

(test-gpu "without an intercept, fits meet the certified rational oracle's bounds (S1-S2)"
  (check-elastic-net-laws (list (correlated-case 20 7 0.9 3 #:weighted? #t)
                                (correlated-case 2000 30 0 4))
                          '(#f)
                          #:tols '(1e-2 1e-3)))
