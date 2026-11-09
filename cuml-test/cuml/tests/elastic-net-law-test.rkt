#lang racket/base

(require (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/elastic-net-data.rkt" check-elastic-net-laws correlated-case))

(test-gpu "with an intercept, fits meet the certified rational oracle's bounds (S1-S3)"
  (check-elastic-net-laws (list (correlated-case 20 2 0 0)
                                (correlated-case 200 7 0.9 1 #:weighted? #t)
                                (correlated-case 200 7 0.99 2))
                          '(#t)))
