#lang racket/base

(require (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/real-linear-data.rkt" check-real-refusal real-cases test-real-set))

(test-real-set 'wine)

(test-gpu "wine: l1-ratio 0 at the ridge path's start is refused at the default tol, exact at a tight one"
  (for-each check-real-refusal (real-cases 'wine)))
