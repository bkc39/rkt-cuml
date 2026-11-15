#lang racket/base

(require (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/real-mbsgd-data.rkt" check-breast-cancer))

(test-gpu "float32: standardised breast-cancer string labels classify as accurately as measured"
  (check-breast-cancer 'float32))
