#lang racket/base

(require (only-in cuml/tests/private/linear-data dtypes)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/gaussian-glmnet-data.rkt" check-gaussian law-trials))

(test-gpu "lasso, elastic-net and ridge fits on the 200x7 draws are glmnet's"
  (for* ([dt (in-list dtypes)]
         [t (in-list (law-trials '((200 7))))])
    (check-gaussian t dt)))
