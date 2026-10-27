#lang racket/base

(require (only-in "private/python-env.rkt" test-twin)
         (only-in "private/spectral-embedding-determinism-data.rkt" check-se-repeatable dense-configs))

(test-twin "L4: dense fits on digits repeat bitwise in and across processes, at 10 and 256 neighbours"
  (check-se-repeatable (dense-configs 'digits)))
