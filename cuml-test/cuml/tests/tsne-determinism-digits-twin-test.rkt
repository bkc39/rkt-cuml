#lang racket/base

(require (only-in "private/python-env.rkt" test-twin)
         (only-in "private/tsne-determinism-data.rkt" check-tsne-repeatable))

(test-twin "L4: FFT fits on digits repeat bitwise in and across processes"
  (check-tsne-repeatable 'digits '(euclidean)))
