#lang racket/base

(require (only-in "private/pca-twin-data.rkt" check-against-cuml named)
         (only-in "private/python-env.rkt" test-twin))

(test-twin "uniform integers is cuML-Python's within the L14 bounds"
  (check-against-cuml (named "uniform integers")))
