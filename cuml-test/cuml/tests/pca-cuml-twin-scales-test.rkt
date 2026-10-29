#lang racket/base

(require (only-in "private/pca-twin-data.rkt" check-against-cuml named)
         (only-in "private/python-env.rkt" test-twin))

(for ([name (in-list '("scales" "uniform integers"))])
  (test-twin (format "L15: ~a is cuML-Python's within the L14 bounds" name)
    (check-against-cuml (named name))))
