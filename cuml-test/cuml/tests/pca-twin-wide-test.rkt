#lang racket/base

(require (only-in "private/pca-twin-data.rkt" check-against-sklearn named)
         (only-in "private/python-env.rkt" test-twin))

(for ([name (in-list '("uniform integers" "wide"))])
  (test-twin (format "L12, L14: ~a is sklearn's within the analytic bounds" name)
    (check-against-sklearn (named name))))
