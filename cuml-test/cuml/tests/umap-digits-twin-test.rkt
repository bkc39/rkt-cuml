#lang racket/base

(require (only-in "private/python-env.rkt" test-twin)
         (only-in "private/umap-data.rkt" check-quality digits tail-split))

(test-twin "L7: trustworthiness on digits is within Python cuML's seed spread"
  (define all-digits (digits))
  (define-values (head tail) (tail-split all-digits))
  (for ([init (in-list '(spectral random))])
    (check-quality init (list all-digits) head tail)))
