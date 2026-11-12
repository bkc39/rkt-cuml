#lang racket/base

(require (only-in "private/real-linear-data.rkt" test-real-set))

(test-real-set 'breast-cancer
               #:arms '((offset . unweighted) (offset . weighted))
               #:checks '(lstsq))
