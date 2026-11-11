#lang racket/base

(require (only-in "private/real-linear-data.rkt" test-real-set))

(test-real-set 'breast-cancer #:checks '(elastic-net) #:dtypes '(float64) #:arms '((raw #f) (raw #t)))
