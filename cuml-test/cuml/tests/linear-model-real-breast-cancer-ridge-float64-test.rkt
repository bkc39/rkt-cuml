#lang racket/base

(require (only-in "private/real-linear-data.rkt" test-real-set))

(test-real-set 'breast-cancer #:checks '(ridge) #:dtypes '(float64))
