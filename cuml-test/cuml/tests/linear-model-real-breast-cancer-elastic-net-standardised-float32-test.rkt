#lang racket/base

(require (only-in "private/real-linear-data.rkt" test-real-set))

(test-real-set 'breast-cancer
               #:checks '(elastic-net)
               #:dtypes '(float32)
               #:arms '((standardised #f) (standardised #t)))
