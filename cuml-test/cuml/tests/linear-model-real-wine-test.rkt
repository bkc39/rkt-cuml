#lang racket/base

(require (only-in "private/real-linear-data.rkt" test-real-set))

(test-real-set 'wine #:checks '(lstsq ridge elastic-net score refusal))
