#lang racket/base

(require (only-in "private/real-linear-data.rkt" test-real-set))

(test-real-set 'diabetes #:checks '(elastic-net))
